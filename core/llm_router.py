"""
QuantGuard-MAS
Universal Multi-Provider LLM Router (Groq + Google Gemini)
with Automatic 429 Key Failover, JSON Mode, and SQLite Usage Logging
"""

import time

from groq import Groq
from google import genai
from google.genai import types

from core.key_manager import APIKeyManager
from evaluation.usage_tracker import UsageTracker
from schemas.llm_response import LLMResponse


class LLMRouter:

    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

    def __init__(
        self,
        max_retries=2,
        retry_base_delay=2,
        retry_max_delay=10,
        enable_usage_tracking=True,
    ):
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")

        self.key_manager = APIKeyManager()
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self.enable_usage_tracking = enable_usage_tracking
        self.clients = {}
        self.usage_tracker = UsageTracker() if enable_usage_tracking else None

    def _get_client(self, provider, key_number=None):
        provider = provider.lower()
        selected_key = self.key_manager.get_key_number(provider, key_number)
        api_key = self.key_manager.get_key(provider, selected_key)

        cache_id = (provider, selected_key)
        if cache_id in self.clients:
            return self.clients[cache_id]

        if provider == "groq":
            client = Groq(api_key=api_key)
        elif provider == "gemini":
            client = genai.Client(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        self.clients[cache_id] = client
        return client

    @classmethod
    def _get_status_code(cls, error):
        status = getattr(error, "status_code", None)
        if status is None:
            status = getattr(error, "code", None)
        try:
            return int(status)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _is_retryable(cls, error):
        status = cls._get_status_code(error)
        if status in cls.RETRYABLE_STATUS_CODES:
            return True
        error_name = type(error).__name__.lower()
        return any(
            name in error_name
            for name in (
                "connectionerror",
                "connectionexception",
                "apiconnectionerror",
                "timeout",
                "timeoutexception",
            )
        )

    def _record_usage(
        self,
        provider,
        model,
        key_number,
        latency_ms,
        attempts,
        status,
        response=None,
        error=None,
    ):
        if not self.usage_tracker:
            return

        self.usage_tracker.record(
            provider=provider,
            model=model,
            key_number=key_number,
            input_tokens=response.input_tokens if response else 0,
            output_tokens=response.output_tokens if response else 0,
            total_tokens=response.total_tokens if response else 0,
            latency_ms=latency_ms,
            finish_reason=response.finish_reason if response else None,
            request_id=response.request_id if response else None,
            attempts=attempts,
            status=status,
            error_type=type(error).__name__ if error else None,
        )

    def generate(
        self,
        provider,
        model,
        system_prompt,
        user_prompt,
        temperature=None,
        max_output_tokens=None,
        reasoning_effort=None,
        key_number=None,
        json_mode=False,
    ):
        provider = provider.lower()
        if provider not in {"groq", "gemini"}:
            raise ValueError(f"Unsupported provider: {provider}")
        if not model or not model.strip():
            raise ValueError("A model name is required")
        if not system_prompt or not user_prompt:
            raise ValueError("System and user prompts are required")
        if temperature is not None and not 0 <= temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        if max_output_tokens is not None and max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if reasoning_effort is not None:
            if provider != "groq":
                raise ValueError("reasoning_effort is supported only on Groq")
            if reasoning_effort not in {"low", "medium", "high"}:
                raise ValueError("reasoning_effort must be low, medium, or high")

        selected_key = self.key_manager.get_key_number(provider, key_number)
        started = time.perf_counter()
        last_error = None

        for attempt in range(1, self.max_retries + 2):
            client = self._get_client(provider, selected_key)
            try:
                if provider == "groq":
                    result = self._generate_groq(
                        client,
                        model,
                        system_prompt,
                        user_prompt,
                        temperature,
                        max_output_tokens,
                        reasoning_effort,
                        json_mode=json_mode,
                    )
                else:
                    result = self._generate_gemini(
                        client,
                        model,
                        system_prompt,
                        user_prompt,
                        temperature,
                        max_output_tokens,
                        json_mode=json_mode,
                    )

                latency_ms = (time.perf_counter() - started) * 1000
                result.provider = provider
                result.model = model
                result.key_number = selected_key
                result.latency_ms = round(latency_ms, 2)
                result.attempts = attempt

                self._record_usage(
                    provider=provider,
                    model=model,
                    key_number=selected_key,
                    latency_ms=latency_ms,
                    attempts=attempt,
                    status="success",
                    response=result,
                )
                return result

            except Exception as error:
                last_error = error
                if attempt > self.max_retries or not self._is_retryable(error):
                    break

                # Automatically rotate API key on 429 rate limits when key_number wasn't pinned
                if self._get_status_code(error) == 429 and key_number is None:
                    selected_key = self.key_manager.rotate_key(provider)

                delay = min(
                    self.retry_base_delay * (2 ** (attempt - 1)),
                    self.retry_max_delay,
                )
                time.sleep(delay)

        latency_ms = (time.perf_counter() - started) * 1000
        self._record_usage(
            provider=provider,
            model=model,
            key_number=selected_key,
            latency_ms=latency_ms,
            attempts=min(self.max_retries + 1, attempt),
            status="failed",
            error=last_error,
        )
        raise last_error

    @staticmethod
    def _generate_groq(
        client,
        model,
        system_prompt,
        user_prompt,
        temperature,
        max_output_tokens,
        reasoning_effort,
        json_mode=False,
    ):
        params = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if temperature is not None:
            params["temperature"] = temperature
        if max_output_tokens is not None:
            params["max_completion_tokens"] = max_output_tokens
        if reasoning_effort is not None:
            params["reasoning_effort"] = reasoning_effort
        if json_mode:
            params["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**params)
        choice = response.choices[0]
        text = choice.message.content or ""
        if not text.strip():
            raise RuntimeError("Groq returned no text content")

        usage = response.usage
        return LLMResponse(
            text=text,
            provider="groq",
            model=model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            finish_reason=choice.finish_reason,
            request_id=response.id,
        )

    @staticmethod
    def _generate_gemini(
        client,
        model,
        system_prompt,
        user_prompt,
        temperature,
        max_output_tokens,
        json_mode=False,
    ):
        config_params = {"system_instruction": system_prompt}
        if temperature is not None:
            config_params["temperature"] = temperature
        if max_output_tokens is not None:
            config_params["max_output_tokens"] = max_output_tokens
        if json_mode:
            config_params["response_mime_type"] = "application/json"

        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(**config_params),
        )
        text = response.text or ""
        if not text.strip():
            raise RuntimeError("Gemini returned no text content")

        usage = response.usage_metadata
        in_tok = usage.prompt_token_count if usage and usage.prompt_token_count is not None else 0
        out_tok = usage.candidates_token_count if usage and usage.candidates_token_count is not None else 0
        tot_tok = usage.total_token_count if usage and usage.total_token_count is not None else in_tok + out_tok
        finish_reason = str(response.candidates[0].finish_reason) if response.candidates else None

        return LLMResponse(
            text=text,
            provider="gemini",
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            total_tokens=tot_tok,
            finish_reason=finish_reason,
            request_id=getattr(response, "response_id", None),
        )
