"""
QuantGuard-MAS
Multi-Provider API Key Manager with Automatic 429 Failover Rotation
"""

import os
from dotenv import load_dotenv

load_dotenv()


class APIKeyManager:

    SUPPORTED_PROVIDERS = ("groq", "gemini")

    def __init__(self):
        self.keys = {
            "groq": self._load_keys("GROQ"),
            "gemini": self._load_keys("GEMINI"),
        }

        self.active_keys = {
            "groq": int(os.getenv("GROQ_ACTIVE_KEY", "1")),
            "gemini": int(os.getenv("GEMINI_ACTIVE_KEY", "1")),
        }

        self._validate_active_keys()

    @staticmethod
    def _load_keys(prefix):
        available = {}
        for number in range(1, 21):
            value = os.getenv(f"{prefix}_API_KEY_{number}")
            if value and value.strip():
                available[number] = value.strip()

        # Fallback to unnumbered key if numbered keys are not defined
        if not available:
            single = os.getenv(f"{prefix}_API_KEY")
            if single and single.strip():
                available[1] = single.strip()

        return available

    def _validate_active_keys(self):
        for provider in self.SUPPORTED_PROVIDERS:
            if not self.keys[provider]:
                continue
            active = self.active_keys[provider]
            if active not in self.keys[provider]:
                # Fall back to first available key number cleanly
                self.active_keys[provider] = sorted(self.keys[provider].keys())[0]

    def get_key(self, provider, key_number=None):
        provider = provider.lower()
        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}")

        selected = (
            key_number if key_number is not None else self.active_keys[provider]
        )
        if selected not in self.keys[provider]:
            raise ValueError(f"Key #{selected} is not configured for {provider}")
        return self.keys[provider][selected]

    def get_key_number(self, provider, key_number=None):
        provider = provider.lower()
        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError("Unsupported provider")

        selected = (
            key_number if key_number is not None else self.active_keys[provider]
        )
        if selected not in self.keys[provider]:
            raise ValueError(f"Key #{selected} is not configured for {provider}")
        return selected

    def set_active_key(self, provider, number):
        provider = provider.lower()
        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError("Unsupported provider")
        if number not in self.keys[provider]:
            raise ValueError(
                f"Key number {number} is not configured for {provider}"
            )
        self.active_keys[provider] = number

    def rotate_key(self, provider):
        """Rotate to the next configured key number on rate-limit (HTTP 429)."""
        provider = provider.lower()
        key_nums = sorted(self.keys.get(provider, {}).keys())
        if len(key_nums) <= 1:
            return self.active_keys.get(provider, 1)
        curr = self.active_keys[provider]
        idx = key_nums.index(curr) if curr in key_nums else 0
        next_num = key_nums[(idx + 1) % len(key_nums)]
        self.active_keys[provider] = next_num
        return next_num

    def list_keys(self):
        return {
            provider: {
                "available_key_numbers": list(self.keys[provider].keys()),
                "active_key": self.active_keys[provider],
            }
            for provider in self.SUPPORTED_PROVIDERS
        }
