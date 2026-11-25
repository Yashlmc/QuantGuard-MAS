
from core.llm_router import LLMRouter

router = LLMRouter()

print("Testing upgraded Gemini router...\n")

try:

    result = router.generate(
        provider="gemini",
        model="gemini-3.6-flash",
        system_prompt=(
            "You are QuantGuard, a financial "
            "intelligence assistant."
        ),
        user_prompt=(
            "Explain portfolio diversification "
            "in two sentences."
        ),
        temperature=0.2,
        max_output_tokens=500
    )

    print("RESPONSE")
    print("=" * 50)
    print(result.text)

    print("\nMETADATA")
    print("=" * 50)

    print("Provider:", result.provider)
    print("Model:", result.model)
    print("Input tokens:", result.input_tokens)
    print("Output tokens:", result.output_tokens)
    print("Total tokens:", result.total_tokens)
    print("Latency:", result.latency_ms, "ms")
    print("Finish reason:", result.finish_reason)
    print("Attempts:", result.attempts)

except Exception as error:

    print("Gemini request failed:")
    print(type(error).__name__, error)
