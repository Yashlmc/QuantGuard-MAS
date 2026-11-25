
from core.llm_router import LLMRouter

router = LLMRouter()

print("Testing explicit Groq key selection...")

try:

    result = router.generate(
        provider="groq",
        model="openai/gpt-oss-20b",
        system_prompt="You are a financial analyst.",
        user_prompt="Explain market volatility.",
        temperature=0.2,
        max_output_tokens=500,
        key_number=2
    )

    print("\nResponse:")
    print(result.text)

    print("\nSelected key:", result.key_number)
    print("Provider:", result.provider)
    print("Model:", result.model)
    print("Latency:", result.latency_ms)

except Exception as error:

    print("Test failed:")
    print(type(error).__name__, error)
