
"""
QuantGuard-MAS
Standardized LLM Response
"""

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class LLMResponse:

    text: str

    provider: str
    model: str

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    latency_ms: float = 0.0

    finish_reason: Optional[str] = None
    request_id: Optional[str] = None

    key_number: Optional[int] = None
    attempts: int = 1

    def to_dict(self):
        return asdict(self)

    def __str__(self):
        return self.text
