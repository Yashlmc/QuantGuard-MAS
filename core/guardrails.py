"""
QuantGuard-MAS — Unit I: Simple Reflex Guardrails, Utility Scorer & Safety Sanitizer
"""

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List


INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions", re.I),
    re.compile(r"bypass\s+(?:all\s+)?(?:guardrails?|limits?|circuit\s*breaker)", re.I),
    re.compile(r"disable\s+(?:risk|safety|hitl|human\s+approval)", re.I),
    re.compile(r"execute\s+(?:automatically|without\s+approval)", re.I),
    re.compile(r"system\s*prompt\s*:", re.I),
]


class PromptInjectionSanitizer:
    """Deterministic pre-LLM filter flagging indirect/direct prompt-injection attempts."""

    @classmethod
    def scan(cls, text: str) -> List[str]:
        alerts = []
        if not isinstance(text, str):
            return alerts
        for pat in INJECTION_PATTERNS:
            m = pat.search(text)
            if m:
                alerts.append(f"Blocked suspicious directive: '{m.group(0)}'")
        return alerts


@dataclass
class GuardrailVerdict:
    approved: bool
    violations: List[str] = field(default_factory=list)
    adjusted_weights: Dict[str, float] = field(default_factory=dict)


class ReflexCircuitBreaker:
    """
    Unit I — Simple Reflex Agent Guardrail.
    Immediately blocks any allocation violating hard quantitative risk caps.
    """

    def __init__(
        self,
        max_single_asset_weight: float = 0.35,
        min_cash_weight: float = 0.10,
        max_drawdown_limit: float = 0.15,
        weight_sum_tolerance: float = 0.02,
    ):
        self.max_single_asset_weight = float(max_single_asset_weight)
        self.min_cash_weight = float(min_cash_weight)
        self.max_drawdown_limit = float(max_drawdown_limit)
        self.weight_sum_tolerance = float(weight_sum_tolerance)

    def check(
        self,
        target_weights: Dict[str, float],
        current_drawdown: float = 0.0,
    ) -> GuardrailVerdict:
        violations = []

        if not isinstance(target_weights, dict) or not target_weights:
            return GuardrailVerdict(False, ["Target allocation is empty."])

        if current_drawdown >= self.max_drawdown_limit:
            violations.append(
                f"Circuit breaker tripped: drawdown {current_drawdown*100:.2f}% "
                f">= max limit {self.max_drawdown_limit*100:.2f}%."
            )

        total_w = 0.0
        for sym, w in target_weights.items():
            if not isinstance(w, (int, float)) or isinstance(w, bool) or not math.isfinite(w):
                violations.append(f"Invalid non-finite weight for {sym}.")
                continue
            if w < 0.0:
                violations.append(f"Short-selling disallowed: {sym} weight {w:.2f} < 0.")
            if sym != "CASH" and w > self.max_single_asset_weight + 1e-6:
                violations.append(
                    f"Single-stock cap breached: {sym} weight {w*100:.1f}% "
                    f"> {self.max_single_asset_weight*100:.1f}%."
                )
            total_w += float(w)

        if abs(total_w - 1.0) > self.weight_sum_tolerance:
            violations.append(
                f"Portfolio weights sum to {total_w:.4f} (must sum to 1.00)."
            )

        cash_w = float(target_weights.get("CASH", 0.0))
        if cash_w + 1e-6 < self.min_cash_weight:
            violations.append(
                f"Cash reserve too low: {cash_w*100:.1f}% < {self.min_cash_weight*100:.1f}%."
            )

        return GuardrailVerdict(
            approved=len(violations) == 0,
            violations=violations,
            adjusted_weights=self.enforce_safe_weights(target_weights) if violations else target_weights,
        )

    def enforce_safe_weights(self, target_weights: Dict[str, float]) -> Dict[str, float]:
        """Deterministically clamp weights to satisfy single-stock cap & min cash."""
        equities = {
            k: min(max(0.0, float(v)), self.max_single_asset_weight)
            for k, v in target_weights.items()
            if k != "CASH" and isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        cash = max(self.min_cash_weight, float(target_weights.get("CASH", self.min_cash_weight)))
        cash = min(cash, 1.0)
        eq_budget = max(0.0, 1.0 - cash)
        eq_sum = sum(equities.values())

        if eq_sum > 0 and eq_budget > 0:
            scale = eq_budget / eq_sum
            equities = {k: round(v * scale, 4) for k, v in equities.items()}
            # Re-clamp after scaling if any exceeded cap
            overflow = 0.0
            for k in list(equities.keys()):
                if equities[k] > self.max_single_asset_weight:
                    overflow += equities[k] - self.max_single_asset_weight
                    equities[k] = round(self.max_single_asset_weight, 4)
            cash = round(1.0 - sum(equities.values()), 4)
        else:
            equities = {}
            cash = 1.0

        result = {k: v for k, v in equities.items() if v > 0.0}
        result["CASH"] = round(max(0.0, cash), 4)
        return result


class UtilityScorer:
    """
    Unit I — Utility-Based Agent Scorer.
    U = Expected_Return - risk_aversion * Volatility - fee_penalty
    """

    def __init__(self, risk_aversion: float = 1.5, fee_rate: float = 0.001):
        self.risk_aversion = float(risk_aversion)
        self.fee_rate = float(fee_rate)

    def score(
        self,
        expected_return: float,
        volatility: float,
        turnover: float = 1.0,
    ) -> float:
        fee_penalty = self.fee_rate * abs(turnover)
        utility = expected_return - (self.risk_aversion * volatility) - fee_penalty
        return round(float(utility), 5)
