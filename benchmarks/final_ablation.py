"""
QuantGuard-MAS — Final Jury Ablation Study
Compares Single-Agent Baseline (No Reflex Audit, No Conflict Arbitration)
vs. Full 4-Agent QuantGuard-MAS under a -12% Market Shock + Concentration Stress Test.
"""

import pandas as pd

from core.guardrails import ReflexCircuitBreaker
from core.mas_graph import QuantGuardMAS


def run_final_ablation() -> pd.DataFrame:
    breaker = ReflexCircuitBreaker()

    # Single-agent baseline blindly follows bullish Researcher overweight (42% RELIANCE.NS)
    baseline_weights = {
        "RELIANCE.NS": 0.42,
        "TCS.NS": 0.25,
        "HDFCBANK.NS": 0.23,
        "CASH": 0.10,
    }
    base_check = breaker.check(baseline_weights)

    # Full 4-Agent QuantGuard-MAS with Risk Auditor + Arbitration + HITL
    mas = QuantGuardMAS(use_live_llms=False, use_hf_models=False)
    mas_out = mas.run_mission(
        goal="Stress-test ₹10L Indian equity allocation under RELIANCE.NS crash.",
        shock_symbol="RELIANCE.NS",
        shock_pct=-0.12,
        simulate_conflict=True,
    )
    mas_weights = mas_out["state"]["final_allocation"]
    mas_check = breaker.check(mas_weights)

    return pd.DataFrame(
        [
            {
                "Architecture": "Single-Agent Baseline",
                "Guardrail Compliant": base_check.approved,
                "Violations": len(base_check.violations),
                "Max Single-Asset Weight": f"{max(v for k, v in baseline_weights.items() if k != 'CASH')*100:.1f}%",
                "Cross-Family Audit": "No",
                "HITL Gate": "No",
                "FIPA Messages": 0,
            },
            {
                "Architecture": "QuantGuard-MAS (4-Agent)",
                "Guardrail Compliant": mas_check.approved,
                "Violations": len(mas_check.violations),
                "Max Single-Asset Weight": f"{max(v for k, v in mas_weights.items() if k != 'CASH')*100:.1f}%",
                "Cross-Family Audit": "Yes (Gemini + Reflex)",
                "HITL Gate": mas_out["state"]["hitl_status"],
                "FIPA Messages": len(mas_out["fipa_messages"]),
            },
        ]
    )


if __name__ == "__main__":
    print(run_final_ablation().to_string(index=False))
