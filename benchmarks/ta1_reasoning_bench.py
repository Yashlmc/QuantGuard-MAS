"""
QuantGuard-MAS — TA1 Benchmark (Units I & II)
Compares CoT vs. ToT vs. ReAct + Mid-Task Market Shock Dynamic Replanning.
"""

import pandas as pd

from core.reasoning import ReasoningEngine
from core.tools import fetch_market_data
from environment.peas import QuantGuardEnvironment


def run_ta1_benchmark(seed: int = 42) -> pd.DataFrame:
    env = QuantGuardEnvironment(initial_capital=1_000_000, seed=seed)
    for _ in range(35):
        env.advance_market()

    engine = ReasoningEngine()
    goal = "Maximize risk-adjusted Indian equity returns under 35% single-stock cap."
    mdata = fetch_market_data(list(env.market.get_prices().keys()), env=env)

    cot = engine.run_cot(env, goal)
    tot = engine.run_tot(env, mdata, goal)
    react = engine.run_react_with_replanning(
        env, goal, shock_symbol="RELIANCE.NS", shock_pct=-0.12
    )

    df = pd.DataFrame(
        [
            {
                "Engine": "CoT (Linear)",
                "Steps": cot["steps"],
                "Latency (ms)": cot["latency_ms"],
                "Constraint Compliant": cot["compliant"],
                "Shock Replanned": False,
                "RELIANCE.NS Weight": cot["weights"].get("RELIANCE.NS", 0.0),
            },
            {
                "Engine": f"ToT ({tot['selected_branch']})",
                "Steps": tot["steps"],
                "Latency (ms)": tot["latency_ms"],
                "Constraint Compliant": tot["compliant"],
                "Shock Replanned": False,
                "RELIANCE.NS Weight": tot["weights"].get("RELIANCE.NS", 0.0),
            },
            {
                "Engine": "ReAct + Dynamic Replanning",
                "Steps": react["steps"],
                "Latency (ms)": react["latency_ms"],
                "Constraint Compliant": react["compliant"],
                "Shock Replanned": react["replanned"],
                "RELIANCE.NS Weight": react["weights"].get("RELIANCE.NS", 0.0),
            },
        ]
    )
    return df


if __name__ == "__main__":
    print(run_ta1_benchmark().to_string(index=False))
