"""
QuantGuard-MAS — Unit II: CoT vs. ToT vs. ReAct Reasoning & Dynamic Shock Replanning
"""

import time
from typing import Dict, List, Optional

from core.guardrails import ReflexCircuitBreaker, UtilityScorer
from core.tools import compute_portfolio_quant_metrics, fetch_market_data, finbert_sentiment


BRANCH_TEMPLATES = {
    "Conservative": {
        "ITC.NS": 0.25,
        "TCS.NS": 0.20,
        "HDFCBANK.NS": 0.20,
        "CASH": 0.35,
    },
    "Balanced": {
        "RELIANCE.NS": 0.25,
        "TCS.NS": 0.20,
        "HDFCBANK.NS": 0.20,
        "ITC.NS": 0.15,
        "CASH": 0.20,
    },
    "Aggressive": {
        "RELIANCE.NS": 0.35,
        "INFY.NS": 0.25,
        "SBIN.NS": 0.20,
        "LT.NS": 0.10,
        "CASH": 0.10,
    },
}


class ReasoningEngine:
    """
    Implements and benchmarks Unit II reasoning paradigms on identical market states:
      - Chain-of-Thought (CoT): Single-path linear heuristic/LLM allocation
      - Tree-of-Thought (ToT): 3-branch generation, utility scoring, & guardrail pruning
      - ReAct: Tool-interleaved reasoning (market data + FinBERT + sandbox) + dynamic shock replanning
    """

    def __init__(
        self,
        breaker: Optional[ReflexCircuitBreaker] = None,
        scorer: Optional[UtilityScorer] = None,
    ):
        self.breaker = breaker or ReflexCircuitBreaker()
        self.scorer = scorer or UtilityScorer()

    def run_cot(self, env, goal: str) -> Dict:
        t0 = time.perf_counter()
        weights = dict(BRANCH_TEMPLATES["Balanced"])
        obs = env.observe()
        verdict = self.breaker.check(weights, (obs["performance"].get("max_drawdown_pct", 0.0) / 100.0))
        return {
            "method": "CoT",
            "goal": goal,
            "steps": 1,
            "weights": weights,
            "compliant": verdict.approved,
            "violations": verdict.violations,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    def run_tot(self, env, market_data: Dict, goal: str) -> Dict:
        t0 = time.perf_counter()
        obs = env.observe()
        drawdown = (obs["performance"].get("max_drawdown_pct", 0.0) / 100.0)
        evaluated_branches = []

        for name, weights in BRANCH_TEMPLATES.items():
            verdict = self.breaker.check(weights, drawdown)
            quant = compute_portfolio_quant_metrics(
                weights,
                market_data,
                chart_filename=f"tot_{name.lower()}.png",
            )["metrics"]
            utility = self.scorer.score(
                expected_return=quant["annualized_return"],
                volatility=quant["annualized_volatility"],
            )
            evaluated_branches.append(
                {
                    "branch": name,
                    "weights": weights,
                    "compliant": verdict.approved,
                    "violations": verdict.violations,
                    "utility": utility,
                    "metrics": quant,
                    "pruned": not verdict.approved,
                }
            )

        viable = [b for b in evaluated_branches if not b["pruned"]]
        best = (
            max(viable, key=lambda b: b["utility"])
            if viable
            else max(evaluated_branches, key=lambda b: b["utility"])
        )

        return {
            "method": "ToT",
            "goal": goal,
            "steps": len(evaluated_branches) + 1,
            "branches": evaluated_branches,
            "selected_branch": best["branch"],
            "weights": best["weights"],
            "utility": best["utility"],
            "compliant": best["compliant"],
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    def run_react_with_replanning(
        self,
        env,
        goal: str,
        shock_symbol: Optional[str] = None,
        shock_pct: float = -0.12,
        headlines: Optional[List[str]] = None,
    ) -> Dict:
        """
        Interleaves Thought -> Action -> Observation and dynamically replans
        when a mid-task market shock or negative FinBERT signal is observed.
        """
        t0 = time.perf_counter()
        trace = []

        # Step 1: Observe market & fetch data
        symbols = list(env.market.get_prices().keys())
        mdata = fetch_market_data(symbols, env=env, use_live_yfinance=False)
        trace.append("Thought 1: Fetch latest market returns and run ToT baseline.")
        tot_res = self.run_tot(env, mdata, goal)
        weights = dict(tot_res["weights"])
        trace.append(
            f"Observation 1: ToT selected '{tot_res['selected_branch']}' (U={tot_res['utility']})."
        )

        # Step 2: Perception via FinBERT
        sample_headlines = headlines or [
            "TCS posts strong quarterly margin expansion and stable guidance.",
            "Reliance faces sudden crude volatility and sharp sell-off warning.",
        ]
        sentiments = finbert_sentiment(sample_headlines, use_hf_model=False)
        trace.append(f"Observation 2 (FinBERT): Scored {len(sentiments)} headlines.")

        # Step 3: Optional mid-task Market Shock & Dynamic Replanning
        replanned = False
        if shock_symbol and shock_symbol in env.market.get_prices():
            env.advance_market(shocks={shock_symbol: shock_pct})
            replanned = True
            trace.append(
                f"Shock Event: {shock_symbol} dropped {shock_pct*100:.1f}%! Triggering dynamic replanning."
            )
            # Reallocate capital out of shocked asset into defensive ITC.NS & CASH
            shocked_w = weights.pop(shock_symbol, 0.0)
            if shocked_w > 0:
                weights["ITC.NS"] = round(weights.get("ITC.NS", 0.0) + shocked_w * 0.4, 4)
                weights["CASH"] = round(weights.get("CASH", 0.0) + shocked_w * 0.6, 4)

        verdict = self.breaker.check(weights, (env.observe()["performance"].get("max_drawdown_pct", 0.0) / 100.0))
        if not verdict.approved:
            weights = verdict.adjusted_weights
            trace.append("Reflex Guardrail adjusted allocation to restore compliance.")

        final_quant = compute_portfolio_quant_metrics(
            weights,
            fetch_market_data(symbols, env=env, use_live_yfinance=False),
            chart_filename="react_replan_chart.png",
        )

        return {
            "method": "ReAct",
            "goal": goal,
            "steps": len(trace),
            "replanned": replanned,
            "weights": weights,
            "compliant": self.breaker.check(weights).approved,
            "quant": final_quant["metrics"],
            "chart_path": final_quant["chart_path"],
            "sentiments": sentiments,
            "trace": trace,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
