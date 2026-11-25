"""
QuantGuard-MAS — Unit IV: 4-Agent Multi-Agent Orchestration, Conflict Arbitration & HITL Gate

Models orchestrated:
  - Model 1: openai/gpt-oss-20b (Groq) -> ResearcherAgent & QuantCoderAgent
  - Model 2: openai/gpt-oss-120b (Groq) -> OrchestratorAgent (Planner & Mediator)
  - Model 3: gemini-2.5-flash (Gemini) -> RiskAuditorAgent (Cross-family Critic)
  - Model 4: ProsusAI/finbert (Local HF) -> Perception sentiment classifier
  - Model 5: all-MiniLM-L6-v2 (Local HF) -> FAISS Long-Term Memory embeddings
"""

import os
import time
from typing import Dict, List, Optional

from core.fipa_protocol import ContractNetBroker, SharedBlackboard
from core.guardrails import (
    PromptInjectionSanitizer,
    ReflexCircuitBreaker,
    UtilityScorer,
)
from core.memory_faiss import FAISSLongTermMemory
from core.reasoning import ReasoningEngine
from core.tools import (
    compute_portfolio_quant_metrics,
    fetch_market_data,
    finbert_sentiment,
)
from environment.peas import QuantGuardEnvironment


class QuantGuardMAS:
    """
    End-to-end 4-Agent Collaborative Financial Intelligence Platform with
    FIPA-ACL Blackboard, Contract Net task allocation, Conflict Arbitration,
    Dynamic Shock Replanning, and Human-in-the-Loop (HITL) approval gate.
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        seed: int = 42,
        use_live_llms: bool = False,
        use_hf_models: bool = False,
        max_single_asset: float = 0.35,
        max_drawdown_limit: float = 0.15,
    ):
        self.env = QuantGuardEnvironment(initial_capital=initial_capital, seed=seed)
        # Warm up 35 trading days so Sharpe, Volatility, and VaR have >=30 observations
        for _ in range(35):
            self.env.advance_market()

        self.use_live_llms = use_live_llms
        self.use_hf_models = use_hf_models
        self.memory = FAISSLongTermMemory(use_local_embeddings=use_hf_models)
        self.breaker = ReflexCircuitBreaker(
            max_single_asset_weight=max_single_asset,
            max_drawdown_limit=max_drawdown_limit,
        )
        self.scorer = UtilityScorer(risk_aversion=1.5)
        self.reasoning = ReasoningEngine(breaker=self.breaker, scorer=self.scorer)

        self.router = None
        if use_live_llms:
            try:
                from core.llm_router import LLMRouter

                self.router = LLMRouter(enable_usage_tracking=True)
            except Exception:
                self.router = None

        self.active_runs: Dict[str, SharedBlackboard] = {}

    def _safe_llm_note(self, provider: str, model: str, system_prompt: str, user_prompt: str, fallback: str) -> str:
        if not self.use_live_llms or self.router is None:
            return fallback
        try:
            res = self.router.generate(
                provider=provider,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.2,
                max_output_tokens=300,
            )
            return res.text.strip() or fallback
        except Exception:
            return fallback

    def run_mission(
        self,
        goal: str,
        shock_symbol: Optional[str] = None,
        shock_pct: float = -0.12,
        simulate_conflict: bool = True,
    ) -> Dict:
        """
        Execute full 4-agent pipeline and pause at HITL gate (status='AWAITING_HITL').
        """
        t0 = time.perf_counter()
        injection_alerts = PromptInjectionSanitizer.scan(goal)
        if injection_alerts:
            raise ValueError(f"Security guardrail blocked input: {injection_alerts}")

        bb = SharedBlackboard()
        broker = ContractNetBroker(bb)
        bb.state["goal"] = goal

        # 1. Orchestrator decomposes goal & runs Contract Net auctions
        bb.post(
            performative="inform",
            sender="OrchestratorAgent",
            receiver="SharedBlackboard",
            ontology="goal_decomposition",
            content={
                "goal": goal,
                "subtasks": ["market_research_rag", "quant_sandbox_analytics", "cross_family_risk_audit"],
            },
        )

        broker.run_auction(
            "market_research_rag",
            [
                {"agent": "ResearcherAgent", "confidence": 0.94, "est_latency_ms": 420},
                {"agent": "QuantCoderAgent", "confidence": 0.55, "est_latency_ms": 600},
            ],
        )
        broker.run_auction(
            "quant_sandbox_analytics",
            [
                {"agent": "QuantCoderAgent", "confidence": 0.96, "est_latency_ms": 380},
                {"agent": "ResearcherAgent", "confidence": 0.45, "est_latency_ms": 700},
            ],
        )

        # 2. Optional mid-task Market Shock injection
        if shock_symbol and shock_symbol in self.env.market.get_prices():
            self.env.advance_market(shocks={shock_symbol: shock_pct})
            bb.post(
                performative="inform",
                sender="MarketEnvironment",
                receiver="OrchestratorAgent",
                ontology="market_shock",
                content={"symbol": shock_symbol, "shock_pct": shock_pct},
            )

        # 3. Researcher Agent (Model 1 + FinBERT + FAISS RAG)
        rag_hits = self.memory.search(goal, top_k=4, active_only=True)
        headlines = [
            "RELIANCE.NS beats quarterly Jio revenue estimates with strong retail growth.",
            "TCS.NS posts steady operating margins and resilient order book.",
            f"{shock_symbol or 'SBIN.NS'} experiences sharp intraday volatility and risk warning.",
        ]
        sentiments = finbert_sentiment(headlines, use_hf_model=self.use_hf_models)
        researcher_stance = "BULLISH_RELIANCE" if simulate_conflict else "BALANCED"

        bb.state["rag_memories"] = rag_hits
        bb.state["finbert_sentiments"] = sentiments
        bb.post(
            performative="inform",
            sender="ResearcherAgent",
            receiver="OrchestratorAgent",
            ontology="market_intelligence",
            content={
                "rag_hits": len(rag_hits),
                "sentiments": sentiments,
                "recommendation": researcher_stance,
            },
        )

        # 4. Quant Coder Agent (Model 1 + Sandboxed Python + ToT)
        symbols = list(self.env.market.get_prices().keys())
        mdata = fetch_market_data(symbols, env=self.env, use_live_yfinance=False)
        tot_result = self.reasoning.run_tot(self.env, mdata, goal)
        proposed_weights = dict(tot_result["weights"])

        # If simulating bullish-vs-bearish conflict, Researcher pushes RELIANCE.NS over cap (42%)
        if simulate_conflict:
            proposed_weights = {
                "RELIANCE.NS": 0.42,
                "TCS.NS": 0.25,
                "HDFCBANK.NS": 0.23,
                "CASH": 0.10,
            }

        quant_bundle = compute_portfolio_quant_metrics(
            proposed_weights,
            mdata,
            chart_filename=f"run_{bb.run_id}_initial.png",
        )
        bb.state["tot_branches"] = tot_result["branches"]
        bb.state["quant_report"] = quant_bundle
        bb.post(
            performative="propose",
            sender="QuantCoderAgent",
            receiver="RiskAuditorAgent",
            ontology="portfolio_proposal",
            content={"weights": proposed_weights, "metrics": quant_bundle["metrics"]},
        )

        # 5. Risk Auditor Agent (Model 3: Gemini Cross-Family Critic + Simple Reflex Circuit Breaker)
        obs = self.env.observe()
        guard_verdict = self.breaker.check(
            proposed_weights, current_drawdown=(obs["performance"].get("max_drawdown_pct", 0.0) / 100.0)
        )
        auditor_note = self._safe_llm_note(
            provider="gemini",
            model=os.getenv("GEMINI_AUDITOR_MODEL", "gemini-2.5-flash"),
            system_prompt="You are an independent cross-family Risk Auditor reviewing an Indian equity allocation.",
            user_prompt=f"Goal: {goal}\nProposed weights: {proposed_weights}\nViolations: {guard_verdict.violations}\nGive a 2-sentence risk audit.",
            fallback=(
                f"Risk Auditor flagged {len(guard_verdict.violations)} violation(s): "
                + "; ".join(guard_verdict.violations)
                if guard_verdict.violations
                else "Allocation complies with single-stock, cash reserve, and drawdown guardrails."
            ),
        )

        bb.state["audit_verdict"] = {
            "approved": guard_verdict.approved,
            "violations": guard_verdict.violations,
            "auditor_note": auditor_note,
        }

        final_weights = dict(proposed_weights)
        if not guard_verdict.approved:
            bb.post(
                performative="refuse",
                sender="RiskAuditorAgent",
                receiver="OrchestratorAgent",
                ontology="risk_audit",
                content={"violations": guard_verdict.violations, "note": auditor_note},
            )

            # 6. Orchestrator Utility-Based Conflict Arbitration
            safe_weights = guard_verdict.adjusted_weights
            if shock_symbol and shock_symbol in safe_weights:
                cut = safe_weights[shock_symbol] * 0.5
                safe_weights[shock_symbol] = round(safe_weights[shock_symbol] - cut, 4)
                safe_weights["ITC.NS"] = round(safe_weights.get("ITC.NS", 0.0) + cut * 0.5, 4)
                safe_weights["CASH"] = round(safe_weights.get("CASH", 0.0) + cut * 0.5, 4)
                safe_weights = self.breaker.enforce_safe_weights(safe_weights)

            re_quant = compute_portfolio_quant_metrics(
                safe_weights,
                mdata,
                chart_filename=f"run_{bb.run_id}_arbitrated.png",
            )
            arbitration_note = self._safe_llm_note(
                provider="groq",
                model=os.getenv("GROQ_PLANNER_MODEL", "openai/gpt-oss-120b"),
                system_prompt="You are the Orchestrator Mediator resolving a Researcher vs Risk Auditor conflict.",
                user_prompt=f"Original: {proposed_weights}\nArbitrated: {safe_weights}\nExplain the rebalance in 2 sentences.",
                fallback="Orchestrator resolved the Researcher vs. Risk Auditor conflict by capping single-stock concentration and shifting risk weight into defensive holdings and cash.",
            )
            final_weights = safe_weights
            bb.state["quant_report"] = re_quant
            bb.state["conflict_arbitration"] = {
                "conflict_triggered": True,
                "original_weights": proposed_weights,
                "arbitrated_weights": final_weights,
                "utility_score": self.scorer.score(
                    re_quant["metrics"]["annualized_return"],
                    re_quant["metrics"]["annualized_volatility"],
                ),
                "mediation_note": arbitration_note,
            }
            bb.post(
                performative="agree",
                sender="OrchestratorAgent",
                receiver="ALL_AGENTS",
                ontology="conflict_arbitration",
                content=bb.state["conflict_arbitration"],
            )
        else:
            bb.post(
                performative="agree",
                sender="RiskAuditorAgent",
                receiver="OrchestratorAgent",
                ontology="risk_audit",
                content={"approved": True, "note": auditor_note},
            )
            bb.state["conflict_arbitration"] = {"conflict_triggered": False}

        # 7. Pause at HITL Gate (Never execute automatically without human approval)
        bb.state["final_allocation"] = final_weights
        bb.state["hitl_status"] = "AWAITING_HITL"
        bb.state["market_snapshot"] = obs["market"]
        bb.state["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        bb.post(
            performative="request",
            sender="OrchestratorAgent",
            receiver="HumanOperator",
            ontology="hitl_approval",
            content={"proposed_allocation": final_weights},
        )

        self.active_runs[bb.run_id] = bb
        return {
            "run_id": bb.run_id,
            "state": bb.state,
            "fipa_messages": bb.get_messages(),
        }

    def apply_hitl_decision(
        self,
        run_id: str,
        decision: str,
        feedback_note: str = "",
        new_max_drawdown: Optional[float] = None,
    ) -> Dict:
        """
        Resume from HITL interrupt: APPROVE, REJECT, or MODIFY.
        When feedback_note or new_max_drawdown is supplied, triggers Unit III
        Feedback-Based Adaptation in FAISS (is_correction=True).
        """
        if run_id not in self.active_runs:
            raise ValueError(f"Unknown run_id: {run_id}")

        bb = self.active_runs[run_id]
        decision = decision.strip().upper()
        if decision not in {"APPROVE", "REJECT", "MODIFY"}:
            raise ValueError("Decision must be APPROVE, REJECT, or MODIFY.")

        execution_result = None
        if decision == "APPROVE":
            execution_result = self.env.apply_allocation(bb.state["final_allocation"])
            bb.state["hitl_status"] = "APPROVED_AND_EXECUTED"
        elif decision == "REJECT":
            bb.state["hitl_status"] = "REJECTED_BY_HUMAN"
        else:
            bb.state["hitl_status"] = "MODIFIED_BY_HUMAN"

        memory_update = None
        if new_max_drawdown is not None:
            self.breaker.max_drawdown_limit = float(new_max_drawdown)
            memory_update = self.memory.add_memory(
                text=f"Updated user risk profile: Max drawdown limit changed to {new_max_drawdown*100:.1f}%. {feedback_note}".strip(),
                memory_type="user_preference",
                symbol="PORTFOLIO",
                is_correction=True,
            )
        elif feedback_note.strip():
            memory_update = self.memory.add_memory(
                text=f"Human HITL feedback ({decision}): {feedback_note.strip()}",
                memory_type="user_preference",
                symbol="PORTFOLIO",
                is_correction=(decision in {"REJECT", "MODIFY"}),
            )

        bb.state["hitl_feedback"] = feedback_note
        bb.post(
            performative="inform",
            sender="HumanOperator",
            receiver="OrchestratorAgent",
            ontology="hitl_decision",
            content={
                "decision": decision,
                "feedback": feedback_note,
                "memory_updated": memory_update["id"] if memory_update else None,
            },
        )

        return {
            "run_id": run_id,
            "hitl_status": bb.state["hitl_status"],
            "execution_result": execution_result,
            "memory_update": memory_update,
            "fipa_messages": bb.get_messages(),
        }
