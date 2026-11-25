"""
QuantGuard-MAS — Final Jury FastAPI Backend
Endpoints:
  GET  /health
  POST /analyze
  POST /shock-replan
  POST /hitl-decision
  GET  /memory-audit
  GET  /usage-summary
  GET  /fipa-log/{run_id}
"""

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.config import Config
from core.fipa_protocol import SharedBlackboard
from core.mas_graph import QuantGuardMAS
from evaluation.usage_tracker import UsageTracker


app = FastAPI(
    title="QuantGuard-MAS API",
    description="Autonomous Multi-Agent Financial Intelligence & Risk Advisory Platform",
    version="1.0.0",
)

mas = QuantGuardMAS(use_live_llms=False, use_hf_models=False)
usage_tracker = UsageTracker()


class AnalyzeRequest(BaseModel):
    goal: str = Field(min_length=5)
    simulate_conflict: bool = True
    use_live_llms: bool = False


class ShockReplanRequest(BaseModel):
    goal: str = Field(min_length=5)
    shock_symbol: str = "RELIANCE.NS"
    shock_pct: float = Field(default=-0.12, ge=-0.50, le=0.50)


class HITLDecisionRequest(BaseModel):
    run_id: str
    decision: str  # APPROVE | REJECT | MODIFY
    feedback_note: str = ""
    new_max_drawdown: Optional[float] = Field(default=None, ge=0.02, le=0.50)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "app": Config.APP_NAME,
        "currency": Config.DEFAULT_CURRENCY,
        "hitl_enabled": Config.ENABLE_HUMAN_APPROVAL,
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    try:
        mas.use_live_llms = req.use_live_llms
        return mas.run_mission(
            goal=req.goal,
            simulate_conflict=req.simulate_conflict,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/shock-replan")
def shock_replan(req: ShockReplanRequest):
    try:
        return mas.run_mission(
            goal=req.goal,
            shock_symbol=req.shock_symbol,
            shock_pct=req.shock_pct,
            simulate_conflict=True,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/hitl-decision")
def hitl_decision(req: HITLDecisionRequest):
    try:
        return mas.apply_hitl_decision(
            run_id=req.run_id,
            decision=req.decision,
            feedback_note=req.feedback_note,
            new_max_drawdown=req.new_max_drawdown,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/memory-audit")
def memory_audit(active_only: bool = False):
    return {
        "records": mas.memory.list_all(active_only=active_only),
    }


@app.get("/usage-summary")
def usage_summary():
    return {"summary": usage_tracker.get_summary()}


@app.get("/fipa-log/{run_id}")
def fipa_log(run_id: str):
    return {"run_id": run_id, "messages": SharedBlackboard.fetch_logged_messages(run_id)}
