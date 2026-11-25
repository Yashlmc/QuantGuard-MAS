"""
QuantGuard-MAS — TA2 Benchmark (Units III & IV)
Evaluates FAISS retrieval & feedback superseding, Python sandbox security,
FIPA-ACL logging, and Bullish-vs-Bearish conflict arbitration.
"""

import tempfile
from pathlib import Path

import pandas as pd

from core.mas_graph import QuantGuardMAS
from core.memory_faiss import FAISSLongTermMemory
from core.tools import SecurityError, python_sandbox


def run_ta2_benchmark() -> pd.DataFrame:
    rows = []

    # 1. FAISS Feedback Adaptation (is_correction=True)
    with tempfile.TemporaryDirectory() as tmp:
        mem = FAISSLongTermMemory(store_dir=Path(tmp), use_local_embeddings=False)
        mem.add_memory(
            "Updated user risk profile: Max drawdown limit is now 8%.",
            memory_type="user_preference",
            symbol="PORTFOLIO",
            is_correction=True,
        )
        active_prefs = [
            r for r in mem.list_all(active_only=True) if r["memory_type"] == "user_preference"
        ]
        archived_prefs = [
            r for r in mem.list_all(active_only=False) if not r["is_active"]
        ]
        rows.append(
            {
                "Subsystem": "FAISS LTM Feedback Overwrite",
                "Status": "PASS" if len(active_prefs) == 1 and len(archived_prefs) >= 1 else "FAIL",
                "Metric / Evidence": f"Active prefs={len(active_prefs)}, Archived={len(archived_prefs)}",
            }
        )

    # 2. Python Sandbox Security Block Test
    blocked = False
    try:
        python_sandbox("import os\nos.system('echo hacked')")
    except SecurityError:
        blocked = True
    rows.append(
        {
            "Subsystem": "AST Python Sandbox Security",
            "Status": "PASS" if blocked else "FAIL",
            "Metric / Evidence": "Blocked unauthorized `import os` call",
        }
    )

    # 3. 4-Agent MAS Conflict Arbitration + FIPA-ACL + HITL
    mas = QuantGuardMAS(use_live_llms=False, use_hf_models=False)
    out = mas.run_mission(
        goal="Allocate ₹10L across Indian equities with strict risk control.",
        shock_symbol="RELIANCE.NS",
        shock_pct=-0.12,
        simulate_conflict=True,
    )
    arb = out["state"]["conflict_arbitration"]
    fipa_perfs = {m["performative"] for m in out["fipa_messages"]}
    rows.append(
        {
            "Subsystem": "FIPA-ACL + Contract Net + Arbitration",
            "Status": "PASS" if arb.get("conflict_triggered") and {"cfp", "propose", "refuse", "agree", "request"}.issubset(fipa_perfs) else "FAIL",
            "Metric / Evidence": f"Messages={len(out['fipa_messages'])}, HITL={out['state']['hitl_status']}",
        }
    )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(run_ta2_benchmark().to_string(index=False))
