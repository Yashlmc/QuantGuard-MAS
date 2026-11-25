"""
QuantGuard-MAS — Final Jury 4-Tab Streamlit Mission Control Dashboard
Run: python -m streamlit run app/dashboard.py
"""

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.final_ablation import run_final_ablation
from benchmarks.ta1_reasoning_bench import run_ta1_benchmark
from benchmarks.ta2_memory_mas_bench import run_ta2_benchmark
from core.mas_graph import QuantGuardMAS


st.set_page_config(
    page_title="QuantGuard-MAS Mission Control",
    page_icon="🛡️",
    layout="wide",
)


if "mas" not in st.session_state:
    st.session_state["mas"] = QuantGuardMAS(use_live_llms=False, use_hf_models=False)

mas: QuantGuardMAS = st.session_state["mas"]

st.title("🛡️ QuantGuard-MAS — Autonomous Multi-Agent Financial Advisory Platform")
st.caption(
    "23CSE4018 Agentic AI Capstone | 5-Model Architecture | Units I–IV: "
    "Reflex Guardrails · ToT/ReAct Replanning · FAISS LTM & Sandbox · FIPA-ACL & HITL"
)

with st.sidebar:
    st.header("⚙️ Runtime Controls")
    use_live = st.toggle("Enable Live Groq + Gemini LLMs", value=False)
    use_hf = st.toggle("Enable Local HuggingFace (FinBERT + MiniLM)", value=False)
    mas.use_live_llms = use_live
    mas.use_hf_models = use_hf
    st.divider()
    st.metric("Max Single-Stock Cap", f"{mas.breaker.max_single_asset_weight*100:.0f}%")
    st.metric("Max Drawdown Limit", f"{mas.breaker.max_drawdown_limit*100:.0f}%")
    st.metric("Min Cash Reserve", f"{mas.breaker.min_cash_weight*100:.0f}%")


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🚀 Live Agent Mission Control & HITL",
        "🧠 Memory & RAG Inspector",
        "📊 Sandboxed Quant Lab",
        "🧪 Course Evaluation Benchmarks",
    ]
)


# ==================================================
# TAB 1: MISSION CONTROL & HITL
# ==================================================

with tab1:
    st.subheader("Configure Portfolio Advisory Mission")
    goal = st.text_input(
        "Investment Goal",
        value="Build a balanced ₹10,00,000 Indian equity portfolio targeting steady growth with strict drawdown protection.",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        inject_shock = st.checkbox("Inject Mid-Task Market Shock (-12%)", value=True)
    with c2:
        shock_sym = st.selectbox(
            "Shocked Asset",
            list(mas.env.market.get_prices().keys()),
            index=0,
        )
    with c3:
        sim_conflict = st.checkbox("Simulate Researcher vs. Risk Auditor Conflict", value=True)

    if st.button("🚀 Launch 4-Agent MAS Mission", type="primary", use_container_width=True):
        try:
            with st.spinner("Running Contract Net -> Research -> Quant Sandbox -> Risk Audit -> Arbitration..."):
                res = mas.run_mission(
                    goal=goal,
                    shock_symbol=shock_sym if inject_shock else None,
                    shock_pct=-0.12,
                    simulate_conflict=sim_conflict,
                )
                st.session_state["last_mission"] = res
            st.success(f"Mission `{res['run_id']}` paused at HITL Approval Gate.")
        except Exception as e:
            st.error(f"Mission blocked or failed: {e}")

    if "last_mission" in st.session_state:
        res = st.session_state["last_mission"]
        state = res["state"]
        metrics = state["quant_report"]["metrics"]

        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("HITL Status", state["hitl_status"])
        m2.metric("Annualized Return", f"{metrics['annualized_return']*100:.2f}%")
        m3.metric("Annualized Volatility", f"{metrics['annualized_volatility']*100:.2f}%")
        m4.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']}")

        if state["conflict_arbitration"].get("conflict_triggered"):
            st.warning(
                "**Conflict Arbitration Triggered:** "
                + state["conflict_arbitration"]["mediation_note"]
            )

        st.subheader("Proposed Final Allocation")
        alloc_df = pd.DataFrame(
            [
                {"Asset": k, "Target Weight (%)": round(v * 100, 2)}
                for k, v in state["final_allocation"].items()
            ]
        )
        st.dataframe(alloc_df, use_container_width=True, hide_index=True)

        # HITL Approval Gate
        st.subheader("🧑‍⚖️ Human-in-the-Loop (HITL) Approval Gate")
        h1, h2 = st.columns([2, 1])
        with h1:
            fb_note = st.text_input(
                "Optional feedback / preference correction (persisted to FAISS LTM):",
                placeholder="e.g., Tighten my max drawdown limit to 8%.",
            )
        with h2:
            new_dd = st.number_input(
                "Update Max Drawdown Limit (%)",
                min_value=5,
                max_value=30,
                value=int(mas.breaker.max_drawdown_limit * 100),
            )

        b1, b2, b3 = st.columns(3)
        for col, dec in zip([b1, b2, b3], ["APPROVE", "MODIFY", "REJECT"]):
            if col.button(f"{dec} Allocation", use_container_width=True):
                hitl_out = mas.apply_hitl_decision(
                    run_id=res["run_id"],
                    decision=dec,
                    feedback_note=fb_note,
                    new_max_drawdown=(new_dd / 100.0) if new_dd != int(mas.breaker.max_drawdown_limit * 100) else None,
                )
                st.session_state["last_mission"]["state"]["hitl_status"] = hitl_out["hitl_status"]
                st.session_state["last_mission"]["fipa_messages"] = hitl_out["fipa_messages"]
                st.success(f"HITL Decision recorded: {hitl_out['hitl_status']}")

        with st.expander("📨 Inspect FIPA-ACL Message Bus Trace (SQLite Logged)"):
            fipa_df = pd.DataFrame(
                [
                    {
                        "Performative": m["performative"].upper(),
                        "Sender": m["sender"],
                        "Receiver": m["receiver"],
                        "Ontology": m["ontology"],
                        "Content": str(m["content"])[:140],
                    }
                    for m in res["fipa_messages"]
                ]
            )
            st.dataframe(fipa_df, use_container_width=True, hide_index=True)


# ==================================================
# TAB 2: MEMORY & RAG INSPECTOR
# ==================================================

with tab2:
    st.subheader("🧠 Persistent FAISS Long-Term Memory (Active vs. Superseded)")
    q = st.text_input("Semantic Memory Search Query", value="risk drawdown limit")
    if q.strip():
        hits = mas.memory.search(q, top_k=5, active_only=True)
        st.dataframe(
            pd.DataFrame(hits)[["id", "memory_type", "symbol", "similarity", "text"]],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Complete Memory Version History")
    all_recs = mas.memory.list_all(active_only=False)
    st.dataframe(
        pd.DataFrame(all_recs)[
            ["id", "is_active", "memory_type", "symbol", "superseded_by", "text"]
        ],
        use_container_width=True,
        hide_index=True,
    )


# ==================================================
# TAB 3: SANDBOXED QUANT LAB
# ==================================================

with tab3:
    st.subheader("📊 AST-Sandboxed Quantitative Execution & Charts")
    if "last_mission" in st.session_state:
        qr = st.session_state["last_mission"]["state"]["quant_report"]
        st.code(qr["sandbox_code"], language="python")
        st.caption(f"Sandbox stdout: {qr['sandbox_stdout']}")
        if os.path.exists(qr["chart_path"]):
            st.image(qr["chart_path"], use_container_width=True)
    else:
        st.info("Launch a mission in Tab 1 to inspect generated sandbox code and risk charts.")


# ==================================================
# TAB 4: COURSE EVALUATION BENCHMARKS
# ==================================================

with tab4:
    st.subheader("🧪 Automated Units I–IV Evaluation Suites")
    c1, c2, c3 = st.columns(3)
    if c1.button("Run TA1: CoT vs ToT vs ReAct + Shock", use_container_width=True):
        st.session_state["ta1_df"] = run_ta1_benchmark()
    if c2.button("Run TA2: FAISS + Sandbox + Conflict MAS", use_container_width=True):
        st.session_state["ta2_df"] = run_ta2_benchmark()
    if c3.button("Run Final: Single-Agent vs Multi-Agent", use_container_width=True):
        st.session_state["ablation_df"] = run_final_ablation()

    if "ta1_df" in st.session_state:
        st.markdown("**TA1 Reasoning & Dynamic Replanning Benchmark**")
        st.dataframe(st.session_state["ta1_df"], use_container_width=True, hide_index=True)

    if "ta2_df" in st.session_state:
        st.markdown("**TA2 Memory, Sandbox Security & Conflict Arbitration Benchmark**")
        st.dataframe(st.session_state["ta2_df"], use_container_width=True, hide_index=True)

    if "ablation_df" in st.session_state:
        st.markdown("**Final Jury Ablation: Single-Agent Baseline vs. QuantGuard-MAS**")
        st.dataframe(st.session_state["ablation_df"], use_container_width=True, hide_index=True)