"""
QuantGuard-MAS — Unit III: Tools & Sandboxed Environment Interaction
Tools:
  1. fetch_market_data (live yfinance with automatic PEAS simulator fallback)
  2. finbert_sentiment (Model 4: ProsusAI/finbert with deterministic lexicon fallback)
  3. safe_calculator (AST-validated arithmetic evaluator)
  4. python_sandbox (AST-restricted quantitative risk & Matplotlib chart executor)
"""

import ast
import io
import math
import operator
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.config import Config


# ==================================================
# 1. TOOL 1 — MARKET DATA (yfinance + PEAS FALLBACK)
# ==================================================

def fetch_market_data(symbols: List[str], env=None, use_live_yfinance: bool = False) -> Dict:
    """
    Fetch current prices and recent daily returns for Indian NSE equities.
    Uses live yfinance when requested & available; falls back seamlessly to PEAS simulator.
    """
    data = {}
    if use_live_yfinance:
        try:
            import yfinance as yf

            for sym in symbols:
                hist = yf.Ticker(sym).history(period="3mo")
                if not hist.empty and len(hist) >= 5:
                    closes = hist["Close"].dropna().tolist()
                    rets = [
                        (closes[i] / closes[i - 1]) - 1.0
                        for i in range(1, len(closes))
                    ]
                    data[sym] = {
                        "price": round(float(closes[-1]), 2),
                        "returns": rets,
                        "source": "yfinance_live",
                    }
        except Exception:
            pass

    if env is not None:
        prices = env.market.get_prices()
        for sym in symbols:
            if sym not in data and sym in prices:
                data[sym] = {
                    "price": round(float(prices[sym]), 2),
                    "returns": env.market.get_return_history(sym),
                    "source": "peas_simulator",
                }
    return data


# ==================================================
# 2. TOOL 2 — MODEL 4: ProsusAI/finbert SENTIMENT
# ==================================================

_FINBERT_PIPE = None

POSITIVE_LEXICON = {
    "beat", "beats", "surge", "growth", "profit", "upgrade", "strong",
    "record", "rally", "expansion", "dividend", "outperform", "bullish", "gains",
}
NEGATIVE_LEXICON = {
    "crash", "drop", "plunge", "loss", "downgrade", "weak", "warning",
    "default", "fraud", "cut", "shock", "volatility", "bearish", "probe", "decline",
}


def finbert_sentiment(headlines: List[str], use_hf_model: bool = True) -> List[Dict]:
    """
    Traditional/Discriminative AI Perception Tool (ProsusAI/finbert).
    Classifies financial headlines into Positive / Negative / Neutral with confidence.
    Falls back to a deterministic financial lexicon if transformers/model weights are offline.
    """
    global _FINBERT_PIPE
    results = []

    if use_hf_model and _FINBERT_PIPE is None:
        try:
            from transformers import pipeline

            _FINBERT_PIPE = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
            )
        except Exception:
            _FINBERT_PIPE = False

    if use_hf_model and _FINBERT_PIPE:
        preds = _FINBERT_PIPE(headlines, truncation=True)
        for h, p in zip(headlines, preds):
            results.append(
                {
                    "headline": h,
                    "label": str(p["label"]).capitalize(),
                    "score": round(float(p["score"]), 4),
                    "engine": "ProsusAI/finbert",
                }
            )
        return results

    # Deterministic offline fallback
    for h in headlines:
        toks = set(h.lower().replace(",", " ").replace(".", " ").split())
        pos = len(toks & POSITIVE_LEXICON)
        neg = len(toks & NEGATIVE_LEXICON)
        if pos > neg:
            label, score = "Positive", min(0.95, 0.65 + 0.1 * pos)
        elif neg > pos:
            label, score = "Negative", min(0.95, 0.65 + 0.1 * neg)
        else:
            label, score = "Neutral", 0.60
        results.append(
            {
                "headline": h,
                "label": label,
                "score": round(score, 4),
                "engine": "finbert_offline_lexicon_fallback",
            }
        )
    return results


# ==================================================
# 3. TOOL 3 — SAFE AST ARITHMETIC CALCULATOR
# ==================================================

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def safe_calculator(expression: str) -> float:
    """Evaluate a pure arithmetic expression safely via AST (no eval())."""
    tree = ast.parse(expression, mode="eval")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsupported arithmetic node: {type(node).__name__}")

    return round(float(_eval(tree)), 6)


# ==================================================
# 4. TOOL 4 — AST-RESTRICTED PYTHON QUANT SANDBOX
# ==================================================

FORBIDDEN_AST_NODES = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)
FORBIDDEN_NAMES = {
    "__import__", "open", "exec", "eval", "compile", "input",
    "os", "sys", "subprocess", "socket", "shutil", "pathlib",
}


def validate_sandbox_code(code: str) -> None:
    """Reject any code containing imports, file/OS calls, or dunder escapes."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_AST_NODES):
            raise SecurityError(f"Blocked AST node in sandbox: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise SecurityError(f"Blocked identifier in sandbox: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise SecurityError(f"Blocked dunder attribute access: {node.attr}")


class SecurityError(Exception):
    pass


def python_sandbox(code: str, context: Optional[Dict] = None) -> Dict:
    """
    Execute restricted quantitative Python code with only safe math/numpy
    primitives injected, capturing stdout and returning computed `result`.
    """
    validate_sandbox_code(code)

    safe_builtins = {
        "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
        "round": round, "range": range, "sorted": sorted, "enumerate": enumerate,
        "zip": zip, "float": float, "int": int, "dict": dict, "list": list,
        "print": print,
    }
    env_globals = {
        "__builtins__": safe_builtins,
        "math": math,
        "np": np,
    }
    env_ns = {**env_globals, **(context or {})}

    buf = io.StringIO()
    with redirect_stdout(buf):
        exec(compile(code, "<quantguard_sandbox>", "exec"), env_ns)

    return {
        "stdout": buf.getvalue().strip(),
        "result": env_ns.get("result"),
    }


def compute_portfolio_quant_metrics(
    weights: Dict[str, float],
    market_data: Dict,
    risk_free_rate: float = 0.06,
    chart_filename: str = "portfolio_risk_chart.png",
) -> Dict:
    """
    Run deterministic sandboxed portfolio analytics (Annualized Return, Volatility,
    Sharpe Ratio, Historical 95% VaR) and generate a Matplotlib allocation & risk chart.
    """
    code = """
symbols = [s for s in weights if s != "CASH" and s in market_data and len(market_data[s].get("returns", [])) >= 5]
if not symbols:
    result = {
        "annualized_return": 0.06,
        "annualized_volatility": 0.02,
        "sharpe_ratio": 0.0,
        "var_95_daily": 0.0,
        "observations": 0,
    }
else:
    min_len = min(len(market_data[s]["returns"]) for s in symbols)
    port_rets = np.zeros(min_len, dtype=float)
    for s in symbols:
        r = np.array(market_data[s]["returns"][-min_len:], dtype=float)
        port_rets += float(weights[s]) * r
    mean_daily = float(np.mean(port_rets))
    std_daily = float(np.std(port_rets, ddof=1)) if min_len > 1 else 0.0
    ann_ret = mean_daily * 252.0 + float(weights.get("CASH", 0.0)) * risk_free_rate
    ann_vol = std_daily * math.sqrt(252.0)
    sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 1e-9 else 0.0
    var_95 = float(np.percentile(port_rets, 5))
    result = {
        "annualized_return": round(ann_ret, 4),
        "annualized_volatility": round(ann_vol, 4),
        "sharpe_ratio": round(sharpe, 3),
        "var_95_daily": round(var_95, 4),
        "observations": int(min_len),
    }
print(f"Computed Quant Metrics over {result['observations']} trading days: Sharpe={result['sharpe_ratio']}")
"""
    sandbox_out = python_sandbox(
        code,
        context={
            "weights": weights,
            "market_data": market_data,
            "risk_free_rate": risk_free_rate,
        },
    )

    # Generate risk & allocation chart
    charts_dir = Path(Config.CHART_DIRECTORY)
    charts_dir.mkdir(parents=True, exist_ok=True)
    chart_path = charts_dir / chart_filename

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    labels = list(weights.keys())
    vals = [weights[k] * 100 for k in labels]
    ax1.barh(labels, vals, color="#1D4ED8")
    ax1.set_title("Proposed Portfolio Allocation (%)")
    ax1.set_xlabel("Weight (%)")
    ax1.grid(axis="x", linestyle="--", alpha=0.4)

    metrics = sandbox_out["result"]
    m_names = ["Ann. Return %", "Ann. Vol %", "95% Daily VaR %"]
    m_vals = [
        metrics["annualized_return"] * 100,
        metrics["annualized_volatility"] * 100,
        abs(metrics["var_95_daily"]) * 100,
    ]
    ax2.bar(m_names, m_vals, color=["#0F766E", "#D97706", "#DC2626"])
    ax2.set_title(f"Risk Profile (Sharpe: {metrics['sharpe_ratio']})")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close(fig)

    return {
        "metrics": metrics,
        "sandbox_code": code.strip(),
        "sandbox_stdout": sandbox_out["stdout"],
        "chart_path": str(chart_path),
    }
