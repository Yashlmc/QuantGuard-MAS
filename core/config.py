"""
QuantGuard-MAS
Central Application Configuration
"""

import os
from pathlib import Path
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_FILE)


def get_env(name, default=None):
    return os.getenv(name, default)


def get_bool(name, default=False):
    value = get_env(name)
    if value is None:
        return default
    return value.strip().lower() in {"true", "1", "yes", "on"}


def get_int(name, default):
    value = get_env(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{name} must be an integer")


def get_float(name, default):
    value = get_env(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"{name} must be a float")


class Config:

    # Application
    APP_NAME = get_env("APP_NAME", "QuantGuard-MAS")
    APP_ENV = get_env("APP_ENV", "development")
    DEBUG = get_bool("DEBUG", True)

    APP_HOST = get_env("APP_HOST", "127.0.0.1")
    APP_PORT = get_int("APP_PORT", 8000)

    # 5-Model Architecture Defaults
    GROQ_FAST_MODEL = get_env("GROQ_FAST_MODEL", "openai/gpt-oss-20b")
    GROQ_PLANNER_MODEL = get_env("GROQ_PLANNER_MODEL", "openai/gpt-oss-120b")
    GEMINI_AUDITOR_MODEL = get_env("GEMINI_AUDITOR_MODEL", "gemini-2.5-flash")
    FINBERT_MODEL = get_env("FINBERT_MODEL", "ProsusAI/finbert")
    EMBEDDING_MODEL = get_env(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Unit I Risk Guardrail Defaults
    MAX_SINGLE_ASSET_WEIGHT = get_float("MAX_SINGLE_ASSET_WEIGHT", 0.35)
    MIN_CASH_WEIGHT = get_float("MIN_CASH_WEIGHT", 0.10)
    MAX_DRAWDOWN_LIMIT = get_float("MAX_DRAWDOWN_LIMIT", 0.15)

    # Security
    ENABLE_HUMAN_APPROVAL = get_bool("ENABLE_HUMAN_APPROVAL", True)
    ENABLE_PYTHON_SANDBOX = get_bool("ENABLE_PYTHON_SANDBOX", True)
    ENABLE_REFLEX_GUARDRAILS = get_bool("ENABLE_REFLEX_GUARDRAILS", True)
    ALLOW_AUTOMATIC_TRADE_EXECUTION = get_bool(
        "ALLOW_AUTOMATIC_TRADE_EXECUTION", False
    )
    ALLOW_EXTERNAL_CODE_EXECUTION = get_bool(
        "ALLOW_EXTERNAL_CODE_EXECUTION", False
    )

    # Storage
    DATA_DIRECTORY = BASE_DIR / get_env("DATA_DIRECTORY", "data")
    LOG_DIRECTORY = BASE_DIR / get_env("LOG_DIRECTORY", "logs")
    CHART_DIRECTORY = BASE_DIR / get_env("CHART_DIRECTORY", "charts")
    SQLITE_DATABASE_PATH = BASE_DIR / get_env(
        "SQLITE_DATABASE_PATH", "data/quantguard.db"
    )
    FAISS_INDEX_PATH = BASE_DIR / get_env("FAISS_INDEX_PATH", "data/faiss_index")

    # Market
    DEFAULT_CURRENCY = get_env("DEFAULT_CURRENCY", "INR")
    DEFAULT_MARKET = get_env("DEFAULT_MARKET", "INDIA")
    MARKET_DATA_PROVIDER = get_env("MARKET_DATA_PROVIDER", "yfinance")
    ENABLE_OFFLINE_CACHE = get_bool("ENABLE_OFFLINE_CACHE", True)

    # Memory
    MEMORY_TOP_K = get_int("MEMORY_TOP_K", 5)
    ENABLE_SHORT_TERM_MEMORY = get_bool("ENABLE_SHORT_TERM_MEMORY", True)
    ENABLE_LONG_TERM_MEMORY = get_bool("ENABLE_LONG_TERM_MEMORY", True)

    # Logging
    LOG_LEVEL = get_env("LOG_LEVEL", "INFO")
    ENABLE_API_USAGE_TRACKING = get_bool("ENABLE_API_USAGE_TRACKING", True)
    ENABLE_AGENT_TRACE = get_bool("ENABLE_AGENT_TRACE", True)
    ENABLE_LATENCY_TRACKING = get_bool("ENABLE_LATENCY_TRACKING", True)
    ENABLE_TOKEN_TRACKING = get_bool("ENABLE_TOKEN_TRACKING", True)

    @classmethod
    def validate(cls):
        if not 1 <= cls.APP_PORT <= 65535:
            raise ValueError("APP_PORT must be between 1 and 65535")
        if cls.MEMORY_TOP_K < 1:
            raise ValueError("MEMORY_TOP_K must be at least 1")
        if cls.APP_ENV not in {"development", "testing", "production"}:
            raise ValueError("Invalid APP_ENV")
        return True


if __name__ == "__main__":
    Config.validate()
    print("QuantGuard Configuration Loaded")
    print("--------------------------------")
    print("Application:", Config.APP_NAME)
    print("Models:", Config.GROQ_FAST_MODEL, "|", Config.GROQ_PLANNER_MODEL, "|", Config.GEMINI_AUDITOR_MODEL)
    print("Configuration validation: SUCCESS")
