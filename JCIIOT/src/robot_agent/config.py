"""Configuration helpers for the robot agent.

All env-var-backed fields use ``field(default_factory=...)`` so the
environment is re-read on **each instantiation**, not cached at import time.
"""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path

from robot_agent.core.feature_gates import FeatureGates


def _load_llm_params() -> dict:
    """Read LLM configuration from ``knowledge/robot_params.json``.

    Returns a dict with keys ``ollama_base_url``, ``ollama_model``,
    ``vision_model``.  Missing file / keys silently fall back to the
    factory defaults defined here.
    """
    _ROOT = Path(__file__).resolve().parents[2]
    _path = _ROOT / "knowledge" / "robot_params.json"
    _defaults = {
        "ollama_base_url": "http://localhost:11434",
        "ollama_model": "qwen3.6:27b-mtp-q4_K_M",
        "vision_model": "qwen3-vl:8b",
    }
    try:
        if _path.exists():
            _data = json.loads(_path.read_text(encoding="utf-8"))
            _llm = _data.get("llm", {}) if isinstance(_data, dict) else {}
            if isinstance(_llm, dict):
                for _k in _defaults:
                    _v = _llm.get(_k)
                    if _v and isinstance(_v, str):
                        _defaults[_k] = _v
    except Exception:
        pass
    return _defaults


def _float_env(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


def _int_env(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass
class AgentConfig:
    name: str = "robot-agent-v2"
    max_steps: int = 16
    memory_limit: int = 32

    # ── Feature gates (central switchboard) ───────────────
    feature_gates: FeatureGates = field(default_factory=FeatureGates)

    # ── Ollama / LLM ──────────────────────────────────────
    # Priority: env var > robot_params.json > factory default
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL")
        or _load_llm_params()["ollama_base_url"]
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL")
        or _load_llm_params()["ollama_model"]
    )
    ollama_timeout: float = field(
        default_factory=lambda: _float_env("OLLAMA_TIMEOUT", 120.0)
    )

    # ── Planner ───────────────────────────────────────────
    planner_max_tokens: int = field(
        default_factory=lambda: _int_env("PLANNER_MAX_TOKENS", 4096)
    )
    planner_temperature: float = field(
        default_factory=lambda: _float_env("PLANNER_TEMPERATURE", 0.1)
    )
    planner_json_retries: int = field(
        default_factory=lambda: _int_env("PLANNER_JSON_RETRIES", 2)
    )

    # ── Step execution ────────────────────────────────────
    step_default_timeout: float = field(
        default_factory=lambda: _float_env("STEP_DEFAULT_TIMEOUT", 3000.0)
    )
    step_max_retries: int = field(
        default_factory=lambda: _int_env("STEP_MAX_RETRIES", 0)
    )

