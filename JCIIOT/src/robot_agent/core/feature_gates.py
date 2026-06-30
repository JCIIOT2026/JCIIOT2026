"""Central switchboard for enabling/disabling agent components.

每个开关对应一个 ``GATE_*`` 环境变量，在 **实例化时** 读取。
设为 ``"0"``, ``"false"``, ``"no"``, ``"off"`` 关闭；不设默认启用。

用法::

    export GATE_PLANNER=false
    export GATE_STEP_TIMEOUT=false
    python -c "from robot_agent.core.feature_gates import FeatureGates; print(FeatureGates().as_dict())"

----
新增门控登记规范见 ``docs/feature_gates.md``。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _bool_env(key: str, default: bool = True) -> bool:
    return os.getenv(key, str(default).lower()).lower() in {"1", "true", "yes", "on"}


def _gate(env_key: str, default: bool = True) -> bool:
    """``default_factory`` 简写 — 每次实例化时重新读环境变量。"""
    return _bool_env(env_key, default)


@dataclass
class FeatureGates:
    """机器人 Agent 的 boolean 功能开关。全部默认 *True*（启用）。

    新增开关请遵循登记规范（见 docs/feature_gates.md）：
      1. 确认确实需要门控（非关键组件不需要）
      2. 在此 dataclass 中添加一个 ``field(default_factory=lambda: _gate("GATE_XXX"))`` 字段
      3. 在 ``as_dict()`` 中添加对应条目
      4. 在使用侧通过 ``self.config.feature_gates.xxx`` 检查
    """

    # ── LLM 后端 ──────────────────────────────────────────
    ollama: bool = field(default_factory=lambda: _gate("GATE_OLLAMA"))

    # ── 任务规划 ──────────────────────────────────────────
    planner: bool = field(default_factory=lambda: _gate("GATE_PLANNER"))

    # ── 步骤执行 ──────────────────────────────────────────
    step_retry: bool = field(default_factory=lambda: _gate("GATE_STEP_RETRY"))
    step_timeout: bool = field(default_factory=lambda: _gate("GATE_STEP_TIMEOUT"))
    precondition_validation: bool = field(
        default_factory=lambda: _gate("GATE_PRECONDITION_VALIDATION")
    )
    fail_fast: bool = field(default_factory=lambda: _gate("GATE_FAIL_FAST"))

    # ── helpers ──────────────────────────────────────────

    def as_dict(self) -> dict[str, bool]:
        """返回所有开关的 dict，便于调试和展示。"""
        return {
            "ollama": self.ollama,
            "planner": self.planner,
            "step_retry": self.step_retry,
            "step_timeout": self.step_timeout,
            "precondition_validation": self.precondition_validation,
            "fail_fast": self.fail_fast,
        }
