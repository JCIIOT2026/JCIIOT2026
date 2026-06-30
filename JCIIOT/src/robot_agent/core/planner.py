"""Task planner that uses Ollama when available and requests structured JSON.

The planner sends a carefully-structured prompt (with few-shot examples) to
the LLM, parses & repairs the JSON response, validates each step against the
skill registry, and returns a normalised ``PlanDecision``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robot_agent.config import AgentConfig
from robot_agent.core.ollama_client import OllamaClient
from robot_agent.core.registry import SkillRegistry
from robot_agent.core.scene_context import SceneContext
from robot_agent.core.schema import (
    normalize_planner_output,
    repair_json,
    validate_plan_steps,
)

logger = logging.getLogger(__name__)


def _task_level_from_metadata(scene_metadata: dict | None) -> str | None:
    if not isinstance(scene_metadata, dict):
        return None
    try:
        task_index = int(scene_metadata.get("task_index"))
    except (TypeError, ValueError):
        return None
    if task_index < 0:
        return None
    return f"L{task_index + 1}"


def _current_sop_main_row(knowledge_root: Path, scene_metadata: dict | None) -> str:
    """Return only the current L1-L5 row from sop_main.md."""
    level = _task_level_from_metadata(scene_metadata)
    if not level:
        return ""

    sop_path = knowledge_root / "sop_main.md"
    try:
        lines = sop_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return ""

    header = ""
    separator = ""
    row = ""
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("| Level "):
            header = stripped
            if index + 1 < len(lines):
                separator = lines[index + 1].strip()
            continue
        if not stripped.startswith("|"):
            continue
        cells = [part.strip() for part in stripped.strip("|").split("|")]
        if cells and cells[0] == level:
            row = stripped
            break

    if not row:
        return ""
    if not header:
        header = "| Level | Scene | Pick Station | Pick Coords | Object Name | Place Station | Place Coords |"
    if not separator:
        separator = "| --- | --- | --- | --- | --- | --- | --- |"

    return "\n".join([
        "## Current Task Coordinate Reference (from sop_main.md)",
        header,
        separator,
        row,
        "",
        "Use only this current task row for Pick Station / Place Station mapping.",
    ])


@dataclass(slots=True)
class PlanDecision:
    skill_name: str
    response: str
    raw: str            # normalised JSON (for backward compat)
    raw_llm_text: str   # original LLM response (for audit)
    details: dict[str, Any]


# ── prompt template ───────────────────────────────────────

def _build_plan_prompt(
    task: str,
    skill_names: list[str],
    scene_summary: str = "",
    knowledge_enabled: bool = True,
    scene_metadata: dict | None = None,
) -> str:
    """Build a structured planner prompt with few-shot examples.

    Args:
        task: The user's natural-language command.
        skill_names: Registered skill names (e.g. ``["move", "pick_up", "place_down"]``).
        scene_summary: Optional multi-line scene description (station names, positions).
        scene_metadata: Optional scene metadata including input_object_map.
    """
    names = ", ".join(skill_names)

    # Scene summary only used when knowledge is OFF (no KB mapping available)
    scene_block = ""
    if scene_summary and not knowledge_enabled:
        scene_block = f"""
## 场景信息
{scene_summary}

## 约束
- inputs.target 必须使用上面列出的工位名称
"""

    # Load all knowledge from markdown files (both locked + team) via KnowledgeManager
    _kb_block = ""
    if knowledge_enabled:
        try:
            from robot_agent.core.knowledge_manager import KnowledgeManager
            # Locked competition knowledge
            _kbm = KnowledgeManager("knowledge")
            _kbm.reload()
            _kb_block = _kbm.as_prompt_context(exclude_sources={"sop_main.md"})
            _sop_row = _current_sop_main_row(_kbm.root, scene_metadata)
            if _sop_row:
                _kb_block = f"{_kb_block}\n\n{_sop_row}" if _kb_block else _sop_row
            # Team's own knowledge
            try:
                _team_kbm = KnowledgeManager("team_submission/knowledge")
                _team_kbm.reload()
                _team_block = _team_kbm.as_prompt_context()
                if _team_block:
                    _kb_block += "\n## Team Knowledge\n" + _team_block
            except Exception:
                pass
        except Exception:
            pass

    _kb_section = ""
    if knowledge_enabled and _kb_block:
        _kb_section = f"""
## Knowledge Base (mandatory)
{_kb_block}

## Knowledge Base Rules
- Station names, coordinates, and SOP must follow the knowledge base above
- All constraints and safety rules in the knowledge base must be strictly followed
"""

    # Object mapping injection (always shown when available)
    _obj_map_block = ""
    if scene_metadata:
        input_obj_map = scene_metadata.get("input_object_map", {})
        if input_obj_map:
            _lines = ["## Current Scene Object Mapping (station -> object_name)"]
            for port, obj in sorted(input_obj_map.items()):
                if not port.startswith("line_"):
                    _lines.append(f"- {port} -> {obj}")
            _lines.append("- **CRITICAL: When calling pick_up, set inputs.object_name to the EXACT object name from this mapping!**")
            _obj_map_block = "\n".join(_lines)

    _examples = ""

    return f"""你是一个机器人任务规划器。你的职责是将用户任务分解为有序的技能步骤。
{_kb_section}
{_obj_map_block}

## 可用技能（只能使用这些名字）
{names}

各技能的 inputs 格式：
- **move**:         {{"target": "<工位名称>"}}
- **pick_up**:      {{"target": "<工位名称>", "object_name": "<物体精确名称>"}}
- **place_down**:   {{"target": "<工位名称>"}}
- **analyze_supply**: 自动分析供需并执行完整搬运（无需 inputs）
{scene_block}
## 输出格式
你必须**只输出一个 JSON 对象**，不含任何额外文本、注释或 markdown 标记。
{{
  "understanding": "<一句话总结你对任务的理解>",
  "reason": "<为什么这样拆解任务>",
  "plan": [
    {{
      "skill_name": "move",
      "description": "导航至取货工位接近点",
      "inputs": {{"target": "input_6"}},
      "preconditions": ["底盘静止于起始位置"],
      "expected_output": "机器人到达目标工位停止点",
      "timeout": 300,
      "retries": 0
    }},
    {{
      "skill_name": "pick_up",
      "description": "在取货工位抓取目标物体",
      "inputs": {{"target": "input_6", "object_name": "green_tote_b01_upper"}},
      "preconditions": ["夹爪为空", "底盘已停稳"],
      "expected_output": "成功抓取物体并提升至安全高度",
      "timeout": 300,
      "retries": 0
    }},
    {{
      "skill_name": "move",
      "description": "携带物体导航至放置工位",
      "inputs": {{"target": "output_4"}},
      "preconditions": ["夹爪已持握物体"],
      "expected_output": "机器人到达放置工位停止点",
      "timeout": 300,
      "retries": 0
    }},
    {{
      "skill_name": "place_down",
      "description": "在放置工位放下物体",
      "inputs": {{"target": "output_4"}},
      "preconditions": ["底盘已停稳于放置点"],
      "expected_output": "物体放置成功，夹爪释放并收回",
      "timeout": 300,
      "retries": 0
    }}
  ],
  "explanation": "<整体结果的简短说明>"
}}

## 规则
1. skill_name 必须从可用技能列表中选择
2. move/place_down 的 inputs 只包含 target
3. pick_up 的 inputs 必须同时包含 target 和 object_name
4. object_name 必须使用 Object Mapping 中列出的精确名称，严禁猜测
5. 严格按照 SOP 约束执行（超时 300s，不重试）
6. 必须输出完整的 4 步计划：move → pick_up → move → place_down
{_examples}

现在请为以下任务生成规划 JSON：
用户任务："{task}"
输出："""


# ── JSON extraction (planner's responsibility) ──────────────

def _extract_json(
    raw_text: str,
    base_prompt: str,
    client: OllamaClient,
    *,
    num_predict: int,
    temperature: float,
    max_retries: int,
) -> dict | None:
    """Try to parse JSON from LLM output, with repair and retry nudges."""
    prompt = base_prompt

    for attempt in range(1 + max_retries):
        # 1) direct parse
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            pass

        # 2) repair
        repaired = repair_json(raw_text)
        if repaired is not None:
            logger.info("JSON repaired on attempt %d", attempt + 1)
            return repaired

        # 3) retry with nudge
        if attempt < max_retries:
            logger.warning("Bad JSON on attempt %d, retrying…", attempt + 1)
            prompt = (
                f"{prompt}\n\n"
                f"【重要】上次输出不是合法 JSON，无法解析。"
                f"请严格只输出一个 JSON 对象。"
            )
            try:
                raw_text = client.generate(
                    prompt,
                    num_predict=num_predict,
                    temperature=temperature,
                )
            except RuntimeError:
                continue
            if not raw_text:
                continue

    return None


def _fallback(task: str, reason: str) -> PlanDecision:
    """Return a fallback PlanDecision when LLM planning fails."""
    return PlanDecision(
        skill_name="fallback",
        response="",
        raw="",
        raw_llm_text="",
        details={
            "version": "2.0",
            "understanding": str(task),
            "reason": f"LLM 调用失败: {reason}",
            "plan": [],
            "explanation": "",
            "warnings": [reason],
        },
    )


# ── planner ───────────────────────────────────────────────

class TaskPlanner:
    def __init__(
        self,
        client: OllamaClient,
        registry: SkillRegistry,
        config: AgentConfig | None = None,
        scene_context: SceneContext | None = None,
        knowledge_enabled: bool = True,
    ) -> None:
        self._client = client
        self._registry = registry
        self._config = config or AgentConfig()
        self._scene_context = scene_context
        self._knowledge_enabled = knowledge_enabled

    def plan(self, task: str, scene_metadata: dict | None = None) -> PlanDecision:
        skill_names = [skill.name for skill in self._registry.all()]
        if not skill_names:
            raise RuntimeError("No skills registered — cannot plan.")

        scene_summary = ""
        if self._knowledge_enabled and self._scene_context is not None:
            scene_summary = self._scene_context.as_prompt_context()

        prompt = _build_plan_prompt(
            task, skill_names,
            scene_summary=scene_summary,
            knowledge_enabled=self._knowledge_enabled,
            scene_metadata=scene_metadata,
        )
        logger.debug("Planner prompt length: %d chars", len(prompt))

        # ── call LLM: try json_mode first, then plain ─────
        raw_text: str = ""
        last_error: str = ""
        for mode in (True, False):
            try:
                raw_text = self._client.generate(
                    prompt,
                    num_predict=self._config.planner_max_tokens,
                    temperature=self._config.planner_temperature,
                    json_mode=mode,
                )
            except RuntimeError as exc:
                last_error = str(exc)
                continue
            if raw_text:
                break
            last_error = "Ollama 返回空响应" if mode else "JSON 与非 JSON 模式均返回空"

        if not raw_text:
            return _fallback(task, last_error)

        # ── extract JSON (with repair + retry) ────────────
        raw_dict = _extract_json(
            raw_text, prompt, self._client,
            num_predict=self._config.planner_max_tokens,
            temperature=self._config.planner_temperature,
            max_retries=self._config.planner_json_retries,
        )
        if raw_dict is None:
            return _fallback(task, f"无法从 LLM 输出中提取 JSON: {raw_text[:200]}")

        # ── normalise ──────────────────────────────────────
        details = normalize_planner_output(raw_dict, task, self._registry)

        # ── log warnings ───────────────────────────────────
        for w in details.get("warnings", []):
            logger.warning("Plan validation: %s", w)

        response = details.get("explanation") or ""
        return PlanDecision(
            skill_name=str(details.get("primary_skill", "fallback")),
            response=response,
            raw=details.get("raw", ""),
            raw_llm_text=raw_text,
            details=details,
        )
