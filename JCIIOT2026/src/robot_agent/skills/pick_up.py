"""Pick-up skill — grasp and lift a target object via backend."""

from __future__ import annotations

import logging
import re

from robot_agent.core.scene_context import SceneContext
from robot_agent.core.types import ExecutionContext, SkillResult
from robot_agent.skills.base import BaseSkill

logger = logging.getLogger(__name__)

# Chinese-number → digit
_CN_DIGIT: dict[str, str] = {
    "一": "1", "二": "2", "三": "3", "四": "4",
    "五": "5", "六": "6", "七": "7", "八": "8",
    "九": "9", "十": "10",
}
# Chinese role → role prefix
_CN_ROLE: dict[str, str] = {
    "进料": "input", "输入": "input", "入料": "input",
    "出料": "output", "输出": "output",
}
# Digit-word → index
_CN_INDEX: dict[str, str] = {
    "1": "1", "2": "2", "3": "3", "4": "4",
    "一": "1", "二": "2", "三": "3", "四": "4",
}
# Station kind keywords to strip from target
_CN_KIND: list[str] = ["传送带", "架子", "桌子", "箱子", "料箱", "料斗",
                        "conveyor", "shelf", "table", "bin"]


def _resolve_station_name(target: str, scene: SceneContext) -> str:
    """Resolve a natural-language target to a known station name.

    Examples of what this handles:
        "在1号进料口抓取目标物体" → "input_1"
        "把物品放到3号出料口"     → "output_3"
        "input_1"                  (pass-through — exact match)
    """
    known = scene.all_port_names()
    if not known:
        return target

    # 0) exact match
    if target in known:
        return target

    # 1) known name is a substring of target
    for name in known:
        if name in target:
            return name

    # 2) match by (role, index) — e.g. "1号进料口" → input station #1
    role, idx = _parse_role_index(target)
    if role and idx is not None:
        desired_idx = int(idx)
        for name in known:
            info = (scene.input_ports.get(name) or
                    scene.output_ports.get(name))
            if info is None:
                continue
            if info.role == role and info.index == desired_idx:
                return name

    return target


def _parse_role_index(text: str) -> tuple[str | None, int | None]:
    """Extract (role, index) from Chinese text like "1号进料口" → ("input", 1)."""
    # Normalise Chinese digits → Arabic
    s = text
    for cn, d in _CN_DIGIT.items():
        s = s.replace(cn, d)

    # Find a digit followed by optional characters then a role word
    m = re.search(r"(\d+)\s*[号#]?\s*([进出入输][料料入出])", s)
    if m:
        digit = m.group(1)
        role_cn = m.group(2)
        for cn_word, role_prefix in _CN_ROLE.items():
            if cn_word in role_cn:
                return role_prefix, int(digit)

    # Also try "input_N" / "output_N" pattern directly
    m = re.search(r"(input|output)\s*_?\s*(\d+)", text, re.IGNORECASE)
    if m:
        return m.group(1).lower(), int(m.group(2))

    return None, None


class PickUpSkill(BaseSkill):
    """Grasp a target object through the environment backend.

    Resolves natural-language target descriptions to known station names
    via ``SceneContext``, falling back to substring matching.
    """

    def __init__(self, *, backend, scene_context: SceneContext | None = None) -> None:
        super().__init__(
            name="pick_up",
            description="Grasp or pick up an object",
            keywords=(
                "pick", "grasp", "grab", "lift",
                "grasp", "pick", "grab", "take", "lift", "collect",
            ),
        )
        self._backend = backend
        self._scene = scene_context

    def run(self, context: ExecutionContext) -> SkillResult:
        inputs: dict = context.metadata.get("inputs", {})
        raw_target: str = (
            inputs.get("target")
            or context.task
        )
        object_name = (
            inputs.get("object_name")
            or inputs.get("obj_name")
            or inputs.get("object")
            or inputs.get("target_object")
        )
        object_name = str(object_name).strip() if object_name else None
        initial_base_pose = inputs.get("grasp_initial_base_pose")
        if initial_base_pose is None:
            initial_base_pose = inputs.get("initial_base_pose")
        if initial_base_pose is None:
            initial_base_pose = inputs.get("base_pose")
        target = raw_target
        if self._scene is not None:
            target = _resolve_station_name(raw_target, self._scene)
            logger.info("pick_up target: %r → %r", raw_target, target)

        # Physics grasp (only mode — no teleport fallback)
        if hasattr(self._backend, "grasp_object_physics"):
            try:
                ok = self._backend.grasp_object_physics(
                    target,
                    object_name=object_name,
                    initial_base_pose=initial_base_pose,
                )
                resolved_object = getattr(self._backend, "_held_crate_name", None) or object_name
                return SkillResult(
                    skill_name=self.name,
                    success=ok,
                    message=f"Physics grasp {'OK' if ok else 'FAIL'}: {target}",
                    payload={
                        "action": "pick_up",
                        "target": target,
                        "object_name": resolved_object,
                        "grasp_initial_base_pose": initial_base_pose,
                        "method": "physics",
                        "ok": ok,
                    },
                )
            except Exception as exc:
                logger.exception("physics grasp crashed")
                return SkillResult(
                    skill_name=self.name, success=False,
                    message=f"Physics grasp error: {exc}",
                    payload={
                        "action": "pick_up",
                        "target": target,
                        "object_name": object_name,
                        "grasp_initial_base_pose": initial_base_pose,
                        "error": str(exc),
                    },
                )

        # No physics configured — teleport only
        try:
            self._backend.pick_object(target)
        except Exception:
            pass
        return SkillResult(
            skill_name=self.name, success=True,
            message=f"Grasped (snap): {target}",
            payload={"action": "pick_up", "target": target, "raw_target": raw_target, "method": "teleport"},
        )
