"""Core robot agent orchestrator with retry, timeout, precondition checks,
feature gates, and execution timing.

All output uses the canonical types in ``robot_agent.core.output``.
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robot_agent.config import AgentConfig
from robot_agent.core.memory import InMemoryStore
from robot_agent.core.ollama_client import OllamaClient
from robot_agent.core.output import StepOutput, ThinkingOutput, TaskOutput
import os as _os


def _build_llm_client(config):
    """Auto-select the LLM backend based on environment variables.

    Priority:
    1. LOCAL_LLM_MODEL  → local GGUF file (llama-cpp-python)
    2. GLM_API_KEY       → Zhipu GLM cloud API
    3. GATE_OLLAMA=true  → Ollama server (default)
    """
    import logging
    _log = logging.getLogger(__name__)

    # 1) Local GGUF model file
    local_path = _os.getenv("LOCAL_LLM_MODEL", "")
    if local_path:
        _log.info("Using local LLM: %s", local_path)
        from robot_agent.core.local_llm import LocalLLM
        return LocalLLM(model_path=local_path)

    # 2) GLM cloud API
    glm_key = _os.getenv("GLM_API_KEY", "")
    if glm_key:
        _log.info("Using GLM API")
        from robot_agent.core.glm_client import GlmClient
        return GlmClient(api_key=glm_key)

    # 3) Ollama (default)
    if config.feature_gates.ollama:
        _log.info("Using Ollama: %s / %s", config.ollama_base_url, config.ollama_model)
        return OllamaClient(
            base_url=config.ollama_base_url,
            model=config.ollama_model,
            timeout=config.ollama_timeout,
        )

    raise RuntimeError(
        "No LLM backend configured. Set one of:\n"
        "  LOCAL_LLM_MODEL=/path/to/model.gguf\n"
        "  GLM_API_KEY=your-key\n"
        "  GATE_OLLAMA=true + OLLAMA_BASE_URL=http://..."
    )
from robot_agent.core.planner import PlanDecision, TaskPlanner
from robot_agent.core.registry import Skill, SkillRegistry
from robot_agent.core.scene_context import SceneContext
from robot_agent.core.types import AgentStep, ExecutionContext, SkillResult
from robot_agent.environments.base import EnvBackend
from robot_agent.skills.library import wired_skills

logger = logging.getLogger(__name__)


@dataclass
class RobotAgent:
    config: AgentConfig = field(default_factory=AgentConfig)
    registry: SkillRegistry = field(default_factory=SkillRegistry)

    # ── required wiring ─────────────────────────────────────
    backend: EnvBackend | None = None
    scene_context: SceneContext | None = None
    grid: np.ndarray | None = None
    path_spacing: float = 0.35
    scene_metadata: dict[str, Any] = field(default_factory=dict)
    knowledge_enabled: bool = True

    memory: InMemoryStore = field(init=False)
    planner: TaskPlanner = field(init=False)

    def __post_init__(self) -> None:
        if not self.config.feature_gates.ollama:
            raise RuntimeError(
                "LLM backend is required. Set GATE_OLLAMA=true and configure "
                "Ollama before starting the agent."
            )
        if self.backend is None:
            raise RuntimeError("backend is required. Pass a RobosuiteBackend (or MockBackend for tests).")
        if self.scene_context is None:
            raise RuntimeError("scene_context is required. Pass a SceneContext from your semantic map.")
        if self.grid is None:
            raise RuntimeError("grid is required. Pass a 2D occupancy grid (npy).")

        self.memory = InMemoryStore(limit=self.config.memory_limit)
        self._bootstrap_skills()
        client = _build_llm_client(self.config)
        self.planner = TaskPlanner(
            client, self.registry, self.config,
            scene_context=self.scene_context,
            knowledge_enabled=self.knowledge_enabled,
        )

    # ── skills ────────────────────────────────────────────

    def _bootstrap_skills(self) -> None:
        skills = wired_skills(
            self.backend,
            scene_context=self.scene_context,
            grid=self.grid,
            path_spacing=self.path_spacing,
            memory_store=self.memory,
        )
        for skill in skills:
            self.registry.register(skill)

    def _skill_metadata(self, **extra: Any) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "agent": self.config.name,
            "scene": dict(self.scene_metadata),
        }
        for key in ("scene_name", "env_name", "task_index"):
            if key in self.scene_metadata:
                metadata[key] = self.scene_metadata[key]
        metadata.update(extra)
        return metadata

    def _inputs_with_scene_object(self, inputs: dict[str, Any]) -> dict[str, Any]:
        resolved = dict(inputs)
        object_keys = ("object_name", "obj_name", "object", "target_object")
        if any(resolved.get(key) for key in object_keys):
            return resolved

        target = resolved.get("target")
        if not target:
            return resolved

        input_object_map = self.scene_metadata.get("input_object_map", {})
        if not isinstance(input_object_map, dict):
            return resolved

        object_name = input_object_map.get(str(target))
        if object_name:
            resolved["object_name"] = object_name
        return resolved

    # ── step execution ────────────────────────────────────

    @staticmethod
    def _inputs_with_previous_move_pose(
        inputs: dict[str, Any],
        skill_name: str,
        previous_steps: list[StepOutput],
    ) -> dict[str, Any]:
        if skill_name != "pick_up" or inputs.get("grasp_initial_base_pose"):
            return inputs
        if not previous_steps:
            return inputs

        previous = previous_steps[-1]
        if previous.skill != "move" or not previous.success:
            return inputs

        final_base_pose = (previous.payload or {}).get("final_base_pose")
        if not isinstance(final_base_pose, dict):
            return inputs

        resolved = dict(inputs)
        resolved["grasp_initial_base_pose"] = final_base_pose
        return resolved

    @staticmethod
    def _truthy_step_flag(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @staticmethod
    def _is_soft_pick_failure(step: dict[str, Any], result: StepOutput) -> bool:
        if result.success or result.skill != "pick_up":
            return False

        payload = result.payload or {}
        if payload.get("error"):
            return False

        normal_physics_failure = (
            payload.get("action") == "pick_up"
            and payload.get("method") == "physics"
            and payload.get("ok") is False
        )
        explicit_continue = any(
            RobotAgent._truthy_step_flag(step.get(flag))
            for flag in ("continue_on_failure", "optional_failure", "soft_fail")
        )
        return normal_physics_failure or (explicit_continue and payload.get("action") == "pick_up")

    @staticmethod
    def _previous_step_allows_continuation(previous_steps: list[StepOutput]) -> bool:
        if not previous_steps:
            return True
        previous = previous_steps[-1]
        return bool(previous.success or (previous.payload or {}).get("soft_failure"))

    def _execute_step(
        self,
        step: dict[str, Any],
        step_index: int,
        previous_steps: list[StepOutput],
    ) -> StepOutput:
        skill_name: str = step.get("skill_name", "")
        description: str = step.get("description") or step.get("task") or ""
        raw_inputs = step.get("inputs") or {}
        inputs: dict = dict(raw_inputs) if isinstance(raw_inputs, dict) else {}
        inputs = self._inputs_with_scene_object(inputs)
        inputs = self._inputs_with_previous_move_pose(inputs, skill_name, previous_steps)
        preconditions: list[str] = step.get("preconditions") or []
        expected_output: str = step.get("expected_output") or ""
        # Use max of plan timeout and config timeout (plan can't override below config minimum)
        plan_timeout = step.get("timeout")
        if plan_timeout:
            timeout = max(float(plan_timeout), self.config.step_default_timeout)
        else:
            timeout = self.config.step_default_timeout
        # Cap retries at config limit
        plan_retries = step.get("retries")
        if plan_retries is not None:
            max_retries = min(int(plan_retries), self.config.step_max_retries)
        else:
            max_retries = self.config.step_max_retries

        # resolve skill
        skill: Skill | None = self.registry.get(skill_name)
        if skill is None:
            skill = self.registry.find(description)
        if skill is None:
            return StepOutput(
                skill=skill_name or "unknown",
                description=description,
                success=False,
                message=f"Unknown skill '{skill_name}'",
                inputs=inputs,
                preconditions=preconditions,
                expected_output=expected_output,
                timeout=timeout,
                retries=max_retries,
            )
        inputs = self._inputs_with_previous_move_pose(inputs, skill.name, previous_steps)

        # preconditions (gated) — skipped if last step already succeeded
        _last_ok = self._previous_step_allows_continuation(previous_steps)
        if not _last_ok and self.config.feature_gates.precondition_validation and preconditions:
            ok, msg = self._check_preconditions(preconditions, previous_steps)
            if not ok:
                return StepOutput(
                    skill=skill.name,
                    description=description,
                    success=False,
                    message=f"Precondition not met: {msg}",
                    inputs=inputs,
                    preconditions=preconditions,
                    expected_output=expected_output,
                    timeout=timeout,
                    retries=max_retries,
                )

        # execute with retry (gated) + timeout (gated)
        effective_max_retries = (
            max_retries if self.config.feature_gates.step_retry else 0
        )
        last_result: SkillResult | None = None

        for attempt in range(effective_max_retries + 1):
            ctx = ExecutionContext(
                task=description,
                metadata=self._skill_metadata(
                    from_plan=True,
                    inputs=inputs,
                    step_index=step_index,
                    attempt=attempt,
                ),
            )
            try:
                result = (
                    self._run_with_timeout(skill.run, ctx, timeout=timeout)
                    if self.config.feature_gates.step_timeout
                    else skill.run(ctx)
                )
                if result.success:
                    logger.info("Step %d (%s): success on attempt %d", step_index, skill.name, attempt + 1)
                    return StepOutput(
                        skill=skill.name,
                        description=description,
                        success=True,
                        message=result.message,
                        payload=result.payload or {},
                        inputs=inputs,
                        preconditions=preconditions,
                        expected_output=expected_output,
                        timeout=timeout,
                        retries=max_retries,
                        attempts=attempt + 1,
                    )
                last_result = result
            except TimeoutError:
                last_result = SkillResult(skill_name=skill.name, success=False, message=f"Timed out after {timeout:.0f}s")
                logger.warning("Step %d (%s): timeout (attempt %d/%d)", step_index, skill.name, attempt + 1, effective_max_retries + 1)
            except Exception:
                logger.exception("Step %d (%s): error on attempt %d", step_index, skill.name, attempt + 1)
                last_result = SkillResult(skill_name=skill.name, success=False, message=f"Error on attempt {attempt + 1}")

        logger.error("Step %d (%s): all %d attempts failed", step_index, skill.name, effective_max_retries + 1)
        return StepOutput(
            skill=skill.name,
            description=description,
            success=False,
            message=(last_result.message if last_result else "No attempts made"),
            payload=last_result.payload if last_result else {},
            inputs=inputs,
            preconditions=preconditions,
            expected_output=expected_output,
            timeout=timeout,
            retries=max_retries,
            attempts=effective_max_retries + 1,
        )

    @staticmethod
    def _run_with_timeout(fn, ctx: ExecutionContext, *, timeout: float) -> SkillResult:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fn, ctx)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"Step timed out after {timeout:.0f}s") from None

    @staticmethod
    def _check_preconditions(
        preconditions: list[str], previous_steps: list[StepOutput]
    ) -> tuple[bool, str]:
        """Check preconditions using fuzzy character-level matching.

        Three-tier check:
        1. Exact substring — fast path
        2. Alphanumeric tokens (A/B/C/1/2) — MUST all appear (distinguishes 仓库B vs 仓库C)
        3. Character overlap — ≥40% of chars found in context
        """
        if not preconditions:
            return True, ""

        # build context from all previous steps
        prev_context = ""
        for s in previous_steps:
            prev_context += f" {s.message} "
            prev_context += " ".join(
                str(v) for v in s.payload.values() if isinstance(v, str)
            )
            prev_context += f" {s.description} "
        prev_lower = prev_context.lower()

        for cond in preconditions:
            cond_lower = cond.lower().strip()
            if not cond_lower:
                continue

            # 1) exact or substring match
            if cond_lower in prev_lower:
                continue

            # 2) label tokens MUST appear (A/B/C/1/2 — location markers)
            import re
            label_tokens = set(re.findall(r"[a-z0-9]+", cond_lower))
            if label_tokens:
                prev_labels = set(re.findall(r"[a-z0-9]+", prev_lower))
                missing = label_tokens - prev_labels
                if missing:
                    return False, (
                        f"'{cond}' not satisfied "
                        f"(missing label: {', '.join(sorted(missing))})"
                    )

            # 3) character overlap on Chinese text (fuzzy, handles 已到达 vs 已移动)
            cond_han = re.sub(r"[a-z0-9]+", "", cond_lower).replace(" ", "")
            if cond_han:
                prev_han = re.sub(r"[a-z0-9]+", "", prev_lower)
                overlap = sum(1 for c in set(cond_han) if c in prev_han)
                ratio = overlap / len(set(cond_han))
                if ratio < 0.4:
                    return False, (
                        f"'{cond}' not satisfied "
                        f"(char overlap {ratio:.0%}, need >=40%)"
                    )

        return True, ""

    # ── main entry point ──────────────────────────────────

    def run(self, task: str) -> TaskOutput:
        t_start = time.perf_counter()

        # planning (gated)
        if self.config.feature_gates.planner:
            decision: PlanDecision = self.planner.plan(task, scene_metadata=self.scene_metadata)
        else:
            decision = PlanDecision(
                skill_name="fallback", response="", raw="", raw_llm_text="",
                details={
                    "version": "2.0", "understanding": str(task),
                    "reason": "Planner disabled via GATE_PLANNER",
                    "plan": [], "explanation": "", "warnings": [],
                },
            )

        details: dict[str, Any] = getattr(decision, "details", {}) or {}
        plan: list[dict[str, Any]] | None = (
            details.get("plan") if isinstance(details, dict) else None
        )

        if plan and isinstance(plan, list):
            output = self._run_multi_step(task, decision, plan, details)
        else:
            output = self._run_single_step(task, decision)

        output.elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 1)
        logger.info("Task completed in %.1f ms", output.elapsed_ms)

        self.memory.add(AgentStep(task=task, skill_name=output.skill_name,
                                  result=SkillResult(skill_name=output.skill_name,
                                                     success=output.success,
                                                     message=output.message,
                                                     payload=output.as_dict())))
        return output

    def _run_multi_step(
        self, task: str, decision: PlanDecision,
        plan: list[dict[str, Any]], details: dict[str, Any],
    ) -> TaskOutput:
        gates = self.config.feature_gates
        steps: list[StepOutput] = []
        overall_success = True

        for i, step in enumerate(plan):
            result = self._execute_step(step, i, steps)
            soft_failure = self._is_soft_pick_failure(step, result)
            if soft_failure:
                result.payload = dict(result.payload or {})
                result.payload["soft_failure"] = True
                result.payload["continued_after_failure"] = True
                result.payload["failure_policy"] = "continue_after_pick_up_failure"
            steps.append(result)
            if not result.success:
                overall_success = False
                logger.warning(
                    "Step %d (%s) failed: %s",
                    i + 1, result.skill or "unknown", result.message,
                )
                if soft_failure:
                    logger.warning("Step %d failed softly, continuing after pick_up failure", i + 1)
                elif gates.fail_fast:
                    logger.warning("Step %d failed, fail_fast — stopping", i + 1)
                    break

        messages = [s.message for s in steps]
        thinking = ThinkingOutput.from_details(details, plan)

        return TaskOutput(
            skill_name="composed",
            success=overall_success,
            message=decision.response or ("; ".join(messages)),
            steps=steps,
            thinking=thinking,
            planner_raw=decision.raw_llm_text,
            plan_warnings=details.get("warnings", []),
        )

    def _run_single_step(self, task: str, decision: PlanDecision) -> TaskOutput:
        context = ExecutionContext(task=task, metadata=self._skill_metadata())
        skill: Skill | None = self.registry.get(decision.skill_name)
        if skill is None:
            skill = self.registry.find(task)
        if skill is None:
            raise RuntimeError(f"No skill found for task: {task}")

        result = skill.run(context)
        if decision.response:
            result.message = decision.response

        details: dict[str, Any] = getattr(decision, "details", {}) or {}
        thinking = ThinkingOutput.from_details(details)

        return TaskOutput(
            skill_name=skill.name,
            success=result.success,
            message=result.message,
            payload=result.payload or {},
            thinking=thinking,
            planner_raw=decision.raw_llm_text,
            plan_warnings=details.get("warnings", []),
        )
