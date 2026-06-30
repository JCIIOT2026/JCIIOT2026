"""Move skill — navigate the robot base to a target via A* + backend."""

from __future__ import annotations

import logging
import re

import numpy as np

from robot_agent.core.types import ExecutionContext, SkillResult
from robot_agent.skills.base import BaseSkill

logger = logging.getLogger(__name__)


class MoveSkill(BaseSkill):
    """Navigate the mobile base to a named station or world coordinate.

    Requires a backend, scene context, and occupancy grid — no mock fallback.
    """

    def __init__(
        self,
        *,
        backend,
        scene_context,
        grid: np.ndarray,
        path_spacing: float = 0.35,
    ) -> None:
        super().__init__(
            name="move",
            description="Move to a specified location",
            keywords=(
                "move", "go", "navigate",
                "move", "go", "navigate", "travel", "drive", "approach",
            ),
        )
        self._backend = backend
        self._scene = scene_context
        self._grid = grid
        self._path_spacing = path_spacing

    # ── public API ──────────────────────────────────────────

    def run(self, context: ExecutionContext) -> SkillResult:
        target: str = (
            context.metadata.get("inputs", {}).get("target")
            or context.task
        )

        goal_xy = self._resolve_target(target)
        if goal_xy is None:
            return SkillResult(
                skill_name=self.name,
                success=False,
                message=f"Cannot resolve target location: {target}",
                payload={"action": "move", "target": target},
            )

        start_xy, start_yaw = self._backend.get_base_pose()
        path = self._plan(start_xy, goal_xy)
        if path is None:
            return SkillResult(
                skill_name=self.name,
                success=False,
                message=f"A* planning failed: {target}",
                payload={"action": "move", "target": target, "start": start_xy.tolist()},
            )

        reached = self._backend.follow_path(path)
        final_xy, final_yaw = self._backend.get_base_pose()
        return SkillResult(
            skill_name=self.name,
            success=reached,
            message=f"Moved to: {target}" if reached else f"Failed to reach: {target}",
            payload={
                "action": "move",
                "target": target,
                "goal_xy": goal_xy.tolist(),
                "start_base_pose": {
                    "xy": start_xy.tolist(),
                    "yaw": float(start_yaw),
                    "robot_base_pos": [float(start_xy[0]), float(start_xy[1]), 0.0],
                    "robot_base_ori": [0.0, 0.0, float(start_yaw)],
                },
                "final_base_pose": {
                    "xy": final_xy.tolist(),
                    "yaw": float(final_yaw),
                    "robot_base_pos": [float(final_xy[0]), float(final_xy[1]), 0.0],
                    "robot_base_ori": [0.0, 0.0, float(final_yaw)],
                },
                "waypoints": len(path),
                "reached": reached,
            },
        )

    # ── internal ────────────────────────────────────────────

    def _resolve_target(self, target: str) -> np.ndarray | None:
        """Convert a target description to a (2,) world xy position.

        Resolution order:
        1. Known station name via ``SceneContext.approach_xy()``
        2. Direct (x, y) tuple in the target string
        """
        # 1) named station
        for name in self._scene.all_port_names():
            if name in target:
                return self._scene.approach_xy(name)

        # 2) numeric "x, y"
        nums = re.findall(r"[-+]?\d*\.?\d+", target)
        if len(nums) >= 2:
            try:
                return np.array([float(nums[0]), float(nums[1])], dtype=float)
            except ValueError:
                pass

        return None

    def _plan(
        self, start_xy: np.ndarray, goal_xy: np.ndarray,
    ) -> list[np.ndarray] | None:
        """Run A* and return a world-frame path, or None on failure."""
        from robot_agent.core.map_loader import plan_world_path

        try:
            scene_dict = {
                "bounds": self._scene.bounds,
                "resolution": self._scene.resolution,
            }
            return plan_world_path(
                scene_dict, self._grid, start_xy, goal_xy,
                min_spacing=self._path_spacing,
            )
        except Exception:
            logger.exception("A* planning failed")
            return None
