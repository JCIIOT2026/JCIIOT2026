"""
Abstract environment backend for robot skills.

Every concrete backend (robosuite, ROS2, mock) implements this protocol
so that skills like ``move`` / ``pick_up`` / ``place_down`` work without
knowing which backend is wired in.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class EnvBackend(Protocol):
    """Minimal interface that every simulation or real-robot backend must satisfy.

    All methods are synchronous and blocking by default; a backend wrapping
    async hardware should handle that internally.
    """

    # ── lifecycle ───────────────────────────────────────────

    def reset(self) -> None:
        """Reset the environment to its initial state."""
        ...

    def close(self) -> None:
        """Tear down the environment, free resources."""
        ...

    # ── robot state ─────────────────────────────────────────

    def get_base_pose(self) -> tuple[np.ndarray, float]:
        """Return the robot's 2D base position (x, y) and yaw (rad)."""
        ...

    # ── navigation ──────────────────────────────────────────

    def follow_path(
        self,
        path: list[np.ndarray],
        *,
        max_steps: int = 3000,
        waypoint_tolerance: float = 0.18,
        stop_on_collision: bool = True,
        debug: bool = False,
    ) -> bool:
        """Drive the robot base along a world-frame path.

        Args:
            path: Sequence of (2,) waypoints in world frame.
            max_steps: Maximum simulation steps before giving up.
            waypoint_tolerance: Distance (m) threshold to consider a waypoint reached.
            stop_on_collision: If True, abort early on unexpected contact.
            debug: Print per-step diagnostics.

        Returns:
            True if the final waypoint was reached, False otherwise.
        """
        ...

    # ── manipulation (optional — may raise NotImplementedError) ─

    def pick_object(self, target: str) -> bool:
        """Grasp the object identified by *target*."""
        raise NotImplementedError("pick_object not available on this backend")

    def place_object(self, target: str) -> bool:
        """Release the held object at *target*."""
        raise NotImplementedError("place_object not available on this backend")

    # ── rendering ───────────────────────────────────────────

    def render(self) -> None:
        """Update the on-screen viewer (no-op if headless)."""
        ...

    # ── action space ────────────────────────────────────────

    @property
    def action_spec(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (low, high) action bounds."""
        ...


class MockBackend:
    """A no-op backend for offline testing of the planner → skill pipeline.

    Every method succeeds immediately with plausible fake data.
    """

    def __init__(self, *, base_pos: np.ndarray | None = None, base_yaw: float = 0.0):
        self._pos = base_pos.copy() if base_pos is not None else np.array([0.0, 0.0])
        self._yaw = float(base_yaw)
        self._step_count = 0

    def reset(self) -> None:
        self._step_count = 0

    def close(self) -> None:
        pass

    def get_base_pose(self) -> tuple[np.ndarray, float]:
        return self._pos.copy(), self._yaw

    def follow_path(
        self,
        path: list[np.ndarray],
        *,
        max_steps: int = 3000,
        waypoint_tolerance: float = 0.18,
        stop_on_collision: bool = True,
        debug: bool = False,
    ) -> bool:
        if not path:
            return True
        self._pos = path[-1].copy()
        self._step_count += len(path)
        return True

    def pick_object(self, target: str) -> bool:
        return True

    def place_object(self, target: str) -> bool:
        return True

    def render(self) -> None:
        pass

    @property
    def action_spec(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.zeros(1, dtype=float)
        hi = np.ones(1, dtype=float)
        return lo, hi
