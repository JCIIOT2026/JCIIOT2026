"""
Robosuite backend for the robot agent.

Wraps the ``FactorySorting`` / ``TiagoSortingTable`` MuJoCo environments
and exposes the ``EnvBackend`` protocol via the ``RobosuiteBackend`` class.

Derived from ``llm_task_navigator.py`` — all simulation coupling lives here.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

# ── monkey-patch: fix robosuite.__file__ ───────────────────
# The outer robosuite/ dir has no __init__.py, making robosuite a namespace
# package.  Its __file__ is None, breaking controller config loading.  We patch
# it to point to the inner robosuite/robosuite/__init__.py.
_THIS_FILE = Path(__file__).resolve()
_INNER_INIT_DIR = _THIS_FILE.parents[3] / "robosuite" / "robosuite"
_INNER_INIT = _INNER_INIT_DIR / "__init__.py"
if not _INNER_INIT.exists():
    # Fallback: walk up until we find robosuite/robosuite/__init__.py
    for _p in _THIS_FILE.parents:
        _candidate = _p / "robosuite" / "robosuite" / "__init__.py"
        if _candidate.exists():
            _INNER_INIT_DIR = _candidate.parent
            _INNER_INIT = _candidate
            break

# Ensure v4 inner is first in sys.path so submodules resolve here
if str(_INNER_INIT_DIR) not in sys.path:
    sys.path.insert(0, str(_INNER_INIT_DIR))

import robosuite as _rs
if _rs.__file__ is None and _INNER_INIT.exists():
    _rs.__file__ = str(_INNER_INIT)
    _rs.__path__ = [str(_INNER_INIT_DIR)]  # force submodule resolution to v4
    # Also set __version__ if the namespace package missed it
    import runpy as _runpy
    _ns = _runpy.run_path(str(_INNER_INIT))
    for _attr in ("__version__", "__logo__"):
        if _attr in _ns and not hasattr(_rs, _attr):
            setattr(_rs, _attr, _ns[_attr])
    # Also import make from the real module
    _rs.make = _ns.get("make")
    _rs.FactorySorting = _ns.get("FactorySorting")

DEFAULT_IGNORED_COLLISION_GEOMS: set[str | None] = {
    "floor", "ground", "world", None,
}

# Substring patterns for scene fixtures that the arm may touch in its default
# pose.  Unlike DEFAULT_IGNORED_COLLISION_GEOMS (exact match), these match
# any geom whose name *contains* the pattern.
DEFAULT_IGNORED_COLLISION_PATTERNS: tuple[str, ...] = (
    "table_top",       # station tables near the robot
    "conveyor",        # central conveyor rails / belts
    "shelf",           # warehouse shelves
    "container",       # Siemens containers on tables
    "tote",            # Siemens totes on tables
    "cardbox",         # Siemens cardbox on tables
    "plastic_crate",   # plastic crates on tables
)


# ── robot parameter loading ─────────────────────────────────

_ROBOT_PARAMS_CACHE: dict | None = None


def _load_robot_params() -> dict:
    """Load robot execution parameters from ``knowledge/robot_params.json``.

    Missing keys or unparseable JSON silently fall back to the factory
    defaults defined here.  Every numeric value is clamped to a safe range
    so the simulation stays stable.
    """
    global _ROBOT_PARAMS_CACHE
    if _ROBOT_PARAMS_CACHE is not None:
        return _ROBOT_PARAMS_CACHE

    import json as _json

    _THIS_FILE = Path(__file__).resolve()
    _KNOWLEDGE_DIR = _THIS_FILE.parents[3] / "knowledge"
    _PARAMS_PATH = _KNOWLEDGE_DIR / "robot_params.json"

    # ── factory defaults (same as the original hardcoded values) ──
    _defaults: dict = {
        "llm": {
            "ollama_base_url": "https://pa1l785z-a9z3rfcp-11434.zj02restapi.gpufree.cn:8443",
            "ollama_model": "qwen3.6:27b-mtp-q4_K_M",
            "vision_model": "qwen3-vl:8b",
        },
        "navigation": {
            "max_steps": 3000,
            "waypoint_tolerance": 0.01,
            "settle_steps": 5,
            "max_linear": 0.70,
            "max_angular": 1.20,
            "k_linear": 1.2,
            "k_angular": 1.8,
            "holonomic_base": True,
            "yaw_control": False,
            "turn_in_place_angle": 0.65,
            "drive_mode": "direct",
            "control_freq": 20,
        },
        "grasp_policy": {
            "checkpoint_path": "robosuite/robosuite/model_epoch_500.pth",
            "checkpoint_fallback_path": "robosuite/robosuite/model_epoch_150.pth",
            "eval_steps": 360,
            "post_hold_steps": 5,
            "initial_view_steps": 5,
            "debug_policy": False,
            "debug_every": 25,
            "record_frame_interval": 5,
        },
        "lift": {
            "lift_height": 0.15,
            "max_steps": 300,
            "hold_steps": 20,
            "tolerance": 0.02,
            "max_action": 0.80,
        },
        "turn": {
            "tolerance": 0.02,
            "max_iters": 8,
            "turn_steps": 40,
            "settle_steps": 10,
        },
        "place": {
            "lower_delta": 0.18,
            "lower_steps": 25,
            "release_steps": 60,
            "release_clearance": 0.04,
        },
    }

    # ── valid ranges (min, max) ──
    _ranges: dict[tuple[str, str], tuple[float, float]] = {
        ("navigation", "max_steps"): (500, 10000),
        ("navigation", "waypoint_tolerance"): (0.001, 0.10),
        ("navigation", "settle_steps"): (0, 20),
        ("navigation", "max_linear"): (0.1, 2.0),
        ("navigation", "max_angular"): (0.1, 3.0),
        ("navigation", "k_linear"): (0.1, 5.0),
        ("navigation", "k_angular"): (0.1, 5.0),
        ("navigation", "turn_in_place_angle"): (0.1, 3.0),
        ("navigation", "control_freq"): (10, 50),
        ("grasp_policy", "eval_steps"): (100, 1000),
        ("grasp_policy", "post_hold_steps"): (0, 30),
        ("grasp_policy", "initial_view_steps"): (0, 30),
        ("grasp_policy", "debug_every"): (1, 100),
        ("grasp_policy", "record_frame_interval"): (1, 50),
        ("lift", "lift_height"): (0.05, 0.50),
        ("lift", "max_steps"): (50, 1000),
        ("lift", "hold_steps"): (5, 100),
        ("lift", "tolerance"): (0.001, 0.10),
        ("lift", "max_action"): (0.10, 2.0),
        ("turn", "tolerance"): (0.001, 0.10),
        ("turn", "max_iters"): (1, 30),
        ("turn", "turn_steps"): (10, 200),
        ("turn", "settle_steps"): (0, 50),
        ("place", "lower_delta"): (0.05, 0.50),
        ("place", "lower_steps"): (5, 100),
        ("place", "release_steps"): (10, 300),
        ("place", "release_clearance"): (0.01, 0.20),
    }

    # ── allowed enum values ──
    _enums: dict[tuple[str, str], set] = {
        ("navigation", "drive_mode"): {"direct", "action"},
        ("navigation", "holonomic_base"): {True, False},
        ("navigation", "yaw_control"): {True, False},
        ("grasp_policy", "debug_policy"): {True, False},
    }

    # ── load user JSON ──
    _user: dict = {}
    if _PARAMS_PATH.exists():
        try:
            _user = _json.loads(_PARAMS_PATH.read_text(encoding="utf-8"))
            if not isinstance(_user, dict):
                _user = {}
        except Exception as exc:
            logger.warning("robot_params.json parse failed (%s) — using factory defaults", exc)

    # ── deep-merge with clamping ──
    _result: dict = {}
    for _section, _sec_defaults in _defaults.items():
        _result[_section] = {}
        _user_section = _user.get(_section, {}) if isinstance(_user, dict) else {}
        if not isinstance(_user_section, dict):
            _user_section = {}
        for _key, _default in _sec_defaults.items():
            _user_val = _user_section.get(_key)
            if _user_val is None:
                _result[_section][_key] = _default
                continue

            _range = _ranges.get((_section, _key))
            _enum = _enums.get((_section, _key))

            if _enum is not None:
                if _user_val in _enum:
                    _result[_section][_key] = _user_val
                else:
                    logger.warning(
                        "robot_params: %s.%s=%r ∉ %s → default %r",
                        _section, _key, _user_val, _enum, _default,
                    )
                    _result[_section][_key] = _default
            elif _range is not None:
                try:
                    _clamped = max(_range[0], min(float(_user_val), _range[1]))
                    if isinstance(_default, int):
                        _clamped = int(_clamped)
                    _result[_section][_key] = _clamped
                except (TypeError, ValueError):
                    logger.warning(
                        "robot_params: %s.%s=%r invalid → default %r",
                        _section, _key, _user_val, _default,
                    )
                    _result[_section][_key] = _default
            else:
                _result[_section][_key] = _user_val

    _ROBOT_PARAMS_CACHE = _result
    logger.info("Robot params loaded from %s", _PARAMS_PATH if _PARAMS_PATH.exists() else "factory defaults")
    return _result


# ── backend class ───────────────────────────────────────────

_TRAJECTORY_EXCLUDED_JOINTS: set[str] = {
    "mobilebase0_joint_mobile_forward",
    "mobilebase0_joint_mobile_side",
    "mobilebase0_joint_mobile_yaw",
}


def _without_trajectory_excluded_joints(joint_positions: dict[str, float]) -> dict[str, float]:
    return {
        name: value
        for name, value in joint_positions.items()
        if name not in _TRAJECTORY_EXCLUDED_JOINTS
    }


class RobosuiteBackend:
    """Concrete ``EnvBackend`` that drives a robosuite simulation.

    Usage::

        backend = RobosuiteBackend(
            env_name="FactorySorting",
            camera="birdview",
            drive_mode="direct",
            seed=42,
        )
        backend.reset()
        reached = backend.follow_path(path)
        backend.close()
    """

    def __init__(
        self,
        *,
        env_name: str = "FactorySorting1_3FO3ERFHISEM",
        robots: str = "Tiago",
        camera: str = "birdview",
        headless: bool = False,
        control_freq: int | None = None,
        drive_mode: str | None = None,       # "direct" | "action"; None → from robot_params.json
        seed: int | None = None,
        # collision (NOT in robot_params.json — safety constraint)
        stop_on_collision: bool = True,
        collision_warmup_steps: int = 5,
        ignore_collision_geom: Sequence[str] = (),
        max_collision_pairs: int = 8,
        # action-mode gains (None → from robot_params.json)
        k_linear: float | None = None,
        k_angular: float | None = None,
        max_linear: float | None = None,
        max_angular: float | None = None,
        holonomic_base: bool | None = None,
        yaw_control: bool | None = None,
        turn_in_place_angle: float | None = None,
    ):
        # ── resolve defaults from robot_params.json ──
        self._rp = _load_robot_params()
        _nav = self._rp["navigation"]

        self._env_name = env_name
        self._robots = robots
        self._camera = camera
        self._headless = headless
        self._control_freq = control_freq if control_freq is not None else _nav["control_freq"]
        self._drive_mode = drive_mode if drive_mode is not None else _nav["drive_mode"]
        self._seed = seed

        # collision settings (hardcoded — NOT in robot_params.json)
        self._stop_on_collision = stop_on_collision
        self._collision_warmup_steps = collision_warmup_steps
        self._ignore_collision_geom = list(ignore_collision_geom)
        self._max_collision_pairs = max_collision_pairs

        # action-mode settings
        self._k_linear = k_linear if k_linear is not None else _nav["k_linear"]
        self._k_angular = k_angular if k_angular is not None else _nav["k_angular"]
        self._max_linear = max_linear if max_linear is not None else _nav["max_linear"]
        self._max_angular = max_angular if max_angular is not None else _nav["max_angular"]
        self._holonomic_base = holonomic_base if holonomic_base is not None else _nav["holonomic_base"]
        self._yaw_control = yaw_control if yaw_control is not None else _nav["yaw_control"]
        self._turn_in_place_angle = turn_in_place_angle if turn_in_place_angle is not None else _nav["turn_in_place_angle"]

        self._env = None
        self._robot_geom_names: set[str] | None = None
        self._recorded_frames: list[np.ndarray] = []
        self._trajectory: list[dict] = []       # trajectory JSON frames
        self._trajectory_events: list[dict] = []
        self._trajectory_start_time: float = 0.0
        self._trajectory_autosave_path: Path | None = None
        self._trajectory_autosave_every: int = 0
        self._trajectory_autosave_last_count: int = 0
        self._held_crate_name: str | None = None
        self._held_crate_body_id: int | None = None
        self._capture_grasp_frames = False
        self._grasp_frames: list[np.ndarray] = []

    # ── lifecycle ───────────────────────────────────────────

    def reset(self) -> None:
        if self._env is not None:
            self._env.close()
        self._wrapped_env = None
        self._nav_env = None
        has_physics = getattr(self, "_has_physics", False)
        # Nav env: viewer on for physics, offscreen always available
        show_win = has_physics
        self._env = _make_env(
            self._env_name, robots=self._robots, camera=self._camera,
            headless=not show_win, control_freq=self._control_freq, seed=self._seed,
            force_offscreen=show_win,
            use_camera_obs=False, camera_names="agentview",
            camera_heights=256, camera_widths=256,
        )
        self._env.reset()
        if has_physics:
            self._ensure_physics_policy()
            try:
                _set_viewer_camera(self._env, "birdview", render_once=True)
            except Exception:
                pass
        self._recorded_frames = []
        self._robot_geom_names = None

    def close(self) -> None:
        self._wrapped_env = None
        self._nav_env = None
        if self._env is not None:
            self._env.close()
            self._env = None

    @property
    def env(self):
        """Direct access to the robosuite environment (escape hatch)."""
        if self._env is None:
            raise RuntimeError("Backend not reset — call .reset() first.")
        return self._env

    # ── robot state ─────────────────────────────────────────

    def get_base_pose(self) -> tuple[np.ndarray, float]:
        return _get_base_pose(self._env)

    # ── manipulation ──────────────────────────────────────────

    def pick_object(self, target: str) -> bool:
        """Grasp an object at the named input station."""
        logger.info("pick_object called: target=%r", target)
        crate_name = self._find_crate_for_target(target)
        if crate_name is None:
            logger.warning(
                "pick_object: no object matching '%s'. Available: %s",
                target, sorted(self.env.obj_body_id.keys()),
            )
            # Best-effort: try to find ANY object at this port
            for name in self.env.obj_body_id:
                if name.startswith(target + "_"):
                    crate_name = name
                    logger.info("pick_object: fallback match '%s'", crate_name)
                    break
        if crate_name is None:
            logger.warning("pick_object: truly no match — ignoring")
            return False

        body_id = self.env.obj_body_id[crate_name]
        self._held_crate_name = crate_name
        self._held_crate_body_id = body_id
        self._update_held_crate_position()
        logger.info("pick_object: grasped '%s' (resolved from '%s')", crate_name, target)
        return True

    def get_available_crates(self) -> dict[str, str]:
        """Return {port_name: obj_name} for objects still at their input stations.

        An object is "available" if it's near the table surface (z < 0.55).
        Works with both PlasticCrateObject and Siemens mesh objects.
        """
        available: dict[str, str] = {}
        for port_name in self.env.input_ports:
            obj_name = self._find_crate_for_target(port_name)
            if obj_name is None:
                continue
            body_id = self.env.obj_body_id[obj_name]
            pos = self.env.sim.data.body_xpos[body_id]
            if pos[2] < 0.55:  # on table, not held
                available[port_name] = obj_name
        return available

    def _find_crate_for_target(self, target: str) -> str | None:
        """Find a material object name matching *target*.

        Works with both old-style PlasticCrateObject names and new-style
        Siemens mesh object names (cardbox, container, tote, etc.).

        Tries (in order):
        1. Exact match in obj_body_id.
        2. target + '_plastic_crate' suffix (legacy).
        3. Any material object whose name *starts with* target.
        4. Any material object whose name *contains* target.
        5. Fallback: any object starting with target+'_' (Siemens naming).
        """
        objs = self.env.obj_body_id
        # Filter out static scene objects — material objects contain port prefix + spec name
        static_names = {name for name in objs if any(
            name.endswith(suffix) for suffix in (
                '_table_top', '_table_leg_0', '_table_leg_1', '_table_leg_2', '_table_leg_3',
                '_base', '_belt_visual',
            )
        )}
        # Also skip objects ending with known static suffixes
        def _is_material(name: str) -> bool:
            if name in static_names:
                return False
            return not any(name.endswith(s) for s in (
                '_table_top', '_table_leg_0', '_table_leg_1', '_table_leg_2', '_table_leg_3',
                '_base', '_belt_visual', '_roller_', '_rail_',
            ))

        # 1) exact
        if target in objs:
            return target

        # 2) legacy suffix
        for suffix in ('_plastic_crate', '_cardbox_c2', '_container_h10', '_container_h01', '_tote_b01'):
            suffixed = f"{target}_{suffix}" if not target.endswith(suffix) else target
            if suffixed in objs:
                return suffixed

        # 3) prefix — e.g. target="input_1" matches "input_1_conveyor_cardbox_c2"
        for name in objs:
            if _is_material(name) and name.startswith(target + "_"):
                return name

        # 4) substring
        for name in objs:
            if _is_material(name) and target in name:
                return name

        # 5) material_metadata fallback (for hidden replacement objects without port prefix)
        metadata = getattr(self.env, "material_metadata", {}) or {}
        # First try: exact port_name match in metadata
        for obj_name, info in metadata.items():
            if not isinstance(info, dict):
                continue
            port_name = str(info.get("port_name") or "")
            if port_name == target:
                return obj_name
        # Second try: substring match in metadata keys
        for obj_name in metadata:
            if _is_material(obj_name) and target in obj_name:
                return obj_name

        return None

    def _has_object_joint(self, obj_name: str) -> bool:
        for suffix in ("_free", "_joint0"):
            try:
                self.env.sim.model.get_joint_qpos_addr(f"{obj_name}{suffix}")
                return True
            except Exception:
                continue
        return False

    def _has_grasp_sites(self, obj_name: str) -> bool:
        for arm in ("right", "left"):
            try:
                self.env.sim.model.site_name2id(f"{obj_name}_{arm}_grasp_site")
            except Exception:
                return False
        return True

    def _available_grasp_objects(self) -> list[str]:
        objects = list(getattr(self.env, "material_objects", []) or [])
        if not objects:
            metadata = getattr(self.env, "material_metadata", {}) or {}
            objects = list(metadata.keys())
        if not objects:
            objects = list(getattr(self.env, "obj_body_id", {}).keys())
        seen: set[str] = set()
        available: list[str] = []
        for name in objects:
            if name in seen:
                continue
            seen.add(name)
            if self._has_grasp_sites(name):
                available.append(name)
        return available

    def _env_grasp_object_candidates(self, source: str) -> list[str]:
        """Return grasp-object candidates discovered from the current env."""
        metadata = getattr(self.env, "material_metadata", {}) or {}
        material_objects = list(getattr(self.env, "material_objects", []) or [])
        source_names = [source]
        if source.startswith("line_"):
            source_names.append("input_" + source.split("_", 1)[1])
        elif source.startswith("input_"):
            source_names.append("line_" + source.split("_", 1)[1])

        candidates: list[str] = []
        for obj_name, info in metadata.items():
            if not isinstance(info, dict):
                continue
            port_name = str(info.get("port_name") or "")
            if port_name in source_names:
                candidates.append(obj_name)

        for obj_name in material_objects:
            if any(
                obj_name == name
                or obj_name.startswith(f"{name}_")
                for name in source_names
            ):
                candidates.append(obj_name)
        return candidates

    def _grasp_resolution_context(self) -> str:
        env_cls = type(self.env).__name__ if self._env is not None else "None"
        scene_meta = getattr(self, "_scene_metadata", {}) or {}
        expected_env = scene_meta.get("env_name") if isinstance(scene_meta, dict) else None
        return (
            f"backend_env={self._env_name}, actual_env_class={env_cls}, "
            f"expected_scene_env={expected_env}"
        )

    @staticmethod
    def _default_physics_object_map() -> dict[str, str]:
        """Return fallback object map from llm_task_navigator if available.

        NOTE: The old hardcoded per-scene mappings have been removed.  Object
        resolution now relies on ``material_metadata`` (which includes
        ``port_name`` for every object) and the dynamic ``input_object_map``
        built by ``task_subprocess_runner``.
        """
        mapping: dict[str, str] = {}
        try:
            from robosuite.environments.factory_sorting.llm_task_navigator import (
                FACTORY_SORTING_CURRENT_INPUT_OBJECTS,
            )
            mapping.update(FACTORY_SORTING_CURRENT_INPUT_OBJECTS)
        except Exception:
            pass
        return mapping

    def _is_valid_grasp_candidate(self, obj_name: str) -> bool:
        """Check if *obj_name* is a known graspable object from scene metadata.

        Validates against ``material_metadata`` first (the single source of
        truth), then falls back to the MuJoCo model joint/site checks.
        """
        metadata = getattr(self.env, "material_metadata", {}) or {}
        if obj_name in metadata:
            info = metadata[obj_name]
            if not isinstance(info, dict):
                return False
            # XML-based objects have model + joint_name
            if info.get("model") is not None and info.get("joint_name"):
                return True
            # Non-XML objects still registered with a joint_name
            if info.get("joint_name"):
                return True
        # Fallback: check MuJoCo model (for objects not in material_metadata)
        return self._has_object_joint(obj_name) and self._has_grasp_sites(obj_name)

    @staticmethod
    def _line_name_for_station(source: str) -> str | None:
        if source.startswith("input_"):
            return "line_" + source.split("_", 1)[1]
        return None

    def _resolve_grasp_object_name(self, source: str, object_name: str | None = None) -> str:
        object_map = self._default_physics_object_map()
        object_map.update(getattr(self, "_physics_object_map", {}) or {})

        candidates: list[str] = []
        if object_name:
            candidates.append(str(object_name).strip())
        candidates.extend(self._env_grasp_object_candidates(source))
        mapped = object_map.get(source)
        if mapped:
            candidates.append(mapped)

        found = self._find_crate_for_target(source)
        if found:
            candidates.append(found)

        candidates.append(source)

        line_name = self._line_name_for_station(source)
        suffixes = (
            "container_h01_near",
            "container_h01_far",
            "container_h10",
            "container_h01",
            "tote_b01",
            "plastic_crate",
        )
        for base in (source, line_name):
            if not base:
                continue
            for suffix in suffixes:
                candidates.append(base if base.endswith(suffix) else f"{base}_{suffix}")

        seen: set[str] = set()
        unique_candidates: list[str] = []
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            unique_candidates.append(candidate)

        for candidate in unique_candidates:
            # Primary: validate against material_metadata (trusted source of truth)
            if self._is_valid_grasp_candidate(candidate):
                return candidate
            # Fallback: traditional nav-env joint + grasp-site check
            if self._has_object_joint(candidate) and self._has_grasp_sites(candidate):
                return candidate

        if object_name:
            available = ", ".join(self._available_grasp_objects()[:12])
            raise RuntimeError(
                f"Explicit grasp object '{object_name}' is not available or has no grasp sites. "
                f"{self._grasp_resolution_context()}. Available grasp objects: {available}"
            )

        available = ", ".join(self._available_grasp_objects()[:12])
        raise RuntimeError(
            f"Could not resolve grasp object for source '{source}'. "
            f"{self._grasp_resolution_context()}. "
            f"Tried: {unique_candidates}. Available grasp objects: {available}"
        )

    def _find_output_station_entry(self, target: str) -> tuple[str | None, dict | None]:
        """Find an output station name and dict matching *target*."""
        ports = self.env.output_ports
        if target in ports:
            return target, ports[target]
        for name in ports:
            if name.startswith(target):
                return name, ports[name]
        for name in ports:
            if target in name:
                return name, ports[name]
        return None, None

    def _find_output_station(self, target: str) -> dict | None:
        """Find an output station dict matching *target*.

        Tries: exact match, then prefix match (e.g. ``output_3`` matches
        ``output_3_conveyor``), then substring match.
        """
        _, station = self._find_output_station_entry(target)
        return station

    @staticmethod
    def _output_station_index(name: str | None) -> int | None:
        if not name:
            return None
        text = name.lower()
        marker = "output_"
        start = text.find(marker)
        if start < 0:
            return None
        start += len(marker)
        digits: list[str] = []
        while start < len(text) and text[start].isdigit():
            digits.append(text[start])
            start += 1
        if not digits:
            return None
        return int("".join(digits))

    def _output_table_top_z(
        self,
        target: str,
        station_name: str | None,
        station: dict,
    ) -> float | None:
        """Return the target output table top z when the scene exposes it."""
        station_idx = self._output_station_index(station_name) or self._output_station_index(target)
        surfaces_fn = getattr(self.env, "_siemens_static_table_support_surfaces", None)
        if callable(surfaces_fn):
            try:
                surfaces = list(surfaces_fn())
                if station_idx is not None and 1 <= station_idx <= len(surfaces):
                    support_pose, _ = surfaces[station_idx - 1]
                    return float(np.asarray(support_pose, dtype=float)[2])

                center = np.asarray(station.get("center", []), dtype=float)
                if center.size >= 2 and surfaces:
                    support_pose, _ = min(
                        surfaces,
                        key=lambda item: float(np.linalg.norm(np.asarray(item[0], dtype=float)[:2] - center[:2])),
                    )
                    return float(np.asarray(support_pose, dtype=float)[2])
            except Exception as exc:
                logger.debug("_output_table_top_z: Siemens support lookup failed: %s", exc)

        station_surface_top = getattr(self.env, "_station_surface_top", None)
        if callable(station_surface_top):
            try:
                return float(station_surface_top(station, 0))
            except Exception as exc:
                logger.debug("_output_table_top_z: station surface lookup failed: %s", exc)

        table_top_z = getattr(self.env, "table_top_z", None)
        if table_top_z is not None:
            return float(table_top_z)

        center = np.asarray(station.get("center", []), dtype=float)
        if center.size >= 3 and abs(float(center[2])) > 1e-9:
            return float(center[2])
        return None

    def _object_bottom_offset_z(self, obj_name: str) -> float | None:
        metadata = getattr(self.env, "material_metadata", {}).get(obj_name, {}) or {}
        model = metadata.get("model")
        bottom_offset = getattr(model, "bottom_offset", None)
        if bottom_offset is None:
            return None
        try:
            return float(np.asarray(bottom_offset, dtype=float)[2])
        except Exception as exc:
            logger.debug("_object_bottom_offset_z: failed for %s: %s", obj_name, exc)
            return None

    def _get_object_joint_addr(self, obj_name: str):
        """Get the qpos address for an object's free joint.

        Tries ``_free`` (Siemens objects) first, then ``_joint0`` (legacy
        PlasticCrateObject), then ``_joint0`` suffix on the body name.
        """
        for suffix in ("_free", "_joint0"):
            try:
                return self.env.sim.model.get_joint_qpos_addr(f"{obj_name}{suffix}")
            except Exception:
                continue
        raise RuntimeError(f"No free joint found for object '{obj_name}'")

    def _update_held_crate_position(self) -> None:
        """Teleport the held crate to stay above the robot's current position."""
        if self._held_crate_body_id is None or self._held_crate_name is None:
            return
        try:
            base_xy, _ = _get_base_pose(self._env)
            qpos_addr = self._get_object_joint_addr(self._held_crate_name)
            self.env.sim.data.qpos[qpos_addr[0]:qpos_addr[1]] = np.array([
                base_xy[0], base_xy[1], 1.20,    # x, y, z above robot
                1.0, 0.0, 0.0, 0.0,              # quaternion (identity)
            ])
            self.env.sim.forward()
        except Exception as exc:
            logger.warning("_update_held_crate_position failed: %s", exc)

    def place_object(self, target: str) -> bool:
        """Release the held crate at the named output station.

        *target* may be a short port name (e.g. ``output_3``) or a full
        station name.  The crate is teleported to the matching table surface.
        """
        logger.info("place_object called: target=%r, held=%r", target, self._held_crate_name)
        if self._held_crate_name is None or self._held_crate_body_id is None:
            logger.warning("place_object: no crate is currently held")
            return False

        station = self._find_output_station(target)
        if station is None:
            logger.warning(
                "place_object: no output station matching '%s'. Available: %s",
                target, sorted(self.env.output_ports.keys()),
            )
            return False

        try:
            center = np.asarray(station["center"])
            qpos_addr = self._get_object_joint_addr(self._held_crate_name)
            self.env.sim.data.qpos[qpos_addr[0]:qpos_addr[1]] = np.array([
                center[0], center[1], 0.70,
                1.0, 0.0, 0.0, 0.0,
            ])
            self.env.sim.forward()
            logger.info("place_object: placed '%s' at '%s'", self._held_crate_name, target)
        except Exception as exc:
            logger.error("place_object failed: %s", exc)
            return False

        self._held_crate_name = None
        self._held_crate_body_id = None
        return True

    # ── physics-based grasp & place (robomimic pipeline) ──────

    def set_physics_grasp_config(
        self, checkpoint: str | Path | None = None, device: str = "cpu",
        object_map: dict | None = None,
        capture_grasp_frames: bool = False,
    ):
        """Enable physics-based grasp/place with a robomimic checkpoint.

        If *checkpoint* is ``None`` (default), the path is resolved from
        ``robot_params.json`` → ``grasp_policy.checkpoint_path`` (with
        ``checkpoint_fallback_path`` as the fallback) when the policy is
        first loaded.
        """
        if checkpoint is not None:
            self._physics_checkpoint = Path(checkpoint)
        # else: resolved lazily in _ensure_physics_policy()
        self._physics_device = device
        self._physics_object_map = object_map or {}
        self._capture_grasp_frames = bool(capture_grasp_frames)
        self._physics_policy = None
        self._physics_config = None
        self._physics_ckpt_dict = None
        self._has_physics = True

    def _ensure_physics_policy(self):
        if self._physics_policy is not None:
            return
        ckpt = getattr(self, "_physics_checkpoint", None)
        if ckpt is None:
            # Resolve from robot_params.json
            _gp = self._rp["grasp_policy"]
            _PROJECT_ROOT = Path(__file__).resolve().parents[3]
            for _key in ("checkpoint_path", "checkpoint_fallback_path"):
                _cp = _gp.get(_key, "")
                if not _cp:
                    continue
                ckpt = Path(_cp)
                if not ckpt.is_absolute():
                    ckpt = (_PROJECT_ROOT / _cp).resolve()
                if ckpt.exists():
                    self._physics_checkpoint = ckpt
                    logger.info("Physics checkpoint resolved from robot_params: %s", ckpt)
                    break
                ckpt = None  # try fallback
        if ckpt is None or not ckpt.exists():
            raise RuntimeError(f"Physics grasp checkpoint not found: {ckpt}")
        from robosuite.environments.factory_sorting.load_factory_sorting_evalization import (
            load_factory_sorting_policy, make_eval_env,
        )
        import argparse
        ns = argparse.Namespace(
            checkpoint=ckpt, device=self._physics_device, verbose=False,
        )
        self._physics_policy, self._physics_config, self._physics_ckpt_dict = (
            load_factory_sorting_policy(checkpoint=ckpt, device=self._physics_device, verbose=False)
        )
        logger.info("Physics grasp policy loaded from %s", ckpt)

    @staticmethod
    def _normalize_grasp_initial_base_pose(initial_base_pose):
        if not initial_base_pose:
            return None

        try:
            if isinstance(initial_base_pose, dict):
                robot_base_pos = initial_base_pose.get("robot_base_pos")
                robot_base_ori = initial_base_pose.get("robot_base_ori")
                if robot_base_pos is not None and robot_base_ori is not None:
                    pos = np.asarray(robot_base_pos, dtype=float).reshape(-1)
                    ori = np.asarray(robot_base_ori, dtype=float).reshape(-1)
                    if pos.size >= 2 and ori.size >= 3:
                        return [float(pos[0]), float(pos[1]), float(pos[2]) if pos.size >= 3 else 0.0], [
                            float(ori[0]),
                            float(ori[1]),
                            float(ori[2]),
                        ]

                xy = initial_base_pose.get("xy")
                if xy is None:
                    xy = initial_base_pose.get("base_world_xy")
                yaw = initial_base_pose.get("yaw")
                if yaw is None:
                    yaw = initial_base_pose.get("base_world_yaw")
                if yaw is None and initial_base_pose.get("orientation_xyzw") is not None:
                    quat = np.asarray(initial_base_pose["orientation_xyzw"], dtype=float).reshape(-1)
                    if quat.size >= 4:
                        x, y, z, w = [float(v) for v in quat[:4]]
                        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
                if xy is not None and yaw is not None:
                    xy_arr = np.asarray(xy, dtype=float).reshape(-1)
                    if xy_arr.size >= 2:
                        return [float(xy_arr[0]), float(xy_arr[1]), 0.0], [0.0, 0.0, float(yaw)]

            pose = np.asarray(initial_base_pose, dtype=float).reshape(-1)
            if pose.size >= 3:
                return [float(pose[0]), float(pose[1]), 0.0], [0.0, 0.0, float(pose[2])]
        except Exception as exc:
            logger.warning("invalid grasp initial base pose ignored: %s", exc)
        return None

    def grasp_object_physics(
        self,
        source: str,
        object_name: str | None = None,
        initial_base_pose=None,
    ) -> bool:
        """Create wrapped env at nav position, run grasp, sync object back."""
        print("[BACKEND v4] grasp_object_physics called", flush=True)
        self._ensure_physics_policy()
        from robosuite.environments.factory_sorting.load_factory_sorting_evalization import (
            base_robosuite_env,
            run_factory_sorting_grasp_in_wrapped_env, make_eval_env,
        )
        from robosuite.environments.factory_sorting.lift_after_grasp import lift_grasped_object
        from robosuite.environments.factory_sorting.transport_attachment import capture_transport_attachment
        import argparse

        obj_name = self._resolve_grasp_object_name(source, object_name=object_name)
        nav_env = self.env

        # Read grasp poses from knowledge/task_config.json (single source of truth)
        import json as _json
        _cfg_path = Path(__file__).resolve().parents[3] / "knowledge" / "task_config.json"
        if not _cfg_path.exists():
            raise RuntimeError(f"task_config.json not found at {_cfg_path}")
        _cfg = _json.loads(_cfg_path.read_text(encoding="utf-8"))
        _GRASP_POSE: dict = {}
        for _src, _entry in _cfg.get("grasp_poses", {}).items():
            _GRASP_POSE[_src] = (_entry["pos"], [0.0, 0.0, _entry["yaw"]])
        for _i in range(1, 7):
            if f"line_{_i}" not in _GRASP_POSE and f"input_{_i}" in _GRASP_POSE:
                _GRASP_POSE[f"line_{_i}"] = _GRASP_POSE[f"input_{_i}"]

        # Use LLM-supplied XY but force correct yaw from config
        _trained_pose = _GRASP_POSE.get(source)
        _initial_pose = self._normalize_grasp_initial_base_pose(initial_base_pose)
        if _initial_pose is not None and _trained_pose is not None:
            _grasp_pos = _initial_pose[0]
            _grasp_ori = _trained_pose[1]  # force correct yaw from config
            logger.info("grasp_object_physics: using supplied XY + config yaw (%.3f,%.3f,yaw=%.3f)",
                        _grasp_pos[0], _grasp_pos[1], _grasp_ori[2])
        elif _initial_pose is not None:
            _grasp_pos, _grasp_ori = _initial_pose
            logger.info("grasp_object_physics: target=%s obj=%s using supplied pose (%.3f,%.3f,yaw=%.3f)",
                        source, obj_name, _grasp_pos[0], _grasp_pos[1], _grasp_ori[2])
        else:
            try:
                base_xy, yaw = self.get_base_pose()
                _grasp_pos = [float(base_xy[0]), float(base_xy[1]), 0.0]
                _grasp_ori = [0.0, 0.0, float(yaw)]
                logger.info("grasp_object_physics: target=%s obj=%s using nav pose (%.3f,%.3f)",
                            source, obj_name, _grasp_pos[0], _grasp_pos[1])
            except Exception:
                _trained_pose = _GRASP_POSE.get(source)
                if _trained_pose is None:
                    raise
                _grasp_pos, _grasp_ori = _trained_pose
                logger.warning("grasp_object_physics: target=%s obj=%s falling back to trained pose (%s, yaw=%s)",
                               source, obj_name, _grasp_pos, _grasp_ori[2])

        ns = argparse.Namespace(
            factory_scene=self._env_name,
            robot_base_pos=_grasp_pos,
            robot_base_ori=_grasp_ori,
            renderer="mjviewer", camera="robot0_robotview",
            camera_height=128, camera_width=128,
            controller=None, gripper_types="Robotiq140Gripper", seed=None,
        )
        wrapped = make_eval_env(
            ns, config=self._physics_config,
            ckpt_dict=self._physics_ckpt_dict, render=True,
        )

        # Dump ALL BC policy inputs for debugging
        from robosuite.environments.factory_sorting.load_factory_sorting_evalization import base_robosuite_env
        _raw = base_robosuite_env(wrapped)
        _robot = _raw.robots[0]
        _base_xy, _base_yaw = _get_base_pose(_raw)
        print(f"[BC_INPUT] robot_base_pos=({_grasp_pos[0]:.6f},{_grasp_pos[1]:.6f},{_grasp_pos[2]:.6f})", flush=True)
        print(f"[BC_INPUT] robot_base_ori=[{_grasp_ori[0]:.6f},{_grasp_ori[1]:.6f},{_grasp_ori[2]:.6f}] (yaw={_grasp_ori[2]/3.14159:.4f}*pi)", flush=True)
        print(f"[BC_INPUT] actual_base_pose=({_base_xy[0]:.6f},{_base_xy[1]:.6f}) yaw={_base_yaw:.6f}", flush=True)
        print(f"[BC_INPUT] object_name={obj_name}", flush=True)
        print(f"[BC_INPUT] torso_lift={_raw.sim.data.qpos[_raw.sim.model.joint_name2id('robot0_torso_lift_joint')]:.4f}", flush=True)
        print(f"[BC_INPUT] arm_right_1={_raw.sim.data.qpos[_raw.sim.model.joint_name2id('robot0_arm_right_1_joint')]:.4f}", flush=True)
        print(f"[BC_INPUT] arm_right_3={_raw.sim.data.qpos[_raw.sim.model.joint_name2id('robot0_arm_right_3_joint')]:.4f}", flush=True)
        print(f"[BC_INPUT] arm_right_4={_raw.sim.data.qpos[_raw.sim.model.joint_name2id('robot0_arm_right_4_joint')]:.4f}", flush=True)
        print(f"[BC_INPUT] gripper_r={_raw.sim.data.qpos[_raw.sim.model.joint_name2id('gripper0_right_finger_joint')]:.4f}", flush=True)
        print(f"[BC_INPUT] scene={self._env_name}", flush=True)
        for _on in _raw.material_objects:
            for _sfx in ('_joint0','_free'):
                try:
                    _q = _raw.sim.data.get_joint_qpos(f'{_on}{_sfx}')
                    print(f"[BC_INPUT] object {_on}: pos=({_q[0]:.4f},{_q[1]:.4f},{_q[2]:.4f})", flush=True)
                    break
                except: pass

        grasp_raw = base_robosuite_env(wrapped)
        # Set grasp window to robotview
        try:
            _set_viewer_camera(grasp_raw, "robot0_robotview", render_once=True)
        except Exception:
            pass

        # Read grasp-policy and lift params from robot_params.json
        _gp = self._rp["grasp_policy"]
        _lp = self._rp["lift"]

        _grasp_frames: list = []
        # Record trajectory frame from wrapped env every N steps
        _record_interval = _gp["record_frame_interval"]
        _cb_step = [0]  # mutable counter for rendering / recording
        def _cb():
            try:
                _cb_step[0] += 1
                if _cb_step[0] % 2 == 0:
                    _refresh_visible_viewer(grasp_raw)
                if self._capture_grasp_frames and _cb_step[0] % 2 == 0:
                    frame = grasp_raw.sim.render(camera_name="robot0_robotview", height=256, width=256)
                    _grasp_frames.append(np.array(frame[::-1], dtype=np.uint8))
                # Record trajectory frame from wrapped env every N steps
                if _cb_step[0] % _record_interval == 0:
                    try:
                        self._record_trajectory_frame(_env=grasp_raw)
                    except Exception:
                        pass
            except Exception:
                pass

        # Record exact wrapped-env grasp start, then mark the frame for replay.
        try:
            self._record_trajectory_frame(_env=grasp_raw)
            self._mark_trajectory_event(
                "grasp_start",
                object_name=obj_name,
                source=source,
            )
        except Exception as exc:
            logger.warning("mark grasp_start failed: %s", exc)

        # Single grasp attempt (eval steps from robot_params.json)
        try:
            result = run_factory_sorting_grasp_in_wrapped_env(
                env=wrapped, policy=self._physics_policy,
                eval_steps=_gp["eval_steps"],
                debug_policy=_gp["debug_policy"],
                debug_every=_gp["debug_every"],
                object_name=obj_name,
                post_hold_steps=_gp["post_hold_steps"],
                initial_view_steps=_gp["initial_view_steps"],
                camera="robot0_robotview",
                render=True, render_callback=_cb,
            )
        except Exception:
            _close_wrapped_eval_env(wrapped, raw_env=grasp_raw)
            try:
                _set_viewer_camera(nav_env, "birdview", render_once=True)
            except Exception:
                pass
            raise
        self._grasp_frames = _grasp_frames
        grasp_success = bool(result.get("success")) if isinstance(result, dict) else bool(result)

        try:
            self._record_trajectory_frame(_env=grasp_raw)
            self._mark_trajectory_event(
                "grasp_end",
                object_name=obj_name,
                source=source,
                success=grasp_success,
            )
        except Exception as exc:
            logger.warning("mark grasp_end failed: %s", exc)

        # Always attempt lift — contact-based grasp check is unreliable
        lift_result = {"success": False, "failure_reason": "lift was not attempted"}
        try:
            lift_result = lift_grasped_object(
                env=wrapped, object_name=obj_name,
                lift_height=_lp["lift_height"],
                max_steps=_lp["max_steps"],
                hold_steps=_lp["hold_steps"],
                tolerance=_lp["tolerance"],
                max_action=_lp["max_action"],
                render=True,
                render_callback=_cb,
            )
        except Exception as exc:
            logger.warning("lift failed: %s", exc)
            lift_result = {"success": False, "failure_reason": f"lift exception: {exc}"}
        lift_success = bool(lift_result.get("success")) if isinstance(lift_result, dict) else bool(lift_result)
        _close_visible_viewer(grasp_raw)

        # Sync object pos + arm joints from wrapped env to nav env
        try:
            grasp_raw = base_robosuite_env(wrapped)  # properly unwraps FrameStackWrapper+EnvRobosuite
            logger.info("sync: grasp_raw type=%s, nav type=%s", type(grasp_raw).__name__, type(nav_env).__name__)
            for obj_n in grasp_raw.material_objects:
                for suffix in ("_free", "_joint0"):
                    jn = f"{obj_n}{suffix}"
                    try:
                        qpos = grasp_raw.sim.data.get_joint_qpos(jn)
                        nav_env.sim.data.set_joint_qpos(jn, qpos)
                        logger.info("sync: %s qpos=(%.3f,%.3f,%.3f)", jn, qpos[0], qpos[1], qpos[2])
                        break
                    except Exception:
                        continue
            nav_env.sim.forward()
            arm_joints = [j for j in grasp_raw.sim.model.joint_names
                          if j.startswith('robot0_') and 'mobilebase' not in j]
            for jn in arm_joints:
                try:
                    nav_env.sim.data.set_joint_qpos(jn, grasp_raw.sim.data.get_joint_qpos(jn))
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("sync obj failed: %s", exc)

        _ok = grasp_success and lift_success
        if _ok:
            print("[BACKEND] grasp and lift succeeded, proceeding with transport", flush=True)
        else:
            lift_failure = lift_result.get("failure_reason", "") if isinstance(lift_result, dict) else ""
            print(
                "[BACKEND] grasp pipeline failed, skipping transport attachment: "
                f"grasp_success={grasp_success}, lift_success={lift_success}, "
                f"lift_failure={lift_failure}",
                flush=True,
            )

        if _ok:
            # Record post-grasp+lift frame
            self._record_trajectory_frame()
            try:
                capture_transport_attachment(nav_env, obj_name)
                logger.info("transport_attach: obj=%s held", obj_name)
                self._held_crate_name = obj_name
            except Exception as exc:
                logger.warning("transport_attach failed: %s", exc)
                _ok = False
            self._record_trajectory_frame()

        _close_wrapped_eval_env(wrapped, raw_env=grasp_raw)
        try:
            _set_viewer_camera(nav_env, "birdview", render_once=True)
        except Exception:
            pass
        return _ok

    @staticmethod
    def _sync_objects(src_env, dst_env):
        """Copy material object qpos from src to dst env."""
        try:
            for obj_name in src_env.material_objects:
                for suffix in ("_free", "_joint0"):
                    try:
                        qpos = src_env.sim.data.get_joint_qpos(f"{obj_name}{suffix}")
                        dst_env.sim.data.set_joint_qpos(f"{obj_name}{suffix}", qpos)
                        break
                    except Exception:
                        continue
            dst_env.sim.forward()
        except Exception as exc:
            logger.warning("_sync_objects: %s", exc)

    def place_object_physics(self, target: str) -> bool:
        """Animated place at *target* output station: turn → place."""
        if self._held_crate_name is None:
            logger.warning("place_object_physics: no object held")
            return False
        from robosuite.environments.factory_sorting.turn_to_station import turn_to_face_xy
        from robosuite.environments.factory_sorting.place_on_table import (
            gripper_release_action,
            zero_action,
        )
        from robosuite.environments.factory_sorting.transport_attachment import (
            TRANSPORT_ATTACHMENT_ATTR,
            clear_transport_attachment,
            get_object_qpos,
            set_object_qpos,
            sync_transport_attachment,
        )

        station_name, station = self._find_output_station_entry(target)
        if station is None:
            logger.warning(
                "place_object_physics: no output station matching '%s'. Available: %s",
                target, sorted(self.env.output_ports.keys()),
            )
            return False

        # Use the station center only as a facing target, not as the drop XY.
        _scene = getattr(self, "_scene_context", None)
        scene_station = None
        if _scene is not None:
            scene_station = _scene.output_ports.get(station_name or target) or _scene.output_ports.get(target)
        if scene_station is not None:
            target_xy = scene_station.center[:2].copy()
        else:
            target_xy = np.asarray(station["center"][:2], dtype=float)
        raw = self.env
        turn_posture = _capture_upper_body_posture(raw, raw.robots[0])

        def _record_turn_frame() -> None:
            _restore_upper_body_posture(raw, turn_posture)
            self._record_trajectory_frame()

        # Record pre-place frame (before turn)
        self._record_trajectory_frame()

        # Read turn and place params from robot_params.json
        _tp = self._rp["turn"]
        _pp = self._rp["place"]

        # Step 1: turn to face the output station (record every step)
        try:
            result = turn_to_face_xy(
                env=raw, target_xy=target_xy,
                tolerance=_tp["tolerance"],
                max_iters=_tp["max_iters"],
                turn_steps=_tp["turn_steps"],
                settle_steps=_tp["settle_steps"],
                render=not self._headless, render_sleep=0.0,
                sync_attachment=True,
                post_step_callback=_record_turn_frame,
            )
            if not result.get("success"):
                logger.warning("turn_to_face failed: final_error=%.4f", result.get("final_error", -1))
        except Exception as exc:
            logger.warning("turn_to_face error: %s", exc)

        # Record post-turn frame
        self._record_trajectory_frame()

        # Step 2: lower in place, detach, and let gravity drop the object.
        try:
            held_name = self._held_crate_name
            sync_transport_attachment(raw)
            joint_name, start_qpos = get_object_qpos(raw, held_name)

            lower_delta = _pp["lower_delta"]
            lower_steps = _pp["lower_steps"]
            release_steps = _pp["release_steps"]
            release_clearance = _pp["release_clearance"]
            start_z = float(start_qpos[2])
            target_z = max(0.05, start_z - lower_delta)
            table_top_z = self._output_table_top_z(target, station_name, station)
            bottom_offset_z = self._object_bottom_offset_z(held_name)
            if table_top_z is not None and bottom_offset_z is not None:
                safe_release_z = max(0.05, table_top_z - bottom_offset_z + release_clearance)
                target_z = max(target_z, safe_release_z)
                logger.info(
                    "place_object_physics: release height object=%s target=%s table_top_z=%.4f "
                    "bottom_offset_z=%.4f clearance=%.3f start_z=%.4f target_z=%.4f",
                    held_name,
                    station_name or target,
                    table_top_z,
                    bottom_offset_z,
                    release_clearance,
                    start_z,
                    target_z,
                )
            else:
                logger.warning(
                    "place_object_physics: using fallback release height for '%s' at '%s' "
                    "(table_top_z=%s, bottom_offset_z=%s)",
                    held_name,
                    station_name or target,
                    table_top_z,
                    bottom_offset_z,
                )
            idle_action = zero_action(raw)
            release_action = gripper_release_action(raw)

            attachment = getattr(raw, TRANSPORT_ATTACHMENT_ATTR, None)
            use_attachment = (
                attachment is not None
                and attachment.get("active", False)
                and attachment.get("object_name") == held_name
            )

            # Pre-compute gripper joint indexes so we can restore arm/torso/head
            # posture while keeping the gripper in its current state (closed during
            # lowering, open during release).
            _gripper_qpos_idx: list[int] = []
            _gripper_qvel_idx: list[int] = []
            for _gj_list in raw.robots[0].gripper_joints.values():
                for _j in _gj_list:
                    _gripper_qpos_idx.append(raw.sim.model.get_joint_qpos_addr(_j))
                    _gripper_qvel_idx.append(raw.sim.model.get_joint_qvel_addr(_j))

            def _hold_posture() -> None:
                """Restore arm+torso+head to turn_posture; keep gripper as-is."""
                _saved_gripper_qpos: np.ndarray | None = None
                _saved_gripper_qvel: np.ndarray | None = None
                if _gripper_qpos_idx:
                    _saved_gripper_qpos = np.array(raw.sim.data.qpos[_gripper_qpos_idx], dtype=float)
                    _saved_gripper_qvel = np.array(raw.sim.data.qvel[_gripper_qvel_idx], dtype=float)
                _restore_upper_body_posture(raw, turn_posture)
                if _gripper_qpos_idx and _saved_gripper_qpos is not None:
                    raw.sim.data.qpos[_gripper_qpos_idx] = _saved_gripper_qpos
                    raw.sim.data.qvel[_gripper_qvel_idx] = _saved_gripper_qvel  # type: ignore[arg-type]
                    raw.sim.forward()

            for step in range(lower_steps):
                alpha = float(step + 1) / float(lower_steps)
                z = float(start_qpos[2] + (target_z - start_qpos[2]) * alpha)
                if use_attachment:
                    attachment["world_z"] = z
                    sync_transport_attachment(raw)
                    raw.step(idle_action)
                    _hold_posture()
                    sync_transport_attachment(raw)
                else:
                    qpos = start_qpos.copy()
                    qpos[2] = z
                    set_object_qpos(raw, joint_name, qpos)
                    raw.step(idle_action)
                    _hold_posture()
                    set_object_qpos(raw, joint_name, qpos)
                self._record_trajectory_frame()
                if not self._headless:
                    raw.render()

            clear_transport_attachment(raw)
            self._held_crate_name = None
            self._held_crate_body_id = None

            for _ in range(release_steps):
                raw.step(release_action)
                _hold_posture()
                self._record_trajectory_frame()
                if not self._headless:
                    raw.render()

            logger.info(
                "place_object_physics: released '%s' near current pose for target '%s'",
                held_name,
                target,
            )
            return True
        except Exception as exc:
            logger.error("place_object_physics failed: %s", exc)
            return False

    # ── arm helpers ─────────────────────────────────────────

    # Qpos slice for the right arm (7 joints: 6 arm + base torso)
    _ARM_RIGHT_SLICE = slice(3, 12)   # torso(1) + right arm(6) + right gripper(2)
    _ARM_LEFT_SLICE = slice(18, 24)   # left arm(6)

    _ARM_REACH_QPOS = np.array([
        # torso, right_arm(6), right_gripper_finger(2)
        0.35,                          # torso lift
        0.75, -0.40, 2.14, 1.80,     # shoulder, elbow
        -0.35, -1.05,                  # wrist
        0.05, 0.05,                    # gripper open
    ])

    # Matches TIAGO_GRIPPER_Z_DOWN_INIT_QPOS slice [3:12]
    _ARM_TUCK_QPOS = np.array([
        0.35,                          # torso
        0.0, -0.9,                     # head (neutral, turned toward table)
        0.74, -0.09,                   # arm_right_1, _2 (shoulder)
        2.14, 2.11,                    # arm_right_3, _4 (elbow)
        -0.35, -1.05,                  # arm_right_5, _6 (wrist)
    ])

    def _animate_arm_reach(self, steps: int = 8) -> None:
        """Briefly extend the right arm toward the target (visual only).

        Uses ``sim.forward()`` only — no physics stepping — to avoid
        infinite-acceleration NaN from teleported joint positions.
        """
        target_qpos = self._ARM_REACH_QPOS
        start_qpos = self._env.sim.data.qpos[self._ARM_RIGHT_SLICE].copy()
        for i in range(steps):
            alpha = (i + 1) / steps
            self._env.sim.data.qpos[self._ARM_RIGHT_SLICE] = (
                start_qpos + (target_qpos - start_qpos) * alpha
            )
            self._env.sim.forward()

    def _tuck_arm(self, steps: int = 8) -> None:
        """Return the right arm to a compact driving pose.

        Uses ``sim.forward()`` only — no physics stepping.
        """
        start_qpos = self._env.sim.data.qpos[self._ARM_RIGHT_SLICE].copy()
        target_qpos = self._ARM_TUCK_QPOS
        for i in range(steps):
            alpha = (i + 1) / steps
            self._env.sim.data.qpos[self._ARM_RIGHT_SLICE] = (
                start_qpos + (target_qpos - start_qpos) * alpha
            )
            self._env.sim.forward()

    # ── video / frame capture ───────────────────────────────

    def capture_frame(
        self,
        camera: str | None = None,
        width: int = 640,
        height: int = 480,
        retries: int = 2,
    ) -> np.ndarray:
        """Render one RGB frame from *camera* (default: self._camera).

        Retries on GLFW resource-contention errors (common in multi-threaded
        environments like Streamlit).
        """
        cam = camera or self._camera
        for attempt in range(retries + 1):
            try:
                img = self._env.sim.render(
                    camera_name=cam,
                    width=width,
                    height=height,
                    depth=False,
                )
                return np.array(img[::-1], dtype=np.uint8)
            except Exception:
                if attempt < retries:
                    import time
                    time.sleep(0.1)
                else:
                    raise

    def _record_trajectory_frame(self, *, _env=None) -> None:
        """Capture one trajectory frame (base pose + joint positions + object states).

        If *_env* is given (e.g. wrapped env during grasp), read state from it instead
        of ``self._env``.  This lets us record every step of grasp/place operations.
        """
        src = _env if _env is not None else self._env
        if src is None:
            return
        base_xy, yaw = _get_base_pose(src)
        qpos = src.sim.data.qpos
        # Robot joints (scalar addr) + object free joints (tuple addr)
        joint_positions: dict[str, float] = {}
        object_positions: dict[str, list[float]] = {}
        for i in range(src.sim.model.njnt):
            name = src.sim.model.joint_id2name(i)
            if name is None:
                continue
            addr = src.sim.model.get_joint_qpos_addr(name)
            if isinstance(addr, tuple):
                # Free joint → 7 DOF [x, y, z, qw, qx, qy, qz]
                vals = [float(qpos[j]) for j in range(addr[0], addr[1])]
                # Strip _joint0 / _free suffix for a clean object name key
                clean = name
                for suffix in ("_joint0", "_free"):
                    if clean.endswith(suffix):
                        clean = clean[: -len(suffix)]
                        break
                object_positions[clean] = [round(v, 6) for v in vals]
            else:
                joint_positions[name] = float(qpos[addr])
        joint_positions = _without_trajectory_excluded_joints(joint_positions)
        import time
        t = time.perf_counter() - self._trajectory_start_time

        # Record transport attachment state
        held: str | None = None
        if self._held_crate_name:
            held = self._held_crate_name
        # Also check transport_attachment
        try:
            from robosuite.environments.factory_sorting.transport_attachment import TRANSPORT_ATTACHMENT_ATTR
            attachment = getattr(self._env, TRANSPORT_ATTACHMENT_ATTR, None)
            if attachment and attachment.get("active") and attachment.get("object_name"):
                held = attachment["object_name"]
        except Exception:
            pass

        frame: dict = {
            "time": round(t, 3),
            "base_pose": {
                "position": [round(float(base_xy[0]), 4), round(float(base_xy[1]), 4), 0.0],
                "orientation_xyzw": [0.0, 0.0, round(float(np.sin(yaw/2)), 4), round(float(np.cos(yaw/2)), 4)],
            },
            "joint_positions": {k: round(v, 6) for k, v in joint_positions.items()},
            "object_positions": object_positions,
        }
        if held:
            frame["held_object"] = held
        # Record collision state from env (new factory scene has built-in collision detection)
        try:
            if hasattr(src, "has_judge_collision") and src.has_judge_collision:
                frame["has_collision"] = True
                last_collision = getattr(src, "_judge_last_collision_pair", None)
                if last_collision:
                    frame["collision_pair"] = list(last_collision)
        except Exception:
            pass
        self._trajectory.append(frame)
        self._autosave_trajectory()

    def start_recording(self) -> None:
        self._recorded_frames = []
        self._trajectory = []
        self._trajectory_events = []
        self._trajectory_autosave_path = None
        self._trajectory_autosave_every = 0
        self._trajectory_autosave_last_count = 0
        import time
        self._trajectory_start_time = time.perf_counter()
        logger.info("Recording started (video + trajectory)")

    def set_trajectory_autosave(self, path: str | Path, *, every_n_frames: int = 50) -> None:
        """Periodically write a crash-recovery trajectory JSON while recording."""
        self._trajectory_autosave_path = Path(path)
        self._trajectory_autosave_every = max(1, int(every_n_frames))
        self._trajectory_autosave_last_count = -self._trajectory_autosave_every
        self._autosave_trajectory(force=True)

    def clear_trajectory_autosave(self) -> None:
        self._trajectory_autosave_path = None
        self._trajectory_autosave_every = 0
        self._trajectory_autosave_last_count = 0

    def _autosave_trajectory(self, *, force: bool = False) -> None:
        path = self._trajectory_autosave_path
        every = self._trajectory_autosave_every
        if path is None or every <= 0:
            return
        frame_count = len(self._trajectory)
        if not force and frame_count - self._trajectory_autosave_last_count < every:
            return
        try:
            self.save_trajectory(path)
            self._trajectory_autosave_last_count = frame_count
        except Exception as exc:
            logger.warning("trajectory autosave failed: %s", exc)

    def stop_recording(self) -> list[np.ndarray]:
        frames = list(self._recorded_frames)
        self._recorded_frames = []
        return frames

    def get_recorded_frames(self) -> list[np.ndarray]:
        return list(self._recorded_frames)

    def get_trajectory(self) -> list[dict]:
        return list(self._trajectory)

    def _mark_trajectory_event(self, name: str, *, frame_index: int | None = None, **details) -> None:
        if frame_index is None:
            frame_index = max(0, len(self._trajectory) - 1)
        event: dict = {"name": str(name), "frame": int(frame_index)}
        if 0 <= frame_index < len(self._trajectory):
            event["time"] = self._trajectory[frame_index].get("time")
        for key, value in details.items():
            if value is not None:
                event[key] = value
        self._trajectory_events.append(event)
        self._autosave_trajectory(force=True)

    def save_trajectory(self, path: str | Path) -> str:
        """Save accumulated trajectory as JSON (replay-compatible). Returns the file path."""
        import json
        joint_names: list[str] = []
        object_names: list[str] = []
        object_joint_map: dict[str, str] = {}
        frames = []
        for frame in self._trajectory:
            clean_frame = dict(frame)
            clean_frame["joint_positions"] = _without_trajectory_excluded_joints(
                frame.get("joint_positions", {})
            )
            frames.append(clean_frame)

        if frames:
            joint_names = list(frames[0].get("joint_positions", {}).keys())
            object_names = list(frames[0].get("object_positions", {}).keys())
            # Build joint-name mapping for replay
            for clean_name in object_names:
                for suffix in ("_joint0", "_free"):
                    jn = f"{clean_name}{suffix}"
                    try:
                        self._env.sim.model.get_joint_qpos_addr(jn)
                        object_joint_map[clean_name] = jn
                        break
                    except Exception:
                        continue
        data = {
            "robot_model": self._env_name,
            "camera": self._camera,
            "units": {"length": "meter", "angle": "radian"},
            "joint_names": joint_names,
            "object_names": object_names,
            "object_joints": object_joint_map,
            "events": list(self._trajectory_events),
            "frames": frames,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f"{p.name}.tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)
        logger.info("Trajectory saved: %s (%d frames)", p, len(self._trajectory))
        return str(p)

    def replay_trajectory(
        self,
        json_path: str | Path,
        output_gif: str | Path | None = None,
        *,
        camera: str | None = None,
        width: int = 640,
        height: int = 480,
        frame_start: int | None = None,
        frame_end: int | None = None,
    ) -> list[np.ndarray]:
        """Replay a saved trajectory JSON in the current simulation and render frames.

        Args:
            json_path: Path to the trajectory JSON file.
            output_gif: If given, save the rendered frames as a GIF.
            camera: Override camera (default: use camera from JSON, fallback birdview).
            width, height: Render resolution.
            frame_start, frame_end: Optional Python-slice frame range to replay.

        Returns:
            List of rendered frames (numpy arrays).
        """
        import json
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        frames_data = data.get("frames", [])
        joint_names: list[str] = data.get("joint_names", [])
        object_names: list[str] = data.get("object_names", [])
        camera = camera or data.get("camera", self._camera)

        if not frames_data:
            logger.warning("replay_trajectory: no frames in %s", json_path)
            return []

        total_frames = len(frames_data)
        start = 0 if frame_start is None else max(0, int(frame_start))
        end = total_frames if frame_end is None else min(total_frames, int(frame_end))
        if start >= end:
            logger.warning(
                "replay_trajectory: empty frame range %s:%s for %s",
                frame_start, frame_end, json_path,
            )
            return []
        if start != 0 or end != total_frames:
            frames_data = frames_data[start:end]

        # Resolve joint/object qpos addresses once.
        # Mobile base xy/yaw are restored from base_pose so replay uses world
        # pose instead of scene-specific joint offsets.
        _xy_joints = {"mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"}
        joint_addrs: list[tuple[str, int]] = []
        _yaw_addr: int | None = None
        try:
            _yaw_addr = int(self.env.sim.model.get_joint_qpos_addr("mobilebase0_joint_mobile_yaw"))
        except Exception:
            pass
        for name in joint_names:
            if name in _xy_joints:
                continue  # handled via _set_base_xy_direct
            if name == "mobilebase0_joint_mobile_yaw":
                continue
            try:
                addr = self.env.sim.model.get_joint_qpos_addr(name)
                # addr may be numpy int64 (not Python int) — isinstance check fails
                if not isinstance(addr, tuple):
                    joint_addrs.append((name, int(addr)))
            except Exception:
                continue

        object_joint_map: dict[str, str] = data.get("object_joints", {})
        object_addrs: list[tuple[str, tuple]] = []  # (clean_name, (start, end))
        for name in object_names:
            # Resolve via mapping first, then try raw name
            joint_name = object_joint_map.get(name, name)
            try:
                addr = self.env.sim.model.get_joint_qpos_addr(joint_name)
                if isinstance(addr, tuple):
                    object_addrs.append((name, addr))
            except Exception:
                # Fallback: try name directly
                try:
                    addr = self.env.sim.model.get_joint_qpos_addr(name)
                    if isinstance(addr, tuple):
                        object_addrs.append((name, addr))
                except Exception:
                    continue

        logger.info("Replaying %d frames (%d:%d of %d), %d arm joints, %d objects",
                    len(frames_data), start, end, total_frames, len(joint_addrs), len(object_addrs))

        robot = self.env.robots[0]
        follow_mode = (camera == "follow")
        # Find a fixed camera for non-follow mode
        if not follow_mode:
            _render_camera = camera

        rendered: list[np.ndarray] = []
        for fi, frm in enumerate(frames_data):
            bp = frm.get("base_pose", {})
            jp = frm.get("joint_positions", {})

            # Restore yaw before xy. The saved orientation is world yaw, while
            # mobile_yaw qpos is relative to the robot's scene base orientation.
            if _yaw_addr is not None:
                if "mobilebase0_joint_mobile_yaw" in jp:
                    self.env.sim.data.qpos[_yaw_addr] = jp["mobilebase0_joint_mobile_yaw"]
                    self.env.sim.forward()
                    _invalidate_base_xy_qpos_mapping(self.env)
                else:
                    ori = bp.get("orientation_xyzw", [])
                    if len(ori) >= 4:
                        qz, qw = float(ori[2]), float(ori[3])
                        _set_base_world_yaw_direct(self.env, robot, 2.0 * math.atan2(qz, qw))

            # Set base position from base_pose (world xy) after yaw so the
            # forward/side mapping is computed in the correct orientation.
            pos_list = bp.get("position", [])
            if len(pos_list) >= 2:
                target_xy = np.array([pos_list[0], pos_list[1]], dtype=float)
                _set_base_xy_direct(self.env, robot, target_xy)

            # Set arm/body joint positions from trajectory.
            for name, addr in joint_addrs:
                if name in jp:
                    self.env.sim.data.qpos[addr] = jp[name]

            # Set object positions (free joints: 7 values)
            op = frm.get("object_positions", {})
            for name, (start, end) in object_addrs:
                if name in op:
                    vals = op[name]
                    self.env.sim.data.qpos[start:end] = vals

            self.env.sim.forward()

            # Render — use follow camera if requested
            if follow_mode:
                # Position the free camera behind the robot using robosuite's own render pipeline
                rctx = self.env.sim._render_context_offscreen
                rctx.cam.type = __import__("mujoco").mjtCamera.mjCAMERA_FREE
                ori = bp.get("orientation_xyzw", [0, 0, 1, 0])
                qz, qw = float(ori[2]), float(ori[3])
                yaw = 2.0 * math.atan2(qz, qw)
                rctx.cam.lookat[:] = [float(target_xy[0]), float(target_xy[1]), 1.0]
                rctx.cam.distance = 5.0
                rctx.cam.azimuth = float(math.degrees(yaw))
                rctx.cam.elevation = -35.0
                img = self._env.sim.render(
                    camera_name=None,  # free camera
                    width=width, height=height,
                    depth=False,
                )
                img = np.array(img[::-1], dtype=np.uint8)
            else:
                img = self._env.sim.render(
                    camera_name=_render_camera,
                    width=width, height=height,
                    depth=False,
                )
                img = np.array(img[::-1], dtype=np.uint8)
            rendered.append(img)

        # Save GIF if requested
        if output_gif and rendered:
            from PIL import Image
            # Use more frames for smoother playback (300 max), 100ms per frame
            _step = max(1, len(rendered) // 300)
            _display = rendered[::_step]
            _last = _display[-1]
            _pause = [Image.fromarray(_last)] * 5
            _all = [Image.fromarray(f) for f in _display] + _pause
            _all[0].save(
                output_gif, format="GIF", save_all=True,
                append_images=_all[1:], duration=100, loop=0,
            )
            logger.info("Replay GIF saved: %s (%d frames, step=%d)", output_gif, len(_display), _step)

        logger.info("Replay complete: %d frames rendered", len(rendered))
        return rendered

    # ── navigation ──────────────────────────────────────────

    def follow_path(
        self,
        path: list[np.ndarray],
        *,
        max_steps: int | None = None,
        waypoint_tolerance: float | None = None,
        stop_on_collision: bool = True,
        debug: bool = False,
        record_every: int = 1,
    ) -> bool:
        """Drive the robot along *path*, optionally recording frames.

        Args:
            record_every: Capture a frame every N steps (1 = every step).
        """
        # Resolve defaults from robot_params.json
        _nav = self._rp["navigation"]
        if max_steps is None:
            max_steps = _nav["max_steps"]
        if waypoint_tolerance is None:
            waypoint_tolerance = _nav["waypoint_tolerance"]
        _settle_steps = _nav["settle_steps"]

        def _capture() -> None:
            # Capture frame FIRST (before any side effects that might fail)
            if record_every > 0 and self._env is not None:
                try:
                    self._recorded_frames.append(self.capture_frame())
                except Exception as exc:
                    logger.warning("capture_frame failed: %s", exc)
            # Then update held crate + record trajectory
            try:
                self._update_held_crate_position()
                if record_every > 0 and self._env is not None:
                    self._record_trajectory_frame()
            except Exception as exc:
                logger.warning("_capture side-effects failed: %s", exc)

        # Lock upper body before navigation (prevents arm drift)
        posture = _capture_upper_body_posture(self._env, self._env.robots[0])

        result: bool
        if self._drive_mode == "direct":
            result = _follow_path_direct(
                self._env, path,
                max_steps=max_steps,
                waypoint_tolerance=waypoint_tolerance,
                control_freq=self._control_freq,
                max_linear=self._max_linear,
                stop_on_collision=stop_on_collision and self._stop_on_collision,
                collision_warmup_steps=self._collision_warmup_steps,
                ignore_collision_geom=self._ignore_collision_geom,
                max_collision_pairs=self._max_collision_pairs,
                headless=True,
                debug=debug,
                frame_callback=_capture if record_every > 0 else None,
                record_every=record_every,
                settle_steps=_settle_steps,
                posture=posture,
            )
        else:
            result = _follow_path_action(
                self._env, path,
                max_steps=max_steps,
                waypoint_tolerance=waypoint_tolerance,
                k_linear=self._k_linear,
                k_angular=self._k_angular,
                max_linear=self._max_linear,
                max_angular=self._max_angular,
                holonomic_base=self._holonomic_base,
                yaw_control=self._yaw_control,
                turn_in_place_angle=self._turn_in_place_angle,
                stop_on_collision=stop_on_collision and self._stop_on_collision,
                collision_warmup_steps=self._collision_warmup_steps,
                ignore_collision_geom=self._ignore_collision_geom,
                max_collision_pairs=self._max_collision_pairs,
                headless=True,
                debug=debug,
                frame_callback=_capture if record_every > 0 else None,
                record_every=record_every,
                settle_steps=_settle_steps,
                posture=posture,
            )
        logger.info("follow_path: %s, frames=%d", "reached" if result else "blocked", len(self._recorded_frames))
        return result

    def render(self) -> None:
        if self._env is not None and not self._headless:
            self._env.render()

    @property
    def action_spec(self) -> tuple[np.ndarray, np.ndarray]:
        return self._env.action_spec


# ── environment factory ─────────────────────────────────────

def _make_env(
    env_name: str,
    *,
    robots: str,
    camera: str,
    headless: bool,
    control_freq: int,
    seed: int | None,
    use_camera_obs: bool = False,
    camera_names: str = "agentview",
    camera_heights: int = 256,
    camera_widths: int = 256,
    force_offscreen: bool = False,
):
    from robosuite.environments.base import make
    # Import the correct factory scene to ensure it's registered
    if "FactorySorting1_3FO3ERFHISEM" in env_name:
        import robosuite.environments.factory_sorting.factory_sorting_1_3fo3erfhisem  # noqa: F401
    elif "FactorySorting3" in env_name:
        import robosuite.environments.factory_sorting.factory_sorting_3_3fo3errph7x9  # noqa: F401
    elif "FactorySorting5" in env_name:
        import robosuite.environments.factory_sorting.factory_sorting_5_3fo3ertpxeut  # noqa: F401
    elif "FactorySorting7" in env_name:
        import robosuite.environments.factory_sorting.factory_sorting_7_3fo3erfky9rn  # noqa: F401
    elif "FactorySorting9" in env_name:
        import robosuite.environments.factory_sorting.factory_sorting_9_3fo3ert2c5fp  # noqa: F401
    else:
        import robosuite.environments.factory_sorting.factory_sorting  # noqa: F401

    render_camera = None if camera == "free" else camera
    # Build kwargs dict — conditionally add Siemens-arena args
    make_kwargs: dict = dict(
        robot_base_pos=[13.5, 0.0, 0.0],  # unified start across all scenes
        robots=robots,
        has_renderer=not headless,
        has_offscreen_renderer=headless or use_camera_obs or force_offscreen,
        render_camera=render_camera,
        use_camera_obs=use_camera_obs,
        camera_names=camera_names if use_camera_obs else "agentview",
        camera_heights=camera_heights,
        camera_widths=camera_widths,
        use_object_obs=True,
        ignore_done=True,
        control_freq=control_freq,
        seed=seed,
    )
    # Pass scene-specific args based on env_name
    import inspect
    try:
        if "FactorySorting1_3FO3ERFHISEM" in env_name:
            from robosuite.environments.factory_sorting.factory_sorting_1_3fo3erfhisem import FactorySorting1_3FO3ERFHISEM
            _env_cls = FactorySorting1_3FO3ERFHISEM
        elif "FactorySorting3" in env_name:
            from robosuite.environments.factory_sorting.factory_sorting_3_3fo3errph7x9 import FactorySorting3_3FO3ERRPH7X9
            _env_cls = FactorySorting3_3FO3ERRPH7X9
        elif "FactorySorting5" in env_name:
            from robosuite.environments.factory_sorting.factory_sorting_5_3fo3ertpxeut import FactorySorting5_3FO3ERTPXEUT
            _env_cls = FactorySorting5_3FO3ERTPXEUT
        elif "FactorySorting7" in env_name:
            from robosuite.environments.factory_sorting.factory_sorting_7_3fo3erfky9rn import FactorySorting7_3FO3ERFKY9RN
            _env_cls = FactorySorting7_3FO3ERFKY9RN
        elif "FactorySorting9" in env_name:
            from robosuite.environments.factory_sorting.factory_sorting_9_3fo3ert2c5fp import FactorySorting9_3FO3ERT2C5FP
            _env_cls = FactorySorting9_3FO3ERT2C5FP
        else:
            from robosuite.environments.factory_sorting.factory_sorting import FactorySorting
            _env_cls = FactorySorting
        _init_params = inspect.signature(_env_cls.__init__).parameters
        if "use_siemens_arena" in _init_params:
            make_kwargs["use_siemens_arena"] = True
        if "include_material_objects" in _init_params:
            make_kwargs["include_material_objects"] = False  # new scene uses material objects
        if "include_siemens_line_objects" in _init_params:
            make_kwargs["include_siemens_line_objects"] = False  # new scene doesn't use line objects
        if "include_legacy_static_scene" in _init_params:
            make_kwargs["include_legacy_static_scene"] = False
    except Exception:
        pass

    return make(env_name, **make_kwargs)


# ── upper-body posture locking (from llm_task_navigator.py) ──

def _capture_upper_body_posture(env, robot) -> dict:
    """Record upper-body joint names, qpos indexes, and qvel indexes.

    Locks the arm + torso + head so they don't drift during base navigation.
    """
    joint_names: list[str] = []
    joint_names.extend(getattr(robot, "robot_arm_joints", []))
    joint_names.extend(getattr(robot.robot_model, "torso_joints", []))
    joint_names.extend(getattr(robot.robot_model, "head_joints", []))
    for gripper_joints in getattr(robot, "gripper_joints", {}).values():
        joint_names.extend(gripper_joints)

    qpos_idx = [env.sim.model.get_joint_qpos_addr(j) for j in joint_names]
    qvel_idx = [env.sim.model.get_joint_qvel_addr(j) for j in joint_names]
    return {
        "joint_names": joint_names,
        "qpos_indexes": qpos_idx,
        "qvel_indexes": qvel_idx,
        "qpos": np.array(env.sim.data.qpos[qpos_idx], dtype=float),
        "qvel": np.array(env.sim.data.qvel[qvel_idx], dtype=float),
    }


def _restore_upper_body_posture(env, posture: dict | None) -> None:
    """Reset upper-body joints to the captured posture."""
    if posture is None:
        return
    env.sim.data.qpos[posture["qpos_indexes"]] = posture["qpos"]
    env.sim.data.qvel[posture["qvel_indexes"]] = posture["qvel"]
    env.sim.forward()


# ── robot state helpers ─────────────────────────────────────

def _refresh_visible_viewer(env) -> None:
    """Pump one visible-viewer frame without changing simulation state."""
    viewer = getattr(env, "viewer", None)
    if viewer is None:
        try:
            env.render()
        except Exception:
            pass
        return

    update = getattr(viewer, "update", None)
    if callable(update):
        update()
        return

    try:
        env.render()
    except Exception:
        pass


def _close_visible_viewer(env) -> None:
    """Close only the visible viewer/window, keeping the MuJoCo sim alive."""
    viewer = getattr(env, "viewer", None)
    if viewer is None:
        return
    try:
        _refresh_visible_viewer(env)
    except Exception:
        pass

    destroy_viewer = getattr(env, "_destroy_viewer", None)
    if callable(destroy_viewer):
        try:
            destroy_viewer()
            return
        except Exception:
            pass

    close = getattr(viewer, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass
    try:
        env.viewer = None
    except Exception:
        pass


def _close_wrapped_eval_env(wrapped_env, *, raw_env=None) -> None:
    """Release the temporary robomimic / robosuite grasp environment."""
    if raw_env is not None:
        _close_visible_viewer(raw_env)
        close = getattr(raw_env, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
    else:
        close = getattr(wrapped_env, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
    try:
        import gc
        gc.collect()
    except Exception:
        pass


def _set_viewer_camera(env, camera_name: str, *, render_once: bool = False) -> None:
    """Set a visible MuJoCo / OpenCV viewer camera if the env has one."""
    camera_id = env.sim.model.camera_name2id(camera_name)
    viewer = getattr(env, "viewer", None)
    if viewer is None and render_once:
        _refresh_visible_viewer(env)
        viewer = getattr(env, "viewer", None)
    if viewer is None:
        return

    viewer.set_camera(camera_id=camera_id)
    active_viewer = getattr(viewer, "viewer", None)
    active_cam = getattr(active_viewer, "cam", None)
    if active_cam is not None:
        active_cam.type = 2
        active_cam.fixedcamid = camera_id
    if render_once:
        _refresh_visible_viewer(env)


def _get_base_pose(env) -> tuple[np.ndarray, float]:
    from robosuite.utils.transform_utils import mat2euler

    robot = env.robots[0]
    site_name = robot.robot_model.base.correct_naming("center")
    pos = np.array(env.sim.data.site_xpos[env.sim.model.site_name2id(site_name)])
    mat = env.sim.data.get_site_xmat(site_name)
    yaw = float(mat2euler(mat)[2])
    return pos[:2], yaw


# ── base qpos ↔ world mapping ──────────────────────────────

_BASE_XY_MAPPING_CACHE_ATTR = "_agent_robosuite_base_xy_qpos_mapping"


def _wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _invalidate_base_xy_qpos_mapping(env) -> None:
    try:
        if hasattr(env, _BASE_XY_MAPPING_CACHE_ATTR):
            delattr(env, _BASE_XY_MAPPING_CACHE_ATTR)
    except Exception:
        pass


def _base_joint_qpos_indexes(env, robot) -> dict:
    indexes: dict[str, int] = {}
    for joint_name in robot.robot_model.base_joints:
        raw = joint_name.lower()
        if "mobile_forward" in raw:
            indexes["forward"] = env.sim.model.get_joint_qpos_addr(joint_name)
        elif "mobile_side" in raw:
            indexes["side"] = env.sim.model.get_joint_qpos_addr(joint_name)
        elif "mobile_yaw" in raw:
            indexes["yaw"] = env.sim.model.get_joint_qpos_addr(joint_name)
    missing = {"forward", "side"} - set(indexes)
    if missing:
        raise RuntimeError(f"Missing mobile base qpos indexes: {sorted(missing)}")
    return indexes


def _base_xy_qpos_mapping(env, robot, eps: float = 1e-4):
    """Compute the linear map between world xy deltas and qpos deltas.

    Cached on the env object for repeated use.
    """
    cached = getattr(env, _BASE_XY_MAPPING_CACHE_ATTR, None)
    if cached is not None:
        return cached

    indexes = _base_joint_qpos_indexes(env, robot)
    qpos = env.sim.data.qpos
    orig_forward = float(qpos[indexes["forward"]])
    orig_side = float(qpos[indexes["side"]])
    base_xy, _ = _get_base_pose(env)

    # perturb forward
    qpos[indexes["forward"]] = orig_forward + eps
    qpos[indexes["side"]] = orig_side
    env.sim.forward()
    forward_xy, _ = _get_base_pose(env)

    # perturb side
    qpos[indexes["forward"]] = orig_forward
    qpos[indexes["side"]] = orig_side + eps
    env.sim.forward()
    side_xy, _ = _get_base_pose(env)

    # restore
    qpos[indexes["forward"]] = orig_forward
    qpos[indexes["side"]] = orig_side
    env.sim.forward()

    qpos_to_world = np.column_stack(
        ((forward_xy - base_xy) / eps, (side_xy - base_xy) / eps),
    )
    det = float(np.linalg.det(qpos_to_world))
    if abs(det) < 1e-8:
        raise RuntimeError(f"Invalid base qpos-to-world mapping: {qpos_to_world}")

    world_to_qpos = np.linalg.inv(qpos_to_world)
    cached = (indexes, qpos_to_world, world_to_qpos)
    setattr(env, _BASE_XY_MAPPING_CACHE_ATTR, cached)
    return cached


def _set_base_world_yaw_direct(env, robot, target_yaw: float):
    indexes = _base_joint_qpos_indexes(env, robot)
    yaw_addr = indexes.get("yaw")
    if yaw_addr is None:
        return

    # The mobile_yaw joint is relative to the robot's initial scene
    # orientation. Correct it by measured world-yaw error instead of assigning
    # target_yaw directly.
    for _ in range(2):
        _, current_yaw = _get_base_pose(env)
        delta = _wrap_angle(float(target_yaw) - current_yaw)
        if abs(delta) < 1e-6:
            break
        env.sim.data.qpos[yaw_addr] += delta
        env.sim.forward()
    _invalidate_base_xy_qpos_mapping(env)


def _set_base_xy_direct(env, robot, target_xy: np.ndarray):
    indexes, _, world_to_qpos = _base_xy_qpos_mapping(env, robot)
    base_xy, _ = _get_base_pose(env)
    delta = np.asarray(target_xy, dtype=float) - base_xy
    delta_qpos = world_to_qpos @ delta
    env.sim.data.qpos[indexes["forward"]] += delta_qpos[0]
    env.sim.data.qpos[indexes["side"]] += delta_qpos[1]
    env.sim.forward()


# ── collision detection ─────────────────────────────────────

def _all_geom_names(sim) -> list[str]:
    names: list[str] = []
    for geom_id in range(sim.model.ngeom):
        name = sim.model.geom_id2name(geom_id)
        if name is not None:
            names.append(name)
    return names


def _robot_geom_names(env, robot) -> set[str]:
    cache_name = "_agent_robosuite_robot_geom_names"
    cached = getattr(env, cache_name, None)
    if cached is not None:
        return cached

    all_geoms = _all_geom_names(env.sim)
    names: set[str] = set(getattr(robot.robot_model, "contact_geoms", []))
    # Main robot prefix (e.g. "robot0_")
    prefix = getattr(robot.robot_model, "naming_prefix", "")
    if prefix:
        names.update(n for n in all_geoms if n.startswith(prefix))

    # Gripper(s) — robot.gripper may be a dict of {name: gripper_obj} or a single object
    grippers = robot.gripper
    if isinstance(grippers, dict):
        for g_obj in grippers.values():
            g_prefix = getattr(g_obj, "naming_prefix", "")
            if g_prefix:
                names.update(n for n in all_geoms if n.startswith(g_prefix))
    else:
        g_prefix = getattr(grippers, "naming_prefix", "")
        if g_prefix:
            names.update(n for n in all_geoms if n.startswith(g_prefix))

    setattr(env, cache_name, names)
    return names


def _ignored_collision(name: str | None, ignore_patterns: Sequence[str]) -> bool:
    if name in DEFAULT_IGNORED_COLLISION_GEOMS:
        return True
    s = name or ""
    # Check both the caller-supplied patterns and the built-in defaults
    return any(p in s for p in ignore_patterns) or any(p in s for p in DEFAULT_IGNORED_COLLISION_PATTERNS)


def _navigation_collisions(
    env, robot, ignore_patterns: Sequence[str],
) -> list[tuple[str, str]]:
    robot_geoms = _robot_geom_names(env, robot)
    collisions: list[tuple[str, str]] = []

    for contact in env.sim.data.contact[: env.sim.data.ncon]:
        g1 = env.sim.model.geom_id2name(contact.geom1)
        g2 = env.sim.model.geom_id2name(contact.geom2)
        g1_is_robot = g1 in robot_geoms
        g2_is_robot = g2 in robot_geoms

        if g1_is_robot == g2_is_robot:
            continue

        other = g2 if g1_is_robot else g1
        if _ignored_collision(other, ignore_patterns):
            continue

        robot_geom = g1 if g1_is_robot else g2
        collisions.append((robot_geom, other))

    return collisions


def _should_stop_for_collision(
    env, robot, ignore_patterns: Sequence[str],
    step: int, warmup: int, max_pairs: int,
) -> bool:
    if step < warmup:
        return False
    collisions = _navigation_collisions(env, robot, ignore_patterns)
    if not collisions:
        return False
    print("collision_detected:")
    for rg, og in collisions[:max_pairs]:
        print(f"  robot_geom={rg}, other_geom={og}")
    if len(collisions) > max_pairs:
        print(f"  ... {len(collisions) - max_pairs} more contact pairs")
    return True


# ── direct drive mode ───────────────────────────────────────

def _try_sync_transport(env):
    """Sync transport attachment if one is active (ignore errors)."""
    try:
        from robosuite.environments.factory_sorting.transport_attachment import sync_transport_attachment
        sync_transport_attachment(env)
    except Exception:
        pass

def _follow_path_direct(
    env,
    path: list[np.ndarray],
    *,
    max_steps: int,
    waypoint_tolerance: float,
    control_freq: int,
    max_linear: float,
    stop_on_collision: bool,
    collision_warmup_steps: int,
    ignore_collision_geom: Sequence[str],
    max_collision_pairs: int,
    headless: bool,
    debug: bool,
    frame_callback=None,
    record_every: int = 1,
    settle_steps: int = 5,
    posture: dict | None = None,
) -> bool:
    robot = env.robots[0]
    waypoint_index = 0
    reached_final = False
    max_step = max_linear / float(control_freq)
    idle_action = np.zeros_like(env.action_spec[0])

    for step in range(max_steps):
        base_xy, _ = _get_base_pose(env)
        goal_xy = path[waypoint_index]
        delta = goal_xy - base_xy
        distance = float(np.linalg.norm(delta))

        if distance < waypoint_tolerance:
            waypoint_index += 1
            if waypoint_index >= len(path):
                reached_final = True
                break
            continue

        step_xy = base_xy + delta / max(distance, 1e-6) * min(distance, max_step)
        _set_base_xy_direct(env, robot, step_xy)
        _try_sync_transport(env)
        env.step(idle_action)
        _restore_upper_body_posture(env, posture)
        _try_sync_transport(env)

        if frame_callback is not None and step % record_every == 0:
            frame_callback()

        if _should_stop_for_collision(
            env, robot, ignore_collision_geom,
            step, collision_warmup_steps, max_collision_pairs,
        ):
            logger.info("collision logged at step %d (navigation continues)", step)

        if not headless:
            env.render()
        if debug and step % 50 == 0:
            new_xy, yaw = _get_base_pose(env)
            print(
                f"nav_direct step={step} wp={waypoint_index}/{len(path)-1} "
                f"base=({base_xy[0]:.3f},{base_xy[1]:.3f}) "
                f"new=({new_xy[0]:.3f},{new_xy[1]:.3f}) "
                f"goal=({goal_xy[0]:.3f},{goal_xy[1]:.3f}) "
                f"dist={distance:.3f} yaw={yaw:.3f}"
            )

    for _ in range(settle_steps):
        env.step(idle_action)
        _restore_upper_body_posture(env, posture)
        _try_sync_transport(env)
        if frame_callback is not None:
            frame_callback()
        if not headless:
            env.render()
    return reached_final


# ── action (velocity-control) drive mode ────────────────────

def _world_velocity_to_base_frame(v_world: np.ndarray, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array(
        [c * v_world[0] + s * v_world[1], -s * v_world[0] + c * v_world[1]],
        dtype=float,
    )


def _shortest_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _build_base_action(robot, vx: float, vy: float, omega: float) -> np.ndarray:
    split = robot.composite_controller._action_split_indexes
    if "base" not in split:
        raise RuntimeError("Robot action space has no 'base' controller.")
    start, end = split["base"]
    dim = end - start
    base_action = np.zeros(dim)
    cmd = np.array([vx, vy, omega], dtype=float)
    base_action[: min(dim, 3)] = cmd[: min(dim, 3)]
    return robot.create_action_vector({"base": base_action})


def _follow_path_action(
    env,
    path: list[np.ndarray],
    *,
    max_steps: int,
    waypoint_tolerance: float,
    k_linear: float,
    k_angular: float,
    max_linear: float,
    max_angular: float,
    holonomic_base: bool,
    yaw_control: bool,
    turn_in_place_angle: float,
    stop_on_collision: bool,
    collision_warmup_steps: int,
    ignore_collision_geom: Sequence[str],
    max_collision_pairs: int,
    headless: bool,
    debug: bool,
    frame_callback=None,
    record_every: int = 1,
    settle_steps: int = 5,
    posture: dict | None = None,
) -> bool:
    robot = env.robots[0]
    waypoint_index = 0
    reached_final = False

    for step in range(max_steps):
        base_xy, yaw = _get_base_pose(env)
        goal_xy = path[waypoint_index]
        delta = goal_xy - base_xy
        distance = float(np.linalg.norm(delta))

        if distance < waypoint_tolerance:
            waypoint_index += 1
            if waypoint_index >= len(path):
                reached_final = True
                break
            continue

        target_yaw = math.atan2(delta[1], delta[0])
        yaw_error = _shortest_angle(target_yaw - yaw)

        speed = min(k_linear * distance, max_linear)
        if holonomic_base:
            v_world = speed * delta / max(distance, 1e-6)
            forward, lateral = _world_velocity_to_base_frame(v_world, yaw)
        else:
            forward, lateral = speed, 0.0

        if yaw_control:
            angular = np.clip(k_angular * yaw_error, -max_angular, max_angular)
        else:
            angular = 0.0

        if yaw_control and not holonomic_base and abs(yaw_error) > turn_in_place_angle:
            forward = 0.0

        action = _build_base_action(robot, forward, lateral, angular)
        env.step(action)
        _restore_upper_body_posture(env, posture)

        if frame_callback is not None and step % record_every == 0:
            frame_callback()

        if _should_stop_for_collision(
            env, robot, ignore_collision_geom,
            step, collision_warmup_steps, max_collision_pairs,
        ):
            logger.info("collision logged at step %d (navigation continues)", step)

        if not headless:
            env.render()
        if debug and step % 50 == 0:
            print(
                f"nav_action step={step} wp={waypoint_index}/{len(path)-1} "
                f"base=({base_xy[0]:.3f},{base_xy[1]:.3f}) "
                f"goal=({goal_xy[0]:.3f},{goal_xy[1]:.3f}) "
                f"dist={distance:.3f} yaw={yaw:.3f} yaw_err={yaw_error:.3f} "
                f"cmd=({forward:.3f},{lateral:.3f},{angular:.3f})"
            )

    stop_action = _build_base_action(robot, 0.0, 0.0, 0.0)
    for _ in range(settle_steps):
        env.step(stop_action)
        _restore_upper_body_posture(env, posture)
        if frame_callback is not None:
            frame_callback()
        if not headless:
            env.render()
    return reached_final
