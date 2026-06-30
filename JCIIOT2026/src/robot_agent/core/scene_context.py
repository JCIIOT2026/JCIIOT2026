"""
Factory-sorting scene knowledge — station names, positions, and approach points.

This module encodes the **fixed layout** of the FactorySorting environment
so the planner / skills can resolve "进料口1号" → ``input_1`` → world xy.

When the semantic map is available at runtime, prefer its values; the constants
here serve as fallbacks and documentation.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np


# ── port kind colours (shared with factory_sorting.py) ──────
PORT_COLORS: OrderedDict = OrderedDict(
    [
        ("conveyor", [0.05, 0.25, 0.85, 1.0]),
        ("shelf", [0.05, 0.55, 0.20, 1.0]),
        ("table", [0.95, 0.75, 0.10, 1.0]),
        ("bin", [0.85, 0.10, 0.08, 1.0]),
    ]
)

# ── station centre positions (world frame) ──────────────────
INPUT_STATION_TYPES = ["conveyor", "shelf", "table", "bin"]
OUTPUT_STATION_TYPES = ["table", "bin", "conveyor", "shelf"]

# Each (center_x, center_y, kind)
_INPUT_CENTERS: list[tuple[float, float, str]] = [
    (-2.0, -3.8, "conveyor"),
    (0.0, -3.8, "shelf"),
    (-2.0, 3.8, "table"),
    (0.0, 3.8, "bin"),
]

_OUTPUT_CENTERS: list[tuple[float, float, str]] = [
    (4.6, -3.0, "table"),
    (4.6, -1.0, "bin"),
    (4.6, 1.0, "conveyor"),
    (4.6, 3.0, "shelf"),
]

ROBOT_START = np.array([1.35, 0.0], dtype=float)
ROBOT_YAW = np.pi  # facing the input side


# ── data object ─────────────────────────────────────────────

@dataclass
class StationInfo:
    name: str
    kind: str          # conveyor / shelf / table / bin
    role: str          # "input" or "output"
    index: int         # 1-based
    center: np.ndarray
    approach: np.ndarray | None = None  # set from semantic map
    display_name: str = ""


@dataclass
class SceneContext:
    """Immutable snapshot of the factory layout.

    Load from a semantic map::

        ctx = SceneContext.from_semantic_map(scene)

    Then use in planner / skills::

        ctx.input_port("input_1")        # → StationInfo
        ctx.approach_xy("input_1")       # → np.ndarray

    The Siemens scene has 6 production lines (line_1..line_6), each with
    one input station (positive-Y side) and one output station (negative-Y side).
    """

    map_name: str = "factory_sorting"
    scene_name: str = "factory_sorting_six_lines"
    bounds: dict = field(default_factory=lambda: {
        "x_min": -18.8, "x_max": 13.8,
        "y_min": -10.8, "y_max": 8.8,
    })
    resolution: float = 0.05
    robot_start: np.ndarray = field(default_factory=lambda: ROBOT_START.copy())
    robot_yaw: float = ROBOT_YAW
    input_ports: dict[str, StationInfo] = field(default_factory=dict)
    output_ports: dict[str, StationInfo] = field(default_factory=dict)

    # ── factory methods ─────────────────────────────────────

    @classmethod
    def from_semantic_map(cls, scene: dict) -> SceneContext:
        """Build a ``SceneContext`` from a loaded semantic map dict."""
        ctx = cls(
            map_name=scene.get("map_name", "factory_sorting"),
            bounds=scene.get("bounds", {}),
            resolution=float(scene.get("resolution", 0.05)),
        )

        if "robot" in scene:
            robot = scene["robot"]
            start = robot.get("start")
            if start is not None:
                ctx.robot_start = np.array(start, dtype=float)

        # Try dict-style (input_ports/output_ports), fall back to objects array
        _ports_found = False
        for role_key in ("input_ports", "output_ports"):
            role = role_key.replace("_ports", "")  # "input" or "output"
            ports_dict = scene.get(role_key, {})
            if ports_dict:
                _ports_found = True
                for name, obj in ports_dict.items():
                    center = np.array(obj["center"], dtype=float)
                    approach = (
                        np.array(obj["approach"], dtype=float)
                        if obj.get("approach") is not None
                        else None
                    )
                    raw_index = obj.get("index")
                    if raw_index is not None:
                        port_index = int(raw_index) + 1
                    else:
                        import re
                        m = re.search(r"(\d+)$", name)
                        port_index = int(m.group(1)) if m else 0

                    info = StationInfo(
                        name=name, kind=obj.get("kind", ""), role=role,
                        index=port_index, center=center, approach=approach,
                        display_name=obj.get("display_name", name),
                    )
                    getattr(ctx, role_key)[name] = info

        # Fallback: read from "objects" array (new map format)
        if not _ports_found:
            for obj in scene.get("objects", []):
                role = obj.get("role", "")
                if role not in ("input", "output"):
                    continue
                name = obj.get("name", "")
                if not name:
                    continue
                role_key = f"{role}_ports"
                center = np.array(obj["center"], dtype=float)
                approach = (
                    np.array(obj["approach"], dtype=float)
                    if obj.get("approach") is not None
                    else None
                )
                import re
                m = re.search(r"(\d+)$", name)
                port_index = int(m.group(1)) if m else 0

                info = StationInfo(
                    name=name, kind=obj.get("kind", "table"), role=role,
                    index=port_index, center=center, approach=approach,
                    display_name=obj.get("display_name", name),
                )
                getattr(ctx, role_key)[name] = info

        return ctx

    # ── lookups ─────────────────────────────────────────────

    def input_port(self, name: str) -> StationInfo:
        p = self.input_ports.get(name)
        if p is None:
            raise KeyError(f"Unknown input port '{name}'. Options: {list(self.input_ports)}")
        return p

    def output_port(self, name: str) -> StationInfo:
        p = self.output_ports.get(name)
        if p is None:
            raise KeyError(f"Unknown output port '{name}'. Options: {list(self.output_ports)}")
        return p

    def approach_xy(self, port_name: str) -> np.ndarray:
        """Return the (2,) approach point for *port_name*.

        Tries input ports first, then output ports.
        """
        for ports in (self.input_ports, self.output_ports):
            info = ports.get(port_name)
            if info is not None:
                if info.approach is not None:
                    return info.approach.copy()
                # fallback: offset center slightly toward robot
                offset = -1.5 if info.role == "input" else -1.5
                return info.center[:2].copy() + np.array([offset, 0.0])

        raise KeyError(f"Unknown port '{port_name}'")

    def all_port_names(self) -> list[str]:
        return list(self.input_ports) + list(self.output_ports)

    def as_prompt_context(self) -> str:
        """One-shot summary text suitable for injection into an LLM planner prompt."""
        lines = [
            f"Scene: Siemens factory line ({getattr(self, 'scene_name', 'factory_sorting')})",
            f"{len(self.input_ports)} production lines, each with 1 input port (input_N) and 1 output port (output_N)",
            "Available stations:",
        ]
        for name, info in self.input_ports.items():
            lines.append(
                f"  Input {name} ({info.kind}) "
                f"中心=({info.center[0]:.1f},{info.center[1]:.1f})"
            )
        for name, info in self.output_ports.items():
            lines.append(
                f"  Output {name} ({info.kind}) "
                f"中心=({info.center[0]:.1f},{info.center[1]:.1f})"
            )
        lines.append(
            f"  Robot start=({self.robot_start[0]:.1f},{self.robot_start[1]:.1f})"
        )
        return "\n".join(lines)
