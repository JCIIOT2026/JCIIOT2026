"""
Load semantic maps and occupancy grids; plan world-frame paths.

Thin orchestration layer that ties ``navigation.py`` to file I/O.
Does **not** import robosuite — pure numpy + stdlib.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from robot_agent.core.navigation import (
    astar,
    grid_to_world,
    simplify_path,
    world_to_grid,
)


# ── map loading ─────────────────────────────────────────────

def load_semantic_map(path: str | Path) -> dict:
    """Load a semantic-map JSON file (e.g. ``factory_sorting_semantic_map.json``)."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_occupancy_grid(path: str | Path) -> np.ndarray:
    """Load an occupancy grid ``.npy`` file."""
    return np.load(str(path))


def load_map_files(
    semantic_map: str | Path,
    occupancy_grid: str | Path,
) -> tuple[dict, np.ndarray]:
    """Convenience: load both semantic map and occupancy grid at once.

    Returns ``(scene_dict, grid_array)``.
    """
    return load_semantic_map(semantic_map), load_occupancy_grid(occupancy_grid)


# ── path planning (world-frame) ─────────────────────────────

def plan_world_path(
    scene: dict,
    grid: np.ndarray,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    min_spacing: float = 0.35,
) -> list[np.ndarray]:
    """Plan a world-frame path from *start_xy* to *goal_xy*.

    1. Convert world → grid
    2. Run A*
    3. Convert grid → world
    4. Simplify (down-sample)

    Args:
        scene: Semantic map dict (must contain ``bounds`` and ``resolution``).
        grid: 2D occupancy grid (uint8).
        start_xy: (2,) world position.
        goal_xy:  (2,) world position.
        min_spacing: Minimum spacing (m) between consecutive waypoints.

    Returns:
        List of (2,) numpy arrays in world frame.
    """
    bounds = scene["bounds"]
    resolution = float(scene["resolution"])

    start_cell = world_to_grid(start_xy[0], start_xy[1], bounds, resolution)
    goal_cell = world_to_grid(goal_xy[0], goal_xy[1], bounds, resolution)

    cell_path = astar(grid, start_cell, goal_cell)
    world_path = [
        grid_to_world(row, col, bounds, resolution) for row, col in cell_path
    ]
    return simplify_path(world_path, min_spacing=min_spacing)


# ── station summary (for LLM) ───────────────────────────────

def summarize_map_for_llm(scene: dict) -> dict:
    """Extract a compact station summary suitable for an LLM prompt.

    Returns ``{name: {role, kind, center, approach, display_name, image_position}}``.
    """
    nodes: dict[str, dict] = {}
    for group in ("input_ports", "output_ports"):
        for name, obj in scene.get(group, {}).items():
            nodes[name] = {
                "role": obj.get("role"),
                "kind": obj.get("kind"),
                "center": obj.get("center"),
                "approach": obj.get("approach"),
                "display_name": obj.get("display_name", name),
                "image_position": obj.get("image_position"),
            }
    return nodes
