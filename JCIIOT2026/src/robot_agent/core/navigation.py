"""
Pure 2D path planning and coordinate transforms for robot navigation.

Zero dependencies beyond numpy.  No simulation, no robot model, no API calls.
Extracted from ``llm_task_navigator.py`` so the agent can plan paths offline.
"""

from __future__ import annotations

import heapq
import math
from typing import Sequence

import numpy as np

# ── grid cell types ─────────────────────────────────────────
FREE = 0
OBSTACLE = 1
STATION = 2
APPROACH = 3
ROBOT = 4

PASSABLE: set[int] = {FREE, APPROACH, ROBOT}


# ── coordinate transforms ───────────────────────────────────

def world_to_grid(
    x: float, y: float, bounds: dict, resolution: float,
) -> tuple[int, int]:
    """Convert world (x, y) to grid (row, col)."""
    row = int(round((x - bounds["x_min"]) / resolution))
    col = int(round((y - bounds["y_min"]) / resolution))
    return row, col


def grid_to_world(
    row: int, col: int, bounds: dict, resolution: float,
) -> np.ndarray:
    """Convert grid (row, col) to world (x, y)."""
    return np.array(
        [
            bounds["x_min"] + row * resolution,
            bounds["y_min"] + col * resolution,
        ],
        dtype=float,
    )


# ── grid queries ────────────────────────────────────────────

def in_grid(grid: np.ndarray, cell: tuple[int, int]) -> bool:
    row, col = cell
    return 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]


def is_passable(grid: np.ndarray, cell: tuple[int, int]) -> bool:
    return in_grid(grid, cell) and int(grid[cell[0], cell[1]]) in PASSABLE


def nearest_passable_cell(
    grid: np.ndarray,
    start_cell: tuple[int, int],
    max_radius: int = 20,
) -> tuple[int, int]:
    """Find the closest passable grid cell within *max_radius*.

    Searches in expanding rings; returns as soon as a candidate is found.
    """
    if is_passable(grid, start_cell):
        return start_cell

    start_row, start_col = start_cell
    best_cell: tuple[int, int] | None = None
    best_dist = float("inf")

    for radius in range(1, max_radius + 1):
        for row in range(start_row - radius, start_row + radius + 1):
            for col in range(start_col - radius, start_col + radius + 1):
                # only check the outer ring
                if abs(row - start_row) != radius and abs(col - start_col) != radius:
                    continue
                cell = (row, col)
                if is_passable(grid, cell):
                    dist = math.hypot(row - start_row, col - start_col)
                    if dist < best_dist:
                        best_cell = cell
                        best_dist = dist
        if best_cell is not None:
            return best_cell

    raise RuntimeError(
        f"No passable cell found near {start_cell}; "
        f"check approach points and safety margin."
    )


# ── A* planner ──────────────────────────────────────────────

def astar(
    grid: np.ndarray,
    start_cell: tuple[int, int],
    goal_cell: tuple[int, int],
) -> list[tuple[int, int]]:
    """Classic A* on an 8-connected grid.

    Returns the shortest path as a list of (row, col) cells from start to goal.
    Both endpoints are snapped to the nearest passable cell first.
    """
    start_cell = nearest_passable_cell(grid, start_cell)
    goal_cell = nearest_passable_cell(grid, goal_cell)

    # 8-connected neighbours: (drow, dcol, cost)
    neighbours: list[tuple[int, int, float]] = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    ]

    def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    open_heap: list[tuple[float, float, tuple[int, int]]] = [
        (_heuristic(start_cell, goal_cell), 0.0, start_cell),
    ]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], float] = {start_cell: 0.0}

    while open_heap:
        _, cost, current = heapq.heappop(open_heap)
        if current == goal_cell:
            return reconstruct_path(came_from, current)
        if cost > g_score.get(current, float("inf")):
            continue

        for drow, dcol, step_cost in neighbours:
            nxt = (current[0] + drow, current[1] + dcol)
            if not is_passable(grid, nxt):
                continue
            tentative = cost + step_cost
            if tentative < g_score.get(nxt, float("inf")):
                came_from[nxt] = current
                g_score[nxt] = tentative
                priority = tentative + _heuristic(nxt, goal_cell)
                heapq.heappush(open_heap, (priority, tentative, nxt))

    raise RuntimeError(f"A* failed from {start_cell} to {goal_cell}.")


def reconstruct_path(
    came_from: dict[tuple[int, int], tuple[int, int]],
    current: tuple[int, int],
) -> list[tuple[int, int]]:
    """Back-track through *came_from* to produce an ordered cell path."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


# ── path post-processing ────────────────────────────────────

def simplify_path(
    world_path: Sequence[np.ndarray],
    min_spacing: float = 0.20,
) -> list[np.ndarray]:
    """Down-sample a world-frame path so consecutive points are ≥ *min_spacing* apart.

    Always keeps the first and last waypoints.
    """
    if len(world_path) <= 2:
        return list(world_path)

    simplified: list[np.ndarray] = [world_path[0]]
    last = world_path[0]
    for point in world_path[1:-1]:
        if np.linalg.norm(point - last) >= min_spacing:
            simplified.append(point)
            last = point
    simplified.append(world_path[-1])
    return simplified


# ── angle helpers ───────────────────────────────────────────

def shortest_angle(angle: float) -> float:
    """Wrap *angle* (rad) to [-π, π)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
