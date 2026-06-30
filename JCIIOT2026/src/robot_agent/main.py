"""CLI entrypoint for the robot agent.

Two modes:
    python -m robot_agent <task>     —  one-shot task (full pipeline)
    python -m robot_agent wired      —  factory-sorting hardcoded demo
"""

from __future__ import annotations

import sys
from pathlib import Path

from robot_agent.core.agent import RobotAgent
from robot_agent.core.map_loader import load_map_files
from robot_agent.core.scene_context import SceneContext
from robot_agent.environments import RobosuiteBackend


def run_wired(task: str) -> int:
    """Full pipeline: load maps, create backend, plan & execute."""
    _map_dir = (
        Path(__file__).resolve().parents[2]
        / "robosuite" / "robosuite" / "environments" / "factory_sorting" / "generated_maps"
    )
    # Prefer Siemens regenerated maps; fall back to legacy maps
    siemens_semantic = _map_dir / "factory_sorting_scene_regenerated_semantic_map.json"
    siemens_grid = _map_dir / "factory_sorting_scene_regenerated_occupancy_grid.npy"
    legacy_semantic = _map_dir / "factory_sorting_semantic_map.json"
    legacy_grid = _map_dir / "factory_sorting_occupancy_grid.npy"

    if siemens_semantic.exists() and siemens_grid.exists():
        semantic_map, occupancy_grid = siemens_semantic, siemens_grid
    else:
        semantic_map, occupancy_grid = legacy_semantic, legacy_grid

    if not semantic_map.exists() or not occupancy_grid.exists():
        print("Map files not found. Run get_map.py first:")
        print(f"  python robosuite/environments/factory_sorting/get_map.py")
        return 1

    scene, grid = load_map_files(semantic_map, occupancy_grid)
    scene_ctx = SceneContext.from_semantic_map(scene)

    backend = RobosuiteBackend(
        env_name="FactorySorting",
        camera="birdview",
        drive_mode="direct",
    )
    backend.reset()

    try:
        agent = RobotAgent(
            backend=backend,
            scene_context=scene_ctx,
            grid=grid,
        )
        result = agent.run(task)

        print(f"skill   = {result.skill_name}")
        print(f"success = {result.success}")
        print(f"message = {result.message}")
        for i, step in enumerate(result.steps):
            print(f"  step[{i}]: {step.skill} ok={step.success}  {step.message}")
        return 0 if result.success else 2
    finally:
        backend.close()


def build_agent() -> RobotAgent:
    """Build a RobotAgent with default maps and backend."""
    _map_dir = (
        Path(__file__).resolve().parents[2]
        / "robosuite" / "robosuite" / "environments" / "factory_sorting" / "generated_maps"
    )
    scene, grid = load_map_files(
        _map_dir / "factory_sorting_scene_regenerated_semantic_map.json",
        _map_dir / "factory_sorting_scene_regenerated_occupancy_grid.npy",
    )
    scene_ctx = SceneContext.from_semantic_map(scene)

    backend = RobosuiteBackend(
        env_name="FactorySorting",
        camera="birdview",
        drive_mode="direct",
    )
    backend.reset()

    return RobotAgent(backend=backend, scene_context=scene_ctx, grid=grid)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: robot-agent <task>")
        print("Example: robot-agent 把1号进料口的物体送到3号出料口")
        return 1

    task = " ".join(sys.argv[1:]).strip()
    return run_wired(task)


if __name__ == "__main__":
    raise SystemExit(main())
