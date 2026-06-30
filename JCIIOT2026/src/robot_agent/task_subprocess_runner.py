"""Run a robot-agent MuJoCo task in an isolated OS process.

This module is launched by the Streamlit app via ``python -m``.  It keeps all
robosuite / MuJoCo env creation, rendering, stepping, and cleanup out of the
Streamlit process so native viewer crashes do not take down the UI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


def _load_task_config():
    import json as _json
    from pathlib import Path
    _p = Path(__file__).resolve().parents[2] / "knowledge" / "task_config.json"
    return _json.loads(_p.read_text(encoding="utf-8")) if _p.exists() else {}

_TASK_CFG = _load_task_config()
_TASK_LIST = _TASK_CFG.get("tasks", [])
SCENE_MAP: dict[int, tuple[str, str]] = {
    i: (t["scene_prefix"], t["env_name"]) for i, t in enumerate(_TASK_LIST)
}
SCENE_INPUT_OBJECT_MAP: dict[str, dict[str, str]] = {
    t["env_name"]: {t["source"]: t["object"]} for t in _TASK_LIST
}


def _app_dir_from_args(value: str | None) -> Path:
    if value:
        return Path(value).resolve()
    return Path(__file__).resolve().parents[2]


def _configure_paths(app_dir: Path) -> None:
    src_dir = app_dir / "src"
    robosuite_inner_dir = app_dir / "robosuite" / "robosuite"
    for path in (src_dir, app_dir, app_dir / "robomimic", robosuite_inner_dir):
        s = str(path)
        if s not in sys.path:
            sys.path.insert(0, s)

    robosuite_inner = robosuite_inner_dir / "__init__.py"
    if robosuite_inner.exists():
        import robosuite as rs_patch

        rs_patch.__file__ = str(robosuite_inner)
        rs_patch.__path__ = [str(robosuite_inner_dir)]
        with open(robosuite_inner, encoding="utf-8") as handle:
            code = compile(handle.read(), str(robosuite_inner), "exec")
        exec(code, rs_patch.__dict__)


def _scene_env_name(task_index: int) -> str:
    return SCENE_MAP.get(task_index, SCENE_MAP[0])[1]


def _choose_map_files(app_dir: Path, task_index: int) -> tuple[Path, Path]:
    map_dir = (
        app_dir / "robosuite" / "robosuite" / "environments"
        / "factory_sorting" / "generated_maps"
    )
    prefix = SCENE_MAP.get(task_index, SCENE_MAP[0])[0]
    semantic = map_dir / f"{prefix}_scene_regenerated_semantic_map.json"
    grid = map_dir / f"{prefix}_scene_regenerated_occupancy_grid.npy"
    if semantic.exists() and grid.exists():
        return semantic, grid
    fallback = SCENE_MAP[0][0]
    return (
        map_dir / f"{fallback}_scene_regenerated_semantic_map.json",
        map_dir / f"{fallback}_scene_regenerated_occupancy_grid.npy",
    )


def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def _build_agent(app_dir: Path, task_index: int, knowledge_enabled: bool = True):
    from robot_agent.core.agent import RobotAgent
    from robot_agent.core.map_loader import load_map_files
    from robot_agent.core.scene_context import SceneContext
    from robot_agent.environments import RobosuiteBackend

    os.environ["GATE_OLLAMA"] = "true"
    os.environ["GATE_STEP_TIMEOUT"] = "false"
    # LLM config from robot_params.json (only set env if not already set)
    from robot_agent.config import _load_llm_params as _load_llm
    _llm = _load_llm()
    os.environ.setdefault("OLLAMA_BASE_URL", _llm["ollama_base_url"])
    os.environ.setdefault("OLLAMA_MODEL", _llm["ollama_model"])

    env_name = _scene_env_name(task_index)
    semantic, grid_file = _choose_map_files(app_dir, task_index)
    if not semantic.exists():
        raise RuntimeError(f"semantic map not found: {semantic}")

    scene, grid = load_map_files(semantic, grid_file)
    scene_ctx = SceneContext.from_semantic_map(scene)

    backend = RobosuiteBackend(
        env_name=env_name,
        camera="birdview",
        drive_mode="direct",
    )
    backend._scene_context = scene_ctx

    # First reset: headless, just to populate material_metadata for object map
    backend.reset()

    # Build dynamic input_object_map from material_metadata (all objects with port_name)
    dynamic_input_object_map: dict[str, str] = {}
    raw_metadata = getattr(backend.env, "material_metadata", {}) or {}
    for obj_name, info in raw_metadata.items():
        if not isinstance(info, dict):
            continue
        port_name = str(info.get("port_name") or "")
        if port_name:
            dynamic_input_object_map[port_name] = obj_name
            # Also register the line_* variant
            if port_name.startswith("input_"):
                line_name = "line_" + port_name.split("_", 1)[1]
                dynamic_input_object_map[line_name] = obj_name
            elif port_name.startswith("line_"):
                input_name = "input_" + port_name.split("_", 1)[1]
                dynamic_input_object_map[input_name] = obj_name

    # Merge task_config entry as override/fallback
    full_object_map = dict(dynamic_input_object_map)
    full_object_map.update(SCENE_INPUT_OBJECT_MAP.get(env_name, {}))

    # Checkpoint path is resolved from knowledge/robot_params.json at
    # policy-load time — no hardcoded path here.
    backend.set_physics_grasp_config(
        device="cpu",
        object_map=full_object_map,
    )

    # Second reset: now _has_physics=True, so visible birdview viewer is created
    backend.reset()

    scene_metadata = {
        "task_index": task_index,
        "env_name": env_name,
        "scene_name": getattr(scene_ctx, "scene_name", ""),
        "map_name": getattr(scene_ctx, "map_name", ""),
        "map_prefix": SCENE_MAP.get(task_index, SCENE_MAP[0])[0],
        "semantic_map": str(semantic),
        "occupancy_grid": str(grid_file),
        "input_object_map": full_object_map,
    }
    backend._scene_metadata = scene_metadata
    return RobotAgent(
        backend=backend,
        scene_context=scene_ctx,
        grid=grid,
        scene_metadata=scene_metadata,
        knowledge_enabled=knowledge_enabled,
    )


def _persist_memory(app_dir: Path, agent) -> None:
    from robot_agent.core.memory import InMemoryStore

    store = InMemoryStore(limit=32, persist_path=app_dir / "memory.json")
    for step in agent.memory.items():
        store.add(step)


def _manifest(
    *,
    task: str,
    task_index: int,
    env_name: str,
    status: str,
    success: bool,
    trajectory: str | None,
    running_trajectory: str,
    result: dict | None,
    elapsed_sec: float,
    error: str | None = None,
    traceback_text: str | None = None,
) -> dict:
    return {
        "task": task,
        "task_index": task_index,
        "env_name": env_name,
        "status": status,
        "success": bool(success),
        "trajectory": trajectory,
        "running_trajectory": running_trajectory,
        "result": result,
        "elapsed_sec": round(float(elapsed_sec), 3),
        "error": error,
        "traceback": traceback_text,
        "phase": "before_close",
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def run_task(args: argparse.Namespace) -> int:
    app_dir = _app_dir_from_args(args.app_dir)
    _configure_paths(app_dir)

    task_index = int(args.task_index)
    env_name = _scene_env_name(task_index)
    rec_dir = app_dir / "recordings" / env_name
    rec_dir.mkdir(parents=True, exist_ok=True)
    ts = args.timestamp
    running_path = rec_dir / f"trajectory_{ts}_RUNNING.json"
    result_path = Path(args.result_json) if args.result_json else rec_dir / f"result_{ts}.json"

    start = time.perf_counter()
    agent = None
    trajectory_path: str | None = None
    manifest: dict | None = None
    exit_code = 0

    try:
        kb_enabled = getattr(args, "knowledge_enabled", True)
        agent = _build_agent(app_dir, task_index, knowledge_enabled=kb_enabled)

        # Write scene_ready signal so the Streamlit app can close the "creating" dialog
        scene_ready_path = rec_dir / f"scene_ready_{ts}.json"
        _write_json_atomic(scene_ready_path, {
            "status": "ready",
            "env_name": env_name,
            "task_index": task_index,
            "timestamp": ts,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        agent.backend.start_recording()
        if hasattr(agent.backend, "set_trajectory_autosave"):
            agent.backend.set_trajectory_autosave(running_path, every_n_frames=50)

        result = agent.run(args.task)
        try:
            agent.backend._record_trajectory_frame()
        except Exception:
            pass

        status = "OK" if result.success else "FAIL"
        trajectory_path = agent.backend.save_trajectory(rec_dir / f"trajectory_{ts}_{status}.json")
        _persist_memory(app_dir, agent)
        try:
            if hasattr(agent.backend, "clear_trajectory_autosave"):
                agent.backend.clear_trajectory_autosave()
            if running_path.exists():
                running_path.unlink()
        except Exception:
            pass

        manifest = _manifest(
            task=args.task,
            task_index=task_index,
            env_name=env_name,
            status=status,
            success=bool(result.success),
            trajectory=trajectory_path,
            running_trajectory=str(running_path),
            result=result.as_dict(),
            elapsed_sec=time.perf_counter() - start,
        )
        _write_json_atomic(result_path, manifest)
    except Exception as exc:
        exit_code = 1
        tb = traceback.format_exc()
        print(tb, flush=True)
        try:
            if agent is not None:
                trajectory_path = agent.backend.save_trajectory(rec_dir / f"trajectory_{ts}_CRASH.json")
        except Exception:
            trajectory_path = str(running_path) if running_path.exists() else None

        manifest = _manifest(
            task=args.task,
            task_index=task_index,
            env_name=env_name,
            status="CRASH",
            success=False,
            trajectory=trajectory_path,
            running_trajectory=str(running_path),
            result=None,
            elapsed_sec=time.perf_counter() - start,
            error=str(exc),
            traceback_text=tb,
        )
        _write_json_atomic(result_path, manifest)
    finally:
        if agent is not None:
            agent.backend.close()

    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--app-dir", default=None)
    parser.add_argument("--knowledge-enabled", type=lambda v: v.lower() in ("true", "1", "yes"), default=True)
    args = parser.parse_args(argv)
    return run_task(args)


if __name__ == "__main__":
    raise SystemExit(main())
