"""Robot Agent dashboard wired to robosuite with video output,
knowledge-base browser, and memory inspector."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib import request

import numpy as np
import streamlit as st
from PIL import Image

# ── Ensure v4 paths are first (v2 may leak from venv .pth) ──
_THIS_FILE = Path(__file__).resolve()
_APP_DIR = _THIS_FILE.parent
_SRC_DIR = _APP_DIR / "src"
_ROBOSUITE_INNER_DIR = _APP_DIR / "robosuite" / "robosuite"
_ROBOSUITE_INNER = _ROBOSUITE_INNER_DIR / "__init__.py"

# v4/src must precede v2/src so robot_agent resolves correctly
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

# Purge any cached v2 robot_agent modules
for _m in list(sys.modules):
    if _m.startswith("robot_agent"):
        del sys.modules[_m]

# ── monkey-patch robosuite namespace-package issue ──
if _ROBOSUITE_INNER.exists():
    # Ensure v4 inner is first in sys.path so submodules resolve here, not v2
    if str(_ROBOSUITE_INNER_DIR) not in sys.path:
        sys.path.insert(0, str(_ROBOSUITE_INNER_DIR))
    import robosuite as _rs_patch
    _rs_patch.__file__ = str(_ROBOSUITE_INNER)
    _rs_patch.__path__ = [str(_ROBOSUITE_INNER_DIR)]  # force submodule resolution to v4
    # Re-run the real __init__.py in robosuite's namespace to export
    # make, FactorySorting, __version__, etc.
    with open(_ROBOSUITE_INNER, encoding="utf-8") as _f:
        _code = compile(_f.read(), str(_ROBOSUITE_INNER), "exec")
    exec(_code, _rs_patch.__dict__)

# Use this file's location to find maps; robust regardless of cwd.
_MAP_DIR = (
    _APP_DIR / "robosuite" / "robosuite" / "environments"
    / "factory_sorting" / "generated_maps"
)
_KNOWLEDGE_ROOT = _APP_DIR / "knowledge"

# LLM config is read from knowledge/robot_params.json at import time.
# Env vars OLLAMA_BASE_URL / OLLAMA_MODEL still take priority if set.
from robot_agent.config import _load_llm_params as _load_llm_params
_llm_defaults = _load_llm_params()
DEFAULT_OLLAMA_BASE_URL = _llm_defaults["ollama_base_url"]
DEFAULT_OLLAMA_MODEL = _llm_defaults["ollama_model"]
DEFAULT_OPENAI_BASE_URL = _llm_defaults.get("openai_base_url", "https://api.deepseek.com")
DEFAULT_OPENAI_MODEL = _llm_defaults.get("openai_model", "deepseek-v4-flash")
DEFAULT_VISION_MODEL = _llm_defaults["vision_model"]
AUTO_GENERATE_REPLAY_GIFS = False
SCORE_RULE_VERSION = "grasp_success_gate_l5_multi_v2"

L5_INPUT1_OBJECTS = (
    "white_tote_b01_left_center",
    "white_tote_b01_left_front",
    "white_tote_b01_left_back",
)


# =====================================================================
#  Diagnostics
# =====================================================================

# Load unified task config (single source of truth) — all data in knowledge/task_config.json
def _load_task_config() -> dict:
    import json as _json
    _path = _APP_DIR / "knowledge" / "task_config.json"
    if _path.exists():
        return _json.loads(_path.read_text(encoding="utf-8"))
    return {}

_TASK_CFG = _load_task_config()
_TASK_LIST = _TASK_CFG.get("tasks", [])

def _task_for_index(task_index: int) -> dict:
    return _TASK_LIST[min(task_index, len(_TASK_LIST) - 1)] if _TASK_LIST else {}

def _scene_prefix(task_index: int) -> str:
    return _task_for_index(task_index).get("scene_prefix", "factory_sorting_1_3fo3erfhisem")

def _scene_env_name(task_index: int) -> str:
    return _task_for_index(task_index).get("env_name", "FactorySorting1_3FO3ERFHISEM")

def _task_source_name(task_index: int) -> str:
    return _task_for_index(task_index).get("source", "input_5")

def _task_target_name(task_index: int) -> str:
    return _task_for_index(task_index).get("target", "output_4")

def _task_object_name(task_index: int) -> str:
    return _task_for_index(task_index).get("object", "")

def _task_grasp_pose(source: str) -> tuple | None:
    poses = _TASK_CFG.get("grasp_poses", {})
    entry = poses.get(source)
    if entry:
        return (entry["pos"], [0.0, 0.0, entry["yaw"]])
    return None

def _task_default_object_map() -> dict:
    return _TASK_CFG.get("default_object_map", {})

def _recordings_root() -> Path:
    return _APP_DIR / "recordings"

def _scene_recording_dir(env_name: str) -> Path:
    path = _recordings_root() / env_name
    path.mkdir(parents=True, exist_ok=True)
    return path

def _task_recording_dir(task_index: int | None = None) -> Path:
    return _scene_recording_dir(_scene_env_name(task_index))

def _score_path_for_trajectory(traj_path: Path) -> Path:
    name = traj_path.name
    if name.startswith("trajectory_"):
        name = "score_" + name[len("trajectory_"):]
    else:
        name = traj_path.stem + "_score.json"
    return traj_path.with_name(name)

def _json_safe(value):
    """Convert numpy / Path values to plain JSON-serializable Python types."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value

def _write_score_file(
    *,
    task_index: int,
    details: dict,
    trajectory_path: Path,
    status: str,
    elapsed: float | None = None,
) -> Path:
    data = {
        "task_index": int(task_index),
        "env_name": _scene_env_name(task_index),
        "status": str(status),
        "trajectory": str(trajectory_path),
        "elapsed_sec": round(float(elapsed), 3) if elapsed is not None else None,
        "score": details.get("total", 0),
        "score_rule_version": SCORE_RULE_VERSION,
        "details": details,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    data = _json_safe(data)
    score_path = _score_path_for_trajectory(trajectory_path)
    tmp = score_path.with_name(f"{score_path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(score_path)
    return score_path

def _load_score_file(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        details = data.get("details")
        if isinstance(details, dict):
            return data
    except Exception:
        pass
    return None

def _latest_score_for_task(task_index: int) -> dict | None:
    rec_dir = _task_recording_dir(task_index)
    files = sorted(rec_dir.glob("score_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in files:
        data = _load_score_file(path)
        if (
            data
            and int(data.get("task_index", -1)) == int(task_index)
            and data.get("score_rule_version") == SCORE_RULE_VERSION
        ):
            data["_score_path"] = str(path)
            return data
    return None

def _latest_trajectory_for_task(task_index: int) -> Path | None:
    rec_dir = _task_recording_dir(task_index)
    files = sorted(rec_dir.glob("trajectory_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None

def _ensure_scene_recording_dirs() -> None:
    for _t in _TASK_LIST:
        _scene_recording_dir(_t["env_name"])

def _trajectory_env_name(json_path: Path) -> str:
    known_envs = {t["env_name"] for t in _TASK_LIST}
    if json_path.parent.name in known_envs:
        return json_path.parent.name
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        env_name = data.get("robot_model")
        if env_name in known_envs:
            return env_name
    except Exception:
        pass
    return _scene_env_name(0)

def _trajectory_json_files() -> list[Path]:
    root = _recordings_root()
    if not root.exists():
        return []
    files = list(root.glob("trajectory_*.json"))
    for _t in _TASK_LIST:
        scene_dir = root / _t["env_name"]
        if scene_dir.exists():
            files.extend(scene_dir.glob("trajectory_*.json"))
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)

def _infer_grasp_replay_range(json_path: Path) -> tuple[int, int, str] | None:
    """Find a compact grasp segment from a trajectory JSON.

    New trajectories use explicit grasp_start / grasp_end events. Older files
    fall back to a heuristic based on object lift / transport movement.
    """
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    frames = data.get("frames", [])
    if len(frames) < 2:
        return None

    events = data.get("events", [])
    if isinstance(events, list):
        clean_events = []
        for event in events:
            if not isinstance(event, dict):
                continue
            try:
                frame = int(event.get("frame"))
            except Exception:
                continue
            if 0 <= frame < len(frames):
                clean_events.append((frame, event))
        clean_events.sort(key=lambda item: item[0])

        for start_frame, start_event in clean_events:
            if start_event.get("name") != "grasp_start":
                continue
            for end_frame, end_event in clean_events:
                if end_frame < start_frame:
                    continue
                if end_event.get("name") != "grasp_end":
                    continue
                obj_name = (
                    start_event.get("object_name")
                    or end_event.get("object_name")
                    or start_event.get("source")
                    or "grasp"
                )
                return start_frame, min(len(frames), end_frame + 1), str(obj_name)

    object_names = data.get("object_names", [])
    if not object_names:
        return None

    best: tuple[float, str, int, int | None] | None = None
    for obj_name in object_names:
        first = None
        for f in frames:
            pos = f.get("object_positions", {}).get(obj_name)
            if pos and len(pos) >= 3:
                first = (float(pos[0]), float(pos[1]), float(pos[2]))
                break
        if first is None:
            continue

        x0, y0, z0 = first
        max_score = 0.0
        trigger_idx = None
        move_idx = None
        for idx, f in enumerate(frames):
            pos = f.get("object_positions", {}).get(obj_name)
            if not pos or len(pos) < 3:
                continue
            dx = float(pos[0]) - x0
            dy = float(pos[1]) - y0
            dz = float(pos[2]) - z0
            xy = float((dx * dx + dy * dy) ** 0.5)
            score = xy + max(0.0, dz) * 2.0
            if score > max_score:
                max_score = score
            if trigger_idx is None and dz > 0.03:
                trigger_idx = idx
            if move_idx is None and xy > 0.20:
                move_idx = idx

        if trigger_idx is None:
            trigger_idx = move_idx
        if trigger_idx is None or max_score < 0.05:
            continue
        candidate = (max_score, obj_name, trigger_idx, move_idx)
        if best is None or candidate[0] > best[0]:
            best = candidate

    if best is None:
        return None

    _, obj_name, trigger_idx, move_idx = best
    bp = frames[trigger_idx].get("base_pose", {})
    base_pos = bp.get("position", [])
    if len(base_pos) >= 2:
        bx, by = float(base_pos[0]), float(base_pos[1])
        start = trigger_idx
        while start > 0:
            prev_bp = frames[start - 1].get("base_pose", {})
            prev_pos = prev_bp.get("position", [])
            if len(prev_pos) < 2:
                break
            dist = ((float(prev_pos[0]) - bx) ** 2 + (float(prev_pos[1]) - by) ** 2) ** 0.5
            if dist > 0.45:
                break
            start -= 1
        start = max(0, start - 5)
    else:
        start = max(0, trigger_idx - 100)

    end = (move_idx + 15) if move_idx is not None else (trigger_idx + 60)
    end = min(len(frames), max(end, trigger_idx + 20))
    if end <= start:
        end = min(len(frames), start + 2)
    return start, end, obj_name

def _choose_map_files(task_index: int | None = None) -> tuple[Path, Path]:
    """Pick map files for the given task index."""
    idx = task_index if task_index is not None else 0
    prefix = _scene_prefix(idx)
    semantic = _MAP_DIR / f"{prefix}_scene_regenerated_semantic_map.json"
    grid = _MAP_DIR / f"{prefix}_scene_regenerated_occupancy_grid.npy"
    if semantic.exists() and grid.exists():
        return semantic, grid
    # Fallback to default scene
    fallback = _scene_prefix(0)
    return _MAP_DIR / f"{fallback}_scene_regenerated_semantic_map.json", _MAP_DIR / f"{fallback}_scene_regenerated_occupancy_grid.npy"


def _check_map_files() -> dict:
    """Return a dict describing the map-file situation for the sidebar."""
    semantic, grid = _choose_map_files()
    info: dict = {
        "dir": str(_MAP_DIR),
        "dir_ok": _MAP_DIR.exists(),
        "semantic": str(semantic),
        "semantic_ok": semantic.exists(),
        "grid": str(grid),
        "grid_ok": grid.exists(),
        "new_scene_map_ok": (_MAP_DIR / "factory_sorting_1_3fo3erfhisem_scene_regenerated_semantic_map.json").exists(),
        "new_scene_grid_ok": (_MAP_DIR / "factory_sorting_1_3fo3erfhisem_scene_regenerated_occupancy_grid.npy").exists(),
    }
    info["all_ok"] = info["dir_ok"] and info["semantic_ok"] and info["grid_ok"]
    return info


# =====================================================================
#  LLM backend helpers
# =====================================================================

def _build_llm_config() -> dict[str, str]:
    """Return the effective LLM backend configuration for display.

    Reads from session state first, then env vars, then defaults.
    """
    backend = st.session_state.get("_llm_backend", "ollama")
    if backend == "local":
        local_path = st.session_state.get("_local_model_path", "") or os.getenv("LOCAL_LLM_MODEL", "")
        return {"backend": "Local GGUF", "model": Path(local_path).name if local_path else "(not set)"}
    if backend == "openai":
        model = st.session_state.get("_openai_model", "") or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        return {"backend": "OpenAI API", "model": model}
    # ollama (default)
    model = st.session_state.get("_ollama_model", "") or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    return {"backend": "Ollama", "model": model}


def _get_llm_backend() -> str:
    """Return the currently selected backend name."""
    return st.session_state.get("_llm_backend", "ollama")


def _create_llm_client_from_session():
    """Build an LLM client from the current sidebar session state.

    Returns a client with the ``generate(prompt, *, num_predict, temperature, json_mode) -> str``
    interface (OllamaClient / OpenAIClient / GlmClient / LocalLLM).
    """
    backend = st.session_state.get("_llm_backend", "ollama")

    if backend == "local":
        local_path = st.session_state.get("_local_model_path", "") or os.getenv("LOCAL_LLM_MODEL", "")
        if not local_path:
            raise RuntimeError("No GGUF model path set. Configure Local GGUF in the sidebar.")
        from robot_agent.core.local_llm import LocalLLM
        return LocalLLM(model_path=local_path)

    if backend == "openai":
        api_key = st.session_state.get("_openai_api_key", "") or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("No OpenAI API key set. Configure OpenAI API in the sidebar.")
        from robot_agent.core.openai_client import OpenAIClient
        return OpenAIClient(
            api_key=api_key,
            base_url=st.session_state.get("_openai_base_url", "") or os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL),
            model=st.session_state.get("_openai_model", "") or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
            timeout=120.0,
        )

    # ollama (default)
    from robot_agent.core.ollama_client import OllamaClient
    return OllamaClient(
        base_url=st.session_state.get("_ollama_url", "") or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
        model=st.session_state.get("_ollama_model", "") or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        timeout=120.0,
    )


def _get_ollama_url() -> str:
    """Return the effective Ollama base URL (from session state, env, or default)."""
    return st.session_state.get("_ollama_url", "") or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)


def _detect_vision_config_for_sidebar() -> dict:
    """Return VLM config dict from current sidebar VLM fields or backend fallback."""
    _vlm_url = st.session_state.get("_vlm_api_url", "")
    _vlm_key = st.session_state.get("_vlm_api_key", "")
    _vlm_model = st.session_state.get("_vlm_model", "")

    if _vlm_url:
        from robot_agent.core.vision_client import _detect_api_type
        return {
            "base_url": _vlm_url,
            "model": _vlm_model or DEFAULT_VISION_MODEL,
            "api_type": "openai" if _vlm_key else _detect_api_type(_vlm_url),
            "api_key": _vlm_key,
        }

    _backend = st.session_state.get("_llm_backend", "ollama")
    if _backend == "openai":
        return {
            "base_url": st.session_state.get(
                "_openai_base_url",
                os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            ),
            "model": _vlm_model or "gpt-4o",
            "api_type": "openai",
            "api_key": st.session_state.get(
                "_openai_api_key",
                os.getenv("OPENAI_API_KEY", ""),
            ),
        }
    return {
        "base_url": _get_ollama_url(),
        "model": _vlm_model or DEFAULT_VISION_MODEL,
        "api_type": "ollama",
        "api_key": "",
    }


def _record_active_backend() -> None:
    """Record the currently selected backend + model in session state for display."""
    _backend = st.session_state.get("_llm_backend", "ollama")
    _backend_names = {"ollama": "Ollama", "openai": "OpenAI API", "local": "Local GGUF"}
    if _backend == "ollama":
        _model = st.session_state.get("_ollama_model", DEFAULT_OLLAMA_MODEL)
    elif _backend == "openai":
        _model = st.session_state.get("_openai_model", DEFAULT_OPENAI_MODEL)
    elif _backend == "local":
        _model = Path(st.session_state.get("_local_model_path", "")).name
    else:
        _model = ""
    st.session_state["_active_llm_backend"] = _backend_names.get(_backend, _backend)
    st.session_state["_active_llm_model"] = _model


def _call_generate(prompt: str, num_predict: int) -> tuple[dict, float]:
    gen_url = f"{DEFAULT_OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    payload = json.dumps({
        "model": DEFAULT_OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": num_predict, "temperature": 0},
    }).encode("utf-8")
    t0 = time.perf_counter()
    req = request.Request(gen_url, data=payload,
                          headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data, (time.perf_counter() - t0) * 1000


def diagnose_llm() -> dict:
    """Diagnose the currently selected LLM backend (from sidebar session state).

    Reads backend choice and config from ``st.session_state``, falls back to
    env vars and defaults.
    """
    result: dict = {"ok": False, "message": ""}
    backend = st.session_state.get("_llm_backend", "ollama")

    # ── Local GGUF ──
    if backend == "local":
        local_path = st.session_state.get("_local_model_path", "") or os.getenv("LOCAL_LLM_MODEL", "")
        if not local_path:
            result["message"] = "No GGUF model path set. Enter a path in the sidebar."
            return result
        from pathlib import Path as _Path
        if not _Path(local_path).exists():
            result["message"] = f"Model file not found: {local_path}"
            return result
        t0 = time.perf_counter()
        try:
            from robot_agent.core.local_llm import LocalLLM
            client = LocalLLM(model_path=local_path)
            resp = client.generate("Say hello.", num_predict=16, temperature=0)
            result["api_latency_ms"] = (time.perf_counter() - t0) * 1000
            if resp:
                result["ok"] = True
                result["message"] = f"Local GGUF loaded: {_Path(local_path).name}"
            else:
                result["message"] = "Local LLM returned empty response"
        except Exception as exc:
            result["message"] = f"Local LLM error: {exc}"
        return result

    # ── OpenAI API ──
    if backend == "openai":
        api_key = st.session_state.get("_openai_api_key", "") or os.getenv("OPENAI_API_KEY", "")
        base_url = st.session_state.get("_openai_base_url", "") or os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)
        model = st.session_state.get("_openai_model", "") or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        if not api_key:
            result["message"] = "No API key set. Enter your OpenAI-compatible API key in the sidebar."
            return result
        from robot_agent.core.openai_client import OpenAIClient
        t0 = time.perf_counter()
        try:
            client = OpenAIClient(api_key=api_key, base_url=base_url, model=model, timeout=30.0)
            hc = client.healthcheck()
            result["api_latency_ms"] = (time.perf_counter() - t0) * 1000
            if hc.get("ok") == "true":
                result["ok"] = True
                result["message"] = hc.get("message", f"Connected to {base_url}")
                result["models"] = hc.get("models", "")
            else:
                result["message"] = hc.get("message", "Healthcheck failed")
            return result
        except Exception as exc:
            result["message"] = f"Cannot connect to OpenAI API: {exc}"
            return result

    # ── Ollama (default) ──
    ollama_url = st.session_state.get("_ollama_url", "") or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
    ollama_model = st.session_state.get("_ollama_model", "") or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)

    tags_url = f"{ollama_url.rstrip('/')}/api/tags"
    t0 = time.perf_counter()
    try:
        req = request.Request(tags_url, method="GET",
                              headers={"Content-Type": "application/json"})
        with request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        result["message"] = f"Cannot connect to Ollama at {ollama_url}: {exc}"
        return result
    result["api_latency_ms"] = (time.perf_counter() - t0) * 1000

    models = data.get("models", []) if isinstance(data, dict) else []
    model_names = {str(item.get("name", "")) for item in models if isinstance(item, dict)}
    if ollama_model not in model_names:
        result["message"] = (
            f"Model '{ollama_model}' not found at {ollama_url}\n"
            f"Available models: {', '.join(sorted(model_names))}"
        )
        return result

    # Quick generate test
    try:
        gen_url = f"{ollama_url.rstrip('/')}/api/generate"
        payload = json.dumps({
            "model": ollama_model,
            "prompt": "Say hello.",
            "stream": False,
            "options": {"num_predict": 16, "temperature": 0},
        }).encode("utf-8")
        t1 = time.perf_counter()
        req = request.Request(gen_url, data=payload,
                              headers={"Content-Type": "application/json"}, method="POST")
        with request.urlopen(req, timeout=120) as resp:
            json.loads(resp.read().decode("utf-8"))
        result["hello_roundtrip_ms"] = (time.perf_counter() - t1) * 1000
    except Exception as exc:
        result["message"] = f"Hello test failed: {exc}"
        return result

    result["ok"] = True
    result["message"] = f"Ollama connected — model: {ollama_model}"
    return result


# =====================================================================
#  Agent factory
# =====================================================================

def build_agent(task_index: int = 0) -> RobotAgent | None:
    """Build a wired agent for *task_index* (0-based)."""
    from robot_agent.core.agent import RobotAgent
    from robot_agent.core.map_loader import load_map_files
    from robot_agent.core.scene_context import SceneContext
    from robot_agent.environments import RobosuiteBackend

    os.environ["GATE_OLLAMA"] = "true"
    os.environ["GATE_STEP_TIMEOUT"] = "false"

    # ── Apply backend selection from sidebar session state ──
    _backend = st.session_state.get("_llm_backend", "ollama")

    if _backend == "local":
        _local_path = st.session_state.get("_local_model_path", "")
        if _local_path:
            os.environ["LOCAL_LLM_MODEL"] = _local_path
        os.environ["OPENAI_API_KEY"] = ""  # clear so it doesn't override
        os.environ["GLM_API_KEY"] = ""
    elif _backend == "openai":
        _api_key = st.session_state.get("_openai_api_key", "")
        _base = st.session_state.get("_openai_base_url", DEFAULT_OPENAI_BASE_URL)
        _model = st.session_state.get("_openai_model", DEFAULT_OPENAI_MODEL)
        if _api_key:
            os.environ["OPENAI_API_KEY"] = _api_key
        os.environ["OPENAI_BASE_URL"] = _base
        os.environ["OPENAI_MODEL"] = _model
        os.environ["LOCAL_LLM_MODEL"] = ""
        os.environ["GLM_API_KEY"] = ""
    else:  # ollama (default)
        _url = st.session_state.get("_ollama_url", DEFAULT_OLLAMA_BASE_URL)
        _model = st.session_state.get("_ollama_model", DEFAULT_OLLAMA_MODEL)
        os.environ["OLLAMA_BASE_URL"] = _url
        os.environ["OLLAMA_MODEL"] = _model
        os.environ["LOCAL_LLM_MODEL"] = ""
        os.environ["OPENAI_API_KEY"] = ""
        os.environ["GLM_API_KEY"] = ""

    # ── Record active backend for display ──
    _record_active_backend()

    env_name = _scene_env_name(task_index)
    _semantic, _grid_file = _choose_map_files(task_index)
    if not _semantic.exists():
        return None

    try:
        scene, grid = load_map_files(_semantic, _grid_file)
        scene_ctx = SceneContext.from_semantic_map(scene)

        backend = RobosuiteBackend(
            env_name=env_name,
            camera="birdview",
            drive_mode="direct",
        )
        backend._scene_context = scene_ctx  # inject scene for place target resolution

        # Wire physics grasp before reset so env is created with camera obs.
        # Checkpoint path is resolved from knowledge/robot_params.json at
        # policy-load time — no hardcoded path here.
        _siemens_objects = {
            "line_5": "line_5_container_h01_near",
            "input_5": "line_5_container_h01_near",
        }
        backend.set_physics_grasp_config(
            device="cpu",
            object_map=_siemens_objects,
        )

        backend.reset()

        scene_metadata = {
            "task_index": task_index,
            "env_name": env_name,
            "scene_name": getattr(scene_ctx, "scene_name", ""),
            "map_name": getattr(scene_ctx, "map_name", ""),
            "map_prefix": _scene_prefix(task_index),
            "semantic_map": str(_semantic),
            "occupancy_grid": str(_grid_file),
            "input_object_map": {"input_5": _task_object_name(task_index)} if task_index == 0 else {},
        }
        backend._scene_metadata = scene_metadata
        _kb = st.session_state.get("_kb_enabled", True)
        return RobotAgent(
            backend=backend,
            scene_context=scene_ctx,
            grid=grid,
            scene_metadata=scene_metadata,
            knowledge_enabled=_kb,
        )
    except Exception as exc:
        st.error(f"Failed to create agent: {exc}")
        return None


# =====================================================================
#  Sidebar
# =====================================================================

def render_sidebar() -> None:
    st.sidebar.title("Unit test tool")

    # ── LLM Backend selector + config ──
    st.sidebar.subheader("LLM Backend")

    _backends = ["Ollama", "OpenAI API", "Local GGUF"]
    _backend_keys = ["ollama", "openai", "local"]
    _current_backend = st.session_state.get("_llm_backend", "ollama")
    _current_idx = _backend_keys.index(_current_backend) if _current_backend in _backend_keys else 0

    # ── Status indicator ──
    _conn_status = st.session_state.get("_llm_conn_status", "unknown")  # "ok" | "fail" | "unknown"
    _conn_msg = st.session_state.get("_llm_conn_msg", "")
    if _conn_status == "ok":
        st.sidebar.success(f"Connected: {_conn_msg}" if _conn_msg else "Connected")
    elif _conn_status == "fail":
        st.sidebar.error(f"Disconnected: {_conn_msg}" if _conn_msg else "Disconnected")
    else:
        st.sidebar.caption("Status: not tested")

    _selected = st.sidebar.selectbox(
        "Backend", _backends, index=_current_idx,
        key="_llm_backend_select",
        label_visibility="collapsed",
    )
    _selected_key = _backend_keys[_backends.index(_selected)]
    if _selected_key != st.session_state.get("_llm_backend", "ollama"):
        st.session_state["_llm_backend"] = _selected_key
        st.session_state["_llm_conn_status"] = "unknown"  # reset status on backend switch
        st.session_state["_llm_conn_msg"] = ""
        st.rerun()

    # ── Dynamic config per backend ──
    if _selected_key == "ollama":
        st.sidebar.text_input(
            "Ollama URL", key="_ollama_url",
            value=st.session_state.get("_ollama_url", DEFAULT_OLLAMA_BASE_URL),
            placeholder="http://localhost:11434",
        )
        st.sidebar.text_input(
            "Ollama Model", key="_ollama_model",
            value=st.session_state.get("_ollama_model", DEFAULT_OLLAMA_MODEL),
            placeholder="qwen3.6:27b-mtp-q4_K_M",
        )

    elif _selected_key == "openai":
        st.sidebar.text_input(
            "API Key", key="_openai_api_key", type="password",
            value=st.session_state.get("_openai_api_key", os.getenv("OPENAI_API_KEY", "")),
            placeholder="sk-...",
        )
        st.sidebar.text_input(
            "Base URL", key="_openai_base_url",
            value=st.session_state.get("_openai_base_url", DEFAULT_OPENAI_BASE_URL),
            placeholder="https://api.deepseek.com",
        )
        st.sidebar.text_input(
            "Model", key="_openai_model",
            value=st.session_state.get("_openai_model", DEFAULT_OPENAI_MODEL),
            placeholder="deepseek-v4-flash",
        )

    elif _selected_key == "local":
        st.sidebar.text_input(
            "GGUF Model Path", key="_local_model_path",
            value=st.session_state.get("_local_model_path", os.getenv("LOCAL_LLM_MODEL", "")),
            placeholder="/path/to/model.gguf",
        )

    # ── Test connection button ──
    _test_label = f"Test {_selected} Connection"
    if st.sidebar.button(_test_label, use_container_width=True):
        with st.sidebar:
            with st.spinner(f"Testing {_selected}..."):
                diag = diagnose_llm()
            if diag["ok"]:
                st.session_state["_llm_conn_status"] = "ok"
                st.session_state["_llm_conn_msg"] = diag.get("message", "Connected")
                st.rerun()
            else:
                st.session_state["_llm_conn_status"] = "fail"
                st.session_state["_llm_conn_msg"] = diag.get("message", "Connection failed")
                st.rerun()

    # Knowledge toggle
    kb_enabled = st.sidebar.checkbox("Inject Knowledge Base", value=True,
                                     help="When disabled, LLM planning will not load knowledge/ files")
    st.session_state["_kb_enabled"] = kb_enabled

    # Skill test panel
    st.sidebar.subheader("Skill Test")
    _docx_files = sorted(_APP_DIR.glob("sop+prompt/JCIIOT 2026 case *.docx"))
    _docx_names = [f.name for f in _docx_files]
    if _docx_names:
        _selected_docx = st.sidebar.selectbox("SOP Document", _docx_names, key="skill_test_docx")
        _docx_path = _APP_DIR / "sop+prompt" / _selected_docx
        if _docx_path.exists():
            _use_vlm = st.sidebar.checkbox("Enable VLM Visual Analysis", value=True, key="skill_test_vlm")
            if st.sidebar.button("Test ReadDocumentSkill", use_container_width=True):
                with st.sidebar, st.spinner("Reading document + VLM analysis..."):
                    from robot_agent.skills.read_document import ReadDocumentSkill
                    from robot_agent.core.types import ExecutionContext
                    _vlm_cfg = _detect_vision_config_for_sidebar()
                    skill = ReadDocumentSkill(
                        ollama_base_url=_vlm_cfg["base_url"],
                        vision_model=_vlm_cfg["model"],
                        api_type=_vlm_cfg["api_type"],
                        api_key=_vlm_cfg["api_key"],
                    )
                    ctx = ExecutionContext(task="test", metadata={
                        "inputs": {"file": str(_docx_path), "use_vision": _use_vlm}
                    })
                    result = skill.run(ctx)
                    st.session_state["_skill_test_result"] = result
                    st.session_state["_skill_test_docx"] = _selected_docx
                    st.rerun()

    _test_result = st.session_state.pop("_skill_test_result", None)
    _test_docx = st.session_state.pop("_skill_test_docx", "")
    if _test_result:
        with st.sidebar.expander(f"Result — {_test_docx}", expanded=True):
            st.caption(f"Success: {_test_result.success}")
            st.caption(f"Paragraphs: {_test_result.payload.get('paragraph_count', 0)}")
            st.caption(f"Images: {_test_result.payload.get('image_count', 0)}")
            st.caption(f"VLM analyzed: {_test_result.payload.get('images_analyzed', 0)}")
            text = _test_result.payload.get("text", "")
            if text:
                st.text_area("Text", text[:500], height=120)
            for name, desc in _test_result.payload.get("image_descriptions", {}).items():
                with st.expander(f"Image: {name}"):
                    st.caption(desc[:800])

    # Map status
    map_info = _check_map_files()
    st.sidebar.subheader("Map Status")
    if map_info["all_ok"]:
        st.sidebar.success("Loaded")
    else:
        st.sidebar.warning("Not Found")
        with st.sidebar.expander("Details"):
            for k, v in map_info.items():
                st.caption(f"{k}: {v}")
        st.sidebar.code(
            "python robosuite/robosuite/environments/factory_sorting/get_map.py"
        )

    st.sidebar.divider()
    st.sidebar.subheader("Vision Model (VLM)")
    _render_vlm_section()


    # 鈹€鈹€ Grasp test 鈹€鈹€
    st.sidebar.divider()
    st.sidebar.subheader("Grasp Test")
    _render_grasp_test()

    # Quick actions
    st.sidebar.subheader("Quick Actions")
    render_quick_actions()


def _render_grasp_test() -> None:
    """Sidebar: run grasp evaluation and show terminal output."""
    # ── scene selection ──
    _SCENE_OPTIONS = [
        ("L1 — FactorySorting1_3FO3ERFHISEM", "FactorySorting1_3FO3ERFHISEM"),
        ("L2 — FactorySorting3_3FO3ERRPH7X9",   "FactorySorting3_3FO3ERRPH7X9"),
        ("L3 — FactorySorting5_3FO3ERTPXEUT",   "FactorySorting5_3FO3ERTPXEUT"),
        ("L4 — FactorySorting7_3FO3ERFKY9RN",   "FactorySorting7_3FO3ERFKY9RN"),
        ("L5 — FactorySorting9_3FO3ERT2C5FP",   "FactorySorting9_3FO3ERT2C5FP"),
    ]
    _scene_sel = st.sidebar.selectbox(
        "Scene", [_s[0] for _s in _SCENE_OPTIONS], key="grasp_scene_sel",
    )
    _scene_env = {_s[0]: _s[1] for _s in _SCENE_OPTIONS}[_scene_sel]

    obj_name = st.sidebar.text_input("Object Name", value="line_5_container_h01_near", key="grasp_obj")
    base_x = st.sidebar.text_input("robot base x,y", value="7.95,3.93", key="grasp_base_xy")
    base_yaw = st.sidebar.number_input("base yaw", value=3.139, key="grasp_base_yaw")
    checkpoint = _APP_DIR / "robosuite" / "robosuite" / "model_epoch_150.pth"

    if not checkpoint.exists():
        st.sidebar.warning(f"Checkpoint missing: {checkpoint}")
        return

    if st.sidebar.button("Test Grasp", use_container_width=True):
        msg = st.sidebar.empty()
        with msg:
            with st.spinner("Grasp evaluation..."):
                import subprocess, sys, os
                try:
                    bx, by = [float(x.strip()) for x in base_x.split(",")]
                except Exception:
                    st.error("base x,y format error")
                    return

                script = str(
                    _APP_DIR / "robosuite" / "robosuite" / "environments"
                    / "factory_sorting" / "load_factory_sorting_evalization.py"
                )
                cmd = [
                    sys.executable, script,
                    "--checkpoint", str(checkpoint),
                    "--factory-scene", _scene_env,
                    "--object-name", obj_name,
                    "--robot-base-pos", str(bx), str(by), "0.0",
                    "--robot-base-ori", "0.0", "0.0", str(base_yaw),
                    "--renderer", "mjviewer",
                    "--device", "cpu",
                    "--debug-policy",
                    "--debug-every", "25",
                ]
                exit_code = 0
                try:
                    # Inherit parent env and inject v4 paths so robomimic is found
                    sub_env = os.environ.copy()
                    extra_paths = [
                        str(_APP_DIR),
                        str(_APP_DIR / "robomimic"),
                        str(_APP_DIR / "robosuite" / "robosuite"),
                        str(_APP_DIR / "src"),
                    ]
                    existing = sub_env.get("PYTHONPATH", "")
                    sub_env["PYTHONPATH"] = os.pathsep.join(extra_paths + ([existing] if existing else []))
                    result = subprocess.run(cmd, capture_output=True, text=True,
                                           timeout=180, cwd=str(_APP_DIR), env=sub_env)
                    exit_code = result.returncode
                    output = result.stdout + "\n" + result.stderr
                    if exit_code != 0:
                        output = f"[Exit code {exit_code}]\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
                except subprocess.TimeoutExpired:
                    exit_code = -1
                    output = "(Timeout 180s)"
                except Exception as exc:
                    exit_code = -1
                    output = f"Execution error: {exc}"

            # Show exit code and key grep lines first, then full output
            st.caption(f"Script: `{script}`")
            if exit_code != 0:
                st.error(f"Exit code: {exit_code}")
            with st.expander("Terminal Output", expanded=True):
                st.code(output[-12000:] if len(output) > 12000 else output, language="text")

    # Store last grasp test output for the main panel
    if "_grasp_test_output" not in st.session_state:
        st.session_state["_grasp_test_output"] = ""


def _render_vlm_section() -> None:
    """Sidebar: VLM configuration, status, and test button."""

    # ── Auto-detect default URL from active LLM backend ──
    _backend = st.session_state.get("_llm_backend", "ollama")
    if _backend == "openai":
        _default_url = st.session_state.get(
            "_openai_base_url",
            os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
        _default_key = st.session_state.get(
            "_openai_api_key",
            os.getenv("OPENAI_API_KEY", ""),
        )
        _default_model = "gpt-4o"
    else:
        _default_url = _get_ollama_url()
        _default_key = ""
        _default_model = DEFAULT_VISION_MODEL

    # ── VLM API configuration (independent of text LLM backend) ──
    st.sidebar.caption("VLM API Configuration")

    _vlm_url = st.sidebar.text_input(
        "VLM Base URL",
        value=st.session_state.get("_vlm_api_url", _default_url),
        key="_vlm_api_url_input",
        placeholder=_default_url,
        label_visibility="visible",
    )
    if _vlm_url:
        st.session_state["_vlm_api_url"] = _vlm_url

    _vlm_key = st.sidebar.text_input(
        "VLM API Key",
        value=st.session_state.get("_vlm_api_key", _default_key),
        key="_vlm_api_key_input",
        type="password",
        placeholder="sk-... (leave empty for Ollama)",
        label_visibility="visible",
    )
    if _vlm_key:
        st.session_state["_vlm_api_key"] = _vlm_key

    _vlm_model = st.sidebar.text_input(
        "VLM Model",
        value=st.session_state.get("_vlm_model", _default_model),
        key="_vlm_model_input",
        placeholder=_default_model,
        label_visibility="visible",
    )
    if _vlm_model:
        st.session_state["_vlm_model"] = _vlm_model

    # ── Detect API type ──
    from robot_agent.core.vision_client import _detect_api_type
    _vlm_api_type = _detect_api_type(_vlm_url)
    if _vlm_key:
        _vlm_api_type = "openai"

    # ── Status indicator ──
    _vlm_status = st.session_state.get("_vlm_conn_status", "unknown")
    _vlm_msg = st.session_state.get("_vlm_conn_msg", "")
    if _vlm_status == "ok":
        st.sidebar.success(f"VLM: {_vlm_msg}" if _vlm_msg else "VLM: Connected")
    elif _vlm_status == "fail":
        st.sidebar.error(f"VLM: {_vlm_msg}" if _vlm_msg else "VLM: Disconnected")
    else:
        st.sidebar.caption("VLM Status: not tested")

    # ── Show last test result ──
    _last_img = st.session_state.get("_vlm_test_image")
    _last_resp = st.session_state.get("_vlm_test_response")
    if _last_img is not None and _last_resp is not None:
        st.sidebar.image(_last_img, caption="Factory Scene — Birdview",
                         use_container_width=True)
        st.sidebar.caption(_last_resp[:500])

    # ── Test button ──
    if st.sidebar.button("Test VLM with Factory Scene", use_container_width=True,
                          help="Load factory birdview → VLM describes the scene"):
        with st.sidebar:
            with st.spinner(f"Loading factory scene + testing VLM ({_vlm_api_type})..."):
                try:
                    from robot_agent.environments import RobosuiteBackend
                    from robot_agent.core.vision_client import ask_vision
                    from PIL import Image

                    backend = RobosuiteBackend(
                        env_name="FactorySorting1_3FO3ERFHISEM",
                        camera="birdview",
                        drive_mode="direct",
                    )
                    backend.reset()
                    frame = backend.capture_frame()
                    backend.close()

                    img = Image.fromarray(frame)
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=80)

                    response = ask_vision(
                        "Describe this factory scene from a top-down birdview. "
                        "What do you see? Tables, conveyors, stations, robot, "
                        "production lines? Keep it under 100 words.",
                        buf.getvalue(),
                        base_url=_vlm_url,
                        model=_vlm_model,
                        api_type=_vlm_api_type,
                        api_key=_vlm_key,
                        timeout=120.0,
                    )
                    st.session_state["_vlm_conn_status"] = "ok"
                    st.session_state["_vlm_conn_msg"] = response[:80]
                    st.session_state["_vlm_test_image"] = frame
                    st.session_state["_vlm_test_response"] = response
                    st.rerun()
                except Exception as exc:
                    st.session_state["_vlm_conn_status"] = "fail"
                    st.session_state["_vlm_conn_msg"] = str(exc)[:100]
                    st.session_state["_vlm_test_image"] = None
                    st.session_state["_vlm_test_response"] = None
                    st.rerun()


def _render_vision_test() -> None:
    """Legacy sidebar: test vision model (kept for backward compatibility)."""
    _render_vlm_section()


def render_quick_actions() -> None:
    """Sidebar quick-action buttons for memory / knowledge management."""
    col1, col2 = st.sidebar.columns(2)

    with col1:
        if st.button("Clear Memory", use_container_width=True,
                     help="Clear agent runtime memory"):
            store = st.session_state.get("_mem_store")
            if store is not None:
                store.clear()
                st.session_state["_mem_op"] = ("clear", 0)
            st.rerun()

    with col2:
        if st.button("Refresh Knowledge Base", use_container_width=True,
                     help="Rescan knowledge/ folder"):
            from robot_agent.core.knowledge_manager import KnowledgeManager
            mgr = KnowledgeManager(_KNOWLEDGE_ROOT)
            added = mgr.reload()
            st.session_state["_kb_op"] = ("reload", added)
            st.rerun()

    # Show feedback from quick actions
    if "_mem_op" in st.session_state:
        action, count = st.session_state.pop("_mem_op")
        if action == "clear":
            st.sidebar.success(f"Memory cleared ({count} -> 0)")
    if "_kb_op" in st.session_state:
        action, added = st.session_state.pop("_kb_op")
        st.sidebar.success(f"Knowledge base refreshed: +{added} docs")

    # 鈹€鈹€ Trajectory replay 鈹€鈹€
    st.sidebar.divider()
    st.sidebar.subheader("Trajectory Replay")
    _render_replay_section()


def _render_replay_section() -> None:
    """Sidebar section: pick a trajectory JSON, replay in sim, download GIF."""
    from robot_agent.environments import RobosuiteBackend

    _ensure_scene_recording_dirs()
    rec_root = _recordings_root()
    json_files = _trajectory_json_files()
    if not json_files:
        st.sidebar.caption("No trajectory files yet")
        return

    options = [str(f.relative_to(rec_root)) for f in json_files]
    selected = st.sidebar.selectbox("Select Trajectory", options, label_visibility="collapsed")

    replay_scope = st.sidebar.selectbox(
        "Replay Scope", ["Full Trajectory", "Grasp Segment"], index=0, label_visibility="collapsed",
    )
    camera_options = ["birdview", "robot0_robotview", "all"]
    camera_default = 1 if replay_scope == "Grasp Segment" else 0
    camera = st.sidebar.selectbox(
        "Camera", camera_options,
        index=camera_default, label_visibility="collapsed",
        key=f"replay_camera_{replay_scope}",
    )

    button_label = "Grasp Segment → GIF" if replay_scope == "Grasp Segment" else "Replay → GIF"
    if st.sidebar.button(button_label, use_container_width=True):
        json_path = rec_root / selected
        env_name = _trajectory_env_name(json_path)
        cameras_to_render = camera_options[:-1] if camera == "all" else [camera]
        frame_start = None
        frame_end = None
        grasp_obj = None
        if replay_scope == "Grasp Segment":
            inferred = _infer_grasp_replay_range(json_path)
            if inferred is None:
                st.sidebar.error("Cannot infer grasp segment from this JSON")
                return
            frame_start, frame_end, grasp_obj = inferred

        with st.sidebar:
            generated: list[str] = []
            for cam in cameras_to_render:
                cam_suffix = "" if cam == "birdview" else f"_{cam}"
                if replay_scope == "Grasp Segment":
                    gif_path = json_path.parent / json_path.name.replace("trajectory_", "grasp_replay_").replace(".json", f"{cam_suffix}.gif")
                else:
                    gif_path = json_path.parent / json_path.name.replace("trajectory_", "replay_").replace(".json", f"{cam_suffix}.gif")
                with st.spinner(f"Replaying {cam}..."):
                    try:
                        backend = RobosuiteBackend(
                            env_name=env_name,
                            camera=cam,
                            drive_mode="direct",
                            headless=True,
                        )
                        backend.reset()
                        frames = backend.replay_trajectory(
                            json_path,
                            gif_path,
                            camera=cam,
                            frame_start=frame_start,
                            frame_end=frame_end,
                        )
                        backend.close()
                        generated.append(str(gif_path))
                    except Exception as exc:
                        st.error(f"Replay failed ({cam}): {exc}")
            if generated:
                if replay_scope == "Grasp Segment":
                    st.success(
                        f"Generated {len(generated)} GIF(s): "
                        f"{', '.join(generated)} "
                        f"({frame_start}-{frame_end - 1}, {grasp_obj})"
                    )
                else:
                    st.success(f"Generated {len(generated)} GIF(s): {', '.join(generated)}")


# =====================================================================
#  Physics-based grasp pipeline (factory_sorting)
# =====================================================================

def _verify_placement(source: str, target: str, task_index: int = 0) -> dict:
    """Check if any material object is sitting on the *target* output table.

    Success = object x,y is within the table's horizontal projection AND
    z is above the table surface (0.35-0.80m indicates placed, not held).
    """
    try:
        from robot_agent.environments import RobosuiteBackend
        from robot_agent.core.map_loader import load_map_files
        from robot_agent.core.scene_context import SceneContext
        import numpy as np

        scene, _grid = load_map_files(*_choose_map_files(task_index))
        scene_ctx = SceneContext.from_semantic_map(scene)

        backend = RobosuiteBackend(env_name=_scene_env_name(task_index), camera="birdview", drive_mode="direct")
        backend.reset()
        env = backend.env

        target_info = scene_ctx.output_ports.get(target)
        if target_info is None:
            backend.close()
            return {"ok": False, "reason": f"Target station {target} does not exist"}

        table_cx, table_cy = float(target_info.center[0]), float(target_info.center[1])
        # Table half-size: 0.8m x 0.8m; check within 0.35m radius.
        table_radius = 0.35

        # Use env support surface z if available, else fallback
        table_z = 0.40
        try:
            if hasattr(env, "_siemens_static_table_support_surfaces"):
                surfaces = list(env._siemens_static_table_support_surfaces())
                # Match output index: output_4 -> index 3 (0-based).
                import re
                _m = re.search(r"(\d+)$", target)
                _idx = int(_m.group(1)) - 1 if _m else 0
                if 0 <= _idx < len(surfaces):
                    table_z = float(surfaces[_idx][0][2])  # support z center
        except Exception:
            pass
        if hasattr(env, "table_top_z") and float(env.table_top_z) > table_z:
            table_z = float(env.table_top_z)

        found = []
        all_positions = {}
        for obj_name in env.material_objects:
            body_id = env.obj_body_id.get(obj_name)
            if body_id is None:
                continue
            pos = env.sim.data.body_xpos[body_id]
            px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
            all_positions[obj_name] = [round(px, 4), round(py, 4), round(pz, 4)]

            # Object on table if within horizontal radius AND z near table surface
            xy_dist = float(np.linalg.norm(np.array([px, py]) - np.array([table_cx, table_cy])))
            on_table = xy_dist < table_radius and pz > table_z - 0.02
            if on_table:
                found.append({"name": obj_name, "xy_dist": round(xy_dist, 3), "z": round(pz, 3)})

        backend.close()

        if found:
            return {
                "ok": True,
                "objects_on_table": found,
                "all_objects": all_positions,
                "table_center": [table_cx, table_cy],
                "table_z": table_z,
            }
        else:
            return {
                "ok": False,
                "reason": f"No object is above the {target} table surface",
                "all_objects": all_positions,
                "table_center": [table_cx, table_cy],
                "table_z": table_z,
            }
    except Exception as exc:
        return {"ok": False, "reason": f"Verification failed: {exc}"}



def _build_physics_args(source: str, target: str) -> argparse.Namespace:
    """Build an argparse.Namespace that llm_task_navigator.execute_navigation expects."""
    from robosuite.environments.factory_sorting import llm_task_navigator as nav

    base_dir = _MAP_DIR.parent
    checkpoint = base_dir.parent.parent / "model_epoch_150.pth"

    ns = argparse.Namespace()
    ns.command = f"Move object from {source} to {target}"
    ns.semantic_map, ns.occupancy_grid = _choose_map_files()
    ns.path_spacing = 0.35
    ns.waypoint_tolerance = 0.18

    # grasp: use Siemens object mapping
    ns.grasp_checkpoint = checkpoint if checkpoint.exists() else None
    ns.grasp_steps = 360
    ns.grasp_device = "cpu"
    ns.grasp_object_name = nav.FACTORY_SORTING_CURRENT_INPUT_OBJECTS.get(source)
    ns.grasp_policy_env = True
    ns.grasp_visual_window = True
    ns.grasp_visual_camera = "robot0_robotview"
    ns.grasp_visual_height = 600
    ns.grasp_visual_width = 600
    ns.grasp_visual_window_name = "grasp view"
    ns.grasp_post_hold_steps = 10
    ns.grasp_initial_view_steps = 30
    ns.reset_robot_to_grasp_init = False  # stay at navigated position, don't teleport to training pos
    ns.grasp_init_state = None
    ns.grasp_robot_base_pos = [0.756088, -3.787826, 0.0]
    ns.grasp_robot_base_ori = [0.0, 0.0, 3.139422]
    ns.grasp_init_settle_steps = 5
    ns.render_grasp = False
    ns.keep_grasp_viewer_open = False
    ns.debug_grasp = False
    ns.debug_grasp_every = 25
    ns.verbose_grasp = False

    # lift
    ns.lift_after_grasp = True
    ns.lift_height = 0.03
    ns.lift_max_steps = 200
    ns.lift_hold_steps = 20
    ns.lift_tolerance = 0.02
    ns.lift_max_action = 0.80
    ns.debug_lift = False

    # transport
    ns.transport_attach = True

    # turn
    ns.turn_to_output = True
    ns.turn_tolerance = 0.02
    ns.turn_max_iters = 8
    ns.turn_steps = 40
    ns.turn_settle_steps = 10
    ns.turn_render_sleep = 0.0
    ns.debug_turn = False

    # place
    ns.place_on_table = True
    ns.place_table_z = None
    ns.place_clearance = 0.01
    ns.place_xy_steps = 25
    ns.place_lower_steps = 50
    ns.place_release_steps = 20
    ns.place_hold_steps = 20
    ns.place_render_sleep = 0.0
    ns.debug_place = False

    # env
    ns.camera = "birdview"
    ns.controller = None
    ns.use_camera_obs = False
    ns.policy_camera = "robot0_robotview"
    ns.policy_camera_height = 128
    ns.policy_camera_width = 128
    ns.gripper_types = "Robotiq140Gripper"
    ns.headless = False
    ns.seed = None
    ns.drive_mode = "direct"
    ns.control_freq = 20
    ns.max_nav_steps = 3000
    ns.k_linear = 1.2
    ns.k_angular = 1.8
    ns.max_linear = 0.70
    ns.max_angular = 1.20
    ns.turn_in_place_angle = 0.65
    ns.holonomic_base = True
    ns.yaw_control = False
    ns.debug_nav = False
    ns.debug_every = 50
    ns.stop_on_collision = True
    ns.collision_warmup_steps = 5
    ns.ignore_collision_geom = ["container", "tote", "cardbox", "plastic_crate"]
    ns.max_collision_pairs = 8
    ns.lock_upper_body = True
    ns.pause_between_stages = 0.1

    # tracking
    ns.turn_target_xy = None
    ns.place_target_xy = None
    ns.failure_reason = None

    return ns


def _execute_physics_pipeline(task: str, task_index: int = 0) -> None:
    """Run the full physics-based grasp pipeline (no teleport)."""
    import datetime
    import numpy as np
    from pathlib import Path

    # ── Dependency check ──
    _missing: list[str] = []
    for _mod in ("torch", "robomimic", "h5py", "tqdm", "imageio", "scipy",
                 "huggingface_hub", "packaging", "psutil"):
        try:
            __import__(_mod)
        except ImportError:
            _missing.append(_mod)
    if _missing:
        st.error(f"Physics pipeline missing dependencies: **{', '.join(_missing)}**\n\n"
                 "Install dependencies in venv:\n"
                 "```bash\n"
                 f"pip install {' '.join(_missing)}\n"
                 "```")
        st.session_state["last_error"] = f"Missing dependencies: {', '.join(_missing)}"
        return

    from robot_agent.core.map_loader import load_map_files
    from robosuite.environments.factory_sorting import llm_task_navigator as nav

    # 1) LLM planning via Ollama (same as Agent)
    os.environ["OLLAMA_BASE_URL"] = _get_ollama_url()
    os.environ["OLLAMA_MODEL"] = st.session_state.get("_ollama_model", DEFAULT_OLLAMA_MODEL)

    scene, grid = load_map_files(*_choose_map_files(task_index))

    try:
        plan = nav.call_ollama_plan(
            command=task,
            scene=scene,
            base_url=_get_ollama_url(),
            model=st.session_state.get("_ollama_model", DEFAULT_OLLAMA_MODEL),
            timeout=120.0,
        )
    except Exception as exc:
        st.error(f"LLM planning failed: {exc}")
        st.session_state["last_error"] = f"LLM planning failed: {exc}"
        return

    try:
        source, target = nav.validate_plan(plan, scene)
    except Exception as exc:
        st.error(f"Plan validation failed: {exc}")
        st.session_state["last_error"] = f"Plan validation failed: {exc}"
        return

    st.info(f"**Plan result**: {source} -> {target}")
    st.caption(f"Plan steps: {json.dumps(plan.get('steps', []), ensure_ascii=False)[:500]}")

    # 2) Build physics args
    args = _build_physics_args(source, target)

    # Store plan
    args.save_plan = _MAP_DIR / "latest_llm_plan.json"
    args.save_path = _MAP_DIR / "latest_navigation_paths.json"
    args.save_plan.parent.mkdir(parents=True, exist_ok=True)
    args.save_plan.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

    # 3) Plan world paths
    robot_start = np.array(scene["robot"]["start"], dtype=float)
    source_goal = np.array(scene["input_ports"][source]["approach"], dtype=float)
    target_goal = np.array(scene["output_ports"][target]["approach"], dtype=float)
    target_center = np.array(scene["output_ports"][target].get("center") or target_goal, dtype=float)
    args.turn_target_xy = target_center[:2].tolist()
    args.place_target_xy = target_center[:2].tolist()

    path_to_source = nav.plan_world_path(scene, grid, robot_start, source_goal, args.path_spacing)
    path_to_target = nav.plan_world_path(scene, grid, source_goal, target_goal, args.path_spacing)
    paths = [("navigate_to_source", path_to_source), ("navigate_to_target", path_to_target)]

    st.caption(f"Path: source={len(path_to_source)}wp, target={len(path_to_target)}wp")

    # 4) Execute with stdout capture for diagnostics
    import contextlib

    captured = io.StringIO()
    with st.spinner("Simulation running (Navigate -> Grasp -> Lift -> Transport -> Turn -> Place)..."):
        try:
            with contextlib.redirect_stdout(captured):
                ok = nav.execute_navigation(paths, args)
        except Exception as exc:
            st.error(f"Physics pipeline crashed: {exc}")
            import traceback
            st.code(traceback.format_exc())
            st.session_state["last_error"] = str(exc)
            return

    fail_reason = nav.failure_reason(args) if hasattr(nav, 'failure_reason') else ""
    captured_text = captured.getvalue()

    # Verify placement: is the object actually near the target?
    placement_ok = False
    placement_info: dict = {}
    try:
        placement_info = _verify_placement(source, target, task_index)
        placement_ok = placement_info.get("ok", False)
        if placement_ok:
            ok = True  # Override: object reached target even if pipeline reported issues.
    except Exception:
        pass

    # 鈹€鈹€ Extract key diagnostics from stdout 鈹€鈹€
    _diag_lines: list[str] = []
    for _kw in ("grasp_status", "fingerpad contact", "gripper end distance",
                "gripper end deltas", "gripper end targets", "gripper end positions",
                "object collision geoms", "lift_initial_grasp_status", "lift_result",
                "place_on_table_result", "turn_to_output_result",
                "failure_reason", "grasp_policy success", "grasp_policy failure",
                "same-env wrapped grasp return", "same-env wrapped grasp success",
                "collision_detected", "executing ", "reached",
                "source_reached", "navigate_to"):
        for _line in captured_text.splitlines():
            if _kw in _line.lower() and _line.strip() not in _diag_lines:
                _diag_lines.append(_line.strip())

    # Determine failure stage
    failed_stage = ""
    if not ok:
        if "lift_after_grasp success: False" in captured_text or "lift timeout" in captured_text.lower():
            if "final_grasp_status={'right': False" in captured_text or \
               "final_grasp_status={'left': False" in captured_text:
                failed_stage = "Lift failed - gripper slipped, object dropped during lifting"
            else:
                failed_stage = "Lift failed - insufficient lift force, target height not reached"
        elif "grasp_policy success: False" in captured_text or "grasp_policy failure" in captured_text:
            if "fingerpad contact status: {'right': False" in captured_text:
                failed_stage = "Grasp no-contact - policy inference succeeded but gripper did not touch object"
            else:
                failed_stage = "Grasp failed - policy completed but grasp not achieved"
        elif "collision_detected" in captured_text or "collision at step" in captured_text.lower():
            failed_stage = "Navigation aborted due to collision"
        elif "navigate_to_source" in fail_reason.lower():
            failed_stage = "Source station navigation failed"
        elif "turn_to" in fail_reason.lower():
            failed_stage = "Turn to target station failed"
        elif "place_on_table" in fail_reason.lower():
            failed_stage = "Object placement failed"
        else:
            failed_stage = fail_reason or "Unknown stage"

    if ok:
        if placement_ok:
            objs = placement_info.get("objects_on_table", [])
            st.success(f"Task complete! Object is on {target} table (xy offset {objs[0]['xy_dist']}m, height {objs[0]['z']}m)")
        else:
            st.success("Pipeline completed (object not confirmed at target)")
    else:
        st.error(f"**Failed stage**: {failed_stage}")
        if fail_reason:
            st.caption(f"Details: {fail_reason}")
        if placement_ok:
            st.info("But object has reached near the target")
        if not _diag_lines and not placement_ok:
            st.warning("Pipeline stuck or no diagnostic output — possible model loading timeout or inference blocking")

    # Place placement info into last_result for scoring
    st.session_state["_placement_ok"] = placement_ok
    st.session_state["_placement_info"] = placement_info

    # Show diagnostics
    if _diag_lines:
        with st.expander("Pipeline Diagnostic Output", expanded=not ok):
            for line in _diag_lines[:30]:
                st.caption(line)

    # 5) Save all output to recordings, even on failure.
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rec_dir = _task_recording_dir(task_index)

    # 5a) Diagnostic text file
    diag_path = rec_dir / f"physics_diag_{ts}.txt"
    diag_path.write_text(
        f"# Physics Pipeline Diagnostic Output\n"
        f"# timestamp: {ts}\n"
        f"# task: {task}\n"
        f"# source: {source}  target: {target}\n"
        f"# success: {ok}\n"
        f"# failed_stage: {failed_stage}\n"
        f"# failure_reason: {fail_reason}\n"
        f"# source_path: {len(path_to_source)}wp  target_path: {len(path_to_target)}wp\n"
        f"\n## LLM Plan\n{json.dumps(plan, indent=2, ensure_ascii=False)}\n"
        f"\n## Captured stdout\n{captured_text}\n",
        encoding="utf-8",
    )
    st.caption(f"Diagnostics saved: {diag_path.name}")

    # 5b) Summary JSON
    summary_path = rec_dir / f"physics_summary_{ts}.json"
    summary_path.write_text(json.dumps({
        "timestamp": ts,
        "task": task,
        "source": source,
        "target": target,
        "success": ok,
        "failed_stage": failed_stage,
        "failure_reason": fail_reason,
        "path_waypoints": {"source": len(path_to_source), "target": len(path_to_target)},
        "diagnostics": _diag_lines[:30],
        "llm_plan": plan,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # 5c) Optionally generate replay GIFs for the full source-to-target path.
    _status_tag = "OK" if ok else "FAIL"
    if AUTO_GENERATE_REPLAY_GIFS:
        try:
            from robot_agent.environments import RobosuiteBackend
            for _cam, _suffix, _session_key in [
                ("birdview", "", "last_frames"),
                ("robot0_robotview", "_robotview", "_robotview_frames"),
            ]:
                gif_path = rec_dir / f"physics_replay_{ts}_{_status_tag}{_suffix}.gif"
                backend = RobosuiteBackend(env_name=_scene_env_name(task_index), camera=_cam, drive_mode="direct")
                backend.reset()
                backend.start_recording()
                # Walk source path (approach)
                backend.follow_path(path_to_source, max_steps=3000)
                # Walk target path (full journey)
                if path_to_target:
                    backend.follow_path(path_to_target, max_steps=3000)
                frames = backend.get_recorded_frames()
                # On failure, add a long pause at end so user can see final state
                if not ok and frames:
                    _last = Image.fromarray(frames[-1])
                    for _ in range(20):  # 20 * 250ms = 5s pause
                        frames.append(np.array(_last))
                if frames:
                    _save_gif(frames, gif_path)
                    st.session_state[_session_key] = frames
                backend.close()
            st.caption("Recording auto-generated")
        except Exception as exc:
            st.caption(f"Recording generation skipped: {exc}")

    st.session_state["_plan_only"] = {
        "understanding": plan.get("understanding", task),
        "plan": [{"skill_name": s.get("action", "?"), "description": s.get("goal", "")}
                 for s in plan.get("steps", [])],
        "raw_llm_text": json.dumps(plan, indent=2, ensure_ascii=False),
    }
    st.session_state["last_error"] = None
    st.session_state["last_result"] = type("Result", (), {
        "success": ok,
        "elapsed_ms": 0,
        "message": "Physics pipeline: " + ("Success" if ok else f"Failed [{failed_stage}]: {fail_reason or 'unknown'}"),
        "steps": [],
        "thinking": None,
        "planner_raw": json.dumps(plan, indent=2, ensure_ascii=False),
        "plan_warnings": [],
    })()
    st.rerun()


# =====================================================================
#  Scoring helpers
# =====================================================================

def _event_success_value(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "ok",
        "success",
        "succeeded",
    }


def _trajectory_object_position(object_positions: dict, object_name: str) -> tuple[float, float, float] | None:
    if not isinstance(object_positions, dict) or not object_name:
        return None

    pos = object_positions.get(object_name)
    if pos is None:
        for candidate_name, candidate_pos in object_positions.items():
            candidate_name = str(candidate_name)
            if object_name in candidate_name or candidate_name in object_name:
                pos = candidate_pos
                break

    try:
        if pos is None or len(pos) < 2:
            return None
        z = float(pos[2]) if len(pos) >= 3 else 0.0
        return float(pos[0]), float(pos[1]), z
    except Exception:
        return None


def _l5_match_object(event_object: str, tracked_objects: list[str]) -> str | None:
    event_object = str(event_object or "")
    if not event_object:
        return None
    for object_name in tracked_objects:
        if object_name == event_object or object_name in event_object or event_object in object_name:
            return object_name
    return None


def _l5_left_source_after_grasp(
    frames: list,
    object_name: str,
    src_xy,
    start_frame: int,
) -> tuple[bool, float | None, float | None]:
    last_dx = None
    last_dy = None
    start_frame = max(0, min(int(start_frame), max(0, len(frames) - 1)))

    for frame in frames[start_frame:]:
        if not isinstance(frame, dict):
            continue
        pos = _trajectory_object_position(frame.get("object_positions", {}), object_name)
        if pos is None:
            continue
        last_dx = abs(pos[0] - float(src_xy[0]))
        last_dy = abs(pos[1] - float(src_xy[1]))
        if last_dx > 1.0 or last_dy > 1.0:
            return True, last_dx, last_dy
    return False, last_dx, last_dy


def _score_l5_multi_object(task_index: int, src_xy, tgt_xy, tgt_z: float) -> dict:
    """L5 scores the three white totes on input_1 independently."""
    empty = {"total": 0, "items": []}
    source_name = _task_source_name(task_index)
    target_name = _task_target_name(task_index)

    try:
        last_traj = st.session_state.get("_last_trajectory")
        if not last_traj:
            return empty
        traj_path = Path(str(last_traj))
        if not traj_path.exists():
            return empty
        traj = json.loads(traj_path.read_text(encoding="utf-8"))
        frames = traj.get("frames", [])
        if not isinstance(frames, list) or not frames:
            return empty
        events = traj.get("events", [])
        if not isinstance(events, list):
            events = []
    except Exception:
        return empty

    first_positions = frames[0].get("object_positions", {}) if isinstance(frames[0], dict) else {}
    last_positions = frames[-1].get("object_positions", {}) if isinstance(frames[-1], dict) else {}
    tracked_objects = [
        name for name in L5_INPUT1_OBJECTS
        if _trajectory_object_position(first_positions, name) is not None
        or _trajectory_object_position(last_positions, name) is not None
    ]
    if not tracked_objects:
        tracked_objects = list(L5_INPUT1_OBJECTS)

    grasp_frame_by_object: dict[str, int] = {}
    for event in events:
        if not isinstance(event, dict) or event.get("name") != "grasp_end":
            continue
        if not _event_success_value(event.get("success")):
            continue
        object_name = _l5_match_object(str(event.get("object_name") or ""), tracked_objects)
        if object_name is None:
            continue
        try:
            frame_index = int(event.get("frame", 0))
        except Exception:
            frame_index = 0
        if object_name not in grasp_frame_by_object:
            grasp_frame_by_object[object_name] = frame_index

    items = []
    debug_lines = [
        f"L5 source {source_name}: {', '.join(tracked_objects)}",
        f"L5 target {target_name}: x={float(tgt_xy[0]):.3f} y={float(tgt_xy[1]):.3f} z={float(tgt_z):.3f}",
    ]

    for object_name in tracked_objects:
        grasped = object_name in grasp_frame_by_object
        left_ok = False
        left_dx = None
        left_dy = None
        if grasped:
            left_ok, left_dx, left_dy = _l5_left_source_after_grasp(
                frames,
                object_name,
                src_xy,
                grasp_frame_by_object[object_name],
            )

        final_pos = _trajectory_object_position(last_positions, object_name)
        dist_tgt = None
        placed_ok = False
        if grasped and final_pos is not None:
            dist_tgt = float(np.linalg.norm(np.array(final_pos[:2]) - tgt_xy))
            placed_ok = dist_tgt < 0.80

        dx_text = f"{left_dx:.2f}m" if left_dx is not None else "n/a"
        dy_text = f"{left_dy:.2f}m" if left_dy is not None else "n/a"
        if final_pos is None:
            final_text = "final=n/a"
            dist_text = "dist=n/a"
        else:
            final_text = f"x={final_pos[0]:.2f}, y={final_pos[1]:.2f}, z={final_pos[2]:.2f}"
            dist_text = f"dist={dist_tgt:.2f}m" if dist_tgt is not None else "dist=n/a"

        items.append({
            "label": (
                f"L5 {object_name}: grasped and left {source_name} "
                f"(grasp={'yes' if grasped else 'no'}, dx_src={dx_text}, dy_src={dy_text})"
            ),
            "score": 5,
            "ok": left_ok,
        })
        items.append({
            "label": (
                f"L5 {object_name}: placed at {target_name} "
                f"(grasp={'yes' if grasped else 'no'}, {dist_text}, {final_text})"
            ),
            "score": 5,
            "ok": placed_ok,
        })
        debug_lines.append(
            f"{object_name}: grasp={grasped}, left={left_ok}, placed={placed_ok}, {dist_text}, {final_text}"
        )

    total = sum(item["score"] for item in items if item["ok"])

    collision = any(isinstance(frame, dict) and frame.get("has_collision") for frame in frames)
    if collision:
        total = max(0, total - 5)
        items.append({
            "label": "Collision penalty (collision detected during L5 task)",
            "score": -5,
            "ok": True,
            "is_penalty": True,
        })
        debug_lines.append("collision: True (-5)")
    else:
        debug_lines.append("collision: False")

    st.session_state["_score_debug"] = debug_lines
    return {"total": total, "items": items}


def _score_steps(task_index: int) -> dict:
    """Score based on ground-truth object positions from the simulation.

    Opens a fresh headless env, reads material object x/y/z, and compares
    against the task's source/target station coordinates.

    Per-task max: L1=10, L2=15, L3=20, L4=25, L5=30.
    """
    import numpy as np

    _MAX_SCORES = [10, 15, 20, 25, 30]
    _max = _MAX_SCORES[min(task_index, len(_MAX_SCORES) - 1)]
    # Weights: leave source (30%), arrive near target (30%), rest on table (40%)
    empty = {"total": 0, "items": []}

    # 鈹€鈹€ Source/target: read dynamically from each scene's map 鈹€鈹€
    _SRC_NAMES = [_task_source_name(i) for i in range(5)]
    _TGT_NAMES = [_task_target_name(i) for i in range(5)]
    try:
        from robot_agent.core.map_loader import load_map_files
        from robot_agent.core.scene_context import SceneContext
        _sem, _grid = _choose_map_files(task_index)
        _scene_dict, _ = load_map_files(_sem, _grid)
        _ctx = SceneContext.from_semantic_map(_scene_dict)
        _src = _ctx.input_ports.get(_SRC_NAMES[task_index])
        _tgt = _ctx.output_ports.get(_TGT_NAMES[task_index])
        if _src is None or _tgt is None:
            return empty
        src_xy = _src.center[:2].copy()
        tgt_xy = _tgt.center[:2].copy()
        # Read target table z from map or env
        _tgt_z = 1.09  # default for new Siemens scenes
        try:
            if hasattr(_tgt, "center") and len(_tgt.center) >= 3:
                _tgt_z = float(_tgt.center[2])
        except Exception:
            pass
        obj_hint = _task_object_name(task_index)
        if task_index == 4:
            return _score_l5_multi_object(task_index, src_xy, tgt_xy, _tgt_z)
    except Exception:
        return empty

    # 鈹€鈹€ Read object positions from the LAST TRAJECTORY FRAME 鈹€鈹€
    # A fresh env reset sends objects back to spawn; use trajectory JSON instead.
    grasp_success = False
    try:
        import json as _json
        _last_traj = st.session_state.get("_last_trajectory")
        if _last_traj and Path(_last_traj).exists():
            with open(_last_traj, "r") as _f:
                _traj = _json.load(_f)
            _events = _traj.get("events", [])
            if isinstance(_events, list):
                for _event in _events:
                    if not isinstance(_event, dict) or _event.get("name") != "grasp_end":
                        continue
                    _event_source = str(_event.get("source") or "")
                    _event_object = str(_event.get("object_name") or "")
                    _source_ok = not _event_source or _event_source == _SRC_NAMES[task_index]
                    _object_ok = (
                        not obj_hint
                        or not _event_object
                        or obj_hint in _event_object
                        or _event_object in obj_hint
                    )
                    _success_value = _event.get("success")
                    _success_ok = _event_success_value(_success_value)
                    if _source_ok and _object_ok and _success_ok:
                        grasp_success = True
                        break
            _frames = _traj.get("frames", [])
            if _frames:
                # Last frame has the final object positions
                _last_frame = _frames[-1]
                _obj_positions = _last_frame.get("object_positions", {})
                # Find the right object
                px = py = pz = None
                for obj_name, pos in _obj_positions.items():
                    if obj_hint and obj_hint in obj_name:
                        px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
                        best_obj = obj_name
                        break
                if px is None and _obj_positions:
                    # Pick the one nearest to target
                    best_dist = float("inf")
                    for obj_name, pos in _obj_positions.items():
                        d = float(np.linalg.norm(np.array(pos[:2]) - tgt_xy))
                        if d < best_dist:
                            best_dist, best_obj = d, obj_name
                            px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
                if px is None:
                    return empty
            else:
                return empty
        else:
            return empty
    except Exception:
        return empty

    # 鈹€鈹€ Compute distances first 鈹€鈹€
    dx_src = abs(px - src_xy[0])
    dy_src = abs(py - src_xy[1])
    dist_tgt = float(np.linalg.norm(np.array([px, py]) - tgt_xy))  # XY only, z checked separately

    # 鈹€鈹€ Debug: dump coordinates 鈹€鈹€
    try:
        debug_lines = [
            f"Object x={px:.3f} y={py:.3f} z={pz:.3f}",
            f"Target x={tgt_xy[0]:.3f} y={tgt_xy[1]:.3f} z={_tgt_z:.3f}",
            f"dist_xy: {dist_tgt:.3f}m",
            f"grasp_success: {grasp_success}",
        ]
        st.session_state["_score_debug"] = debug_lines
    except Exception:
        st.session_state["_score_debug"] = []

    # 鈹€鈹€ Score: 2 checkpoints 鈹€鈹€
    _half = max(1, _max // 2)
    _w_leave = _half
    _w_place = _max - _w_leave

    left_source_position = dx_src > 1.0 or dy_src > 1.0
    left_source = grasp_success and left_source_position
    on_target_table = grasp_success and dist_tgt < 0.80

    items = [
        {"label": f"Grasp success & left source (grasp={'yes' if grasp_success else 'no'}, dx_src={dx_src:.2f}m, dy_src={dy_src:.2f}m)",
         "score": _w_leave, "ok": left_source},
        {"label": f"Object reached target table after grasp (grasp={'yes' if grasp_success else 'no'}, dist={dist_tgt:.2f}m, x={px:.2f}, y={py:.2f}, z={pz:.2f})",
         "score": _w_place, "ok": on_target_table},
    ]
    total = sum(it["score"] for it in items if it["ok"])

    # 鈹€鈹€ Collision penalty: -5 if collision detected in trajectory 鈹€鈹€
    _collision = False
    try:
        import json as _json2
        _last_traj2 = st.session_state.get("_last_trajectory")
        if _last_traj2 and Path(_last_traj2).exists():
            with open(_last_traj2, "r") as _f2:
                _traj2 = _json2.load(_f2)
            for _frm in _traj2.get("frames", []):
                if _frm.get("has_collision"):
                    _collision = True
                    break
    except Exception:
        pass

    if _collision:
        total = max(0, total - 5)
        items.append({"label": "Collision penalty (collision detected during task)", "score": -5, "ok": True, "is_penalty": True})

    return {"total": total, "items": items}


def _restore_scores_from_disk(task_count: int) -> None:
    """Restore the latest saved score files after a Streamlit/native restart."""
    if st.session_state.get("_score_restore_done"):
        return
    st.session_state["_score_restore_done"] = True
    for task_index in range(task_count):
        if task_index in st.session_state.get("_task_scores", {}):
            continue
        data = _latest_score_for_task(task_index)
        trajectory_path = _latest_trajectory_for_task(task_index)
        score_is_current = False
        if data:
            try:
                score_path = Path(data.get("_score_path", ""))
                score_traj = Path(str(data.get("trajectory", "")))
                score_is_current = (
                    trajectory_path is None
                    or score_traj == trajectory_path
                    or score_path.stat().st_mtime >= trajectory_path.stat().st_mtime
                )
            except Exception:
                score_is_current = trajectory_path is None

        if data and score_is_current:
            details = data.get("details", {})
            st.session_state["_task_scores"][task_index] = details.get("total", data.get("score", 0))
            st.session_state["_task_score_detail"][task_index] = details
            elapsed = data.get("elapsed_sec")
            if isinstance(elapsed, (int, float)) and elapsed > 0:
                st.session_state["_task_times"][task_index] = f"{float(elapsed):.1f}s"
            trajectory = data.get("trajectory")
            if trajectory:
                st.session_state["_last_trajectory"] = trajectory
            continue

        if trajectory_path is None:
            continue
        st.session_state["_last_trajectory"] = str(trajectory_path)
        details = _score_steps(task_index)
        st.session_state["_task_scores"][task_index] = details.get("total", 0)
        st.session_state["_task_score_detail"][task_index] = details
        st.session_state["_task_times"][task_index] = "-"
        try:
            _write_score_file(
                task_index=task_index,
                details=details,
                trajectory_path=trajectory_path,
                status="RECOVERED",
                elapsed=None,
            )
        except Exception:
            pass


# =====================================================================
#  Left panel: task input
# =====================================================================

@st.dialog("Task Executing", width="large")
def _execution_dialog(task_desc: str, task_index: int):
    """Keep the modal open until the MuJoCo subprocess finishes."""
    st.warning(
        "Environment is being created, this may take a moment. Please wait.\n\n"
        "If no popup appears after creation, check your taskbar for a new window."
    )

    with st.spinner("Creating MuJoCo simulation environment..."):
        result = _run_task_in_mujoco_process(task_desc, task_index)

    if result.get("scene_ready"):
        st.success("Environment created successfully! Task is running in background...")
    else:
        st.info("Environment creation timed out, task will continue in background...")

    with st.spinner("Task running, please wait..."):
        completion = _complete_background_task()
    if completion is None:
        st.warning("Task is still running in the background. The page will refresh shortly.")
        time.sleep(1.0)

    # Dialog closes here — the main flow picks up _bg_subprocess
    # and completes scoring + replay outside the dialog, no flicker.


def _queue_latest_replay(task_index: int) -> None:
    if not AUTO_GENERATE_REPLAY_GIFS:
        return
    traj_dir = _task_recording_dir(task_index)
    if not traj_dir.exists():
        return
    trajs = sorted(traj_dir.glob("trajectory_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if trajs:
        st.session_state["_pending_replay"] = str(trajs[0])
        st.session_state["_pending_replay_idx"] = task_index


def render_input_panel() -> None:
    st.subheader("Task Panel")

    # Check if execution dialog should be shown (before rendering task grid)
    if st.session_state.get("_show_exec_dialog"):
        task_desc, task_idx = st.session_state.pop("_exec_task")
        st.session_state.pop("_show_exec_dialog")
        _execution_dialog(task_desc, task_idx)
        # Dialog returned → st.rerun() to close dialog overlay.
        # _bg_subprocess is already set; next render picks it up below.
        st.rerun()

    # Complete any pending background task (subprocess launched from execution dialog)
    if st.session_state.get("_bg_subprocess"):
        bg_task_index = st.session_state["_bg_subprocess"]["task_index"]
        with st.spinner("Task running, please wait..."):
            completion = _complete_background_task()
        if completion is None:
            time.sleep(1.0)
            st.rerun()
        # Auto-generate replay GIFs after background task completes
        _queue_latest_replay(bg_task_index)
        st.rerun()

    map_info = _check_map_files()
    if not map_info["all_ok"]:
        st.warning(
            "Map files not found, cannot load simulation environment.\n\n"
            f"Search directory: `{_MAP_DIR}`\n\n"
            f"semantic.json: {'OK' if map_info['semantic_ok'] else 'MISSING'}\n\n"
            f"occupancy.npy: {'OK' if map_info['grid_ok'] else 'MISSING'}\n\n"
            "Please run:\n"
            "`python robosuite/robosuite/environments/factory_sorting/get_map.py`"
        )
        return


    # 鈹€鈹€ Competition task grid: 5 tasks, 10/15/20/25/30 pts 鈹€鈹€
    TASKS = [
        {
            "level": "L1",
            "desc": "For this task, you need to transport a blue, hollow plastic box. Please move it from the starting point \"Pick Station 2\" to the destination \"Place Station 3\". Please follow the Standard Operating Procedure (SOP).",
            "help": "SOP knowledge base auto-matches coordinates",
            "max_score": 10,
        },
        {
            "level": "L2",
            "desc": "Current Task Material Information:\nMaterial Name: Green-rimmed storage bin\nStarting Location: Pick Station 1\nTarget Location: Place Station 3\nQuantity to Transport: 1",
            "help": "SOP knowledge base auto-matches coordinates",
            "max_score": 15,
        },
        {
            "level": "L3",
            "desc": "Please follow the SOP. The object is a blue material transfer bin. The Pick Station is Pick Station 1, and the Place Station is Place Station 2.",
            "help": "SOP knowledge base auto-matches coordinates",
            "max_score": 20,
        },
        {
            "level": "L4",
            "desc": "Please strictly adhere to the Standard Operating Procedure (SOP) for this task. The object to be handled is a blue, hollow plastic box. The Pick Station is designated as Pick Station 5, and the Place Station is designated as Place Station 2.",
            "help": "SOP knowledge base auto-matches coordinates",
            "max_score": 25,
        },
        {
            "level": "L5",
            "desc": "Move the three white-rimmed storage bins from Pick Station 6 to Place Station 1.",
            "help": "SOP knowledge base auto-matches coordinates",
            "max_score": 30,
        },
    ]

    # Initialise session score/time storage
    if "_task_scores" not in st.session_state:
        st.session_state["_task_scores"] = {}
    if "_task_times" not in st.session_state:
        st.session_state["_task_times"] = {}
    if "_task_score_detail" not in st.session_state:
        st.session_state["_task_score_detail"] = {}

    _restore_scores_from_disk(len(TASKS))

    # Summary row (top)
    st.divider()
    s1, s2, s3, s4 = st.columns([2.5, 1, 0.8, 0.8])
    with s1:
        st.markdown("**Total Score / Time**")
    with s2:
        if st.button("Reset", key="reset_scores", use_container_width=True):
            st.session_state["_task_scores"] = {}
            st.session_state["_task_times"] = {}
            st.session_state["_task_score_detail"] = {}
            st.session_state["_score_restore_done"] = True
            st.rerun()
    with s3:
        scores = st.session_state["_task_scores"]
        total_score = sum(v for v in scores.values() if isinstance(v, (int, float)))
        max_possible = sum(t.get("max_score", 100) for t in TASKS)
        st.markdown(f"**{total_score}**")
    with s4:
        times = st.session_state["_task_times"]
        total_sec = 0.0
        for v in times.values():
            try:
                total_sec += float(str(v).rstrip("s"))
            except ValueError:
                pass
        st.markdown(f"**{total_sec:.1f}s**" if total_sec > 0 else "**-**")

    st.divider()

    # Header row
    h1, h2, h3, h4 = st.columns([2.5, 1, 0.8, 0.8])
    h1.markdown("**Task Description**")
    h2.markdown("**Actions**")
    h3.markdown("**Score**")
    h4.markdown("**Time**")
    st.divider()

    for i, t in enumerate(TASKS):
        c1, c2, c3, c4 = st.columns([2.5, 1, 0.8, 0.8])

        with c1:
            st.caption(f"`{t['level']}`")
            st.markdown(t["desc"])

        with c2:
            btn_key = f"task_btn_{i}"
            plan_key = f"task_plan_{i}"
            if st.button("LLM Plan", key=plan_key, use_container_width=True):
                with st.spinner("LLM thinking..."):
                    try:
                        _raw = _call_llm_plan_only(t["desc"], i)
                        st.session_state[f"_llm_raw_{i}"] = _raw
                        st.session_state["_show_llm_panel"] = i
                    except Exception as exc:
                        st.session_state[f"_llm_raw_{i}"] = f"LLM call failed: {exc}"
                        st.session_state["_show_llm_panel"] = i
                st.rerun()

            # Show LLM popup panel for this task
            _show_idx = st.session_state.get("_show_llm_panel")
            if _show_idx == i:
                _raw = st.session_state.get(f"_llm_raw_{i}", "")
                _active_backend = st.session_state.get("_active_llm_backend", "")
                _active_model = st.session_state.get("_active_llm_model", "")
                _be_info = f" ({_active_backend} / {_active_model})" if _active_backend else ""
                with st.expander(f"LLM Thought Process — {t['level']}{_be_info}", expanded=True):
                    st.text(_raw if _raw else "Waiting for LLM response...")

            if st.button("Execute", key=btn_key, use_container_width=True):
                st.session_state["_show_exec_dialog"] = True
                st.session_state["_exec_task"] = (t["desc"], i)
                st.rerun()

        with c3:
            score_val = st.session_state["_task_scores"].get(i, 0)
            if score_val > 0:
                st.success(f"{score_val}")
            elif st.session_state["_task_times"].get(i):
                st.error("0")
            else:
                st.markdown("0")

        with c4:
            st.caption(st.session_state["_task_times"].get(i, "-"))

    # 鈹€鈹€ Auto-replay: generate GIF after scores are shown 鈹€鈹€
    _pending = st.session_state.pop("_pending_replay", None)
    if AUTO_GENERATE_REPLAY_GIFS and _pending and Path(_pending).exists():
        traj_path = Path(_pending)
        _gif_cam = st.session_state.get("_replay_camera", "birdview")
        _gif_name = traj_path.stem.replace("trajectory_", "replay_") + f"_{_gif_cam}.gif"
        _gif_path = traj_path.parent / _gif_name
        if not _gif_path.exists():
            with st.spinner(f"Generating replay GIF {_gif_cam}..."):
                from robot_agent.environments import RobosuiteBackend
                rv = RobosuiteBackend(env_name=_trajectory_env_name(traj_path), camera=_gif_cam, drive_mode="direct", headless=True)
                rv.reset()
                rv.replay_trajectory(str(traj_path), str(_gif_path), camera=_gif_cam)
                rv.close()
        st.info(f"Replay generated: {_gif_name}")


def _sync_agent_memory(agent: RobotAgent) -> None:
    """Copy agent's runtime memory into the frontend session store."""
    from robot_agent.core.memory import InMemoryStore

    persist_path = str(_APP_DIR / "memory.json")
    store = InMemoryStore(limit=32, persist_path=persist_path)

    for step in agent.memory.items():
        store.add(step)

    st.session_state["_mem_store"] = store


def _save_gif(frames: list, path: Path) -> None:
    """Save frames as a GIF to *path* with a 2s pause on the last frame."""
    display = frames[:: max(1, len(frames) // 60)]
    _last = display[-1]
    _pause_frames = [Image.fromarray(_last)] * 8  # 8 * 250ms = 2s
    _all_frames = [Image.fromarray(f) for f in display] + _pause_frames
    _all_frames[0].save(
        path, format="GIF", save_all=True,
        append_images=_all_frames[1:], duration=50, loop=0,
    )


def _namespace_from_dict(value):
    from types import SimpleNamespace

    if isinstance(value, dict):
        return SimpleNamespace(**{k: _namespace_from_dict(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_namespace_from_dict(v) for v in value]
    return value


def _latest_trajectory_since(task_index: int, started_at: float) -> Path | None:
    rec_dir = _task_recording_dir(task_index)
    candidates = []
    for path in rec_dir.glob("trajectory_*.json"):
        try:
            if path.stat().st_mtime >= started_at - 2.0:
                candidates.append(path)
        except OSError:
            continue
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _call_llm_plan_only(task: str, task_index: int) -> str:
    """Call the LLM planner only — no MuJoCo, no execution. Returns raw response text."""
    import json as _json
    kb = st.session_state.get("_kb_enabled", True)

    # Record active backend for display
    _record_active_backend()

    from robot_agent.core.registry import SkillRegistry
    from robot_agent.skills.library import wired_skills
    from robot_agent.core.map_loader import load_map_files
    from robot_agent.core.scene_context import SceneContext

    _sem, _grid = _choose_map_files(task_index)
    scene_dict, grid = load_map_files(_sem, _grid)
    ctx = SceneContext.from_semantic_map(scene_dict)

    client = _create_llm_client_from_session()

    registry = SkillRegistry()
    for s in wired_skills(backend=None, scene_context=ctx, grid=grid):
        registry.register(s)

    from robot_agent.core.planner import TaskPlanner
    planner = TaskPlanner(client, registry, scene_context=ctx, knowledge_enabled=kb)
    scene_metadata = {
        "task_index": task_index,
        "env_name": _scene_env_name(task_index),
        "scene_name": getattr(ctx, "scene_name", ""),
        "map_name": getattr(ctx, "map_name", ""),
        "map_prefix": _scene_prefix(task_index),
        "semantic_map": str(_sem),
        "occupancy_grid": str(_grid),
        "input_object_map": {
            _task_source_name(task_index): _task_object_name(task_index)
        } if _task_object_name(task_index) else {},
    }
    decision = planner.plan(task, scene_metadata=scene_metadata)
    return getattr(decision, "raw_llm_text", "") or _json.dumps(
        getattr(decision, "details", {}), indent=2, ensure_ascii=False
    )


# Module-level storage for background subprocess Popen objects.
# We keep these OUT of st.session_state to avoid pickling issues.
_bg_processes: dict[str, Any] = {}


def _run_task_in_mujoco_process(task: str, task_index: int, *, update_score: bool = True) -> dict:
    """Run the MuJoCo task in a subprocess and score from the saved JSON."""
    import datetime
    import subprocess

    started_wall = time.time()
    started_perf = time.perf_counter()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rec_dir = _task_recording_dir(task_index)
    result_path = rec_dir / f"result_{ts}.json"
    log_path = rec_dir / f"subprocess_{ts}.log"
    scene_ready_path = rec_dir / f"scene_ready_{ts}.json"

    kb = st.session_state.get("_kb_enabled", True)
    cmd = [
        sys.executable,
        str(_SRC_DIR / "robot_agent" / "task_subprocess_runner.py"),
        "--task",
        task,
        "--task-index",
        str(task_index),
        "--timestamp",
        ts,
        "--result-json",
        str(result_path),
        "--app-dir",
        str(_APP_DIR),
        "--knowledge-enabled",
        str(kb).lower(),
    ]

    child_env = os.environ.copy()
    extra_paths = [
        str(_APP_DIR / "src"),
        str(_APP_DIR),
        str(_APP_DIR / "robomimic"),
        str(_APP_DIR / "robosuite" / "robosuite"),
    ]
    existing = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = os.pathsep.join(extra_paths + ([existing] if existing else []))
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["GATE_OLLAMA"] = "true"
    child_env["GATE_STEP_TIMEOUT"] = "false"

    # ── Pass backend selection from sidebar to child process ──
    _backend = st.session_state.get("_llm_backend", "ollama")
    if _backend == "local":
        _local_path = st.session_state.get("_local_model_path", "")
        if _local_path:
            child_env["LOCAL_LLM_MODEL"] = _local_path
        child_env["OPENAI_API_KEY"] = ""
        child_env["GLM_API_KEY"] = ""
        child_env["OLLAMA_BASE_URL"] = DEFAULT_OLLAMA_BASE_URL
        child_env["OLLAMA_MODEL"] = DEFAULT_OLLAMA_MODEL
    elif _backend == "openai":
        _api_key = st.session_state.get("_openai_api_key", "")
        _base = st.session_state.get("_openai_base_url", DEFAULT_OPENAI_BASE_URL)
        _model = st.session_state.get("_openai_model", DEFAULT_OPENAI_MODEL)
        if _api_key:
            child_env["OPENAI_API_KEY"] = _api_key
        child_env["OPENAI_BASE_URL"] = _base
        child_env["OPENAI_MODEL"] = _model
        child_env["LOCAL_LLM_MODEL"] = ""
        child_env["GLM_API_KEY"] = ""
        child_env["OLLAMA_BASE_URL"] = DEFAULT_OLLAMA_BASE_URL
        child_env["OLLAMA_MODEL"] = DEFAULT_OLLAMA_MODEL
    else:  # ollama
        _url = st.session_state.get("_ollama_url", DEFAULT_OLLAMA_BASE_URL)
        _model = st.session_state.get("_ollama_model", DEFAULT_OLLAMA_MODEL)
        child_env["OLLAMA_BASE_URL"] = _url
        child_env["OLLAMA_MODEL"] = _model
        child_env["LOCAL_LLM_MODEL"] = ""
        child_env["OPENAI_API_KEY"] = ""
        child_env["GLM_API_KEY"] = ""

    # ── Pass VLM sidebar settings to subprocess ──
    _vlm_url = st.session_state.get("_vlm_api_url", "")
    _vlm_key = st.session_state.get("_vlm_api_key", "")
    _vlm_model = st.session_state.get("_vlm_model", "")
    if _vlm_url:
        child_env["VLM_BASE_URL"] = _vlm_url
    if _vlm_key:
        child_env["VLM_API_KEY"] = _vlm_key
    if _vlm_model:
        child_env["VLM_MODEL"] = _vlm_model

    # Use a real log file instead of PIPE. Streamlit reruns can lose the
    # Popen object; if stdout/stderr are pipes, the child may crash when it
    # writes diagnostics after the parent-side pipe handle is gone.
    log_file = log_path.open("w", encoding="utf-8", errors="replace")
    try:
        log_file.write(f"command: {subprocess.list2cmdline(cmd)}\n\n===== OUTPUT =====\n")
        log_file.flush()
        process = subprocess.Popen(
            cmd,
            cwd=str(_APP_DIR),
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    finally:
        log_file.close()

    # ---- Phase 1: wait for MuJoCo scene to be created ----
    _scene_ready = False
    _scene_timeout = 120  # max seconds to wait for scene creation
    _scene_start = time.perf_counter()
    while process.poll() is None:
        if scene_ready_path.exists():
            _scene_ready = True
            break
        if time.perf_counter() - _scene_start > _scene_timeout:
            break
        time.sleep(0.3)

    # Store background task info so we can complete after dialog closes.
    # Keep the Popen object in module-level dict (avoids pickling issues).
    _bg_processes[ts] = process
    st.session_state["_bg_subprocess"] = {
        "pid": process.pid,
        "result_path": str(result_path),
        "log_path": str(log_path),
        "scene_ready_path": str(scene_ready_path),
        "ts": ts,
        "task_index": task_index,
        "started_wall": started_wall,
        "started_perf": started_perf,
        "update_score": update_score,
    }

    # Return immediately — caller (dialog) closes when scene is ready.
    # _complete_background_task() handles the rest after the dialog closes.
    return {
        "phase": "scene_ready",
        "scene_ready": _scene_ready,
        "result_path": str(result_path),
        "log_path": str(log_path),
        "ts": ts,
        "task_index": task_index,
    }


def _complete_background_task() -> dict | None:
    """Wait for the background subprocess to finish and process results.

    Call this after the execution dialog has closed (scene is ready).
    Returns the same result dict as _run_task_in_mujoco_process, or None
    if there is no pending background task.
    """
    import subprocess

    info = st.session_state.get("_bg_subprocess")
    if info is None:
        return None

    ts = info["ts"]
    process = _bg_processes.pop(ts, None)

    result_path = Path(info["result_path"])
    log_path = Path(info["log_path"])
    task_index = info["task_index"]
    started_wall = info["started_wall"]
    started_perf = info["started_perf"]
    update_score = info["update_score"]

    # ---- Phase 2: wait for the subprocess to finish ----
    returncode = 0
    if process is not None:
        try:
            returncode = process.wait(timeout=6000)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            returncode = -1

        try:
            with log_path.open("a", encoding="utf-8", errors="replace") as _log:
                _log.write(f"\n===== PROCESS EXIT =====\nreturncode: {returncode}\n")
        except Exception:
            pass
    else:
        # Process already completed — just check returncode from result file
        if not result_path.exists():
            return None
        try:
            _existing = json.loads(result_path.read_text(encoding="utf-8"))
            returncode = 1 if _existing.get("status") == "CRASH" else 0
        except Exception:
            returncode = -1
        try:
            with log_path.open("a", encoding="utf-8", errors="replace") as _log:
                _log.write(
                    "\n===== PROCESS EXIT =====\n"
                    f"returncode: {returncode} (inferred; process handle unavailable)\n"
                )
        except Exception:
            pass

    elapsed = time.perf_counter() - started_perf
    st.session_state["_last_physics_files"] = {"diag": str(log_path), "result": str(result_path)}

    manifest = None
    if result_path.exists():
        try:
            manifest = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = None

    trajectory_path = None
    if manifest and manifest.get("trajectory"):
        candidate = Path(str(manifest["trajectory"]))
        if candidate.exists():
            trajectory_path = candidate
    if trajectory_path is None:
        trajectory_path = _latest_trajectory_since(task_index, started_wall)

    if trajectory_path and trajectory_path.exists():
        st.session_state["_last_trajectory"] = str(trajectory_path)
        status = str((manifest or {}).get("status") or ("OK" if returncode == 0 else "SUBPROCESS_CRASH"))
        if update_score:
            details = _score_steps(task_index)
            st.session_state["_task_scores"][task_index] = details["total"]
            st.session_state["_task_score_detail"][task_index] = details
            st.session_state["_task_times"][task_index] = f"{elapsed:.1f}s"
            score_path = _write_score_file(
                task_index=task_index,
                details=details,
                trajectory_path=trajectory_path,
                status=status if returncode == 0 else f"{status}_RET{returncode}",
                elapsed=elapsed,
            )
            st.session_state["_last_score_path"] = str(score_path)
        else:
            details = {"total": 0, "items": []}

        if returncode != 0:
            msg = (
                "MuJoCo subprocess exited abnormally, but trajectory JSON was saved and scored."
                if update_score else
                "MuJoCo subprocess exited abnormally, but trajectory JSON was saved."
            )
            st.session_state["_mujoco_subprocess_warning"] = msg
            st.session_state["last_error"] = None
        else:
            st.session_state["_mujoco_subprocess_warning"] = None
            st.session_state["last_error"] = None
    else:
        st.session_state["_mujoco_subprocess_warning"] = None
        st.session_state["last_error"] = (
            f"MuJoCo subprocess exited abnormally, and no scorable trajectory JSON was found. returncode={returncode}"
            if returncode != 0 else "MuJoCo subprocess did not generate a trajectory JSON."
        )
        details = {"total": 0, "items": []}

    result_dict = (manifest or {}).get("result")
    if isinstance(result_dict, dict):
        st.session_state["last_result"] = _namespace_from_dict(result_dict)
    elif returncode == 0:
        st.session_state["last_result"] = None
    else:
        st.session_state["last_result"] = None

    st.session_state["_plan_only"] = None

    st.session_state.pop("_bg_subprocess", None)

    return {
        "returncode": returncode,
        "manifest": manifest,
        "trajectory": str(trajectory_path) if trajectory_path else None,
        "log": str(log_path),
        "details": details,
        "elapsed": elapsed,
    }


def execute(task: str, agent: RobotAgent, task_index: int = 0) -> None:
    import datetime
    execute_start = time.perf_counter()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rec_dir = _task_recording_dir(task_index)
    running_path = rec_dir / f"trajectory_{ts}_RUNNING.json"
    crash_traj_saved = False
    try:
        agent.backend.start_recording()
        if hasattr(agent.backend, "set_trajectory_autosave"):
            agent.backend.set_trajectory_autosave(running_path, every_n_frames=50)
        result = agent.run(task)
        # Record final frame to capture post-place object position
        try:
            agent.backend._record_trajectory_frame()
        except Exception:
            pass
        ok = result.success
        st.session_state["last_result"] = result
        st.session_state["last_error"] = None
        st.session_state["_plan_only"] = None
        _sync_agent_memory(agent)
        status = "OK" if ok else "FAIL"
        traj_path = agent.backend.save_trajectory(rec_dir / f"trajectory_{ts}_{status}.json")
        st.session_state["_last_trajectory"] = str(traj_path)  # for score verification
        details = _score_steps(task_index)
        elapsed = time.perf_counter() - execute_start
        st.session_state["_task_scores"][task_index] = details["total"]
        st.session_state["_task_score_detail"][task_index] = details
        st.session_state["_task_times"][task_index] = f"{elapsed:.1f}s"
        score_path = _write_score_file(
            task_index=task_index,
            details=details,
            trajectory_path=Path(traj_path),
            status=status,
            elapsed=elapsed,
        )
        st.session_state["_last_score_path"] = str(score_path)
        try:
            if hasattr(agent.backend, "clear_trajectory_autosave"):
                agent.backend.clear_trajectory_autosave()
            if running_path.exists():
                running_path.unlink()
        except Exception:
            pass
        # Keep task execution lightweight: grasp GIFs are generated manually
        # from the saved trajectory JSON in the sidebar replay section.
        st.session_state["_grasp_frames"] = []
        # Generate replay GIFs only when explicitly enabled; the sidebar button is the default path.
        if AUTO_GENERATE_REPLAY_GIFS:
            try:
                from robot_agent.environments import RobosuiteBackend
                for _cam, _suffix, _key in [
                    ("birdview", "_birdview", "last_frames"),
                    ("robot0_robotview", "_robotview", "_robotview_frames"),
                ]:
                    rv_backend = RobosuiteBackend(env_name=_scene_env_name(task_index), camera=_cam, drive_mode="direct", headless=True)
                    rv_backend.reset()
                    rv_frames = rv_backend.replay_trajectory(
                        traj_path,
                        rec_dir / f"recording_{ts}_{status}{_suffix}.gif",
                        camera=_cam,
                    )
                    if rv_frames:
                        st.session_state[_key] = rv_frames
                    rv_backend.close()
            except Exception:
                pass
    except Exception as exc:
        st.session_state["last_error"] = str(exc)
        # Try to salvage trajectory + replay GIFs
        try:
            import traceback as _tb
            _tb.print_exc()
            traj_path = agent.backend.save_trajectory(rec_dir / f"trajectory_{ts}_CRASH.json")
            crash_traj_saved = True
            st.session_state["_last_trajectory"] = str(traj_path)
            details = _score_steps(task_index)
            elapsed = time.perf_counter() - execute_start
            st.session_state["_task_scores"][task_index] = details["total"]
            st.session_state["_task_score_detail"][task_index] = details
            st.session_state["_task_times"][task_index] = f"{elapsed:.1f}s"
            score_path = _write_score_file(
                task_index=task_index,
                details=details,
                trajectory_path=Path(traj_path),
                status="CRASH",
                elapsed=elapsed,
            )
            st.session_state["_last_score_path"] = str(score_path)
            if AUTO_GENERATE_REPLAY_GIFS:
                from robot_agent.environments import RobosuiteBackend
                for _cam, _suffix, _key in [
                    ("birdview", "_birdview", "last_frames"),
                    ("robot0_robotview", "_robotview", "_robotview_frames"),
                ]:
                    try:
                        rv_backend = RobosuiteBackend(env_name=_scene_env_name(task_index), camera=_cam, drive_mode="direct", headless=True)
                        rv_backend.reset()
                        rv_frames = rv_backend.replay_trajectory(
                            traj_path, rec_dir / f"recording_{ts}_CRASH{_suffix}.gif", camera=_cam)
                        if rv_frames:
                            st.session_state[_key] = rv_frames
                        rv_backend.close()
                    except Exception:
                        pass
        except Exception:
            st.session_state["last_frames"] = []
        try:
            if not crash_traj_saved:
                traj_path = agent.backend.save_trajectory(rec_dir / f"trajectory_{ts}_CRASH.json")
                st.session_state["_last_trajectory"] = str(traj_path)
                details = _score_steps(task_index)
                elapsed = time.perf_counter() - execute_start
                st.session_state["_task_scores"][task_index] = details["total"]
                st.session_state["_task_score_detail"][task_index] = details
                st.session_state["_task_times"][task_index] = f"{elapsed:.1f}s"
                score_path = _write_score_file(
                    task_index=task_index,
                    details=details,
                    trajectory_path=Path(traj_path),
                    status="CRASH",
                    elapsed=elapsed,
                )
                st.session_state["_last_score_path"] = str(score_path)
        except Exception:
            pass
        st.session_state["last_result"] = None
    finally:
        try:
            agent.backend.close()
        except Exception:
            pass


def plan_only(task: str) -> None:
    """Show planner output without running the simulation."""
    from robot_agent.config import AgentConfig
    from robot_agent.core.registry import SkillRegistry
    from robot_agent.core.map_loader import load_map_files
    from robot_agent.core.scene_context import SceneContext
    from robot_agent.core.planner import TaskPlanner
    from robot_agent.skills.library import wired_skills

    # Record active backend for display
    _record_active_backend()

    map_info = _check_map_files()
    if not map_info["all_ok"]:
        st.error("Map not loaded, cannot plan")
        return

    with st.spinner("LLM planning..."):
        try:
            scene, grid = load_map_files(
                *_choose_map_files(),
            )
            scene_ctx = SceneContext.from_semantic_map(scene)
            registry = SkillRegistry()

            # Register skills (without backend -- names only for planning)
            class _FakeBackend:
                pass
            fake = _FakeBackend()
            fake.env = None
            skills = wired_skills(backend=fake, scene_context=scene_ctx, grid=grid)
            for s in skills:
                registry.register(s)

            config = AgentConfig()
            config.feature_gates.ollama = True

            client = _create_llm_client_from_session()

            planner = TaskPlanner(client, registry, config, scene_context=scene_ctx)
            decision = planner.plan(task)

            st.session_state["_plan_only"] = decision
        except Exception as exc:
            st.error(f"Planning failed: {exc}")
            st.session_state["_plan_only"] = None


# =====================================================================
#  Right panel: results
# =====================================================================

def render_result_panel() -> None:
    st.subheader("Execution Results")
    subprocess_warning = st.session_state.get("_mujoco_subprocess_warning")
    if subprocess_warning:
        st.warning(subprocess_warning)

    # 鈹€鈹€ Show last physics diag output if available 鈹€鈹€
    _last_diag_files = st.session_state.get("_last_physics_files", {})
    if _last_diag_files.get("diag"):
        import pathlib
        _diag_p = pathlib.Path(_last_diag_files["diag"])
        if _diag_p.exists():
            with st.expander("Terminal Output", expanded=False):
                st.code(_diag_p.read_text(encoding="utf-8")[-5000:], language="text")

    frames = st.session_state.get("last_frames")
    if frames:
        render_video_panel(frames)

    # Show plan-only result if available
    plan_decision = st.session_state.get("_plan_only")
    if plan_decision:
        with st.container(border=True):
            st.markdown("#### Plan Result (LLM planning only, not executed)")
            details = getattr(plan_decision, "details", {}) or {}
            plan = details.get("plan", []) if isinstance(details, dict) else []
            if plan:
                for i, s in enumerate(plan, start=1):
                    st.caption(f"{i}. **{s.get('skill_name','?')}**: {s.get('description','')} -> {s.get('expected_output','')}")
            st.code(getattr(plan_decision, "raw_llm_text", "")[:2000] or "(empty)", language="json")
        return

    result = st.session_state.get("last_result")
    error = st.session_state.get("last_error")

    if error:
        st.error(f"Execution failed: {error}")
        return

    if result is None:
        st.info("Click Execute to run the simulation, or click LLM Plan to preview the planning result.")
        return

    # ── Active backend badge ──
    _active_backend = st.session_state.get("_active_llm_backend", "")
    _active_model = st.session_state.get("_active_llm_model", "")
    if _active_backend:
        st.caption(f"LLM: {_active_backend} / {_active_model}")

    elapsed = getattr(result, "elapsed_ms", 0)
    if elapsed:
        st.metric("Elapsed", f"{elapsed:.0f} ms")

    thinking = getattr(result, "thinking", None)
    reason = thinking.selection_reason if thinking else ""
    warnings = getattr(result, "plan_warnings", []) or []
    if reason and "LLM" in reason:
        st.warning(reason)
        for w in warnings:
            st.caption(f"-> {w}")
    elif reason:
        st.caption(f"Plan: {reason}")

    if result.success:
        st.success("Task Succeeded")
    else:
        st.error("Task Failed")
    st.write(result.message)

    steps = getattr(result, "steps", []) or []
    if steps:
        st.markdown("### Execution Steps")
        for i, s in enumerate(steps, start=1):
            status = "[OK]" if s.success else "[FAIL]"
            label = f"{status} {i}. {s.skill}: {s.description}"
            _step_expander(label, s)

    raw_text = getattr(result, "planner_raw", "")
    _active_backend = st.session_state.get("_active_llm_backend", "")
    _active_model = st.session_state.get("_active_llm_model", "")
    _backend_info = f" ({_active_backend} / {_active_model})" if _active_backend else ""
    with st.expander(f"LLM Thought Process{_backend_info}", expanded=True):
        _render_raw_llm(raw_text)


def _frames_to_gif_bytes(frames: list[np.ndarray]) -> io.BytesIO | None:
    """Convert numpy frames to in-memory GIF bytes."""
    if not frames:
        return None
    max_display = 60
    if len(frames) > max_display:
        step = len(frames) // max_display
        display_frames = frames[::step]
    else:
        display_frames = frames
    pil_frames = [Image.fromarray(f) for f in display_frames]
    _last_pil = pil_frames[-1]
    pil_frames.extend([_last_pil.copy()] * 8)
    buf = io.BytesIO()
    pil_frames[0].save(buf, format="GIF", save_all=True,
                       append_images=pil_frames[1:], duration=50, loop=0)
    buf.seek(0)
    return buf


def render_video_panel(frames: list[np.ndarray]) -> None:
    if not frames:
        return

    st.markdown("### Replay Video")

    # Get multi-camera frames from session state
    birdview_frames = st.session_state.get("last_frames", frames)
    robotview_frames = st.session_state.get("_robotview_frames", [])

    # Show grasp process frames if available (robot0_robotview — chest camera)
    grasp_frames = st.session_state.get("_grasp_frames")
    if grasp_frames:
        buf = _frames_to_gif_bytes(grasp_frames)
        if buf:
            st.image(buf, caption=f"Grasp process — robot0_robotview (chest camera) — {len(grasp_frames)} frames", use_container_width=True)

    # Robot chest view (full width)
    if robotview_frames:
        buf = _frames_to_gif_bytes(robotview_frames)
        if buf:
            st.image(buf, caption=f"robot0_robotview (chest camera) — {len(robotview_frames)} frames", use_container_width=True)

    # Birdview overview (full width)
    buf = _frames_to_gif_bytes(birdview_frames)
    if buf:
        cols = st.columns([1, 1, 1])
        cols[0].metric("Total Frames", len(frames))
        cols[1].metric("FPS", "4")
        cols[2].caption("birdview")
        st.image(buf, caption=f"birdview (top-down overview) — {len(birdview_frames)} frames", use_container_width=True)


def _step_expander(label: str, step: Any) -> None:
    with st.expander(label, expanded=False):
        if getattr(step, "success", False):
            st.success(getattr(step, "message", ""))
        else:
            st.error(getattr(step, "message", ""))
        for field in ("inputs", "preconditions", "expected_output"):
            val = getattr(step, field, None)
            if val:
                st.caption(f"{field}: {val}")
        attempts = getattr(step, "attempts", 0)
        timeout = getattr(step, "timeout", None)
        retries = getattr(step, "retries", 0)
        if timeout or retries:
            st.caption(f"timeout={timeout}s  retries={retries}  attempts={attempts}")


def _render_raw_llm(raw_text: str) -> None:
    _active_backend = st.session_state.get("_active_llm_backend", "")
    _active_model = st.session_state.get("_active_llm_model", "")
    if _active_backend:
        st.caption(f"LLM Backend: {_active_backend} / {_active_model} | Visualization is for reference only.")
    else:
        st.caption("Frontend visualization is for reference only. Final results are based on the raw JSON returned by the LLM.")
    if not raw_text:
        st.info("LLM returned no data")
        return

    cols = st.columns([1, 1, 1])
    cols[0].metric("Chars", len(raw_text))
    cols[1].metric("Lines", raw_text.count("\n") + 1)

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        from robot_agent.core.schema import repair_json
        parsed = repair_json(raw_text)

    if isinstance(parsed, dict):
        cols[2].metric("Parse", "Valid JSON")
        st.markdown("#### Task Understanding")
        st.info(parsed.get("understanding", "(not provided)"))
        reason = parsed.get("reason", "")
        if reason:
            st.caption(f"Decomposition reason: {reason}")
        plan = parsed.get("plan")
        if isinstance(plan, list) and plan:
            st.markdown("#### Plan Steps")
            rows = [{"#": i, "Skill": s.get("skill_name", "?"),
                     "Description": s.get("description", "")}
                    for i, s in enumerate(plan, start=1)]
            st.dataframe(rows, width="stretch", hide_index=True)
        explanation = parsed.get("explanation", "")
        if explanation:
            st.caption(f"Result explanation: {explanation}")
    else:
        cols[2].metric("Parse", "Non-JSON")

    st.markdown("#### Raw Text")
    st.code(raw_text, language="json")


# =====================================================================
#  Knowledge-base panel
# =====================================================================

def render_knowledge_panel() -> None:
    """Browse, search, and manage the knowledge base."""
    from robot_agent.core.knowledge_manager import KnowledgeManager

    if "_kb_mgr" not in st.session_state:
        st.session_state["_kb_mgr"] = KnowledgeManager(_KNOWLEDGE_ROOT)
        st.session_state["_kb_mgr"].reload()

    mgr: KnowledgeManager = st.session_state["_kb_mgr"]

    # Top bar: search + filter + stats
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        kb_query = st.text_input(
            "Search Knowledge Base",
            key="kb_search",
            placeholder="Enter keywords to search...",
            label_visibility="collapsed",
        )
    with c2:
        cat_filter = st.selectbox(
            "Category", ["All", "sop", "reference", "rule", "manual", "inventory", "general"],
            key="kb_cat", label_visibility="collapsed",
        )
    with c3:
        stats = mgr.stats()
        st.metric("Docs", stats["total_docs"])

    # Results
    if kb_query.strip():
        results = mgr.search(kb_query.strip())
    else:
        results = mgr.list_docs(category=None if cat_filter == "All" else cat_filter)

    if not results:
        st.info("No matching documents. Add .md / .txt files to the knowledge/ folder to add them.")
        return

    for doc in results:
        cat_label = doc["category"]
        with st.expander(f"[{cat_label}] **{doc['title']}**", expanded=len(results) <= 3):
            cols = st.columns([3, 1])
            with cols[0]:
                st.caption(f"Source: `{doc['source_file']}`  |  Size: {doc['size_bytes']} B")
                if doc.get("tags"):
                    st.caption("Tags: " + " ".join(f"`{t}`" for t in doc["tags"]))
            with cols[1]:
                if st.button("Delete", key=f"kb_del_{doc['doc_id'][:12]}",
                             use_container_width=True):
                    mgr.remove_doc(doc["doc_id"])
                    st.rerun()

            # Full content
            full_doc = mgr.get_doc(doc["doc_id"])
            if full_doc:
                with st.container(border=True):
                    st.markdown(full_doc.content[:3000])
                    if len(full_doc.content) > 3000:
                        st.caption(f"Showing first {len(full_doc.content)} chars only, truncated")



# =====================================================================
#  Memory-inspector panel
# =====================================================================

def render_memory_panel() -> None:
    """Inspect and manage agent runtime memory."""
    from robot_agent.core.memory import InMemoryStore

    persist_path = str(_APP_DIR / "memory.json")

    if "_mem_store" not in st.session_state:
        st.session_state["_mem_store"] = InMemoryStore(limit=32, persist_path=persist_path)

    store: InMemoryStore = st.session_state["_mem_store"]

    # Summary card
    s = store.stats()
    if s["total_steps"] == 0:
        st.info("Memory is empty. After executing tasks, agent operation records will appear here.")
        return

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", s["total_steps"], f"limit {s['limit']}")
    c2.metric("OK", s["success_count"])
    c3.metric("FAIL", s["fail_count"], delta="-0" if s["fail_count"] == 0 else f"-{s['fail_count']}")
    persist_label = "Enabled" if s["persist_path"] else "Disabled"
    c4.metric("Persistence", persist_label)

    # Most-used skills
    if s["by_skill"]:
        skill_badges = " ".join(f"`{name}` x{cnt}" for name, cnt in list(s["by_skill"].items())[:5])
        st.caption(f"Skill distribution: {skill_badges}")

    # Tabs: recent / search / failures
    tab1, tab2, tab3 = st.tabs(["Recent Ops", "Search Memory", "Failures"])

    with tab1:
        recent = store.recent(10)
        if recent:
            for step in recent:
                status = "[OK]" if step["success"] else "[FAIL]"
                st.caption(
                    f"{status} **{step['skill_name']}** -- {step['message'][:100]}\n"
                    f"  -> task: _{step['task'][:80]}_"
                )
        else:
            st.caption("No records yet")

    with tab2:
        mem_query = st.text_input("Search Memory", key="mem_search",
                                  placeholder="Enter keywords to search history...",
                                  label_visibility="collapsed")
        if mem_query.strip():
            hits = store.recall(mem_query.strip())
            if hits:
                for h in hits:
                    sc = h.get("score", 0)
                    status = "[OK]" if h["success"] else "[FAIL]"
                    st.caption(
                        f"score={sc} {status} **{h['skill_name']}** -- {h['message'][:120]}\n"
                        f"  -> {h['task'][:80]}"
                    )
            else:
                st.info(f"No records matching {mem_query!r} found")
        else:
            st.caption("Enter keywords to search history (matches task / skill / message / payload)")

    with tab3:
        failures = store.failures()
        if failures:
            for f_step in failures:
                st.error(
                    f"**{f_step['skill_name']}** -- {f_step['message'][:150]}\n"
                    f"  -> {f_step['task'][:80]}"
                )
        else:
            st.success("No failure records")

    # Actions
    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Clear All Memory", use_container_width=True):
            store.clear()
            st.rerun()
    with c2:
        forget_q = st.text_input("Forget Keyword", key="mem_forget",
                                 placeholder="e.g.: failed",
                                 label_visibility="collapsed")
    with c3:
        if st.button("Forget Matching", use_container_width=True,
                     disabled=not forget_q.strip()):
            removed = store.forget(forget_q.strip())
            st.success(f"Forgotten {removed} records")
            st.rerun()


# =====================================================================
#  Main
# =====================================================================

def main() -> None:
    st.set_page_config(page_title="JCIIOT2026", layout="wide")
    render_sidebar()

    # Top row: task input + results
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.title("JCIIOT2026")
        render_input_panel()
    with right:
        render_result_panel()


if __name__ == "__main__":
    main()
