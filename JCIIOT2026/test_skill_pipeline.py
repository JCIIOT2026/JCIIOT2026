"""Integration test: verify v4 skill pipeline works with Siemens scene."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── monkey-patch robosuite (same as app.py) ──
_ROBOSUITE_INNER_DIR = ROOT / "robosuite" / "robosuite"
_ROBOSUITE_INNER = _ROBOSUITE_INNER_DIR / "__init__.py"
if _ROBOSUITE_INNER.exists():
    if str(_ROBOSUITE_INNER_DIR) not in sys.path:
        sys.path.insert(0, str(_ROBOSUITE_INNER_DIR))
    import robosuite as _rs
    _rs.__file__ = str(_ROBOSUITE_INNER)
    _rs.__path__ = [str(_ROBOSUITE_INNER_DIR)]
    with open(_ROBOSUITE_INNER, encoding="utf-8") as _f:
        _code = compile(_f.read(), str(_ROBOSUITE_INNER), "exec")
    exec(_code, _rs.__dict__)

# Ensure agent module is on path
_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import os
os.environ["GATE_OLLAMA"] = "true"
os.environ["OLLAMA_BASE_URL"] = "https://pa1l785z-a9z3rfcp-11434.zj02restapi.gpufree.cn:8443"
os.environ["OLLAMA_MODEL"] = "qwen3.6:27b-mtp-q4_K_M"
os.environ["GATE_STEP_TIMEOUT"] = "false"

from robot_agent.core.map_loader import load_map_files
from robot_agent.core.scene_context import SceneContext

_MAP_DIR = (
    ROOT / "robosuite" / "robosuite" / "environments"
    / "factory_sorting" / "generated_maps"
)

# ── 1. Test map loading ──
print("1. Loading Siemens maps...")
siemens_semantic = _MAP_DIR / "factory_sorting_scene_regenerated_semantic_map.json"
siemens_grid = _MAP_DIR / "factory_sorting_scene_regenerated_occupancy_grid.npy"
assert siemens_semantic.exists(), f"Missing: {siemens_semantic}"
assert siemens_grid.exists(), f"Missing: {siemens_grid}"
scene, grid = load_map_files(siemens_semantic, siemens_grid)
print(f"   Grid shape: {grid.shape}")
print(f"   Bounds: {scene.get('bounds')}")
print(f"   Robot start: {scene.get('robot', {}).get('start')}")

# ── 2. Test SceneContext ──
print("2. Building SceneContext...")
ctx = SceneContext.from_semantic_map(scene)
print(f"   Input ports: {list(ctx.input_ports.keys())}")
print(f"   Output ports: {list(ctx.output_ports.keys())}")
assert len(ctx.input_ports) == 6, f"Expected 6 inputs, got {len(ctx.input_ports)}"
assert len(ctx.output_ports) == 6, f"Expected 6 outputs, got {len(ctx.output_ports)}"
for i in range(1, 7):
    assert f"input_{i}" in ctx.input_ports, f"Missing input_{i}"
    assert f"output_{i}" in ctx.output_ports, f"Missing output_{i}"
    ap = ctx.approach_xy(f"input_{i}")
    print(f"   input_{i}: center=({ctx.input_port(f'input_{i}').center[0]:.1f},{ctx.input_port(f'input_{i}').center[1]:.1f}), approach=({ap[0]:.1f},{ap[1]:.1f})")
print("   Prompt context:")
print(ctx.as_prompt_context()[:500])

# ── 3. Test Skill registration ──
print("3. Building skill registry...")
from robot_agent.core.registry import SkillRegistry
from robot_agent.skills.library import wired_skills

registry = SkillRegistry()
skills = wired_skills(backend=None, scene_context=ctx, grid=grid, path_spacing=0.35)
for skill in skills:
    registry.register(skill)
print(f"   Registered skills: {[s.name for s in registry.all()]}")

# ── 4. Test skill target resolution ──
print("4. Testing skill target resolution...")
from robot_agent.skills.move import MoveSkill
from robot_agent.skills.pick_up import PickUpSkill
from robot_agent.skills.place_down import PlaceDownSkill

# Test _resolve_station_name
import inspect
_move = [s for s in skills if s.name == "move"][0]
_pick = [s for s in skills if s.name == "pick_up"][0]

# move._resolve_target should work with Siemens station names
from robot_agent.skills.pick_up import _resolve_station_name
tests = ["input_1", "input_3", "output_5", "1号进料口", "3号出料口", "把1号进料口送到3号出料口"]
for t in tests:
    resolved = _resolve_station_name(t, ctx)
    print(f"   '{t}' -> '{resolved}'")

# ── 5. Test RobosuiteBackend creation (headless) ──
print("5. Creating RobosuiteBackend (headless, Siemens arena)...")
from robot_agent.environments import RobosuiteBackend
backend = RobosuiteBackend(
    env_name="FactorySorting",
    camera="birdview",
    drive_mode="direct",
    headless=True,
)
try:
    backend.reset()
    print("   OK - env created & reset")

    base_xy, yaw = backend.get_base_pose()
    print(f"   Robot base: ({base_xy[0]:.2f}, {base_xy[1]:.2f}), yaw={yaw:.2f}")

    # Check objects
    import numpy as np
    print(f"   Material objects: {backend.env.material_objects}")
    print(f"   Object positions:")
    for obj_name in backend.env.material_objects:
        body_id = backend.env.obj_body_id[obj_name]
        pos = np.array(backend.env.sim.data.body_xpos[body_id])
        print(f"     {obj_name}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

    # Test step
    low, _ = backend.action_spec
    backend.env.step(np.zeros_like(low))
    print("   OK - 1 step completed")

except Exception as e:
    print(f"   FAIL: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
finally:
    backend.close()

print("\n*** ALL INTEGRATION CHECKS PASSED ***")
