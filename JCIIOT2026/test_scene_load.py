"""Minimal test: try to create the FactorySorting env with Siemens arena."""
from __future__ import annotations

import sys
from pathlib import Path

# ── repo root ──
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── monkey-patch robosuite namespace (same as app.py) ──
_ROBOSUITE_INNER_DIR = ROOT / "robosuite" / "robosuite"
_ROBOSUITE_INNER = _ROBOSUITE_INNER_DIR / "__init__.py"
if _ROBOSUITE_INNER.exists():
    # Ensure v4 inner is first in sys.path so submodules resolve here, not v2
    if str(_ROBOSUITE_INNER_DIR) not in sys.path:
        sys.path.insert(0, str(_ROBOSUITE_INNER_DIR))
    import robosuite as _rs
    _rs.__file__ = str(_ROBOSUITE_INNER)
    _rs.__path__ = [str(_ROBOSUITE_INNER_DIR)]  # force submodule resolution to v4
    with open(_ROBOSUITE_INNER, encoding="utf-8") as _f:
        _code = compile(_f.read(), str(_ROBOSUITE_INNER), "exec")
    exec(_code, _rs.__dict__)

# re-import to bind 'robosuite' name in local scope with patched module
import robosuite  # noqa: F811

print("1. robosuite.__version__:", robosuite.__version__)

# ── Check SiemensArena import ──
print("2. Testing SiemensArena import...")
try:
    from robosuite.models.arenas import SiemensArena
    print("   OK - SiemensArena imported")
except ImportError as e:
    print(f"   FAIL: {e}")

# ── Check BoxObject imports used by factory_sorting ──
print("3. Testing FactorySorting imports...")
try:
    from robosuite.models.objects import (
        BoxObject,
        ContainerH01Object,
        ContainerH10Object,
        PlasticCrateObject,
        ToteB01Object,
    )
    print("   OK - all object classes imported")
except ImportError as e:
    print(f"   FAIL: {e}")

# ── Create environment (headless) ──
print("4. Creating FactorySorting env (headless, SiemensArena)...")
import robosuite as suite
try:
    env = suite.make(
        "FactorySorting",
        robots="Tiago",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        use_object_obs=True,
        ignore_done=True,
        control_freq=20,
        use_siemens_arena=True,
    )
    print("   OK - env created")

    print("5. Resetting env...")
    env.reset()
    print("   OK - env reset")

    body_names = [env.sim.model.body_id2name(i) for i in range(env.sim.model.nbody)]
    print(f"   Bodies: {len(body_names)}")
    siemens_bodies = [b for b in body_names if 'line_' in b or 'siemens' in b.lower()]
    print(f"   Siemens-related: {siemens_bodies[:10]}...")

    geom_names = [env.sim.model.geom_id2name(i) for i in range(env.sim.model.ngeom)]
    print(f"   Geoms: {len(geom_names)}")

    # Try stepping
    import numpy as np
    low, high = env.action_spec
    for i in range(10):
        env.step(np.zeros_like(low))
    print("   OK - 10 steps completed")

    env.close()
    print("\n*** ALL CHECKS PASSED ***")

except Exception as e:
    print(f"   FAIL: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
