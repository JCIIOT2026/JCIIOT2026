import argparse
import time
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robosuite.environments.factory_sorting.factory_sorting_3_3fo3errph7x9 import (  # noqa: E402
    FactorySorting3_3FO3ERRPH7X9,
)


MAX_FR = 25


def make_env(args):
    render_camera = None if args.camera == "free" else args.camera
    return FactorySorting3_3FO3ERRPH7X9(
        robots="Tiago",
        has_renderer=not args.headless,
        has_offscreen_renderer=args.headless,
        render_camera=render_camera,
        use_camera_obs=False,
        use_object_obs=True,
        ignore_done=True,
        control_freq=20,
        seed=args.seed,
    )


def main():
    parser = argparse.ArgumentParser(description="Load and visualize the FactorySorting 3-3FO3ERRPH7X9 scene.")
    parser.add_argument(
        "--camera",
        default="free",
        choices=["free", "frontview", "agentview", "birdview", "sideview"],
        help="Use 'free' for the interactive MuJoCo free camera.",
    )
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--headless", action="store_true", help="Create the scene without an on-screen viewer.")
    args = parser.parse_args()

    env = make_env(args)
    env.reset()

    if args.camera != "free" and not args.headless and env.viewer is not None:
        env.viewer.set_camera(camera_id=env.sim.model.camera_name2id(args.camera))

    low, high = env.action_spec
    action = np.zeros_like(low)

    try:
        for _ in range(args.steps):
            start = time.time()
            env.step(action)
            if not args.headless:
                env.render()

            elapsed = time.time() - start
            delay = 1 / MAX_FR - elapsed
            if delay > 0:
                time.sleep(delay)
    finally:
        env.close()


if __name__ == "__main__":
    main()
