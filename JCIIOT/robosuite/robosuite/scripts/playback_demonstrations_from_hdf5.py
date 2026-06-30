"""
A convenience script to playback random demonstrations from
a set of demonstrations stored in a hdf5 file.

Arguments:
    --folder (str): Path to demonstrations
    --filename (str): Name of the HDF5 file (default: demo.hdf5)
    --use-actions (optional): If this flag is provided, the actions are played back
        through the MuJoCo simulator, instead of loading the simulator states
        one by one.
    --visualize-gripper (optional): If set, will visualize the gripper site
    --env-name (str): Name of the environment to use (required if not in HDF5 attributes)
    --robots (str): Name of the robot(s) to use (default: Panda)
    --model-file (str): Path to model XML file (required if not in HDF5 attributes)

Example:
    $ python playback_demonstrations_from_hdf5.py --folder ../models/assets/demonstrations/lift/
    $ python playback_demonstrations_from_hdf5.py --folder ../dataset --filename my_demo.hdf5 --env-name Lift --robots Panda
"""

import argparse
import importlib
import json
import os
import random
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import h5py
import numpy as np

import robosuite
from robosuite.environments.manipulation.lift import Lift
from robosuite.environments.manipulation.stack import Stack
from robosuite.environments.manipulation.pick_place import PickPlace
from robosuite.environments.manipulation.door import Door

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Path to your demonstration folder that contains the demo.hdf5 file, e.g.: "
        "'path_to_assets_dir/demonstrations/YOUR_DEMONSTRATION'",
    ),
    parser.add_argument(
        "--filename",
        type=str,
        default="demo.hdf5",
        help="Name of the HDF5 file (default: demo.hdf5)",
    ),
    parser.add_argument(
        "--use-actions",
        action="store_true",
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default=None,
        help="Name of the environment to use (required if not in HDF5 attributes)",
    )
    parser.add_argument(
        "--robots",
        type=str,
        default="Panda",
        help="Name of the robot(s) to use (default: Panda)",
    )
    parser.add_argument(
        "--model-file",
        type=str,
        default=None,
        help="Path to model XML file (required if not in HDF5 attributes)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode without rendering",
    )
    args = parser.parse_args()

    demo_path = args.folder
    hdf5_path = os.path.join(demo_path, args.filename)

    f = h5py.File(hdf5_path, "r")
    
    # Try to get environment info from HDF5 attributes, fall back to command line args
    if "env" in f["data"].attrs and args.env_name is None:
        env_name = f["data"].attrs["env"]
    elif args.env_name is not None:
        env_name = args.env_name
    else:
        raise ValueError("Environment name not found in HDF5 attributes. Please provide --env-name argument.")
    
    if "env_info" in f["data"].attrs:
        env_info = json.loads(f["data"].attrs["env_info"])
        # Extract env_name from env_info if present
        if "env_name" in env_info:
            env_name = env_info.pop("env_name")
    else:
        env_info = {}

    # Map environment names to classes
    env_classes = {
        "Lift": Lift,
        "Stack": Stack,
        "PickPlace": PickPlace,
        "Door": Door,
    }
    
    if env_name not in env_classes:
        raise ValueError(f"Environment {env_name} not supported. Supported environments: {list(env_classes.keys())}")
    
    env_class = env_classes[env_name]
    env = env_class(
        robots=args.robots,
        **env_info,
        has_renderer=not args.headless,
        has_offscreen_renderer=args.headless,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    # list of all demonstrations episodes
    demos = list(f["data"].keys())

    while True:
        print("Playing back random episode... (press ESC to quit)")

        # select an episode randomly
        ep = random.choice(demos)

        # read the model xml, using the metadata stored in the attribute for this episode, or from command line
        if "model_file" in f["data/{}".format(ep)].attrs:
            model_xml = f["data/{}".format(ep)].attrs["model_file"]
            env.reset()
            xml = env.edit_model_xml(model_xml)
            env.reset_from_xml_string(xml)
            env.sim.reset()
            env.viewer.set_camera(0)
        elif args.model_file is not None:
            with open(args.model_file, 'r') as model_f:
                model_xml = model_f.read()
            env.reset()
            xml = env.edit_model_xml(model_xml)
            env.reset_from_xml_string(xml)
            env.sim.reset()
            env.viewer.set_camera(0)
        else:
            # No model file provided, use the environment's default model
            # Just reset the environment and hope the saved states are compatible
            env.reset()

        # load the flattened mujoco states
        states = f["data/{}/states".format(ep)][()]

        if args.use_actions:

            # load the initial state
            env.sim.set_state_from_flattened(states[0])
            env.sim.forward()

            # load the actions and play them back open-loop
            actions = np.array(f["data/{}/actions".format(ep)][()])
            num_actions = actions.shape[0]

            for j, action in enumerate(actions):
                env.step(action)
                env.render()

                if j < num_actions - 1:
                    # ensure that the actions deterministically lead to the same recorded states
                    state_playback = env.sim.get_state().flatten()
                    if not np.all(np.equal(states[j + 1], state_playback)):
                        err = np.linalg.norm(states[j + 1] - state_playback)
                        print(f"[warning] playback diverged by {err:.2f} for ep {ep} at step {j}")

        else:

            # force the sequence of internal mujoco states one by one
            for state in states:
                env.sim.set_state_from_flattened(state)
                env.sim.forward()
                if env.viewer is not None and env.renderer == "mjviewer":
                    env.viewer.update()
                if not args.headless:
                    env.render()

    f.close()
