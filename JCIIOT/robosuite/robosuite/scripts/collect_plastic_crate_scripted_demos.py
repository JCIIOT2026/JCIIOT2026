"""
Scripted demonstration collector for Tiago in TwoArmPlasticCrateLift.

The policy first opens the viewer, then moves both grippers in two stages:
horizontal XY alignment over the crate site points, followed by vertical
descent to fixed offsets below those sites. Rollouts are rejected if either
gripper touches the crate before closing. Successful trajectories, actions,
states, and current-frame camera images are saved to HDF5.
"""

import argparse
import datetime
import json
import os
import tempfile
import time
from glob import glob

import h5py
import numpy as np

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import DataCollectionWrapper

# Importing the module registers the environment class via robosuite's metaclass.
from robosuite.environments.manipulation.two_arm_plastic_crate_lift import TwoArmPlasticCrateLift  # noqa: F401


DEFAULT_NUM_ROLLOUTS = 20
DEFAULT_XY_STEPS = 120
DEFAULT_DOWN_STEPS = 80
DEFAULT_SITE_BELOW_OFFSET = 0.035
DEFAULT_ARRIVAL_TOLERANCE = 0.025
DEFAULT_GRASP_STEPS = 40
DEFAULT_POST_SUCCESS_HOLD_STEPS = 10
DEFAULT_MAX_ACTION = 0.65
DEFAULT_INITIAL_VIEW_STEPS = 30
DEFAULT_RENDER_SLEEP = 0.02
DEFAULT_CAMERA_HEIGHT = 128
DEFAULT_CAMERA_WIDTH = 128
DEFAULT_ROBOT_LATERAL_OFFSET_RANGE = 0.0
ROBOMIMIC_ROBOSUITE_ENV_TYPE = 1

ARMS = ("right", "left")
SITE_SUFFIXES = {
    "right": "right_grasp_set",
    "left": "left_grasp_set",
}


def find_site_name_by_suffix(sim, suffix):
    matches = []
    for site_id in range(sim.model.nsite):
        site_name = sim.model.site_id2name(site_id)
        if site_name is not None and site_name.endswith(suffix):
            matches.append(site_name)

    if len(matches) != 1:
        raise RuntimeError(f"Expected one site ending with '{suffix}', found: {matches}")
    return matches[0]


def get_target_positions(env, site_below_offset):
    targets = {}
    site_names = {}
    for arm, suffix in SITE_SUFFIXES.items():
        site_name = find_site_name_by_suffix(env.sim, suffix)
        site_names[arm] = site_name
        site_pos = np.array(env.sim.data.get_site_xpos(site_name))
        targets[arm] = site_pos - np.array([0.0, 0.0, site_below_offset])
    return targets, site_names


def get_eef_pos(env, robot, arm):
    return np.array(env.sim.data.site_xpos[robot.eef_site_id[arm]])


def get_crate(env):
    crate = getattr(env, "crate", None)
    if crate is None:
        crate = getattr(env, "pot", None)
    if crate is None:
        raise RuntimeError("Could not find crate object on environment.")
    return crate


def gripper_touches_crate(env, robot, crate):
    return any(env.check_contact(robot.gripper[arm], crate) for arm in ARMS)


def grippers_grasp_crate(env, robot, crate):
    return all(env._check_grasp(gripper=robot.gripper[arm], object_geoms=crate) for arm in ARMS)


def save_images_enabled(args):
    return not args.no_save_images


def image_obs_key(args):
    return f"{args.camera}_image"


def append_current_image(obs, image_buffer, args):
    if image_buffer is None:
        return

    key = image_obs_key(args)
    if key not in obs:
        raise RuntimeError(f"Camera image key '{key}' was not found in observation keys: {list(obs.keys())}")
    image_buffer.append(np.asarray(obs[key]))


def world_delta_to_controller_frame(robot, arm, world_delta):
    controller = robot.part_controllers[arm]
    input_ref_frame = getattr(controller, "input_ref_frame", "world")
    if input_ref_frame == "world":
        return world_delta
    if input_ref_frame == "base":
        origin_ori = controller.origin_ori
        if origin_ori is None:
            _, origin_ori = robot.composite_controller.get_controller_base_pose(controller_name=arm)
        return origin_ori.T @ world_delta
    raise RuntimeError(f"Unsupported input_ref_frame for {arm}: {input_ref_frame}")


def arm_delta_to_normalized_action(robot, arm, delta_pos, max_action):
    controller = robot.part_controllers[arm]
    if controller.name != "OSC_POSE" or controller.input_type != "delta":
        raise RuntimeError(
            f"This scripted policy expects {arm} to use OSC_POSE delta control; "
            f"got {controller.name} with input_type={controller.input_type}."
        )

    pos_scale = np.maximum(np.abs(controller.output_min[:3]), np.abs(controller.output_max[:3]))
    norm_pos = np.divide(delta_pos, pos_scale, out=np.zeros(3), where=pos_scale > 0)
    norm_pos = np.clip(norm_pos, -max_action, max_action)
    return np.concatenate([norm_pos, np.zeros(3)])


def optional_part_action(robot, part_name, fill=None):
    split = robot.composite_controller._action_split_indexes
    if part_name not in split:
        return None
    start, end = split[part_name]
    dim = end - start
    action = np.zeros(dim)
    if fill is not None:
        action[: min(dim, len(fill))] = fill[: min(dim, len(fill))]
    return action


def build_action(env, robot, arm_actions, gripper_value, head_table_view=True):
    action_dict = {}
    for arm in ARMS:
        action_dict[arm] = arm_actions.get(arm, np.zeros(6))
        dof = robot.gripper[arm].dof
        if dof > 0:
            action_dict[f"{arm}_gripper"] = np.full(dof, gripper_value)

    torso_action = optional_part_action(robot, "torso")
    if torso_action is not None:
        action_dict["torso"] = torso_action

    head_fill = np.array([0.0, -0.9]) if head_table_view else None
    head_action = optional_part_action(robot, "head", fill=head_fill)
    if head_action is not None:
        action_dict["head"] = head_action

    base_action = optional_part_action(robot, "base")
    if base_action is not None:
        action_dict["base"] = base_action

    return robot.create_action_vector(action_dict)


def hide_all_sites(env):
    vis_settings = {vis: False for vis in env._visualizations}
    env.visualize(vis_settings=vis_settings)
    env.sim.model.site_rgba[:, 3] = 0.0


def render_frame(env, render, args):
    if not render:
        return

    env.render()
    if args.render_sleep > 0:
        time.sleep(args.render_sleep)


def render_initial_scene(env, render, args):
    for _ in range(args.initial_view_steps):
        render_frame(env, render, args)


def step_towards_targets(env, base_env, robot, targets, max_action, gripper_value, render, args, image_buffer=None):
    robot.composite_controller.update_state()
    arm_actions = {}
    for arm in ARMS:
        world_delta = targets[arm] - get_eef_pos(base_env, robot, arm)
        controller_delta = world_delta_to_controller_frame(robot, arm, world_delta)
        arm_actions[arm] = arm_delta_to_normalized_action(
            robot=robot,
            arm=arm,
            delta_pos=controller_delta,
            max_action=max_action,
        )

    action = build_action(base_env, robot, arm_actions, gripper_value=gripper_value)
    obs, _, _, _ = env.step(action)
    append_current_image(obs, image_buffer, args)
    render_frame(env, render, args)


def rollout_once(env, render, args):
    env.reset()
    image_buffer = [] if save_images_enabled(args) else None
    base_env = env.unwrapped
    robot = base_env.robots[0]
    crate = get_crate(base_env)
    hide_all_sites(base_env)
    render_initial_scene(env, render, args)

    print(f"Robot lateral offset: {getattr(base_env, 'robot_lateral_offset', 0.0):.4f} m")
    final_targets, site_names = get_target_positions(base_env, args.site_below_offset)
    starts = {arm: get_eef_pos(base_env, robot, arm) for arm in ARMS}
    xy_targets = {
        arm: np.array([final_targets[arm][0], final_targets[arm][1], starts[arm][2]])
        for arm in ARMS
    }
    print(f"Tracking sites: {site_names}")
    print(f"XY targets above sites: {xy_targets}")
    print(f"Final targets below sites: {final_targets}")

    failed = False
    failure_reason = ""

    for step in range(1, args.xy_steps + 1):
        alpha = step / float(args.xy_steps)
        targets = {arm: starts[arm] + alpha * (xy_targets[arm] - starts[arm]) for arm in ARMS}
        step_towards_targets(
            env=env,
            base_env=base_env,
            robot=robot,
            targets=targets,
            max_action=args.max_action,
            gripper_value=-1.0,
            render=render,
            args=args,
            image_buffer=image_buffer,
        )
        if gripper_touches_crate(base_env, robot, crate):
            failed = True
            failure_reason = f"gripper touched crate during XY approach at step {step}"
            break

    if not failed:
        distances = {arm: np.linalg.norm(get_eef_pos(base_env, robot, arm) - xy_targets[arm]) for arm in ARMS}
        if any(dist > args.arrival_tolerance for dist in distances.values()):
            failed = True
            failure_reason = f"XY target tolerance failed: {distances}"

    if not failed:
        vertical_starts = {arm: get_eef_pos(base_env, robot, arm) for arm in ARMS}
        vertical_targets = {
            arm: np.array([vertical_starts[arm][0], vertical_starts[arm][1], final_targets[arm][2]])
            for arm in ARMS
        }
        for step in range(1, args.down_steps + 1):
            alpha = step / float(args.down_steps)
            targets = {
                arm: vertical_starts[arm] + alpha * (vertical_targets[arm] - vertical_starts[arm])
                for arm in ARMS
            }
            step_towards_targets(
                env=env,
                base_env=base_env,
                robot=robot,
                targets=targets,
                max_action=args.max_action,
                gripper_value=-1.0,
                render=render,
                args=args,
                image_buffer=image_buffer,
            )
            if gripper_touches_crate(base_env, robot, crate):
                failed = True
                failure_reason = f"gripper touched crate during vertical descent at step {step}"
                break

    if not failed:
        distances = {arm: np.linalg.norm(get_eef_pos(base_env, robot, arm) - final_targets[arm]) for arm in ARMS}
        if any(dist > args.arrival_tolerance for dist in distances.values()):
            failed = True
            failure_reason = f"final target tolerance failed: {distances}"

    if not failed:
        for _ in range(args.grasp_steps):
            action = build_action(base_env, robot, {}, gripper_value=1.0)
            obs, _, _, _ = env.step(action)
            append_current_image(obs, image_buffer, args)
            render_frame(env, render, args)

        success = grippers_grasp_crate(base_env, robot, crate)
        if success:
            env.successful = True
            for _ in range(args.post_success_hold_steps):
                action = build_action(base_env, robot, {}, gripper_value=1.0)
                obs, _, _, _ = env.step(action)
                append_current_image(obs, image_buffer, args)
                render_frame(env, render, args)
            return True, "success", env.ep_directory, image_buffer

        failed = True
        failure_reason = "grippers did not establish a grasp on the crate"

    env.successful = False
    return False, failure_reason, env.ep_directory, image_buffer


def make_robomimic_env_metadata(env_name, env_kwargs):
    return {
        "env_name": env_name,
        "env_version": suite.__version__,
        "type": ROBOMIMIC_ROBOSUITE_ENV_TYPE,
        "env_kwargs": env_kwargs,
    }


def gather_successful_demonstrations_as_hdf5(
    directory,
    out_dir,
    env_name,
    env_kwargs,
    policy_info,
    image_cache=None,
    image_dataset_key=None,
):
    os.makedirs(out_dir, exist_ok=True)
    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")
    grp = f.create_group("data")

    num_eps = 0
    collected_env_name = None
    for ep_directory in os.listdir(directory):
        ep_path = os.path.join(directory, ep_directory)
        if not os.path.isdir(ep_path):
            continue

        states = []
        actions = []
        successful = False
        for state_file in sorted(glob(os.path.join(ep_path, "state_*.npz"))):
            dic = np.load(state_file, allow_pickle=True)
            collected_env_name = str(dic["env"])
            states.extend(dic["states"])
            actions.extend(ai["actions"] for ai in dic["action_infos"])
            successful = successful or bool(dic["successful"])

        if len(states) == 0 or not successful:
            continue

        del states[-1]
        assert len(states) == len(actions)
        num_eps += 1

        ep_data_grp = grp.create_group(f"demo_{num_eps}")
        with open(os.path.join(ep_path, "model.xml"), "r") as xml_file:
            ep_data_grp.attrs["model_file"] = xml_file.read()
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))
        if image_cache is not None:
            images = image_cache.get(ep_path)
            if images is None:
                images = image_cache.get(os.path.normpath(ep_path))
            if images is None:
                raise RuntimeError(f"Missing cached images for successful episode: {ep_path}")
            images = np.asarray(images)
            if len(images) != len(actions):
                raise RuntimeError(
                    f"Image/action length mismatch for {ep_path}: images={len(images)}, actions={len(actions)}"
                )
            obs_grp = ep_data_grp.create_group("obs")
            obs_grp.create_dataset(image_dataset_key, data=images, compression="gzip", compression_opts=4)

    now = datetime.datetime.now()
    grp.attrs["date"] = f"{now.month}-{now.day}-{now.year}"
    grp.attrs["time"] = f"{now.hour}:{now.minute}:{now.second}"
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = collected_env_name if collected_env_name is not None else env_name
    grp.attrs["env_info"] = json.dumps(env_kwargs)
    grp.attrs["env_args"] = json.dumps(make_robomimic_env_metadata(env_name=env_name, env_kwargs=env_kwargs))
    grp.attrs["policy_info"] = json.dumps(policy_info)
    grp.attrs["num_successful_demos"] = num_eps
    f.close()
    return hdf5_path, num_eps


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-rollouts", type=int, default=DEFAULT_NUM_ROLLOUTS)
    parser.add_argument("--xy-steps", type=int, default=DEFAULT_XY_STEPS)
    parser.add_argument("--down-steps", type=int, default=DEFAULT_DOWN_STEPS)
    parser.add_argument("--site-below-offset", type=float, default=DEFAULT_SITE_BELOW_OFFSET)
    parser.add_argument("--arrival-tolerance", type=float, default=DEFAULT_ARRIVAL_TOLERANCE)
    parser.add_argument("--grasp-steps", type=int, default=DEFAULT_GRASP_STEPS)
    parser.add_argument("--post-success-hold-steps", type=int, default=DEFAULT_POST_SUCCESS_HOLD_STEPS)
    parser.add_argument("--max-action", type=float, default=DEFAULT_MAX_ACTION)
    parser.add_argument("--initial-view-steps", type=int, default=DEFAULT_INITIAL_VIEW_STEPS)
    parser.add_argument("--render-sleep", type=float, default=DEFAULT_RENDER_SLEEP)
    parser.add_argument("--camera-height", type=int, default=DEFAULT_CAMERA_HEIGHT)
    parser.add_argument("--camera-width", type=int, default=DEFAULT_CAMERA_WIDTH)
    parser.add_argument("--robot-lateral-offset-range", type=float, default=DEFAULT_ROBOT_LATERAL_OFFSET_RANGE)
    parser.add_argument("--directory", type=str, default=os.path.join(suite.models.assets_root, "demonstrations_private"))
    parser.add_argument("--output-name", type=str, default="scripted_plastic_crate")
    parser.add_argument("--renderer", type=str, default="mjviewer")
    parser.add_argument("--camera", type=str, default="robot0_robotview")
    parser.add_argument("--controller", type=str, default=None)
    parser.add_argument("--gripper-types", type=str, default="Robotiq140Gripper")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-save-images", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    render = not args.no_render
    save_images = save_images_enabled(args)

    env_name = "TwoArmPlasticCrateLift"
    controller_config = load_composite_controller_config(controller=args.controller, robot="Tiago")
    env_kwargs = {
        "robots": "Tiago",
        "env_configuration": "single-robot",
        "controller_configs": controller_config,
        "gripper_types": args.gripper_types,
        "robot_lateral_offset_range": args.robot_lateral_offset_range,
        "has_renderer": render,
        "renderer": args.renderer,
        "has_offscreen_renderer": save_images,
        "render_camera": args.camera,
        "camera_names": args.camera,
        "camera_heights": args.camera_height,
        "camera_widths": args.camera_width,
        "camera_depths": False,
        "ignore_done": True,
        "use_camera_obs": save_images,
        "reward_shaping": True,
        "control_freq": 20,
    }
    dataset_env_kwargs = dict(env_kwargs)
    dataset_env_kwargs.update(
        {
            "has_renderer": False,
        }
    )

    raw_env = suite.make(
        env_name=env_name,
        **env_kwargs,
    )

    tmp_directory = tempfile.mkdtemp(prefix="scripted_crate_raw_")
    env = DataCollectionWrapper(raw_env, tmp_directory, collect_freq=1, flush_freq=1000)

    timestamp = str(time.time()).replace(".", "_")
    out_dir = os.path.join(args.directory, f"{args.output_name}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    successes = 0
    image_cache = {} if save_images else None
    for rollout_idx in range(args.num_rollouts):
        print(f"\nRollout {rollout_idx + 1}/{args.num_rollouts}")
        success, reason, ep_directory, images = rollout_once(env, render=render, args=args)
        successes += int(success)
        if success and image_cache is not None:
            image_cache[os.path.normpath(ep_directory)] = images
        print(f"Result: {reason}")

    env.close()

    hdf5_path, num_saved = gather_successful_demonstrations_as_hdf5(
        tmp_directory,
        out_dir,
        env_name=env_name,
        env_kwargs=dataset_env_kwargs,
        policy_info=vars(args),
        image_cache=image_cache,
        image_dataset_key=image_obs_key(args) if save_images else None,
    )
    print(f"\nAttempts: {args.num_rollouts}, successes: {successes}, saved demos: {num_saved}")
    print(f"HDF5 saved to: {hdf5_path}")
    print(f"Raw trajectory directory: {tmp_directory}")


if __name__ == "__main__":
    main()
