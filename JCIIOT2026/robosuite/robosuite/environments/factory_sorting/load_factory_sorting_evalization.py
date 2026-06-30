"""
Evaluate a robomimic checkpoint in the same FactorySorting scene used for data collection.

This script follows the robomimic rollout pattern:
    policy.start_episode()
    obs = env.reset()
    action = policy(ob=obs)
    obs, reward, done, info = env.step(action)

The environment kwargs match load_factory_sorting_collect.py, while robomimic's
EnvRobosuite wrapper and config wrapper are used for policy-facing observations.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import robosuite as suite  # noqa: E402
from robosuite.controllers import load_composite_controller_config  # noqa: E402
from robosuite.environments.factory_sorting.factory_sorting_1_3fo3erfhisem import (  # noqa: E402,F401
    FactorySorting1_3FO3ERFHISEM,
)
from robosuite.environments.factory_sorting.factory_sorting_3_3fo3errph7x9 import (  # noqa: E402,F401
    FactorySorting3_3FO3ERRPH7X9,
)
from robosuite.environments.factory_sorting.factory_sorting_5_3fo3ertpxeut import (  # noqa: E402,F401
    FactorySorting5_3FO3ERTPXEUT,
)
from robosuite.environments.factory_sorting.factory_sorting_7_3fo3erfky9rn import (  # noqa: E402,F401
    FactorySorting7_3FO3ERFKY9RN,
)
from robosuite.environments.factory_sorting.factory_sorting_9_3fo3ert2c5fp import (  # noqa: E402,F401
    FactorySorting9_3FO3ERT2C5FP,
)


DEFAULT_CAMERA = "robot0_robotview"
DEFAULT_CAMERA_HEIGHT = 128
DEFAULT_CAMERA_WIDTH = 128
DEFAULT_CHECKPOINT = ROOT / "robosuite" / "model_epoch_500.pth"
DEFAULT_DEBUG_EVERY = 25
DEFAULT_EVAL_STEPS = 360
DEFAULT_FACTORY_SCENE = "factory_sorting_1_3fo3erfhisem"
DEFAULT_GRIPPER_TARGET_OFFSET = 0.035
DEFAULT_INITIAL_VIEW_STEPS = 30
DEFAULT_OBJECT_NAME = "line_5_container_h01_near"
DEFAULT_OBJECT_SITE_SIZE = 0.04
DEFAULT_POST_HOLD_STEPS = 10
DEFAULT_RENDER_SLEEP = 0.02
DEFAULT_ROBOT_BASE_ORI = [0.0, 0.0, 3.139422]
DEFAULT_ROBOT_BASE_POS = [8.000001, 4.600000, 0.0]

ARMS = ("right", "left")


FACTORY_SCENE_ENV_NAMES = {
    "factory_sorting": "FactorySorting",
    "FactorySorting": "FactorySorting",
    "factory_sorting_1_3fo3erfhisem": "FactorySorting1_3FO3ERFHISEM",
    "FactorySorting1_3FO3ERFHISEM": "FactorySorting1_3FO3ERFHISEM",
    "factory_sorting_3_3fo3errph7x9": "FactorySorting3_3FO3ERRPH7X9",
    "FactorySorting3_3FO3ERRPH7X9": "FactorySorting3_3FO3ERRPH7X9",
    "factory_sorting_5_3fo3ertpxeut": "FactorySorting5_3FO3ERTPXEUT",
    "FactorySorting5_3FO3ERTPXEUT": "FactorySorting5_3FO3ERTPXEUT",
    "factory_sorting_7_3fo3erfky9rn": "FactorySorting7_3FO3ERFKY9RN",
    "FactorySorting7_3FO3ERFKY9RN": "FactorySorting7_3FO3ERFKY9RN",
    "factory_sorting_9_3fo3ert2c5fp": "FactorySorting9_3FO3ERT2C5FP",
    "FactorySorting9_3FO3ERT2C5FP": "FactorySorting9_3FO3ERT2C5FP",
}


def factory_scene_env_name(args):
    scene = getattr(args, "factory_scene", DEFAULT_FACTORY_SCENE)
    try:
        return FACTORY_SCENE_ENV_NAMES[scene]
    except KeyError as exc:
        raise ValueError(
            f"Unknown factory scene '{scene}'. "
            f"Use one of {sorted(FACTORY_SCENE_ENV_NAMES)}."
        ) from exc


def default_object_name(env):
    if not getattr(env, "material_objects", None):
        raise RuntimeError("FactorySorting has no material objects.")
    return env.material_objects[0]


def site_pos(env, site_name):
    return np.array(env.sim.data.site_xpos[env.sim.model.site_name2id(site_name)])


def object_grasp_site_name(object_name, arm):
    return f"{object_name}_{arm}_grasp_site"


def object_site_names(object_name):
    return [object_grasp_site_name(object_name, arm) for arm in ARMS] + [f"{object_name}_center_site"]


def configure_object_site_markers(env, object_name, visible, site_size):
    positions = {}
    alpha = 1.0 if visible else 0.0
    size = np.full(3, site_size, dtype=float)
    for site_name in object_site_names(object_name):
        try:
            site_id = env.sim.model.site_name2id(site_name)
        except Exception:
            continue
        env.sim.model.site_rgba[site_id, 3] = alpha
        env.sim.model.site_size[site_id] = size
        positions[site_name] = site_pos(env, site_name)
    return positions


def object_collision_geoms(env, object_name):
    exact_name = f"{object_name}_collision"
    try:
        env.sim.model.geom_name2id(exact_name)
        return [exact_name]
    except Exception:
        pass

    prefix = f"{object_name}_"
    geoms = []
    for geom_id in range(env.sim.model.ngeom):
        geom_name = env.sim.model.geom_id2name(geom_id)
        if geom_name is None or not geom_name.startswith(prefix):
            continue
        if geom_name.endswith("_support"):
            continue
        geom_group = int(env.sim.model.geom_group[geom_id])
        if geom_group in {0, 3}:
            geoms.append(geom_name)

    if not geoms:
        raise RuntimeError(f"No collision geoms found for object '{object_name}'.")
    return geoms


def get_target_positions(env, object_name, site_below_offset):
    targets = {}
    site_names = {}
    for arm in ARMS:
        site_name = object_grasp_site_name(object_name, arm)
        try:
            env.sim.model.site_name2id(site_name)
        except Exception as exc:
            raise RuntimeError(f"Missing grasp site '{site_name}'.") from exc
        site_names[arm] = site_name
        targets[arm] = site_pos(env, site_name) - np.array([0.0, 0.0, site_below_offset])
    return targets, site_names


def gripper_end_center_pos(env, robot, arm):
    geom_positions = []
    gripper = robot.gripper[arm]
    important_geoms = getattr(gripper, "important_geoms", {})
    for group_name in ("left_fingerpad", "right_fingerpad"):
        for geom_name in important_geoms.get(group_name, []):
            try:
                geom_id = env.sim.model.geom_name2id(geom_name)
            except Exception:
                continue
            geom_positions.append(np.array(env.sim.data.geom_xpos[geom_id]))

    if geom_positions:
        return np.mean(geom_positions, axis=0)

    site_name = getattr(gripper, "important_sites", {}).get("grip_site")
    if site_name is not None:
        try:
            return site_pos(env, site_name)
        except Exception:
            pass
    return np.array(env.sim.data.site_xpos[robot.eef_site_id[arm]])


def contact_pairs_between(env, geoms_1, geoms_2):
    if isinstance(geoms_1, str):
        geoms_1 = [geoms_1]
    if isinstance(geoms_2, str):
        geoms_2 = [geoms_2]
    geoms_1 = set(geoms_1)
    geoms_2 = set(geoms_2)

    pairs = []
    for contact_idx in range(env.sim.data.ncon):
        contact = env.sim.data.contact[contact_idx]
        geom_1 = env.sim.model.geom_id2name(contact.geom1)
        geom_2 = env.sim.model.geom_id2name(contact.geom2)
        if geom_1 in geoms_1 and geom_2 in geoms_2:
            pairs.append((geom_1, geom_2))
        elif geom_2 in geoms_1 and geom_1 in geoms_2:
            pairs.append((geom_2, geom_1))
    return pairs


def fingerpad_contact_status(env, robot, object_name):
    geoms = object_collision_geoms(env, object_name)
    status = {}
    for arm in ARMS:
        gripper = robot.gripper[arm]
        important_geoms = getattr(gripper, "important_geoms", {})
        status[arm] = {}
        for group_name in ("left_fingerpad", "right_fingerpad"):
            group_geoms = important_geoms.get(group_name, [])
            status[arm][group_name] = bool(group_geoms and env.check_contact(group_geoms, geoms))
    return status


def fingerpad_contact_pairs(env, robot, object_name):
    geoms = object_collision_geoms(env, object_name)
    pairs = {}
    for arm in ARMS:
        gripper = robot.gripper[arm]
        important_geoms = getattr(gripper, "important_geoms", {})
        pairs[arm] = {}
        for group_name in ("left_fingerpad", "right_fingerpad"):
            group_geoms = important_geoms.get(group_name, [])
            pairs[arm][group_name] = contact_pairs_between(env, group_geoms, geoms) if group_geoms else []
    return pairs


def grasp_status(env, robot, object_name):
    geoms = object_collision_geoms(env, object_name)
    return {
        arm: bool(env._check_grasp(gripper=robot.gripper[arm], object_geoms=geoms))
        for arm in ARMS
    }


def print_grasp_debug_info(env, robot, object_name, goal_targets, label):
    positions = {arm: gripper_end_center_pos(env, robot, arm) for arm in ARMS}
    deltas = {arm: positions[arm] - goal_targets[arm] for arm in ARMS}
    distances = {arm: float(np.linalg.norm(deltas[arm])) for arm in ARMS}
    rounded_positions = {arm: np.round(pos, 4).tolist() for arm, pos in positions.items()}
    rounded_targets = {arm: np.round(target, 4).tolist() for arm, target in goal_targets.items()}
    rounded_deltas = {arm: np.round(delta, 4).tolist() for arm, delta in deltas.items()}
    object_geoms = object_collision_geoms(env, object_name)
    contacts = fingerpad_contact_status(env, robot, object_name)
    contact_pairs = fingerpad_contact_pairs(env, robot, object_name)
    grasps = grasp_status(env, robot, object_name)

    print(f"{label} object collision geoms: {object_geoms}")
    print(f"{label} gripper end targets: {rounded_targets}")
    print(f"{label} gripper end positions: {rounded_positions}")
    print(f"{label} gripper end deltas current-minus-target: {rounded_deltas}")
    print(f"{label} gripper end distances: {distances}")
    print(f"{label} fingerpad contact status: {contacts}")
    print(f"{label} fingerpad contact pairs: {contact_pairs}")
    print(f"{label} grasp status: {grasps}")
    return contacts, grasps


def base_robosuite_env(env):
    current = env
    seen = set()
    while True:
        if id(current) in seen:
            return current
        seen.add(id(current))

        if hasattr(current, "base_env"):
            current = current.base_env
            continue
        if hasattr(current, "env") and not hasattr(current, "sim"):
            current = current.env
            continue
        return current


def get_base_world_pose(env, robot=None):
    from robosuite.utils.transform_utils import mat2euler

    robot = robot or env.robots[0]
    site_name = robot.robot_model.base.correct_naming("center")
    site_id = env.sim.model.site_name2id(site_name)
    pos = np.array(env.sim.data.site_xpos[site_id], dtype=float)
    mat = env.sim.data.get_site_xmat(site_name)
    yaw = float(mat2euler(mat)[2])
    return pos[:2], yaw


def joint_state_by_names(env, joint_names):
    joint_names = list(joint_names)
    qpos_indexes = [env.sim.model.get_joint_qpos_addr(joint_name) for joint_name in joint_names]
    qvel_indexes = [env.sim.model.get_joint_qvel_addr(joint_name) for joint_name in joint_names]
    return {
        "joint_names": joint_names,
        "qpos": np.asarray(env.sim.data.qpos[qpos_indexes], dtype=float).tolist(),
        "qvel": np.asarray(env.sim.data.qvel[qvel_indexes], dtype=float).tolist(),
    }


def upper_body_joint_names(robot):
    joint_names = []
    joint_names.extend(getattr(robot, "robot_arm_joints", []))
    joint_names.extend(getattr(robot.robot_model, "torso_joints", []))
    joint_names.extend(getattr(robot.robot_model, "head_joints", []))
    for gripper_joint_names in getattr(robot, "gripper_joints", {}).values():
        joint_names.extend(gripper_joint_names)
    return joint_names


def object_site_positions(env, object_name):
    positions = {}
    for site_name in object_site_names(object_name):
        try:
            positions[site_name] = site_pos(env, site_name).tolist()
        except Exception:
            continue
    return positions


def capture_grasp_initial_state(env, object_name):
    raw_env = base_robosuite_env(env)
    robot = raw_env.robots[0]
    base_xy, base_yaw = get_base_world_pose(raw_env, robot)
    return {
        "version": "factory_sorting_grasp_init_state_v1",
        "object_name": object_name,
        "base_world_xy": base_xy.tolist(),
        "base_world_yaw": base_yaw,
        "base_joint_state": joint_state_by_names(raw_env, getattr(robot.robot_model, "base_joints", [])),
        "upper_body_joint_state": joint_state_by_names(raw_env, upper_body_joint_names(robot)),
        "eef_positions": {
            arm: gripper_end_center_pos(raw_env, robot, arm).tolist()
            for arm in ARMS
        },
        "object_site_positions": object_site_positions(raw_env, object_name),
    }


def save_grasp_initial_state(env, object_name, path):
    if path is None:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = capture_grasp_initial_state(env, object_name)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved_grasp_init_state: {path}")


def make_factory_sorting_env_kwargs(args):
    controller_config = load_composite_controller_config(controller=args.controller, robot="Tiago")
    env_name = factory_scene_env_name(args)
    kwargs = {
        "robots": "Tiago",
        "env_configuration": "single-robot",
        "controller_configs": controller_config,
        "gripper_types": args.gripper_types,
        "robot_base_pos": args.robot_base_pos,
        "robot_base_ori": args.robot_base_ori,
        "renderer": args.renderer,
        "render_camera": args.camera,
        "camera_names": args.camera,
        "camera_heights": args.camera_height,
        "camera_widths": args.camera_width,
        "camera_depths": False,
        "reward_shaping": False,
        "control_freq": 20,
        "seed": args.seed,
    }
    if env_name.startswith("FactorySorting") and env_name != "FactorySorting":
        kwargs["use_siemens_arena"] = True
        kwargs["include_material_objects"] = True
        kwargs["include_siemens_line_objects"] = False
        kwargs["include_legacy_static_scene"] = False
    return kwargs


def load_policy_and_config(args):
    try:
        import torch
        import robomimic.utils.file_utils as FileUtils
        import robomimic.utils.torch_utils as TorchUtils
    except ImportError as exc:
        raise RuntimeError("robomimic and torch are required to evaluate this checkpoint.") from exc

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    if args.device == "auto":
        device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    else:
        device = torch.device(args.device)

    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=str(args.checkpoint),
        device=device,
        verbose=args.verbose,
    )
    config, ckpt_dict = FileUtils.config_from_checkpoint(
        ckpt_dict=ckpt_dict,
        verbose=False,
    )
    print(f"Loaded robomimic checkpoint: {args.checkpoint}")
    print(f"Evaluation device: {device}")
    return policy, config, ckpt_dict


def load_factory_sorting_policy(checkpoint=DEFAULT_CHECKPOINT, device="auto", verbose=False):
    args = argparse.Namespace(
        checkpoint=Path(checkpoint),
        device=device,
        verbose=verbose,
    )
    return load_policy_and_config(args)


def policy_network(policy):
    candidates = [policy, getattr(policy, "policy", None)]
    for candidate in candidates:
        if candidate is None:
            continue
        nets = getattr(candidate, "nets", None)
        if nets is None:
            continue
        try:
            net = nets["policy"]
        except (KeyError, TypeError):
            continue
        return net
    return None


def policy_network_name(policy):
    net = policy_network(policy)
    return net.__class__.__name__ if net is not None else ""


def policy_requires_sequence_obs(policy):
    return "transformer" in policy_network_name(policy).lower()


def policy_sequence_context_length(policy, default=10):
    net = policy_network(policy)
    for attr_name in ("context_length", "seq_length", "sequence_length", "horizon", "max_seq_len"):
        value = getattr(net, attr_name, None) if net is not None else None
        if value is None:
            continue
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            continue
    return default


def policy_required_obs_keys(policy):
    net = policy_network(policy)
    input_shapes = getattr(net, "input_obs_group_shapes", None) if net is not None else None
    if isinstance(input_shapes, dict):
        obs_shapes = input_shapes.get("obs")
        if isinstance(obs_shapes, dict):
            return tuple(obs_shapes.keys())

    candidates = [getattr(policy, "policy", None), policy]
    for candidate in candidates:
        obs_key_shapes = getattr(candidate, "obs_key_shapes", None)
        if not isinstance(obs_key_shapes, dict):
            continue
        obs_shapes = obs_key_shapes.get("obs") if "obs" in obs_key_shapes else obs_key_shapes
        if isinstance(obs_shapes, dict):
            return tuple(obs_shapes.keys())
    return None


def filter_policy_obs(obs, obs_keys):
    if obs_keys is None:
        return obs
    missing = [key for key in obs_keys if key not in obs]
    if missing:
        raise RuntimeError(
            f"Policy requires missing observation keys: {missing}. "
            f"Available keys: {list(obs.keys())}"
        )
    return {key: obs[key] for key in obs_keys}


def stack_obs_history(obs_history, context_length):
    if not obs_history:
        raise RuntimeError("Cannot stack an empty observation history.")
    history = obs_history[-context_length:]
    if len(history) < context_length:
        history = [history[0]] * (context_length - len(history)) + history
    return {
        key: np.stack([np.asarray(obs[key]) for obs in history], axis=0)
        for key in history[-1]
    }


def make_eval_env(args, config, ckpt_dict, render):
    try:
        import robomimic.utils.env_utils as EnvUtils
        try:
            from robomimic.envs.env_robosuite import EnvRobosuite
        except ImportError:
            from robomimic.envs.env_robosuite import RobosuiteEnv as EnvRobosuite
    except ImportError as exc:
        raise RuntimeError("robomimic environment wrappers are required for official rollout evaluation.") from exc

    shape_meta = ckpt_dict["shape_metadata"]
    env_name = factory_scene_env_name(args)
    env = EnvRobosuite(
        env_name=env_name,
        render=render,
        render_offscreen=True,
        use_image_obs=shape_meta.get("use_images", True),
        use_depth_obs=shape_meta.get("use_depths", False),
        **make_factory_sorting_env_kwargs(args),
    )
    print(f"Evaluation factory scene: {env_name}")
    env = EnvUtils.wrap_env_from_config(env, config=config)
    return env


def render_frame(env, render, args):
    if not render:
        return
    camera_name = None if args.camera == "free" else args.camera
    env.render(mode="human", camera_name=camera_name)
    if args.render_sleep > 0:
        time.sleep(args.render_sleep)


def render_raw_env_frame(env, render, render_sleep):
    if not render:
        return
    env.render()
    if render_sleep > 0:
        time.sleep(render_sleep)


def render_frame_or_callback(env, render, args, render_callback=None):
    if render_callback is not None:
        render_callback()
        if args.render_sleep > 0:
            time.sleep(args.render_sleep)
        return
    render_frame(env, render=render, args=args)


def keep_viewer_open(env, render, args):
    if not render:
        print("Renderer disabled; process will exit after evaluation.")
        return

    print("Evaluation finished. Viewer will stay open. Press Ctrl+C in the terminal to exit.")
    try:
        while True:
            render_frame(env, render=True, args=args)
    except KeyboardInterrupt:
        print("Viewer loop interrupted by user.")


def print_reset_debug_info(raw_env, object_name, args):
    robot = raw_env.robots[0]
    below_site_targets, site_names = get_target_positions(raw_env, object_name, args.site_below_offset)
    object_site_positions = configure_object_site_markers(
        raw_env,
        object_name=object_name,
        visible=args.show_object_sites,
        site_size=args.object_site_size,
    )

    base_site_name = robot.robot_model.base.correct_naming("center")
    base_site_id = raw_env.sim.model.site_name2id(base_site_name)
    base_xy = np.array(raw_env.sim.data.site_xpos[base_site_id])[:2]

    print(f"Target object: {object_name}")
    print(f"Robot base xy at reset: ({base_xy[0]:.6f}, {base_xy[1]:.6f})")
    print(f"Tracking sites: {site_names}")
    if args.show_object_sites:
        rounded_sites = {name: np.round(pos, 4).tolist() for name, pos in object_site_positions.items()}
        print(f"Object site markers visible: {rounded_sites}")
    print(f"Below-site grasp targets: {below_site_targets}")
    return below_site_targets


def current_env_policy_obs(raw_env, obs_keys):
    obs = raw_env._get_observations(force_update=True)
    return filter_policy_obs(obs, obs_keys)


def current_wrapped_policy_obs(env):
    raw_env = base_robosuite_env(env)
    action_dim = getattr(env, "action_dimension", None)
    if action_dim is None:
        low, _ = raw_env.action_spec
        idle_action = np.zeros_like(low)
    else:
        idle_action = np.zeros(int(action_dim), dtype=float)

    result = env.step(idle_action)
    if isinstance(result, tuple) and len(result) == 4:
        obs, _, _, _ = result
        return obs

    raise RuntimeError(
        "Unable to read current policy observation from wrapped env without reset_to(). "
        f"Unexpected env.step result: {type(result)}"
    )


def step_env_action(env, action):
    result = env.step(action)
    if isinstance(result, tuple) and len(result) == 4:
        return result
    raise RuntimeError(f"Unexpected env.step result: {type(result)}")


def reset_policy_env_to_raw_env_state(policy_env, source_raw_env):
    obs = policy_env.reset()
    state_dict = policy_env.get_state()
    source_state = source_raw_env.sim.get_state()
    if isinstance(state_dict, dict) and "states" in state_dict and hasattr(source_state, "flatten"):
        source_state_vec = np.asarray(source_state.flatten(), dtype=float)
        policy_state_vec = np.asarray(state_dict["states"], dtype=float)
        if source_state_vec.size == policy_state_vec.size:
            state_dict = dict(state_dict)
            state_dict["states"] = source_state_vec
            obs = policy_env.reset_to(state_dict)
            print("policy_env_state_sync: current_env_sim_state")
            return obs
        print(
            "policy_env_state_sync_warning: "
            f"state_size_mismatch current={source_state_vec.size}, policy={policy_state_vec.size}; "
            "using policy_env reset state"
        )
    else:
        print("policy_env_state_sync_warning: unsupported state dict; using policy_env reset state")
    return policy_env.reset_to(state_dict)


def run_factory_sorting_grasp_on_env(
    env,
    policy,
    eval_steps=DEFAULT_EVAL_STEPS,
    debug_policy=False,
    debug_every=DEFAULT_DEBUG_EVERY,
    object_name=DEFAULT_OBJECT_NAME,
    site_below_offset=DEFAULT_GRIPPER_TARGET_OFFSET,
    post_hold_steps=DEFAULT_POST_HOLD_STEPS,
    initial_view_steps=DEFAULT_INITIAL_VIEW_STEPS,
    render=True,
    render_sleep=DEFAULT_RENDER_SLEEP,
    show_object_sites=False,
    object_site_size=DEFAULT_OBJECT_SITE_SIZE,
):
    raw_env = base_robosuite_env(env)
    robot = raw_env.robots[0]
    object_name = object_name or default_object_name(raw_env)
    args = argparse.Namespace(
        site_below_offset=site_below_offset,
        show_object_sites=show_object_sites,
        object_site_size=object_site_size,
    )

    if hasattr(policy, "start_episode"):
        policy.start_episode()

    below_site_targets = print_reset_debug_info(raw_env, object_name, args)
    obs_keys = policy_required_obs_keys(policy)
    use_sequence_obs = policy_requires_sequence_obs(policy)
    sequence_context_length = policy_sequence_context_length(policy) if use_sequence_obs else None
    print(f"Policy network: {policy_network_name(policy) or 'unknown'}")
    if obs_keys is not None:
        print(f"Policy obs keys: {list(obs_keys)}")
    print(f"Policy sequence obs enabled: {use_sequence_obs}")
    if sequence_context_length is not None:
        print(f"Policy sequence context length: {sequence_context_length}")
    print(f"Executing grasp policy on current environment for {eval_steps} steps")

    for _ in range(initial_view_steps):
        render_raw_env_frame(raw_env, render=render, render_sleep=render_sleep)

    low, _ = raw_env.action_spec
    expected_action_dim = int(np.asarray(low).size)
    last_action = None
    total_reward = 0.0
    obs_history = []

    for step in range(eval_steps):
        obs = current_env_policy_obs(raw_env, obs_keys)
        if use_sequence_obs:
            obs_history.append(obs)
            policy_obs = stack_obs_history(obs_history, context_length=sequence_context_length)
        else:
            policy_obs = obs
        action = np.asarray(policy(ob=policy_obs), dtype=float).reshape(-1)
        if action.size != expected_action_dim:
            raise RuntimeError(
                f"Policy action dimension mismatch: policy={action.size}, "
                f"current_env={expected_action_dim}"
            )
        last_action = action
        _, reward, done, _ = step_env_action(raw_env, action)
        total_reward += float(reward)
        render_raw_env_frame(raw_env, render=render, render_sleep=render_sleep)

        if debug_policy and step % debug_every == 0:
            print(
                f"grasp step={step}/{eval_steps} "
                f"action_norm={float(np.linalg.norm(action)):.3f}"
            )

        if done:
            print(f"Environment returned done=True during grasp at step {step}.")
            break

    if post_hold_steps > 0:
        hold_action = last_action if last_action is not None else np.zeros_like(low)
        print(f"Holding final grasp action for {post_hold_steps} steps")
        for _ in range(post_hold_steps):
            _, reward, _, _ = step_env_action(raw_env, hold_action)
            total_reward += float(reward)
            render_raw_env_frame(raw_env, render=render, render_sleep=render_sleep)

    _, grasps = print_grasp_debug_info(
        env=raw_env,
        robot=robot,
        object_name=object_name,
        goal_targets=below_site_targets,
        label="After current-env policy execution",
    )
    success = all(grasps.values())
    print(f"Current-env grasp return: {total_reward:.6f}")
    print(f"Current-env grasp success: {success}")
    return {
        "success": success,
        "successes": int(success),
        "num_rollouts": 1,
        "return": total_reward,
    }


def run_factory_sorting_grasp_with_policy_env(
    env,
    policy,
    config,
    ckpt_dict,
    factory_scene=DEFAULT_FACTORY_SCENE,
    eval_steps=DEFAULT_EVAL_STEPS,
    debug_policy=False,
    debug_every=DEFAULT_DEBUG_EVERY,
    object_name=DEFAULT_OBJECT_NAME,
    site_below_offset=DEFAULT_GRIPPER_TARGET_OFFSET,
    post_hold_steps=DEFAULT_POST_HOLD_STEPS,
    initial_view_steps=DEFAULT_INITIAL_VIEW_STEPS,
    render=True,
    render_sleep=DEFAULT_RENDER_SLEEP,
    camera_height=DEFAULT_CAMERA_HEIGHT,
    camera_width=DEFAULT_CAMERA_WIDTH,
    show_object_sites=False,
    object_site_size=DEFAULT_OBJECT_SITE_SIZE,
    robot_base_pos=None,
    robot_base_ori=None,
    renderer="mjviewer",
    camera=DEFAULT_CAMERA,
    controller=None,
    gripper_types="Robotiq140Gripper",
    seed=None,
):
    raw_env = base_robosuite_env(env)
    robot = raw_env.robots[0]
    object_name = object_name or default_object_name(raw_env)
    eval_args = argparse.Namespace(
        factory_scene=factory_scene,
        site_below_offset=site_below_offset,
        show_object_sites=show_object_sites,
        object_site_size=object_site_size,
        robot_base_pos=DEFAULT_ROBOT_BASE_POS if robot_base_pos is None else robot_base_pos,
        robot_base_ori=DEFAULT_ROBOT_BASE_ORI if robot_base_ori is None else robot_base_ori,
        renderer=renderer,
        camera=camera,
        camera_height=camera_height,
        camera_width=camera_width,
        controller=controller,
        gripper_types=gripper_types,
        seed=seed,
        render_sleep=render_sleep,
    )
    policy_env = make_eval_env(eval_args, config=config, ckpt_dict=ckpt_dict, render=False)

    try:
        if hasattr(policy, "start_episode"):
            policy.start_episode()
        obs = reset_policy_env_to_raw_env_state(policy_env, raw_env)

        policy_raw_env = base_robosuite_env(policy_env)
        below_site_targets = print_reset_debug_info(raw_env, object_name, eval_args)
        current_base_xy, current_yaw = get_base_world_pose(raw_env, robot)
        policy_base_xy, policy_yaw = get_base_world_pose(policy_raw_env, policy_raw_env.robots[0])
        print(
            "current_env_grasp_start_pose: "
            f"x={current_base_xy[0]:.6f}, y={current_base_xy[1]:.6f}, yaw={current_yaw:.6f}"
        )
        print(
            "policy_env_grasp_start_pose: "
            f"x={policy_base_xy[0]:.6f}, y={policy_base_xy[1]:.6f}, yaw={policy_yaw:.6f}"
        )
        print("Executing grasp policy with robomimic-wrapped observations")

        for _ in range(initial_view_steps):
            render_raw_env_frame(raw_env, render=render, render_sleep=render_sleep)

        low, _ = raw_env.action_spec
        expected_action_dim = int(np.asarray(low).size)
        last_action = None
        total_reward = 0.0

        for step in range(eval_steps):
            action = np.asarray(policy(ob=obs), dtype=float).reshape(-1)
            if action.size != expected_action_dim:
                raise RuntimeError(
                    f"Policy action dimension mismatch: policy={action.size}, "
                    f"current_env={expected_action_dim}"
                )

            last_action = action
            obs, _, policy_done, _ = policy_env.step(action)
            _, reward, current_done, _ = step_env_action(raw_env, action)
            total_reward += float(reward)
            render_raw_env_frame(raw_env, render=render, render_sleep=render_sleep)

            if debug_policy and step % debug_every == 0:
                print(
                    f"grasp step={step}/{eval_steps} "
                    f"action_norm={float(np.linalg.norm(action)):.3f}"
                )

            if policy_done:
                print(f"Policy env returned done=True during grasp at step {step}.")
                break
            if current_done:
                print(f"Current env returned done=True during grasp at step {step}.")
                break

        if post_hold_steps > 0:
            hold_action = last_action if last_action is not None else np.zeros_like(low)
            print(f"Holding final grasp action for {post_hold_steps} steps")
            for _ in range(post_hold_steps):
                obs, _, _, _ = policy_env.step(hold_action)
                _, reward, _, _ = step_env_action(raw_env, hold_action)
                total_reward += float(reward)
                render_raw_env_frame(raw_env, render=render, render_sleep=render_sleep)

        _, grasps = print_grasp_debug_info(
            env=raw_env,
            robot=robot,
            object_name=object_name,
            goal_targets=below_site_targets,
            label="After wrapped-policy execution on current env",
        )
        success = all(grasps.values())
        print(f"Wrapped-policy current-env grasp return: {total_reward:.6f}")
        print(f"Wrapped-policy current-env grasp success: {success}")
        return {
            "success": success,
            "successes": int(success),
            "num_rollouts": 1,
            "return": total_reward,
        }
    finally:
        if hasattr(policy_env, "close"):
            policy_env.close()
            gc.collect()


def run_factory_sorting_grasp_in_wrapped_env(
    env,
    policy,
    eval_steps=DEFAULT_EVAL_STEPS,
    debug_policy=False,
    debug_every=DEFAULT_DEBUG_EVERY,
    object_name=DEFAULT_OBJECT_NAME,
    site_below_offset=DEFAULT_GRIPPER_TARGET_OFFSET,
    post_hold_steps=DEFAULT_POST_HOLD_STEPS,
    initial_view_steps=DEFAULT_INITIAL_VIEW_STEPS,
    render=True,
    render_sleep=DEFAULT_RENDER_SLEEP,
    show_object_sites=False,
    object_site_size=DEFAULT_OBJECT_SITE_SIZE,
    camera=DEFAULT_CAMERA,
    render_callback=None,
):
    raw_env = base_robosuite_env(env)
    robot = raw_env.robots[0]
    object_name = object_name or default_object_name(raw_env)
    eval_args = argparse.Namespace(
        site_below_offset=site_below_offset,
        show_object_sites=show_object_sites,
        object_site_size=object_site_size,
        camera=camera,
        render_sleep=render_sleep,
    )

    if not hasattr(env, "step"):
        raise RuntimeError("run_factory_sorting_grasp_in_wrapped_env requires a robomimic EnvRobosuite wrapper.")

    if hasattr(policy, "start_episode"):
        policy.start_episode()

    obs = current_wrapped_policy_obs(env)
    below_site_targets = print_reset_debug_info(raw_env, object_name, eval_args)
    base_xy, yaw = get_base_world_pose(raw_env, robot)
    print(
        "wrapped_env_grasp_start_pose: "
        f"x={base_xy[0]:.6f}, y={base_xy[1]:.6f}, yaw={yaw:.6f}"
    )
    print("Executing grasp policy on the current robomimic-wrapped environment without reset_to")

    for _ in range(initial_view_steps):
        render_frame_or_callback(env, render=render, args=eval_args, render_callback=render_callback)

    low, _ = raw_env.action_spec
    expected_action_dim = int(np.asarray(low).size)
    last_action = None
    total_reward = 0.0

    for step in range(eval_steps):
        action = np.asarray(policy(ob=obs), dtype=float).reshape(-1)
        if action.size != expected_action_dim:
            raise RuntimeError(
                f"Policy action dimension mismatch: policy={action.size}, "
                f"current_env={expected_action_dim}"
            )

        last_action = action
        obs, reward, done, _ = env.step(action)
        total_reward += float(reward)
        render_frame_or_callback(env, render=render, args=eval_args, render_callback=render_callback)

        if debug_policy and step % debug_every == 0:
            print(
                f"grasp step={step}/{eval_steps} "
                f"action_norm={float(np.linalg.norm(action)):.3f}"
            )

        if done:
            print(f"Wrapped env returned done=True during grasp at step {step}.")
            break

    if post_hold_steps > 0:
        action_dim = getattr(env, "action_dimension", expected_action_dim)
        hold_action = last_action if last_action is not None else np.zeros(action_dim)
        print(f"Holding final grasp action for {post_hold_steps} steps")
        for _ in range(post_hold_steps):
            obs, reward, _, _ = env.step(hold_action)
            total_reward += float(reward)
            render_frame_or_callback(env, render=render, args=eval_args, render_callback=render_callback)

    _, grasps = print_grasp_debug_info(
        env=raw_env,
        robot=robot,
        object_name=object_name,
        goal_targets=below_site_targets,
        label="After same-env wrapped policy execution",
    )
    success = all(grasps.values())
    print(f"Same-env wrapped grasp return: {total_reward:.6f}")
    print(f"Same-env wrapped grasp success: {success}")
    return {
        "success": success,
        "successes": int(success),
        "num_rollouts": 1,
        "return": total_reward,
    }


def evaluate_once(env, policy, render, args, rollout_index):
    policy.start_episode()
    obs = env.reset()

    # This mirrors robomimic's run_trained_agent.py robosuite reset-to-state step.
    state_dict = env.get_state()
    obs = env.reset_to(state_dict)

    raw_env = base_robosuite_env(env)
    robot = raw_env.robots[0]
    object_name = args.object_name or default_object_name(raw_env)

    print(f"\nRollout {rollout_index + 1}/{args.num_rollouts}")
    below_site_targets = print_reset_debug_info(raw_env, object_name, args)
    save_grasp_initial_state(
        env=raw_env,
        object_name=object_name,
        path=getattr(args, "save_grasp_init_state", None),
    )

    for _ in range(args.initial_view_steps):
        render_frame(env, render=render, args=args)

    last_action = None
    total_reward = 0.0
    for step in range(args.eval_steps):
        action = policy(ob=obs)
        last_action = action
        obs, reward, done, info = env.step(action)
        total_reward += reward
        render_frame(env, render=render, args=args)

        if args.debug_policy and step % args.debug_every == 0:
            action_norm = float(np.linalg.norm(np.asarray(action, dtype=float).reshape(-1)))
            print(f"eval step={step}/{args.eval_steps} action_norm={action_norm:.3f}")

        if done:
            print(f"Environment returned done=True at step {step}.")
            break

    if args.post_hold_steps > 0:
        hold_action = last_action if last_action is not None else np.zeros(env.action_dimension)
        print(f"Holding final action for {args.post_hold_steps} steps")
        for _ in range(args.post_hold_steps):
            obs, reward, done, info = env.step(hold_action)
            total_reward += reward
            render_frame(env, render=render, args=args)

    _, grasps = print_grasp_debug_info(
        env=raw_env,
        robot=robot,
        object_name=object_name,
        goal_targets=below_site_targets,
        label="After policy execution",
    )
    success = all(grasps.values())
    print(f"Rollout return: {total_reward:.6f}")
    print(f"Final grasp success: {success}")
    return success


def run_factory_sorting_grasp_from_args(args, render=None, keep_viewer_open_after=False, close_env=True):
    render = not args.no_render if render is None else render

    if args.seed is not None:
        np.random.seed(args.seed)

    policy, config, ckpt_dict = load_policy_and_config(args)
    env = make_eval_env(args, config=config, ckpt_dict=ckpt_dict, render=render)

    successes = 0
    try:
        for rollout_idx in range(args.num_rollouts):
            successes += int(evaluate_once(env, policy, render=render, args=args, rollout_index=rollout_idx))

        print(f"\nAttempts: {args.num_rollouts}, grasp successes: {successes}")
        result = {
            "success": successes == args.num_rollouts,
            "successes": successes,
            "num_rollouts": args.num_rollouts,
        }

        if keep_viewer_open_after:
            keep_viewer_open(env, render=render, args=args)
        return result
    finally:
        if close_env and not keep_viewer_open_after and hasattr(env, "close"):
            env.close()
            gc.collect()


def run_factory_sorting_grasp(
    checkpoint=DEFAULT_CHECKPOINT,
    factory_scene=DEFAULT_FACTORY_SCENE,
    num_rollouts=1,
    eval_steps=DEFAULT_EVAL_STEPS,
    device="auto",
    debug_policy=False,
    debug_every=DEFAULT_DEBUG_EVERY,
    verbose=False,
    object_name=DEFAULT_OBJECT_NAME,
    site_below_offset=DEFAULT_GRIPPER_TARGET_OFFSET,
    post_hold_steps=DEFAULT_POST_HOLD_STEPS,
    initial_view_steps=DEFAULT_INITIAL_VIEW_STEPS,
    render_sleep=DEFAULT_RENDER_SLEEP,
    camera_height=DEFAULT_CAMERA_HEIGHT,
    camera_width=DEFAULT_CAMERA_WIDTH,
    show_object_sites=False,
    object_site_size=DEFAULT_OBJECT_SITE_SIZE,
    robot_base_pos=None,
    robot_base_ori=None,
    renderer="mjviewer",
    camera=DEFAULT_CAMERA,
    controller=None,
    gripper_types="Robotiq140Gripper",
    seed=None,
    render=True,
    keep_viewer_open_after=False,
    close_env=True,
):
    args = argparse.Namespace(
        checkpoint=Path(checkpoint),
        factory_scene=factory_scene,
        num_rollouts=num_rollouts,
        eval_steps=eval_steps,
        device=device,
        debug_policy=debug_policy,
        debug_every=debug_every,
        verbose=verbose,
        object_name=object_name,
        site_below_offset=site_below_offset,
        post_hold_steps=post_hold_steps,
        initial_view_steps=initial_view_steps,
        render_sleep=render_sleep,
        camera_height=camera_height,
        camera_width=camera_width,
        show_object_sites=show_object_sites,
        object_site_size=object_site_size,
        robot_base_pos=DEFAULT_ROBOT_BASE_POS if robot_base_pos is None else robot_base_pos,
        robot_base_ori=DEFAULT_ROBOT_BASE_ORI if robot_base_ori is None else robot_base_ori,
        renderer=renderer,
        camera=camera,
        controller=controller,
        gripper_types=gripper_types,
        seed=seed,
        no_render=not render,
        save_grasp_init_state=None,
    )
    return run_factory_sorting_grasp_from_args(
        args,
        render=render,
        keep_viewer_open_after=keep_viewer_open_after,
        close_env=close_env,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--factory-scene",
        choices=sorted(FACTORY_SCENE_ENV_NAMES),
        default=DEFAULT_FACTORY_SCENE,
    )
    parser.add_argument("--num-rollouts", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=DEFAULT_EVAL_STEPS)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--debug-policy", action="store_true")
    parser.add_argument("--debug-every", type=int, default=DEFAULT_DEBUG_EVERY)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--object-name", type=str, default=DEFAULT_OBJECT_NAME)
    parser.add_argument("--site-below-offset", type=float, default=DEFAULT_GRIPPER_TARGET_OFFSET)
    parser.add_argument("--post-hold-steps", type=int, default=DEFAULT_POST_HOLD_STEPS)
    parser.add_argument("--initial-view-steps", type=int, default=DEFAULT_INITIAL_VIEW_STEPS)
    parser.add_argument("--render-sleep", type=float, default=DEFAULT_RENDER_SLEEP)
    parser.add_argument("--camera-height", type=int, default=DEFAULT_CAMERA_HEIGHT)
    parser.add_argument("--camera-width", type=int, default=DEFAULT_CAMERA_WIDTH)
    parser.add_argument("--show-object-sites", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--object-site-size", type=float, default=DEFAULT_OBJECT_SITE_SIZE)
    parser.add_argument("--robot-base-pos", type=float, nargs=3, default=DEFAULT_ROBOT_BASE_POS)
    parser.add_argument("--robot-base-ori", type=float, nargs=3, default=DEFAULT_ROBOT_BASE_ORI)
    parser.add_argument("--renderer", type=str, default="mjviewer")
    parser.add_argument("--camera", type=str, default=DEFAULT_CAMERA)
    parser.add_argument("--controller", type=str, default=None)
    parser.add_argument("--gripper-types", type=str, default="Robotiq140Gripper")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-grasp-init-state", type=Path, default=None)
    parser.add_argument("--no-render", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    render = not args.no_render
    result = run_factory_sorting_grasp_from_args(
        args,
        render=render,
        keep_viewer_open_after=False,
        close_env=True,
    )
    print(f"Evaluation result: {'success' if result['success'] else 'failure'}")
    return bool(result["success"])


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
