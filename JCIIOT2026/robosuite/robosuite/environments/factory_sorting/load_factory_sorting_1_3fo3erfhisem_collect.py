"""
Collect scripted FactorySorting grasp demonstrations for robomimic.

The robot starts from the navigation pose recorded by llm_task_navigator.py,
then executes a two-arm side grasp around a selected input object. Successful
episodes are converted into a robomimic-style HDF5 with low-dimensional Tiago
EEF / gripper observations and robot0_robotview images.
"""

import argparse
import datetime
import json
import os
import sys
import tempfile
import time
from glob import glob
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import robosuite as suite  # noqa: E402
from robosuite.controllers import load_composite_controller_config  # noqa: E402
from robosuite.environments.factory_sorting.factory_sorting_1_3fo3erfhisem import (  # noqa: E402
    FactorySorting1_3FO3ERFHISEM,
)
from robosuite.wrappers import DataCollectionWrapper  # noqa: E402


DEFAULT_ENV_NAME = FactorySorting1_3FO3ERFHISEM.__name__
DEFAULT_NUM_ROLLOUTS = 20
DEFAULT_UP_STEPS = 60
DEFAULT_XY_STEPS = 120
DEFAULT_DOWN_STEPS = 80
DEFAULT_SAFE_Z = 0.10
DEFAULT_SITE_ABOVE_CLEARANCE = 0.05
DEFAULT_SITE_BELOW_OFFSET = 0.035
DEFAULT_ARRIVAL_TOLERANCE = 0.025
DEFAULT_GRIPPER_END_ARRIVAL_TOLERANCE = 0.03
DEFAULT_SETTLE_STEPS = 80
DEFAULT_GRASP_STEPS = 40
DEFAULT_POST_SUCCESS_HOLD_STEPS = 10
DEFAULT_MAX_ACTION = 0.65
DEFAULT_INITIAL_VIEW_STEPS = 30
DEFAULT_RENDER_SLEEP = 0.02
DEFAULT_CAMERA = "robot0_robotview"
DEFAULT_CAMERA_HEIGHT = 128
DEFAULT_CAMERA_WIDTH = 128
DEFAULT_OBJECT_SITE_SIZE = 0.04
DEFAULT_OBJECT_NAME = "line_5_container_h01_near"
DEFAULT_ROBOT_BASE_POS = [8.000001, 4.600000, 0.0]
DEFAULT_ROBOT_BASE_ORI = [0.0, 0.0, 3.139422]
ROBOMIMIC_ROBOSUITE_ENV_TYPE = 1

ARMS = ("right", "left")
CAMERA_HOLD_PARTS = ("torso", "head")
CAMERA_HOLD_TARGET_ATTR = "_factory_sorting_camera_hold_targets"
OBS_KEYS = (
    "robot0_left_eef_pos",
    "robot0_left_eef_quat",
    "robot0_left_gripper_qpos",
    "robot0_right_eef_pos",
    "robot0_right_eef_quat",
    "robot0_right_gripper_qpos",
    "robot0_robotview_image",
)


def get_eef_pos(env, robot, arm):
    return np.array(env.sim.data.site_xpos[robot.eef_site_id[arm]])


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
    return get_eef_pos(env, robot, arm)


def site_pos(env, site_name):
    return np.array(env.sim.data.site_xpos[env.sim.model.site_name2id(site_name)])


def default_object_name(env):
    if not getattr(env, "material_objects", None):
        raise RuntimeError("FactorySorting has no material objects.")
    return env.material_objects[0]


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


def sync_collection_model_xml(wrapper_env, base_env):
    if hasattr(wrapper_env, "_current_task_instance_xml"):
        wrapper_env._current_task_instance_xml = base_env.sim.model.get_xml()


def get_target_positions(env, object_name, site_below_offset):
    targets = {}
    site_names = {}
    for arm in ARMS:
        site_name = object_grasp_site_name(object_name, arm)
        try:
            env.sim.model.site_name2id(site_name)
        except Exception as exc:
            raise RuntimeError(
                f"Missing grasp site '{site_name}'. Check that FactorySorting adds object grasp sites."
            ) from exc
        site_names[arm] = site_name
        targets[arm] = site_pos(env, site_name) - np.array([0.0, 0.0, site_below_offset])
    return targets, site_names


def gripper_touches_object(env, robot, object_name):
    geoms = object_collision_geoms(env, object_name)
    return any(env.check_contact(robot.gripper[arm], geoms) for arm in ARMS)


def grippers_grasp_object(env, robot, object_name):
    geoms = object_collision_geoms(env, object_name)
    return all(env._check_grasp(gripper=robot.gripper[arm], object_geoms=geoms) for arm in ARMS)


def grasp_status(env, robot, object_name):
    geoms = object_collision_geoms(env, object_name)
    return {
        arm: bool(env._check_grasp(gripper=robot.gripper[arm], object_geoms=geoms))
        for arm in ARMS
    }


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


def print_grasp_debug_info(env, robot, object_name, goal_targets, label):
    positions = {
        arm: gripper_end_center_pos(env, robot, arm)
        for arm in ARMS
    }
    deltas = {
        arm: positions[arm] - goal_targets[arm]
        for arm in ARMS
    }
    distances = {
        arm: float(np.linalg.norm(deltas[arm]))
        for arm in ARMS
    }
    rounded_positions = {
        arm: np.round(pos, 4).tolist()
        for arm, pos in positions.items()
    }
    rounded_targets = {
        arm: np.round(target, 4).tolist()
        for arm, target in goal_targets.items()
    }
    rounded_deltas = {
        arm: np.round(delta, 4).tolist()
        for arm, delta in deltas.items()
    }
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
        fill = np.asarray(fill, dtype=float)
        action[: min(dim, len(fill))] = fill[: min(dim, len(fill))]
    return action


def current_part_qpos(robot, part_name):
    controller = robot.part_controllers.get(part_name)
    if controller is None:
        return None
    controller.update()
    if getattr(controller, "joint_pos", None) is not None:
        return np.asarray(controller.joint_pos, dtype=float).copy()

    qpos_index = getattr(controller, "qpos_index", None)
    if qpos_index is None:
        return None
    return np.asarray(controller.sim.data.qpos[qpos_index], dtype=float).copy()


def capture_camera_hold_targets(robot):
    targets = {}
    for part_name in CAMERA_HOLD_PARTS:
        if part_name not in robot.composite_controller._action_split_indexes:
            continue
        qpos = current_part_qpos(robot, part_name)
        if qpos is not None:
            targets[part_name] = qpos
    return targets


def camera_hold_part_action(robot, part_name):
    targets = getattr(robot, CAMERA_HOLD_TARGET_ATTR, {})
    fill = targets.get(part_name)
    if fill is None:
        fill = current_part_qpos(robot, part_name)
    return optional_part_action(robot, part_name, fill=fill)


def build_action(env, robot, arm_actions, gripper_value):
    action_dict = {}
    for arm in ARMS:
        action_dict[arm] = arm_actions.get(arm, np.zeros(6))
        dof = robot.gripper[arm].dof
        if dof > 0:
            action_dict[f"{arm}_gripper"] = np.full(dof, gripper_value)

    torso_action = camera_hold_part_action(robot, "torso")
    if torso_action is not None:
        action_dict["torso"] = torso_action

    head_action = camera_hold_part_action(robot, "head")
    if head_action is not None:
        action_dict["head"] = head_action

    base_action = optional_part_action(robot, "base")
    if base_action is not None:
        action_dict["base"] = base_action

    return robot.create_action_vector(action_dict)


def render_frame(env, render, args):
    if not render:
        return
    env.render()
    if args.render_sleep > 0:
        time.sleep(args.render_sleep)


def render_initial_scene(env, render, args):
    if not render:
        return
    for _ in range(args.initial_view_steps):
        render_frame(env, render, args)


def append_current_obs(base_env, obs_buffer):
    obs = base_env._get_observations(force_update=True)
    missing = [key for key in OBS_KEYS if key not in obs]
    if missing:
        raise RuntimeError(f"Missing observation keys {missing}. Available keys: {list(obs.keys())}")
    for key in OBS_KEYS:
        obs_buffer[key].append(np.asarray(obs[key]))


def step_with_record(env, base_env, action, obs_buffer, render, args):
    append_current_obs(base_env, obs_buffer)
    env.step(action)
    render_frame(env, render, args)


def step_towards_targets(
    env,
    base_env,
    robot,
    targets,
    max_action,
    gripper_value,
    render,
    args,
    obs_buffer,
):
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
    step_with_record(env, base_env, action, obs_buffer, render, args)


def step_towards_gripper_end_targets(
    env,
    base_env,
    robot,
    targets,
    max_action,
    gripper_value,
    render,
    args,
    obs_buffer,
):
    robot.composite_controller.update_state()
    arm_actions = {}
    for arm in ARMS:
        world_delta = targets[arm] - gripper_end_center_pos(base_env, robot, arm)
        controller_delta = world_delta_to_controller_frame(robot, arm, world_delta)
        arm_actions[arm] = arm_delta_to_normalized_action(
            robot=robot,
            arm=arm,
            delta_pos=controller_delta,
            max_action=max_action,
        )

    action = build_action(base_env, robot, arm_actions, gripper_value=gripper_value)
    step_with_record(env, base_env, action, obs_buffer, render, args)


def move_along_linear_segment(
    env,
    base_env,
    robot,
    object_name,
    goal_targets,
    num_steps,
    gripper_value,
    render,
    args,
    obs_buffer,
    reject_object_contact=False,
    check_arrival=True,
    label="segment",
):
    num_steps = max(1, int(num_steps))
    starts = {arm: get_eef_pos(base_env, robot, arm) for arm in ARMS}
    for step in range(1, num_steps + 1):
        alpha = step / float(num_steps)
        targets = {arm: starts[arm] + alpha * (goal_targets[arm] - starts[arm]) for arm in ARMS}
        step_towards_targets(
            env=env,
            base_env=base_env,
            robot=robot,
            targets=targets,
            max_action=args.max_action,
            gripper_value=gripper_value,
            render=render,
            args=args,
            obs_buffer=obs_buffer,
        )
        if reject_object_contact and gripper_touches_object(base_env, robot, object_name):
            return False, f"gripper touched object during {label} at step {step}"

    if check_arrival:
        distances = {
            arm: np.linalg.norm(get_eef_pos(base_env, robot, arm) - goal_targets[arm])
            for arm in ARMS
        }
        for settle_step in range(1, args.settle_steps + 1):
            if all(dist <= args.arrival_tolerance for dist in distances.values()):
                break
            step_towards_targets(
                env=env,
                base_env=base_env,
                robot=robot,
                targets=goal_targets,
                max_action=args.max_action,
                gripper_value=gripper_value,
                render=render,
                args=args,
                obs_buffer=obs_buffer,
            )
            if reject_object_contact and gripper_touches_object(base_env, robot, object_name):
                return False, f"gripper touched object during {label} settle at step {settle_step}"
            distances = {
                arm: np.linalg.norm(get_eef_pos(base_env, robot, arm) - goal_targets[arm])
                for arm in ARMS
            }

        if any(dist > args.arrival_tolerance for dist in distances.values()):
            return False, f"{label} target tolerance failed after settle: {distances}"
    return True, ""


def move_vertically_below_sites(
    env,
    base_env,
    robot,
    goal_targets,
    site_positions,
    num_steps,
    gripper_value,
    render,
    args,
    obs_buffer,
    label="vertical descent below sites",
):
    num_steps = max(1, int(num_steps))
    starts = {arm: get_eef_pos(base_env, robot, arm) for arm in ARMS}
    for step in range(1, num_steps + 1):
        alpha = step / float(num_steps)
        targets = {
            arm: np.array(
                [
                    site_positions[arm][0],
                    site_positions[arm][1],
                    starts[arm][2] + alpha * (goal_targets[arm][2] - starts[arm][2]),
                ]
            )
            for arm in ARMS
        }
        step_towards_targets(
            env=env,
            base_env=base_env,
            robot=robot,
            targets=targets,
            max_action=args.max_action,
            gripper_value=gripper_value,
            render=render,
            args=args,
            obs_buffer=obs_buffer,
        )

    distances = {
        arm: np.linalg.norm(get_eef_pos(base_env, robot, arm) - goal_targets[arm])
        for arm in ARMS
    }
    for settle_step in range(1, args.settle_steps + 1):
        if all(dist <= args.arrival_tolerance for dist in distances.values()):
            break
        step_towards_targets(
            env=env,
            base_env=base_env,
            robot=robot,
            targets=goal_targets,
            max_action=args.max_action,
            gripper_value=gripper_value,
            render=render,
            args=args,
            obs_buffer=obs_buffer,
        )
        distances = {
            arm: np.linalg.norm(get_eef_pos(base_env, robot, arm) - goal_targets[arm])
            for arm in ARMS
        }

    if any(dist > args.arrival_tolerance for dist in distances.values()):
        return False, f"{label} target tolerance failed after settle: {distances}"
    return True, ""


def settle_gripper_end_centers_at_targets(
    env,
    base_env,
    robot,
    goal_targets,
    gripper_value,
    render,
    args,
    obs_buffer,
    label="gripper end center arrival",
):
    distances = {
        arm: np.linalg.norm(gripper_end_center_pos(base_env, robot, arm) - goal_targets[arm])
        for arm in ARMS
    }
    for settle_step in range(1, args.settle_steps + 1):
        if all(dist <= args.gripper_end_arrival_tolerance for dist in distances.values()):
            break
        step_towards_gripper_end_targets(
            env=env,
            base_env=base_env,
            robot=robot,
            targets=goal_targets,
            max_action=args.max_action,
            gripper_value=gripper_value,
            render=render,
            args=args,
            obs_buffer=obs_buffer,
        )
        distances = {
            arm: np.linalg.norm(gripper_end_center_pos(base_env, robot, arm) - goal_targets[arm])
            for arm in ARMS
        }

    positions = {
        arm: gripper_end_center_pos(base_env, robot, arm)
        for arm in ARMS
    }
    if any(dist > args.gripper_end_arrival_tolerance for dist in distances.values()):
        rounded_positions = {
            arm: np.round(pos, 4).tolist()
            for arm, pos in positions.items()
        }
        return (
            False,
            f"{label} failed before grasp: distances={distances}, positions={rounded_positions}",
        )
    print(f"{label} distances before grasp: {distances}")
    return True, ""


def make_obs_buffer():
    return {key: [] for key in OBS_KEYS}


def rollout_once(env, render, args):
    env.reset()
    base_env = env.unwrapped
    robot = base_env.robots[0]
    setattr(robot, CAMERA_HOLD_TARGET_ATTR, capture_camera_hold_targets(robot))
    object_name = args.object_name or default_object_name(base_env)
    obs_buffer = make_obs_buffer()
    object_site_positions = configure_object_site_markers(
        base_env,
        object_name=object_name,
        visible=args.show_object_sites,
        site_size=args.object_site_size,
    )
    sync_collection_model_xml(env, base_env)

    if args.camera != "free" and render and env.viewer is not None:
        env.viewer.set_camera(camera_id=base_env.sim.model.camera_name2id(args.camera))

    render_initial_scene(env, render, args)

    below_site_targets, site_names = get_target_positions(base_env, object_name, args.site_below_offset)
    starts = {arm: get_eef_pos(base_env, robot, arm) for arm in ARMS}
    site_positions = {
        arm: below_site_targets[arm] + np.array([0.0, 0.0, args.site_below_offset])
        for arm in ARMS
    }
    safe_z = max(
        args.safe_z,
        max(starts[arm][2] for arm in ARMS),
        max(site_positions[arm][2] + args.site_above_clearance for arm in ARMS),
    )
    safe_targets = {arm: np.array([starts[arm][0], starts[arm][1], safe_z]) for arm in ARMS}
    xy_targets = {
        arm: np.array([site_positions[arm][0], site_positions[arm][1], safe_z])
        for arm in ARMS
    }

    base_site_name = robot.robot_model.base.correct_naming("center")
    base_site_id = base_env.sim.model.site_name2id(base_site_name)
    base_xy = np.array(base_env.sim.data.site_xpos[base_site_id])[:2]
    print(f"Target object: {object_name}")
    print(f"Robot base xy at reset: ({base_xy[0]:.6f}, {base_xy[1]:.6f})")
    print(f"Tracking sites: {site_names}")
    if args.show_object_sites:
        rounded_sites = {
            name: np.round(pos, 4).tolist()
            for name, pos in object_site_positions.items()
        }
        print(f"Object site markers visible: {rounded_sites}")
    print(f"Site positions: {site_positions}")
    print(f"Below-site grasp targets: {below_site_targets}")

    failed = False
    failure_reason = ""

    ok, reason = move_along_linear_segment(
        env=env,
        base_env=base_env,
        robot=robot,
        object_name=object_name,
        goal_targets=safe_targets,
        num_steps=args.up_steps,
        gripper_value=-1.0,
        render=render,
        args=args,
        obs_buffer=obs_buffer,
        reject_object_contact=True,
        label="safe vertical lift",
    )
    if not ok:
        failed = True
        failure_reason = reason

    if not failed:
        ok, reason = move_along_linear_segment(
            env=env,
            base_env=base_env,
            robot=robot,
            object_name=object_name,
            goal_targets=xy_targets,
            num_steps=args.xy_steps,
            gripper_value=-1.0,
            render=render,
            args=args,
            obs_buffer=obs_buffer,
            reject_object_contact=True,
            label="XY approach",
        )
        if not ok:
            failed = True
            failure_reason = reason

    if not failed:
        ok, reason = move_vertically_below_sites(
            env=env,
            base_env=base_env,
            robot=robot,
            goal_targets=below_site_targets,
            site_positions=site_positions,
            num_steps=args.down_steps,
            gripper_value=-1.0,
            render=render,
            args=args,
            obs_buffer=obs_buffer,
            label="vertical descent below sites",
        )
        if not ok:
            failed = True
            failure_reason = reason

    if not failed:
        ok, reason = settle_gripper_end_centers_at_targets(
            env=env,
            base_env=base_env,
            robot=robot,
            goal_targets=below_site_targets,
            gripper_value=-1.0,
            render=render,
            args=args,
            obs_buffer=obs_buffer,
            label="gripper end center arrival",
        )
        if not ok:
            failed = True
            failure_reason = reason

    if not failed:
        print_grasp_debug_info(
            env=base_env,
            robot=robot,
            object_name=object_name,
            goal_targets=below_site_targets,
            label="Before grasp close",
        )
        for _ in range(args.grasp_steps):
            action = build_action(base_env, robot, {}, gripper_value=1.0)
            step_with_record(env, base_env, action, obs_buffer, render, args)

        post_contact_status, post_grasp_status = print_grasp_debug_info(
            env=base_env,
            robot=robot,
            object_name=object_name,
            goal_targets=below_site_targets,
            label="After grasp close",
        )
        if not all(post_grasp_status.values()):
            failed = True
            failure_reason = (
                "both grippers did not establish a grasp on the object: "
                f"grasp_status={post_grasp_status}, "
                f"fingerpad_contact_status={post_contact_status}"
            )

    if not failed:
        env.successful = True
        for _ in range(args.post_success_hold_steps):
            action = build_action(base_env, robot, {}, gripper_value=1.0)
            step_with_record(env, base_env, action, obs_buffer, render, args)
        return True, "success: both grippers grasped and held the object", env.ep_directory, obs_buffer

    env.successful = False
    return False, failure_reason, env.ep_directory, obs_buffer


def make_robomimic_env_metadata(env_name, env_kwargs):
    return {
        "env_name": env_name,
        "env_version": suite.__version__,
        "type": ROBOMIMIC_ROBOSUITE_ENV_TYPE,
        "env_kwargs": env_kwargs,
    }


def write_obs_group(ep_data_grp, obs_data, num_actions):
    obs_grp = ep_data_grp.create_group("obs")
    for key in OBS_KEYS:
        values = np.asarray(obs_data[key])
        if len(values) != num_actions:
            raise RuntimeError(
                f"Obs/action length mismatch for {key}: obs={len(values)}, actions={num_actions}"
            )
        if key.endswith("_image"):
            obs_grp.create_dataset(key, data=values, compression="gzip", compression_opts=4)
        else:
            obs_grp.create_dataset(key, data=values)


def gather_successful_demonstrations_as_hdf5(
    directory,
    out_dir,
    hdf5_name,
    env_name,
    env_kwargs,
    policy_info,
    obs_cache,
):
    os.makedirs(out_dir, exist_ok=True)
    hdf5_path = os.path.join(out_dir, hdf5_name)
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

        obs_data = obs_cache.get(os.path.normpath(ep_path))
        if obs_data is None:
            raise RuntimeError(f"Missing cached observations for successful episode: {ep_path}")

        ep_data_grp = grp.create_group(f"demo_{num_eps}")
        with open(os.path.join(ep_path, "model.xml"), "r") as xml_file:
            ep_data_grp.attrs["model_file"] = xml_file.read()
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))
        write_obs_group(ep_data_grp, obs_data, len(actions))

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


def make_env_kwargs(args, render):
    controller_config = load_composite_controller_config(controller=args.controller, robot="Tiago")
    return {
        "robots": "Tiago",
        "env_configuration": "single-robot",
        "controller_configs": controller_config,
        "gripper_types": args.gripper_types,
        "robot_base_pos": args.robot_base_pos,
        "robot_base_ori": args.robot_base_ori,
        "use_siemens_arena": True,
        "include_legacy_static_scene": False,
        "include_material_objects": False,
        "include_siemens_line_objects": False,
        "has_renderer": render,
        "renderer": args.renderer,
        "has_offscreen_renderer": True,
        "render_camera": args.camera,
        "camera_names": args.camera,
        "camera_heights": args.camera_height,
        "camera_widths": args.camera_width,
        "camera_depths": False,
        "ignore_done": True,
        "use_camera_obs": True,
        "use_object_obs": True,
        "reward_shaping": False,
        "control_freq": 20,
        "seed": args.seed,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-rollouts", type=int, default=DEFAULT_NUM_ROLLOUTS)
    parser.add_argument("--object-name", type=str, default=DEFAULT_OBJECT_NAME)
    parser.add_argument("--up-steps", type=int, default=DEFAULT_UP_STEPS)
    parser.add_argument("--xy-steps", type=int, default=DEFAULT_XY_STEPS)
    parser.add_argument("--down-steps", type=int, default=DEFAULT_DOWN_STEPS)
    parser.add_argument("--safe-z", type=float, default=DEFAULT_SAFE_Z)
    parser.add_argument("--site-above-clearance", type=float, default=DEFAULT_SITE_ABOVE_CLEARANCE)
    parser.add_argument("--site-below-offset", type=float, default=DEFAULT_SITE_BELOW_OFFSET)
    parser.add_argument("--arrival-tolerance", type=float, default=DEFAULT_ARRIVAL_TOLERANCE)
    parser.add_argument("--gripper-end-arrival-tolerance", type=float, default=DEFAULT_GRIPPER_END_ARRIVAL_TOLERANCE)
    parser.add_argument("--settle-steps", type=int, default=DEFAULT_SETTLE_STEPS)
    parser.add_argument("--grasp-steps", type=int, default=DEFAULT_GRASP_STEPS)
    parser.add_argument("--post-success-hold-steps", type=int, default=DEFAULT_POST_SUCCESS_HOLD_STEPS)
    parser.add_argument("--max-action", type=float, default=DEFAULT_MAX_ACTION)
    parser.add_argument("--initial-view-steps", type=int, default=DEFAULT_INITIAL_VIEW_STEPS)
    parser.add_argument("--render-sleep", type=float, default=DEFAULT_RENDER_SLEEP)
    parser.add_argument("--camera-height", type=int, default=DEFAULT_CAMERA_HEIGHT)
    parser.add_argument("--camera-width", type=int, default=DEFAULT_CAMERA_WIDTH)
    parser.add_argument("--show-object-sites", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--object-site-size", type=float, default=DEFAULT_OBJECT_SITE_SIZE)
    parser.add_argument("--robot-base-pos", type=float, nargs=3, default=DEFAULT_ROBOT_BASE_POS)
    parser.add_argument("--robot-base-ori", type=float, nargs=3, default=DEFAULT_ROBOT_BASE_ORI)
    parser.add_argument("--directory", type=str, default=os.path.join(suite.models.assets_root, "demonstrations_private"))
    parser.add_argument("--output-name", type=str, default="factory_sorting_grasp")
    parser.add_argument("--renderer", type=str, default="mjviewer")
    parser.add_argument("--camera", type=str, default=DEFAULT_CAMERA)
    parser.add_argument("--controller", type=str, default=None)
    parser.add_argument("--gripper-types", type=str, default="Robotiq140Gripper")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-render", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    render = not args.no_render
    env_name = DEFAULT_ENV_NAME
    env_kwargs = make_env_kwargs(args, render=render)
    dataset_env_kwargs = dict(env_kwargs)
    dataset_env_kwargs["has_renderer"] = False

    raw_env = suite.make(
        env_name=env_name,
        **env_kwargs,
    )

    tmp_directory = tempfile.mkdtemp(prefix="factory_sorting_grasp_raw_")
    env = DataCollectionWrapper(raw_env, tmp_directory, collect_freq=1, flush_freq=1000)

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
    out_dir = os.path.join(args.directory, timestamp)
    hdf5_name = f"{args.output_name}_{timestamp}.hdf5"
    os.makedirs(out_dir, exist_ok=True)

    successes = 0
    obs_cache = {}
    for rollout_idx in range(args.num_rollouts):
        print(f"\nRollout {rollout_idx + 1}/{args.num_rollouts}")
        success, reason, ep_directory, obs_buffer = rollout_once(env, render=render, args=args)
        successes += int(success)
        if success:
            obs_cache[os.path.normpath(ep_directory)] = obs_buffer
        print(f"Result: {reason}")

    env.close()

    hdf5_path, num_saved = gather_successful_demonstrations_as_hdf5(
        tmp_directory,
        out_dir,
        hdf5_name=hdf5_name,
        env_name=env_name,
        env_kwargs=dataset_env_kwargs,
        policy_info=vars(args),
        obs_cache=obs_cache,
    )
    print(f"\nAttempts: {args.num_rollouts}, successes: {successes}, saved demos: {num_saved}")
    print(f"HDF5 saved to: {hdf5_path}")
    print(f"Raw trajectory directory: {tmp_directory}")
    if num_saved == 0:
        print("Warning: no successful grasp demos were saved into the HDF5 file.")


if __name__ == "__main__":
    main()
