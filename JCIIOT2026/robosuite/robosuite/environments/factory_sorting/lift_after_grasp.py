"""
Lift a two-arm grasped FactorySorting object straight upward.

This module is meant to be called from llm_task_navigator.py after the
robomimic grasp policy has established a two-gripper grasp in the same env.
"""

import time

import numpy as np

from robosuite.environments.factory_sorting.load_factory_sorting_evalization import (
    base_robosuite_env,
    grasp_status,
    gripper_end_center_pos,
    site_pos,
)


ARMS = ("right", "left")
CAMERA_HOLD_PARTS = ("torso", "head")
DEFAULT_LIFT_HEIGHT = 0.05
DEFAULT_LIFT_HOLD_STEPS = 20
DEFAULT_LIFT_MAX_ACTION = 0.65
DEFAULT_LIFT_MAX_STEPS = 120
DEFAULT_LIFT_TOLERANCE = 0.01
DEFAULT_RENDER_SLEEP = 0.02


def object_center_pos(env, object_name):
    center_site = f"{object_name}_center_site"
    try:
        return site_pos(env, center_site)
    except Exception:
        pass

    positions = []
    for arm in ARMS:
        grasp_site = f"{object_name}_{arm}_grasp_site"
        try:
            positions.append(site_pos(env, grasp_site))
        except Exception:
            continue
    if positions:
        return np.mean(positions, axis=0)

    raise RuntimeError(f"Cannot find center or grasp sites for object '{object_name}'.")


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
            f"This lift helper expects {arm} to use OSC_POSE delta control; "
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
        action[: min(dim, fill.size)] = fill[: min(dim, fill.size)]
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


def capture_hold_targets(robot, part_names=CAMERA_HOLD_PARTS):
    targets = {}
    for part_name in part_names:
        if part_name not in robot.composite_controller._action_split_indexes:
            continue
        qpos = current_part_qpos(robot, part_name)
        if qpos is not None:
            targets[part_name] = qpos
    return targets


def hold_part_action(robot, part_name, hold_targets):
    fill = hold_targets.get(part_name)
    if fill is None:
        fill = current_part_qpos(robot, part_name)
    return optional_part_action(robot, part_name, fill=fill)


def build_action(robot, arm_actions, gripper_value, hold_targets):
    action_dict = {}
    for arm in ARMS:
        action_dict[arm] = arm_actions.get(arm, np.zeros(6))
        dof = robot.gripper[arm].dof
        if dof > 0:
            action_dict[f"{arm}_gripper"] = np.full(dof, gripper_value)

    for part_name in CAMERA_HOLD_PARTS:
        part_action = hold_part_action(robot, part_name, hold_targets)
        if part_action is not None:
            action_dict[part_name] = part_action

    base_action = optional_part_action(robot, "base")
    if base_action is not None:
        action_dict["base"] = base_action

    return robot.create_action_vector(action_dict)


def render_frame(raw_env, render, render_sleep, render_callback=None):
    if render_callback is not None:
        render_callback()
        if render_sleep > 0:
            time.sleep(render_sleep)
        return
    if not render:
        return
    raw_env.render()
    if render_sleep > 0:
        time.sleep(render_sleep)


def step_env(env, action):
    result = env.step(action)
    if isinstance(result, tuple) and len(result) == 4:
        return result
    raise RuntimeError(f"Unexpected env.step result during lift: {type(result)}")


def lift_grasped_object(
    env,
    object_name,
    lift_height=DEFAULT_LIFT_HEIGHT,
    max_steps=DEFAULT_LIFT_MAX_STEPS,
    hold_steps=DEFAULT_LIFT_HOLD_STEPS,
    tolerance=DEFAULT_LIFT_TOLERANCE,
    max_action=DEFAULT_LIFT_MAX_ACTION,
    gripper_value=1.0,
    render=True,
    render_sleep=DEFAULT_RENDER_SLEEP,
    render_callback=None,
    debug=False,
    debug_every=10,
):
    raw_env = base_robosuite_env(env)
    robot = raw_env.robots[0]
    object_name = object_name or raw_env.material_objects[0]

    initial_grasp_status = grasp_status(raw_env, robot, object_name)
    print(f"lift_initial_grasp_status: {initial_grasp_status}")
    if not all(initial_grasp_status.values()):
        return {
            "success": False,
            "failure_reason": f"object is not grasped by both grippers: {initial_grasp_status}",
        }

    start_object_z = float(object_center_pos(raw_env, object_name)[2])
    target_object_z = start_object_z + float(lift_height)
    start_gripper_positions = {
        arm: gripper_end_center_pos(raw_env, robot, arm)
        for arm in ARMS
    }
    target_gripper_positions = {
        arm: start_gripper_positions[arm] + np.array([0.0, 0.0, lift_height], dtype=float)
        for arm in ARMS
    }
    hold_targets = capture_hold_targets(robot)

    print(
        "lift_start: "
        f"object={object_name}, "
        f"start_object_z={start_object_z:.6f}, "
        f"target_object_z={target_object_z:.6f}, "
        f"lift_height={float(lift_height):.6f}"
    )

    success = False
    last_object_z = start_object_z
    for step in range(max_steps):
        current_object_z = float(object_center_pos(raw_env, object_name)[2])
        last_object_z = current_object_z
        if current_object_z >= target_object_z - tolerance:
            success = True
            break

        robot.composite_controller.update_state()
        arm_actions = {}
        for arm in ARMS:
            world_delta = target_gripper_positions[arm] - gripper_end_center_pos(raw_env, robot, arm)
            controller_delta = world_delta_to_controller_frame(robot, arm, world_delta)
            arm_actions[arm] = arm_delta_to_normalized_action(
                robot=robot,
                arm=arm,
                delta_pos=controller_delta,
                max_action=max_action,
            )

        action = build_action(
            robot=robot,
            arm_actions=arm_actions,
            gripper_value=gripper_value,
            hold_targets=hold_targets,
        )
        step_env(env, action)
        render_frame(raw_env, render=render, render_sleep=render_sleep, render_callback=render_callback)

        if debug and step % debug_every == 0:
            print(
                "lift_debug "
                f"step={step}/{max_steps} "
                f"object_z={current_object_z:.6f} "
                f"target_z={target_object_z:.6f}"
            )

    if success and hold_steps > 0:
        hold_action = build_action(
            robot=robot,
            arm_actions={},
            gripper_value=gripper_value,
            hold_targets=hold_targets,
        )
        for _ in range(hold_steps):
            step_env(env, hold_action)
            render_frame(raw_env, render=render, render_sleep=render_sleep, render_callback=render_callback)

    final_object_z = float(object_center_pos(raw_env, object_name)[2])
    final_grasp_status = grasp_status(raw_env, robot, object_name)
    print(
        "lift_result: "
        f"success={success}, "
        f"final_object_z={final_object_z:.6f}, "
        f"lifted={final_object_z - start_object_z:.6f}, "
        f"target_lift={float(lift_height):.6f}, "
        f"final_grasp_status={final_grasp_status}"
    )

    if not success:
        return {
            "success": False,
            "failure_reason": (
                "lift timeout: "
                f"max_steps={max_steps}, "
                f"last_object_z={last_object_z:.6f}, "
                f"target_object_z={target_object_z:.6f}, "
                f"tolerance={tolerance:.6f}"
            ),
        }

    if not all(final_grasp_status.values()):
        return {
            "success": False,
            "failure_reason": f"object lifted but final grasp was lost: {final_grasp_status}",
        }

    return {
        "success": True,
        "failure_reason": "",
        "start_object_z": start_object_z,
        "final_object_z": final_object_z,
        "lifted": final_object_z - start_object_z,
    }
