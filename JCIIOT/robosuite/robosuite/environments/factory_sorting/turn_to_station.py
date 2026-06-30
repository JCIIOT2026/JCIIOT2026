"""
Turn the Tiago base to face a target station after navigation.

This module is intentionally self-contained so llm_task_navigator can import it
without creating a circular dependency. It uses the same direct base-yaw update
style as navigation and keeps the transport attachment synchronized.
"""

import math
import time

import numpy as np

from robosuite.environments.factory_sorting.load_factory_sorting_evalization import (
    base_robosuite_env,
    get_base_world_pose,
)
from robosuite.environments.factory_sorting.transport_attachment import sync_transport_attachment


DEFAULT_TURN_TOLERANCE = 0.02
DEFAULT_TURN_MAX_ITERS = 8
DEFAULT_TURN_STEPS = 40
DEFAULT_TURN_SETTLE_STEPS = 10
DEFAULT_TURN_RENDER_SLEEP = 0.02


def shortest_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def base_joint_qpos_indexes(env, robot):
    indexes = {}
    for joint_name in robot.robot_model.base_joints:
        raw_name = joint_name.lower()
        if "mobile_forward" in raw_name:
            indexes["forward"] = env.sim.model.get_joint_qpos_addr(joint_name)
        elif "mobile_side" in raw_name:
            indexes["side"] = env.sim.model.get_joint_qpos_addr(joint_name)
        elif "mobile_yaw" in raw_name:
            indexes["yaw"] = env.sim.model.get_joint_qpos_addr(joint_name)
    return indexes


def base_joint_qvel_indexes(env, robot):
    indexes = {}
    for joint_name in robot.robot_model.base_joints:
        raw_name = joint_name.lower()
        if "mobile_forward" in raw_name:
            indexes["forward"] = env.sim.model.get_joint_qvel_addr(joint_name)
        elif "mobile_side" in raw_name:
            indexes["side"] = env.sim.model.get_joint_qvel_addr(joint_name)
        elif "mobile_yaw" in raw_name:
            indexes["yaw"] = env.sim.model.get_joint_qvel_addr(joint_name)
    return indexes


def clear_base_xy_qpos_mapping(env):
    cache_name = "_factory_sorting_base_xy_qpos_mapping"
    if hasattr(env, cache_name):
        delattr(env, cache_name)


def zero_base_velocity(env, robot):
    for qvel_index in base_joint_qvel_indexes(env, robot).values():
        env.sim.data.qvel[qvel_index] = 0.0


def zero_action(env):
    low, _ = env.action_spec
    return np.zeros_like(low)


def base_xy_qpos_mapping(env, robot, eps=1e-4):
    indexes = base_joint_qpos_indexes(env, robot)
    missing = {"forward", "side"} - set(indexes)
    if missing:
        raise RuntimeError(f"Missing mobile base qpos indexes: {sorted(missing)}")

    qpos = env.sim.data.qpos
    original_forward = float(qpos[indexes["forward"]])
    original_side = float(qpos[indexes["side"]])
    base_xy, _ = get_base_world_pose(env, robot)

    qpos[indexes["forward"]] = original_forward + eps
    qpos[indexes["side"]] = original_side
    env.sim.forward()
    forward_xy, _ = get_base_world_pose(env, robot)

    qpos[indexes["forward"]] = original_forward
    qpos[indexes["side"]] = original_side + eps
    env.sim.forward()
    side_xy, _ = get_base_world_pose(env, robot)

    qpos[indexes["forward"]] = original_forward
    qpos[indexes["side"]] = original_side
    env.sim.forward()

    qpos_to_world = np.column_stack(((forward_xy - base_xy) / eps, (side_xy - base_xy) / eps))
    det = float(np.linalg.det(qpos_to_world))
    if abs(det) < 1e-8:
        raise RuntimeError(f"Invalid base qpos-to-world mapping: {qpos_to_world}")

    return indexes, np.linalg.inv(qpos_to_world)


def set_base_xy_direct(env, robot, target_xy):
    indexes, world_to_qpos = base_xy_qpos_mapping(env, robot)
    base_xy, _ = get_base_world_pose(env, robot)
    delta = np.asarray(target_xy, dtype=float)[:2] - base_xy
    delta_qpos = world_to_qpos @ delta

    env.sim.data.qpos[indexes["forward"]] += delta_qpos[0]
    env.sim.data.qpos[indexes["side"]] += delta_qpos[1]
    env.sim.forward()


def lock_base_xy(env, robot, target_xy):
    set_base_xy_direct(env, robot, target_xy)
    zero_base_velocity(env, robot)
    env.sim.forward()


def set_base_world_yaw_direct(env, robot, target_yaw, tolerance=1e-5, max_iters=DEFAULT_TURN_MAX_ITERS, eps=1e-4):
    qpos_indexes = base_joint_qpos_indexes(env, robot)
    if "yaw" not in qpos_indexes:
        base_xy, current_yaw = get_base_world_pose(env, robot)
        if abs(shortest_angle(target_yaw - current_yaw)) <= tolerance:
            return
        raise RuntimeError(
            "Robot base has no mobile_yaw qpos index; "
            f"current=({base_xy[0]:.6f},{base_xy[1]:.6f},{current_yaw:.6f}), "
            f"target_yaw={target_yaw:.6f}"
        )

    yaw_index = qpos_indexes["yaw"]
    for _ in range(max_iters):
        _, current_yaw = get_base_world_pose(env, robot)
        yaw_error = shortest_angle(float(target_yaw) - current_yaw)
        if abs(yaw_error) <= tolerance:
            break

        original_yaw_qpos = float(env.sim.data.qpos[yaw_index])
        env.sim.data.qpos[yaw_index] = original_yaw_qpos + eps
        env.sim.forward()
        _, plus_world_yaw = get_base_world_pose(env, robot)

        env.sim.data.qpos[yaw_index] = original_yaw_qpos
        env.sim.forward()

        gain = shortest_angle(plus_world_yaw - current_yaw) / eps
        if abs(gain) < 1e-8:
            gain = 1.0
        env.sim.data.qpos[yaw_index] = original_yaw_qpos + yaw_error / gain
        zero_base_velocity(env, robot)
        env.sim.forward()

    clear_base_xy_qpos_mapping(env)


def target_yaw_from_xy(base_xy, target_xy):
    target_xy = np.asarray(target_xy, dtype=float)[:2]
    delta = target_xy - np.asarray(base_xy, dtype=float)[:2]
    distance = float(np.linalg.norm(delta))
    if distance < 1e-8:
        raise RuntimeError(
            "Cannot turn to face target: base and target xy are effectively identical "
            f"({base_xy[0]:.6f}, {base_xy[1]:.6f})."
        )
    return float(math.atan2(delta[1], delta[0])), distance


def turn_to_face_xy(
    env,
    target_xy,
    tolerance=DEFAULT_TURN_TOLERANCE,
    max_iters=DEFAULT_TURN_MAX_ITERS,
    turn_steps=DEFAULT_TURN_STEPS,
    settle_steps=DEFAULT_TURN_SETTLE_STEPS,
    render=True,
    render_sleep=DEFAULT_TURN_RENDER_SLEEP,
    sync_attachment=True,
    post_step_callback=None,
    debug=False,
):
    raw_env = base_robosuite_env(env)
    robot = raw_env.robots[0]
    base_xy, start_yaw = get_base_world_pose(raw_env, robot)
    locked_base_xy = base_xy.copy()
    target_yaw, target_distance = target_yaw_from_xy(base_xy, target_xy)
    start_error = shortest_angle(target_yaw - start_yaw)

    idle_action = zero_action(raw_env)
    num_turn_steps = max(1, int(turn_steps))
    actual_turn_steps = 0
    yaw_update_tolerance = min(float(tolerance), 1e-5)

    if abs(start_error) > tolerance:
        for turn_step in range(num_turn_steps):
            actual_turn_steps += 1
            fraction = float(turn_step + 1) / float(num_turn_steps)
            step_yaw = start_yaw + start_error * fraction
            set_base_world_yaw_direct(
                raw_env,
                robot,
                target_yaw=step_yaw,
                tolerance=yaw_update_tolerance,
                max_iters=max_iters,
            )
            lock_base_xy(raw_env, robot, locked_base_xy)
            if post_step_callback is not None:
                post_step_callback()
            if sync_attachment:
                sync_transport_attachment(raw_env)
            raw_env.sim.forward()

            raw_env.step(idle_action)
            lock_base_xy(raw_env, robot, locked_base_xy)
            if post_step_callback is not None:
                post_step_callback()
            if sync_attachment:
                sync_transport_attachment(raw_env)
            if render:
                raw_env.render()
            if render_sleep > 0:
                time.sleep(render_sleep)

    for _ in range(max(0, int(settle_steps))):
        raw_env.step(idle_action)
        lock_base_xy(raw_env, robot, locked_base_xy)
        if post_step_callback is not None:
            post_step_callback()
        if sync_attachment:
            sync_transport_attachment(raw_env)
        if render:
            raw_env.render()
        if render_sleep > 0:
            time.sleep(render_sleep)

    final_xy, final_yaw = get_base_world_pose(raw_env, robot)
    final_error = shortest_angle(target_yaw - final_yaw)
    xy_drift = float(np.linalg.norm(final_xy - locked_base_xy))
    success = abs(final_error) <= tolerance
    result = {
        "success": success,
        "target_xy": np.asarray(target_xy, dtype=float)[:2].tolist(),
        "base_xy": final_xy.tolist(),
        "locked_base_xy": locked_base_xy.tolist(),
        "xy_drift": xy_drift,
        "target_distance": target_distance,
        "start_yaw": start_yaw,
        "target_yaw": target_yaw,
        "final_yaw": final_yaw,
        "start_error": start_error,
        "final_error": final_error,
        "tolerance": tolerance,
        "turn_steps": actual_turn_steps,
        "settle_steps": max(0, int(settle_steps)),
    }
    print(
        "turn_to_output_result: "
        f"success={success}, turn_steps={actual_turn_steps}, target_yaw={target_yaw:.6f}, "
        f"final_yaw={final_yaw:.6f}, final_error={final_error:.6f}, xy_drift={xy_drift:.6f}"
    )
    if debug:
        print(f"turn_to_output_debug: {result}")
    return result


def turn_to_face_station(env, station, **kwargs):
    if isinstance(station, dict):
        target_xy = station.get("center")
        if target_xy is None:
            target_xy = station.get("approach")
    else:
        target_xy = station
    if target_xy is None:
        raise RuntimeError("Station must provide either 'center' or 'approach'.")
    return turn_to_face_xy(env, target_xy=target_xy, **kwargs)
