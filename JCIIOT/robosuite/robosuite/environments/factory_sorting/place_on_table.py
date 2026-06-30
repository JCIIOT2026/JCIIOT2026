"""
Place the transported FactorySorting object onto an output station surface.

The navigation pipeline uses direct base updates and a transport attachment for
carrying the object. This helper takes over the object's freejoint after the
robot has turned to face the output station, moves the object over the station
center, lowers it to the tabletop, and releases the grippers.
"""

import time

import numpy as np

from robosuite.environments.factory_sorting.load_factory_sorting_evalization import (
    base_robosuite_env,
    object_collision_geoms,
    site_pos,
)
from robosuite.environments.factory_sorting.transport_attachment import (
    clear_transport_attachment,
    get_object_qpos,
    set_object_qpos,
    sync_transport_attachment,
)


ARMS = ("right", "left")
DEFAULT_PLACE_CLEARANCE = 0.01
DEFAULT_PLACE_XY_STEPS = 25
DEFAULT_PLACE_LOWER_STEPS = 50
DEFAULT_PLACE_HOLD_STEPS = 20
DEFAULT_PLACE_RELEASE_STEPS = 20
DEFAULT_PLACE_RENDER_SLEEP = 0.02


def output_index(output_name):
    if output_name is None:
        return None
    parts = str(output_name).split("_")
    if len(parts) < 2 or parts[0] != "output":
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def output_info_from_llm_scene(llm_scene, output_name):
    if not llm_scene or output_name is None:
        return None

    output_ports = llm_scene.get("output_ports", {})
    if output_name in output_ports:
        return output_ports[output_name]

    for obj in llm_scene.get("objects", []):
        if obj.get("name") == output_name and obj.get("role") == "output":
            return obj
    return None


def output_support_surface(env, output_name):
    index = output_index(output_name)
    if index is None:
        return None
    if not hasattr(env, "_siemens_static_table_support_surfaces"):
        return None

    surfaces = list(env._siemens_static_table_support_surfaces())
    surface_idx = index - 1
    if surface_idx < 0 or surface_idx >= len(surfaces):
        return None
    support_pose, support_size = surfaces[surface_idx]
    return np.asarray(support_pose, dtype=float), np.asarray(support_size, dtype=float)


def resolve_output_place_target(env, target_xy=None, table_z=None, llm_scene=None, output_name=None):
    output_info = output_info_from_llm_scene(llm_scene, output_name)
    support_surface = output_support_surface(env, output_name)

    if output_info is not None and output_info.get("center") is not None:
        resolved_xy = np.asarray(output_info["center"], dtype=float)[:2]
    elif target_xy is not None:
        resolved_xy = np.asarray(target_xy, dtype=float)[:2]
    elif support_surface is not None:
        resolved_xy = support_surface[0][:2]
    else:
        raise RuntimeError("place_on_table target xy is missing.")

    if table_z is not None:
        resolved_table_z = float(table_z)
    elif support_surface is not None:
        resolved_table_z = float(support_surface[0][2])
    else:
        resolved_table_z = output_surface_z(env, table_z=None)

    return resolved_xy, resolved_table_z


def zero_action(env):
    low, _ = env.action_spec
    return np.zeros_like(low)


def object_center_pos(env, object_name):
    center_site = f"{object_name}_center_site"
    try:
        return site_pos(env, center_site)
    except Exception:
        pass

    positions = []
    for arm in ARMS:
        site_name = f"{object_name}_{arm}_grasp_site"
        try:
            positions.append(site_pos(env, site_name))
        except Exception:
            continue
    if positions:
        return np.mean(positions, axis=0)

    _, qpos = get_object_qpos(env, object_name)
    return qpos[:3].copy()


def object_bottom_z(env, object_name):
    bottoms = []
    for geom_name in object_collision_geoms(env, object_name):
        geom_id = env.sim.model.geom_name2id(geom_name)
        geom_pos = np.asarray(env.sim.data.geom_xpos[geom_id], dtype=float)
        geom_size = np.asarray(env.sim.model.geom_size[geom_id], dtype=float)
        half_z = float(geom_size[2]) if geom_size.size >= 3 and geom_size[2] > 0 else float(env.sim.model.geom_rbound[geom_id])
        bottoms.append(float(geom_pos[2] - half_z))

    if not bottoms:
        raise RuntimeError(f"Cannot infer bottom height for object '{object_name}'.")
    return min(bottoms)


def output_surface_z(env, table_z=None):
    if table_z is not None:
        return float(table_z)
    if hasattr(env, "table_top_z"):
        return float(env.table_top_z)
    return 0.4


def interpolate(start, target, fraction):
    return np.asarray(start, dtype=float) + (np.asarray(target, dtype=float) - np.asarray(start, dtype=float)) * fraction


def render_frame(raw_env, render, render_sleep):
    if not render:
        return
    raw_env.render()
    if render_sleep > 0:
        time.sleep(render_sleep)


def step_env(env, action):
    result = env.step(action)
    if isinstance(result, tuple) and len(result) == 4:
        return result
    raise RuntimeError(f"Unexpected env.step result during place: {type(result)}")


def gripper_release_action(raw_env):
    robot = raw_env.robots[0]
    action_dict = {}
    for arm in ARMS:
        if arm in robot.gripper:
            dof = robot.gripper[arm].dof
            if dof > 0:
                action_dict[f"{arm}_gripper"] = -np.ones(dof)

    split = robot.composite_controller._action_split_indexes
    if "base" in split:
        start, end = split["base"]
        action_dict["base"] = np.zeros(end - start)

    if not action_dict:
        return zero_action(raw_env)
    return robot.create_action_vector(action_dict)


def set_and_step_object_qpos(env, raw_env, joint_name, qpos, action, render, render_sleep):
    set_object_qpos(raw_env, joint_name, qpos)
    step_env(env, action)
    set_object_qpos(raw_env, joint_name, qpos)
    render_frame(raw_env, render=render, render_sleep=render_sleep)


def place_object_on_table(
    env,
    object_name,
    target_xy=None,
    llm_scene=None,
    output_name=None,
    table_z=None,
    clearance=DEFAULT_PLACE_CLEARANCE,
    xy_steps=DEFAULT_PLACE_XY_STEPS,
    lower_steps=DEFAULT_PLACE_LOWER_STEPS,
    hold_steps=DEFAULT_PLACE_HOLD_STEPS,
    release_steps=DEFAULT_PLACE_RELEASE_STEPS,
    render=True,
    render_sleep=DEFAULT_PLACE_RENDER_SLEEP,
    debug=False,
    debug_every=10,
):
    raw_env = base_robosuite_env(env)
    object_name = object_name or raw_env.material_objects[0]
    target_xy, table_z = resolve_output_place_target(
        raw_env,
        target_xy=target_xy,
        table_z=table_z,
        llm_scene=llm_scene,
        output_name=output_name,
    )

    sync_transport_attachment(raw_env)
    clear_transport_attachment(raw_env)

    joint_name, start_qpos = get_object_qpos(raw_env, object_name)
    start_center = object_center_pos(raw_env, object_name)
    center_offset_xy = start_center[:2] - start_qpos[:2]
    target_qpos_xy = target_xy - center_offset_xy

    start_bottom_z = object_bottom_z(raw_env, object_name)
    target_bottom_z = output_surface_z(raw_env, table_z=table_z) + float(clearance)
    target_qpos_z = start_qpos[2] + (target_bottom_z - start_bottom_z)

    above_qpos = start_qpos.copy()
    above_qpos[:2] = target_qpos_xy

    final_qpos = above_qpos.copy()
    final_qpos[2] = target_qpos_z

    idle_action = zero_action(raw_env)
    release_action = gripper_release_action(raw_env)
    xy_steps = max(1, int(xy_steps))
    lower_steps = max(1, int(lower_steps))
    hold_steps = max(0, int(hold_steps))
    release_steps = max(0, int(release_steps))

    print(
        "place_on_table_start: "
        f"object={object_name}, joint={joint_name}, "
        f"output={output_name}, "
        f"target_xy={np.round(target_xy, 4).tolist()}, "
        f"table_z={table_z:.6f}, "
        f"start_bottom_z={start_bottom_z:.6f}, target_bottom_z={target_bottom_z:.6f}"
    )

    for step in range(xy_steps):
        qpos = start_qpos.copy()
        qpos[:2] = interpolate(start_qpos[:2], target_qpos_xy, float(step + 1) / float(xy_steps))
        set_and_step_object_qpos(
            env=env,
            raw_env=raw_env,
            joint_name=joint_name,
            qpos=qpos,
            action=idle_action,
            render=render,
            render_sleep=render_sleep,
        )
        if debug and step % debug_every == 0:
            print(f"place_xy_debug step={step}/{xy_steps}")

    for step in range(lower_steps):
        qpos = above_qpos.copy()
        qpos[2] = float(interpolate(above_qpos[2], target_qpos_z, float(step + 1) / float(lower_steps)))
        set_and_step_object_qpos(
            env=env,
            raw_env=raw_env,
            joint_name=joint_name,
            qpos=qpos,
            action=idle_action,
            render=render,
            render_sleep=render_sleep,
        )
        if debug and step % debug_every == 0:
            current_bottom_z = object_bottom_z(raw_env, object_name)
            print(
                "place_lower_debug "
                f"step={step}/{lower_steps} bottom_z={current_bottom_z:.6f} "
                f"target_bottom_z={target_bottom_z:.6f}"
            )

    for _ in range(release_steps):
        set_and_step_object_qpos(
            env=env,
            raw_env=raw_env,
            joint_name=joint_name,
            qpos=final_qpos,
            action=release_action,
            render=render,
            render_sleep=render_sleep,
        )

    for _ in range(hold_steps):
        set_and_step_object_qpos(
            env=env,
            raw_env=raw_env,
            joint_name=joint_name,
            qpos=final_qpos,
            action=idle_action,
            render=render,
            render_sleep=render_sleep,
        )

    final_center = object_center_pos(raw_env, object_name)
    final_bottom_z = object_bottom_z(raw_env, object_name)
    xy_error = float(np.linalg.norm(final_center[:2] - target_xy))
    z_error = float(abs(final_bottom_z - target_bottom_z))
    success = xy_error <= 0.05 and z_error <= 0.03
    result = {
        "success": success,
        "object_name": object_name,
        "joint_name": joint_name,
        "target_xy": target_xy.tolist(),
        "output_name": output_name,
        "table_z": table_z,
        "final_center": final_center.tolist(),
        "target_bottom_z": target_bottom_z,
        "final_bottom_z": final_bottom_z,
        "xy_error": xy_error,
        "z_error": z_error,
    }
    print(
        "place_on_table_result: "
        f"success={success}, xy_error={xy_error:.6f}, "
        f"final_bottom_z={final_bottom_z:.6f}, target_bottom_z={target_bottom_z:.6f}, "
        f"z_error={z_error:.6f}"
    )
    if debug:
        print(f"place_on_table_debug: {result}")
    return result
