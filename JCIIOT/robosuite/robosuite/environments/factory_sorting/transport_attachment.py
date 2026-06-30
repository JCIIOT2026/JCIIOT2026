"""
Transport attachment helper for FactorySorting.

The navigation script moves the Tiago base by directly editing base qpos. A
free object grasped by the grippers will not automatically follow that direct
base update, so this helper keeps the carried object's freejoint pose at a
fixed offset from the robot base during transport.
"""

import math

import numpy as np

from robosuite.environments.factory_sorting.load_factory_sorting_evalization import (
    base_robosuite_env,
    get_base_world_pose,
)


TRANSPORT_ATTACHMENT_ATTR = "_factory_sorting_transport_attachment"


def rotate_xy(vec_xy, yaw):
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    x, y = vec_xy
    return np.array(
        [
            cos_yaw * x - sin_yaw * y,
            sin_yaw * x + cos_yaw * y,
        ],
        dtype=float,
    )


def yaw_quat_wxyz(yaw):
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], dtype=float)


def quat_conjugate_wxyz(quat):
    quat = np.asarray(quat, dtype=float)
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=float)


def quat_multiply_wxyz(q1, q0):
    w1, x1, y1, z1 = np.asarray(q1, dtype=float)
    w0, x0, y0, z0 = np.asarray(q0, dtype=float)
    quat = np.array(
        [
            w1 * w0 - x1 * x0 - y1 * y0 - z1 * z0,
            w1 * x0 + x1 * w0 + y1 * z0 - z1 * y0,
            w1 * y0 - x1 * z0 + y1 * w0 + z1 * x0,
            w1 * z0 + x1 * y0 - y1 * x0 + z1 * w0,
        ],
        dtype=float,
    )
    norm = np.linalg.norm(quat)
    if norm > 0:
        quat = quat / norm
    return quat


def object_joint_name(env, object_name):
    metadata = getattr(env, "material_metadata", {}).get(object_name, {})
    joint_name = metadata.get("joint_name")
    if joint_name:
        return joint_name
    return f"{object_name}_free"


def get_object_qpos(env, object_name):
    joint_name = object_joint_name(env, object_name)
    qpos = np.asarray(env.sim.data.get_joint_qpos(joint_name), dtype=float)
    if qpos.size != 7:
        raise RuntimeError(f"Expected freejoint qpos for '{joint_name}', got shape {qpos.shape}.")
    return joint_name, qpos.copy()


def set_object_qpos(env, joint_name, qpos):
    env.sim.data.set_joint_qpos(joint_name, np.asarray(qpos, dtype=float))
    env.sim.data.set_joint_qvel(joint_name, np.zeros(6, dtype=float))
    env.sim.forward()


def capture_transport_attachment(env, object_name):
    raw_env = base_robosuite_env(env)
    object_name = object_name or raw_env.material_objects[0]
    joint_name, object_qpos = get_object_qpos(raw_env, object_name)
    base_xy, base_yaw = get_base_world_pose(raw_env)

    world_delta_xy = object_qpos[:2] - base_xy
    relative_xy = rotate_xy(world_delta_xy, -base_yaw)
    base_quat = yaw_quat_wxyz(base_yaw)
    relative_quat = quat_multiply_wxyz(quat_conjugate_wxyz(base_quat), object_qpos[3:7])

    attachment = {
        "active": True,
        "object_name": object_name,
        "joint_name": joint_name,
        "relative_xy": relative_xy,
        "world_z": float(object_qpos[2]),
        "relative_quat": relative_quat,
    }
    setattr(raw_env, TRANSPORT_ATTACHMENT_ATTR, attachment)
    print(
        "transport_attachment_enabled: "
        f"object={object_name}, joint={joint_name}, "
        f"relative_xy={np.round(relative_xy, 4).tolist()}, "
        f"world_z={float(object_qpos[2]):.6f}"
    )
    return attachment


def sync_transport_attachment(env):
    raw_env = base_robosuite_env(env)
    attachment = getattr(raw_env, TRANSPORT_ATTACHMENT_ATTR, None)
    if not attachment or not attachment.get("active", False):
        return False

    base_xy, base_yaw = get_base_world_pose(raw_env)
    relative_xy = np.asarray(attachment["relative_xy"], dtype=float)
    world_xy = base_xy + rotate_xy(relative_xy, base_yaw)
    base_quat = yaw_quat_wxyz(base_yaw)
    object_quat = quat_multiply_wxyz(base_quat, attachment["relative_quat"])
    qpos = np.concatenate(
        [
            np.array([world_xy[0], world_xy[1], attachment["world_z"]], dtype=float),
            object_quat,
        ]
    )
    set_object_qpos(raw_env, attachment["joint_name"], qpos)
    return True


def clear_transport_attachment(env):
    raw_env = base_robosuite_env(env)
    attachment = getattr(raw_env, TRANSPORT_ATTACHMENT_ATTR, None)
    if attachment is not None:
        attachment["active"] = False
