"""
Solve a fixed-torso Tiago bimanual IK target around the table center.

The target is defined from the table top center:
    right gripper: center + (0, -xy_offset, z_offset)
    left gripper:  center + (0,  xy_offset, z_offset)

Both gripper target frames use a vertical orientation with their local z axis
parallel to the world z axis. The torso lift joint is fixed before solving.
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np

import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.utils.ik_utils import IKSolver


def _as_mj_model(sim):
    return getattr(sim.model, "_model", sim.model)


def _as_mj_data(sim):
    return getattr(sim.data, "_data", sim.data)


def _qpos_addr(model, joint_name):
    return int(model.joint(joint_name).qposadr[0])


def _site_pos(data, site_name):
    return np.array(data.site(site_name).xpos, dtype=np.float64)


def _site_mat(data, site_name):
    return np.array(data.site(site_name).xmat, dtype=np.float64).reshape(3, 3)


def _names_containing(model, *parts):
    names = []
    for i in range(model.njnt):
        name = model.joint(i).name
        if name and all(part in name for part in parts):
            names.append(name)
    return names


def _format_array(values):
    return np.array2string(
        np.asarray(values),
        precision=6,
        separator=", ",
        suppress_small=False,
        max_line_width=120,
    )


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Solve Tiago arm qpos for two gripper targets around the table center."
    )
    parser.add_argument("--env", default="TwoArmPlasticCrateLift")
    parser.add_argument("--robots", default="Tiago")
    parser.add_argument("--torso-height", type=float, default=0.30)
    parser.add_argument("--xy-offset", type=float, default=0.30)
    parser.add_argument("--z-offset", type=float, default=0.40)
    parser.add_argument("--right-yaw", type=float, default=0.0)
    parser.add_argument("--left-yaw", type=float, default=0.0)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--damping", type=float, default=5e-2)
    parser.add_argument("--integration-dt", type=float, default=0.1)
    parser.add_argument("--max-dq", type=float, default=4.0)
    parser.add_argument(
        "--nullspace-gain",
        type=float,
        default=0.05,
        help="Posture bias toward the reset arm qpos. Set 0 to disable.",
    )
    parser.add_argument(
        "--controller-config",
        default=None,
        help="Optional composite controller JSON path. Defaults to default_tiago_whole_body_ik.json.",
    )
    parser.add_argument(
        "--table-center",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Override the table top center. Defaults to env.table_offset.",
    )
    return parser


def _make_env(args):
    if args.controller_config is None:
        controller_config_path = (
            Path(robosuite.__file__).resolve().parent
            / "controllers"
            / "config"
            / "robots"
            / "default_tiago_whole_body_ik.json"
        )
    else:
        controller_config_path = Path(args.controller_config)

    controller_config = load_composite_controller_config(controller=str(controller_config_path))
    env = robosuite.make(
        args.env,
        robots=args.robots,
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        initialization_noise=None,
        hard_reset=True,
    )
    env.reset()
    return env


def _build_solver(model, data, joint_names, site_names, args):
    robot_config = {
        "end_effector_sites": site_names,
        "joint_names": joint_names,
        "mocap_bodies": [],
        "nullspace_gains": np.full(len(joint_names), args.nullspace_gain),
    }
    solver = IKSolver(
        model=model,
        data=data,
        robot_config=robot_config,
        damping=args.damping,
        integration_dt=args.integration_dt,
        max_dq=args.max_dq,
        input_action_repr="absolute",
        input_rotation_repr="axis_angle",
        input_ref_frame="world",
    )
    solver.q0 = data.qpos[solver.dof_ids].copy()
    return solver


def _solve(args):
    env = _make_env(args)
    robot = env.robots[0]
    model = _as_mj_model(env.sim)
    data = _as_mj_data(env.sim)

    torso_joint_names = _names_containing(model, "torso_lift_joint")
    if len(torso_joint_names) != 1:
        raise RuntimeError(f"Expected one torso_lift_joint, found {torso_joint_names}")
    torso_joint = torso_joint_names[0]
    data.qpos[_qpos_addr(model, torso_joint)] = args.torso_height
    mujoco.mj_forward(model, data)

    right_joint_names = _names_containing(model, "arm_right_", "_joint")
    left_joint_names = _names_containing(model, "arm_left_", "_joint")
    if len(right_joint_names) != 6 or len(left_joint_names) != 6:
        raise RuntimeError(
            f"Expected 6 joints per Tiago arm, found right={right_joint_names}, left={left_joint_names}"
        )
    joint_names = right_joint_names + left_joint_names

    right_site = robot.gripper["right"].important_sites["grip_site"]
    left_site = robot.gripper["left"].important_sites["grip_site"]
    site_names = [right_site, left_site]

    table_center = (
        np.array(args.table_center, dtype=np.float64)
        if args.table_center is not None
        else np.array(env.table_offset, dtype=np.float64)
    )
    right_target_pos = table_center + np.array([0.0, -args.xy_offset, args.z_offset])
    left_target_pos = table_center + np.array([0.0, args.xy_offset, args.z_offset])

    # Axis-angle rotations about world z. Any yaw keeps the target local z axis
    # parallel to world z; the defaults align the full frame with world axes.
    right_target_rot = np.array([np.pi / 2, np.pi / 2, 0])
    left_target_rot = np.array([np.pi / 2, np.pi / 2, 0])
    target_action = np.concatenate(
        [right_target_pos, right_target_rot, left_target_pos, left_target_rot]
    )

    solver = _build_solver(model, data, joint_names, site_names, args)
    target_pos = np.array([right_target_pos, left_target_pos])

    final_iter = 0
    final_error = np.inf
    for final_iter in range(1, args.iterations + 1):
        q_des = solver.solve(target_action)
        data.qpos[solver.dof_ids] = q_des
        data.qpos[_qpos_addr(model, torso_joint)] = args.torso_height
        mujoco.mj_forward(model, data)

        current_pos = np.array([_site_pos(data, right_site), _site_pos(data, left_site)])
        per_site_error = np.linalg.norm(target_pos - current_pos, axis=1)
        final_error = float(np.max(per_site_error))
        if final_error <= args.tol:
            break

    right_qpos = data.qpos[[model.joint(name).qposadr[0] for name in right_joint_names]].copy()
    left_qpos = data.qpos[[model.joint(name).qposadr[0] for name in left_joint_names]].copy()
    robot_qpos = data.qpos[[model.joint(name).qposadr[0] for name in robot.robot_joints]].copy()

    right_z_axis = _site_mat(data, right_site)[:, 2]
    left_z_axis = _site_mat(data, left_site)[:, 2]

    print("Solved Tiago bimanual IK")
    print(f"iterations: {final_iter}")
    print(f"max_position_error: {final_error:.8f} m")
    print(f"table_center: {_format_array(table_center)}")
    print(f"right_target_pos: {_format_array(right_target_pos)}")
    print(f"left_target_pos: {_format_array(left_target_pos)}")
    print(f"right_actual_pos: {_format_array(_site_pos(data, right_site))}")
    print(f"left_actual_pos: {_format_array(_site_pos(data, left_site))}")
    print(f"right_site_z_axis: {_format_array(right_z_axis)}")
    print(f"left_site_z_axis: {_format_array(left_z_axis)}")
    print("")
    print("right_arm_qpos =", _format_array(right_qpos))
    print("left_arm_qpos  =", _format_array(left_qpos))
    print("")
    print("full_tiago_init_qpos =", _format_array(robot_qpos))
    print("")
    print("joint_values:")
    for name, value in zip(robot.robot_joints, robot_qpos):
        print(f"  {name}: {value:.6f}")


def main():
    args = _build_arg_parser().parse_args()
    _solve(args)


if __name__ == "__main__":
    main()
