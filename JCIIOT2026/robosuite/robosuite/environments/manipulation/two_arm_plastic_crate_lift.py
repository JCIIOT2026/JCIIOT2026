import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.two_arm_lift import TwoArmLift
from robosuite.models.arenas import TableArena
from robosuite.models.objects import PlasticCrateObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
TIAGO_GRIPPER_Z_DOWN_INIT_QPOS = np.array(
    [
        0.35     ,  0.      ,  -0.9      ,  0.742599,
        -0.090401,
        2.137818,
        2.105705,
        -0.349066,
        -1.046345,
        0.994838,
        -0.05236,
        2.199115,
        2.024582,
        -0.349066,
        -1.029744
    ]
)


class TwoArmPlasticCrateLift(TwoArmLift):
    """
    Two-arm lift task using a fixed-pose plastic crate.
    """

    def __init__(self, *args, crate_scale=0.55, robot_lateral_offset_range=0.0, **kwargs):
        self.crate_scale = crate_scale
        self.robot_lateral_offset_range = robot_lateral_offset_range
        self.robot_lateral_offset = 0.0
        kwargs.setdefault("initialization_noise", None)
        super().__init__(*args, **kwargs)

    def _sample_robot_lateral_offset(self):
        offset_range = self.robot_lateral_offset_range
        if offset_range is None:
            return 0.0

        offset_range = np.array(offset_range, dtype=float).flatten()
        if len(offset_range) == 0:
            return 0.0
        if len(offset_range) == 1:
            limit = abs(offset_range[0])
            low, high = -limit, limit
        elif len(offset_range) == 2:
            low, high = np.min(offset_range), np.max(offset_range)
        else:
            raise ValueError("robot_lateral_offset_range should be a scalar or length-2 range.")

        if np.isclose(low, high):
            return float(low)
        return float(self.rng.uniform(low, high))

    def _load_model(self):
        """
        Loads a fixed-pose plastic crate lift model.
        """
        super(TwoArmLift, self)._load_model()

        for robot in self.robots:
            if robot.name == "Tiago":
                robot.init_qpos = TIAGO_GRIPPER_Z_DOWN_INIT_QPOS.copy()

        self.table_offset = np.array((0, 0, 0.4))

        # Adjust base pose(s) accordingly
        self.robot_lateral_offset = self._sample_robot_lateral_offset()
        if self.env_configuration == "single-robot":
            xpos = np.array(self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0]))
            xpos += np.array((0.0, self.robot_lateral_offset, 0.0))
            self.robots[0].robot_model.set_base_xpos(xpos)
        else:
            if self.env_configuration == "opposed":
                for robot, rotation in zip(self.robots, (np.pi / 2, -np.pi / 2)):
                    xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                    rot = np.array((0, 0, rotation))
                    xpos = T.euler2mat(rot) @ np.array(xpos)
                    robot.robot_model.set_base_xpos(xpos)
                    robot.robot_model.set_base_ori(rot)
            else:
                for robot, offset in zip(self.robots, (-0.25, 0.25)):
                    xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                    xpos = np.array(xpos) + np.array((0, offset, 0))
                    robot.robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        self.crate = PlasticCrateObject(name="plastic_crate", scale=self.crate_scale)
        self.pot = self.crate

        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.pot)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.pot,
                x_range=[0.0, 0.0],
                y_range=[0.0, 0.0],
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                rotation=np.pi / 2,
                rng=self.rng,
            )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.pot,
        )
