from collections import OrderedDict
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import EmptyArena
from robosuite.models.arenas.arena import Arena
from robosuite.models.base import MujocoXML
from robosuite.models.objects import (
    BoxObject,
    ContainerH01Object,
    ContainerH10Object,
    PlasticCrateObject,
    ToteB01Object,
)
from robosuite.models.tasks import ManipulationTask
from robosuite.robots import ROBOT_CLASS_MAPPING, WheeledRobot
from robosuite.utils.mjcf_utils import array_to_string, xml_path_completion
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.transform_utils import convert_quat, mat2euler


SIEMENS_MESH_DIR = Path(xml_path_completion("objects/siemens/meshes"))
SIEMENS_MESH_QUAT = [0.707106781, 0.707106781, 0.0, 0.0]
SIEMENS_OBJECT_SPECS = [
    {
        "name": "cardbox_c2",
        "file": "Cardbox_C2.obj",
        "mesh": "siemens_cardbox_c2_mesh",
        "material": "siemens_cardboard",
        "rgba": [0.72, 0.60, 0.43, 1.0],
        "visual_pos": [0.0, 0.0, 0.00017],
        "collision_pos": [0.0, 0.0, 0.129845],
        "half_size": [0.256622, 0.255281, 0.129845],
        "mass": 1.0,
    },
    {
        "name": "container_h10",
        "file": "Container_H10_60x40x17cm_PR_V_NVD_01.obj",
        "mesh": "siemens_container_h10_mesh",
        "material": "siemens_dark_gray_plastic",
        "rgba": [0.18, 0.19, 0.20, 1.0],
        "visual_pos": [0.0, 0.0, 0.0],
        "collision_pos": [0.0, 0.0, 0.084991],
        "half_size": [0.300729, 0.200886, 0.084991],
        "mass": 1.5,
    },
    {
        "name": "container_h01",
        "file": "Container_H01_60x40x25cm_PR_V_NVD_01.obj",
        "mesh": "siemens_container_h01_mesh",
        "material": "siemens_light_gray_plastic",
        "rgba": [0.58, 0.60, 0.62, 1.0],
        "visual_pos": [0.0, 0.0, 0.0],
        "collision_pos": [0.0, 0.0, 0.126610],
        "half_size": [0.302567, 0.202567, 0.126610],
        "mass": 2.0,
    },
    {
        "name": "tote_b01",
        "file": "Tote_B01_60x40x40cm_PR_V_NVD_01.obj",
        "mesh": "siemens_tote_b01_mesh",
        "material": "siemens_blue_plastic",
        "rgba": [0.08, 0.25, 0.70, 1.0],
        "visual_pos": [0.0, 0.0, 0.0],
        "collision_pos": [0.0, 0.0, 0.200001],
        "half_size": [0.300000, 0.200000, 0.200001],
        "mass": 2.5,
    },
]


SIEMENS_XML_OBJECT_CLASSES = {
    "container_h10": ContainerH10Object,
    "container_h01": ContainerH01Object,
    "tote_b01": ToteB01Object,
}


SIEMENS_XML_GRASP_SITE_PARAMS = {
    "container_h10": {
        "grasp_x": 0.165,
        "near_side_y": -0.175,
        "upper_rim_z": 0.085,
    },
    "container_h01": {
        "grasp_x": 0.166,
        "near_side_y": -0.218,
        "upper_rim_z": 0.125,
    },
    "tote_b01": {
        "grasp_x": 0.165,
        "near_side_y": -0.215,
        "upper_rim_z": 0.200,
    },
}


def box_diagonal_inertia(mass, half_size):
    full_size = np.array(half_size, dtype=float) * 2.0
    x, y, z = full_size
    return [
        mass * (y * y + z * z) / 12.0,
        mass * (x * x + z * z) / 12.0,
        mass * (x * x + y * y) / 12.0,
    ]


PORT_COLORS = OrderedDict(
    [
        ("conveyor", [0.05, 0.25, 0.85, 1.0]),
        ("shelf", [0.05, 0.55, 0.20, 1.0]),
        ("table", [0.95, 0.75, 0.10, 1.0]),
        ("bin", [0.85, 0.10, 0.08, 1.0]),
    ]
)


LIGHT_TABLE_COLORS = [
    [0.86, 0.93, 0.98, 1.0],
    [0.88, 0.96, 0.88, 1.0],
    [0.99, 0.93, 0.76, 1.0],
    [0.97, 0.88, 0.92, 1.0],
    [0.90, 0.92, 0.99, 1.0],
    [0.91, 0.97, 0.96, 1.0],
    [0.99, 0.90, 0.82, 1.0],
    [0.94, 0.91, 0.98, 1.0],
]

SIEMENS_UNLOWERED_TABLE_Z_OFFSET = 0.70
SIEMENS_3_3FO3ERRPH7X9_SCENE_XML = "objects/siemens/3-3FO3ERRPH7X9/scene_robosuite.xml"
SIEMENS_3_3FO3ERRPH7X9_ROOM_CENTER = (-9.36, -0.71)
SIEMENS_3_3FO3ERRPH7X9_ROOM_HALF_SIZE = (34.0, 22.0)
SIEMENS_3_3FO3ERRPH7X9_ROOM_HEIGHT = 4.0
SIEMENS_3_3FO3ERRPH7X9_HIDDEN_SCENE_GEOMS = {
    "usd_0008_Assets_Other_Actor_0002_Geom_body_mesh_0000_c7b12fcb",
    "usd_0111_Assets_Other_Actor_0008_Geom_body_mesh_0000_17d47a49",
}
SIEMENS_3_3FO3ERRPH7X9_GREEN_TOTE_REPLACEMENTS = (
    (
        "green_tote_b01_lower",
        np.array([11.867624, 3.195400, 1.200000], dtype=float),
        np.array([0.600000, 0.400000, 0.400001], dtype=float),
    ),
    (
        "green_tote_b01_upper",
        np.array([11.867624, 4.624856, 1.200000], dtype=float),
        np.array([0.600000, 0.400000, 0.400001], dtype=float),
    ),
)
SIEMENS_3_3FO3ERRPH7X9_GREEN_TOTE_RGBA = [0.0126637202, 0.14198029, 0.0126637202, 1.0]
SIEMENS_3_3FO3ERRPH7X9_GREEN_TOTE_SUPPORT_SIZE = np.array([0.340, 0.240, 0.010], dtype=float)

SCENE_AABB_COLLISION_PREFIX = "scene_aabb_proxy_"
JUDGE_ROBOT_GEOM_PREFIXES = ("robot0_", "gripper0_", "mobilebase0_")
SCENE_AABB_COLLISION_HEIGHTS = {
    "station": (0.55, 0.55),
    "production_line_equipment": (0.95, 0.95),
    "production_line_belt": (0.55, 0.55),
    "production_line_local_protrusion": (0.55, 0.55),
    "production_line_local_equipment": (0.95, 0.95),
    "side_station": (0.55, 0.55),
    "right_side_device": (0.55, 0.55),
}
SCENE_AABB_COLLISION_LOWERED_HEIGHT = (0.45, 0.45)
SCENE_AABB_COLLISION_HEIGHT_OVERRIDES = {
    "input_1": SCENE_AABB_COLLISION_LOWERED_HEIGHT,
    "output_1": SCENE_AABB_COLLISION_LOWERED_HEIGHT,
    "input_2": SCENE_AABB_COLLISION_LOWERED_HEIGHT,
    "output_2": SCENE_AABB_COLLISION_LOWERED_HEIGHT,
    "input_3": SCENE_AABB_COLLISION_LOWERED_HEIGHT,
    "output_3": SCENE_AABB_COLLISION_LOWERED_HEIGHT,
    "input_4": SCENE_AABB_COLLISION_LOWERED_HEIGHT,
    "output_4": SCENE_AABB_COLLISION_LOWERED_HEIGHT,
    "input_5": SCENE_AABB_COLLISION_LOWERED_HEIGHT,
    "output_5": SCENE_AABB_COLLISION_LOWERED_HEIGHT,
    "input_6": SCENE_AABB_COLLISION_LOWERED_HEIGHT,
    "output_6": SCENE_AABB_COLLISION_LOWERED_HEIGHT,
    "production_line_1": SCENE_AABB_COLLISION_LOWERED_HEIGHT,
    "production_line_2": SCENE_AABB_COLLISION_LOWERED_HEIGHT,
    "production_line_3": SCENE_AABB_COLLISION_LOWERED_HEIGHT,
    "production_line_4": SCENE_AABB_COLLISION_LOWERED_HEIGHT,
    "production_line_5": SCENE_AABB_COLLISION_LOWERED_HEIGHT,
    "production_line_6": SCENE_AABB_COLLISION_LOWERED_HEIGHT,
    "line_5_crossbar_1": SCENE_AABB_COLLISION_LOWERED_HEIGHT,
    "line_6_output_side_protrusion": SCENE_AABB_COLLISION_LOWERED_HEIGHT,
    "side_table_pos_y_1": SCENE_AABB_COLLISION_LOWERED_HEIGHT,
    "side_table_neg_y_1": SCENE_AABB_COLLISION_LOWERED_HEIGHT,
    "side_table_pos_y_2": SCENE_AABB_COLLISION_LOWERED_HEIGHT,
    "side_table_neg_y_2": SCENE_AABB_COLLISION_LOWERED_HEIGHT,
}
SCENE_AABB_COLLISION_BOXES = (
    ("input_1", "station", [-14.544, 5.010], [0.836, 1.684]),
    ("output_1", "station", [-16.198, -7.290], [0.502, 1.010]),
    ("input_2", "station", [-9.761, 5.010], [0.836, 1.684]),
    ("output_2", "station", [-11.414, -7.135], [0.502, 1.010]),
    ("input_3", "station", [-4.316, 5.010], [0.836, 1.684]),
    ("output_3", "station", [-5.969, -7.077], [0.502, 1.010]),
    ("input_4", "station", [1.487, 5.010], [0.836, 1.684]),
    ("output_4", "station", [-0.166, -7.290], [0.502, 1.010]),
    ("input_5", "station", [7.186, 3.938], [0.500, 0.750]),
    ("output_5", "station", [4.872, -7.261], [0.508, 0.918]),
    ("input_6", "station", [11.937, 3.932], [0.500, 0.750]),
    ("output_6", "station", [10.032, -7.267], [0.508, 0.918]),
    ("production_line_1", "production_line_equipment", [-15.589, -0.920], [2.850, 12.100]),
    ("production_line_2", "production_line_equipment", [-10.805, -0.843], [2.900, 12.000]),
    ("production_line_3", "production_line_equipment", [-5.393, -0.815], [2.750, 11.900]),
    ("production_line_4", "production_line_equipment", [0.410, -0.941], [2.450, 11.200]),
    ("production_line_5", "production_line_equipment", [6.071, -0.946], [2.650, 13.300]),
    ("production_line_6", "production_line_equipment", [10.451, -0.948], [2.900, 12.200]),
    ("line_1_belt", "production_line_belt", [-15.427, -1.170], [0.570, 13.250]),
    ("line_2_belt", "production_line_belt", [-10.643, -1.093], [0.570, 13.095]),
    ("line_3_belt", "production_line_belt", [-5.148, -1.165], [0.570, 12.240]),
    ("line_4_belt", "production_line_belt", [0.605, -1.291], [0.570, 12.493]),
    ("line_5_belt", "production_line_belt", [6.152, -0.946], [0.570, 13.603]),
    ("line_6_belt", "production_line_belt", [10.911, -1.048], [0.570, 13.872]),
    ("line_5_crossbar_1", "production_line_local_protrusion", [6.153, -7.073], [1.960, 0.494]),
    ("line_5_crossbar_2", "production_line_local_protrusion", [6.153, -5.090], [1.960, 0.494]),
    ("line_5_crossbar_3", "production_line_local_protrusion", [6.153, -3.173], [1.960, 0.494]),
    ("line_5_crossbar_4", "production_line_local_protrusion", [6.153, -1.288], [1.960, 0.494]),
    ("line_6_output_side_protrusion", "production_line_local_equipment", [10.901, -7.430], [1.050, 1.650]),
    ("line_6_lower_machine_module", "production_line_local_equipment", [10.750, -3.650], [1.750, 5.150]),
    ("line_6_center_machine_module", "production_line_local_equipment", [10.950, 0.960], [1.550, 3.300]),
    ("line_6_input_side_protrusion", "production_line_local_equipment", [9.530, 3.370], [1.800, 3.500]),
    ("line_6_input_end_module", "production_line_local_equipment", [10.834, 5.624], [0.970, 1.528]),
    ("side_table_pos_y_1", "side_station", [-5.856, 8.473], [1.683, 0.836]),
    ("side_table_neg_y_1", "side_station", [-5.855, -11.030], [1.683, 0.836]),
    ("side_table_pos_y_2", "side_station", [0.144, 8.473], [1.683, 0.836]),
    ("side_table_neg_y_2", "side_station", [0.145, -11.030], [1.683, 0.836]),
    ("right_side_device_1", "right_side_device", [18.820, -6.216], [0.575, 0.553]),
    ("right_side_device_2", "right_side_device", [18.820, -2.416], [0.575, 0.553]),
    ("right_side_device_3", "right_side_device", [18.820, 1.384], [0.575, 0.553]),
    ("right_side_device_4", "right_side_device", [18.820, 5.184], [0.575, 0.553]),
)
SCENE_AABB_LOOSE_FRAME_BOXES = (
    ("loose_frame_between_line_2_3_blue", [-8.858, -1.643, 0.160], [0.482, 0.470, 0.320]),
    ("loose_frame_between_line_2_3_white", [-7.901, -1.670, 0.130], [0.717, 0.717, 0.260]),
    ("loose_frame_input_side_mid", [-0.846, 5.231, 0.127], [0.605, 0.405, 0.253]),
    ("loose_frame_between_line_4_5_blue_1", [3.647, -1.021, 0.160], [0.477, 0.477, 0.320]),
    ("loose_frame_between_line_4_5_blue_2", [3.685, -3.889, 0.160], [0.435, 0.353, 0.320]),
    ("loose_frame_between_line_4_5_blue_3", [3.786, -4.679, 0.160], [0.478, 0.434, 0.320]),
    ("loose_frame_between_line_5_6_gray_1", [8.358, -0.886, 0.127], [0.405, 0.605, 0.253]),
    ("loose_frame_between_line_5_6_gray_2", [8.432, 0.476, 0.127], [0.666, 0.507, 0.253]),
    ("loose_frame_between_line_5_6_gray_3", [8.363, 8.343, 0.127], [0.716, 0.667, 0.253]),
)


TIAGO_GRIPPER_Z_DOWN_INIT_QPOS = np.array(
    [
        0.35,
        0.0,
        -0.9,
        0.742599,
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
        -1.029744,
    ]
)


class Siemens3_3FO3ERRPH7X9Arena(Arena):
    """EmptyArena room + 3-3FO3ERRPH7X9 production-line scene."""

    def __init__(self):
        super().__init__(xml_path_completion("arenas/empty_arena.xml"))
        self._hide_builtin_floor()
        self._resize_room()
        self._merge_scene()
        self._add_scene_aabb_collision_boxes()
        self._add_factory_lights()

    def _hide_builtin_floor(self):
        floor = self.worldbody.find("./geom[@name='floor']")
        if floor is not None:
            floor.attrib.pop("material", None)
            floor.set("rgba", "0 0 0 0")
            floor.set("group", "3")

    def _merge_scene(self):
        factory_scene = MujocoXML(xml_path_completion(SIEMENS_3_3FO3ERRPH7X9_SCENE_XML))
        self.merge(factory_scene)
        self._hide_scene_geoms(SIEMENS_3_3FO3ERRPH7X9_HIDDEN_SCENE_GEOMS)

    def _hide_scene_geoms(self, geom_names):
        for geom_name in geom_names:
            geom = self.worldbody.find(f".//geom[@name='{geom_name}']")
            if geom is None:
                continue
            geom.attrib.pop("material", None)
            geom.set("rgba", "0 0 0 0")
            geom.set("group", "3")

    def _add_scene_aabb_collision_boxes(self):
        for box_name, box_kind, center_xy, size_xy in SCENE_AABB_COLLISION_BOXES:
            z_center, z_half_size = SCENE_AABB_COLLISION_HEIGHT_OVERRIDES.get(
                box_name,
                SCENE_AABB_COLLISION_HEIGHTS[box_kind],
            )
            center_xy = np.asarray(center_xy, dtype=float)
            half_size_xy = np.asarray(size_xy, dtype=float) * 0.5
            self._add_aabb_collision_box(
                box_name,
                [center_xy[0], center_xy[1], z_center],
                [half_size_xy[0], half_size_xy[1], z_half_size],
            )

        for box_name, center_xyz, size_xyz in SCENE_AABB_LOOSE_FRAME_BOXES:
            center_xyz = np.asarray(center_xyz, dtype=float)
            half_size_xyz = np.asarray(size_xyz, dtype=float) * 0.5
            self._add_aabb_collision_box(box_name, center_xyz, half_size_xyz)

    def _add_aabb_collision_box(self, box_name, center_xyz, half_size_xyz):
        ET.SubElement(
            self.worldbody,
            "geom",
            name=f"{SCENE_AABB_COLLISION_PREFIX}{box_name}",
            type="box",
            pos=array_to_string(center_xyz),
            size=array_to_string(half_size_xyz),
            rgba="0 0 0 0",
            contype="1",
            conaffinity="1",
            friction="1 0.005 0.0001",
            group="3",
        )

    def _resize_room(self):
        cx, cy = SIEMENS_3_3FO3ERRPH7X9_ROOM_CENTER
        hx, hy = SIEMENS_3_3FO3ERRPH7X9_ROOM_HALF_SIZE
        hz = SIEMENS_3_3FO3ERRPH7X9_ROOM_HEIGHT / 2.0

        floor = self.worldbody.find("./geom[@name='floor']")
        if floor is not None:
            floor.set("pos", f"{cx} {cy} 0")
            floor.set("size", f"{hx} {hy} 0.125")

        wall_specs = {
            "wall_x_min_visual": (cx - hx, cy, hz, 0.05, hy, hz),
            "wall_x_max_visual": (cx + hx, cy, hz, 0.05, hy, hz),
            "wall_y_min_visual": (cx, cy - hy, hz, hx, 0.05, hz),
            "wall_y_max_visual": (cx, cy + hy, hz, hx, 0.05, hz),
        }

        for name, (x, y, z, sx, sy, sz) in wall_specs.items():
            wall = self.worldbody.find(f"./geom[@name='{name}']")
            if wall is not None:
                wall.set("pos", f"{x} {y} {z}")
                wall.set("size", f"{sx} {sy} {sz}")

    def _add_factory_lights(self):
        cx, cy = SIEMENS_3_3FO3ERRPH7X9_ROOM_CENTER

        ET.SubElement(
            self.worldbody,
            "light",
            name="siemens_5_overhead_light",
            pos=f"{cx} {cy} 18",
            dir="0 0 -1",
            directional="true",
            diffuse="1.0 1.0 0.95",
            ambient="0.45 0.45 0.45",
            specular="0.25 0.25 0.25",
            castshadow="false",
        )

        ET.SubElement(
            self.worldbody,
            "light",
            name="siemens_5_front_fill_light",
            pos=f"{cx + 8.0} {cy - 10.0} 10",
            dir="-0.4 0.5 -1",
            directional="true",
            diffuse="0.7 0.75 0.8",
            ambient="0.25 0.25 0.25",
            specular="0.15 0.15 0.15",
            castshadow="false",
        )


class FactorySorting3_3FO3ERRPH7X9(ManipulationEnv):
    """
    Static electronics-factory sorting scene for Tiago.

    The robot stands in the middle facing the input side. Four input stations
    and four output stations are represented by identical light-colored tables,
    with one plastic crate fixed on each input table.
    """

    def __init__(
        self,
        robots="Tiago",
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise=None,
        use_camera_obs=True,
        use_object_obs=True,
        reward_shaping=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera=None,
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        robot_base_pos=None,
        robot_base_ori=None,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        use_siemens_arena=True,
        include_legacy_static_scene=False,
        include_material_objects=False,
        include_siemens_line_objects=False,
        siemens_line_object_poses=None,
        seed=None,
    ):
        self.use_object_obs = use_object_obs
        self.reward_shaping = reward_shaping
        self.use_siemens_arena = use_siemens_arena
        self.include_legacy_static_scene = include_legacy_static_scene
        self.include_material_objects = include_material_objects
        self.include_siemens_line_objects = include_siemens_line_objects
        self.siemens_line_object_poses = (
            self._default_siemens_line_object_poses()
            if siemens_line_object_poses is None
            else [np.array(pose, dtype=float) for pose in siemens_line_object_poses]
        )

        self.table_full_size = np.array([0.8, 0.8, 0.05])
        self.table_friction = (1.0, 5e-3, 1e-4)
        self.table_offset = np.array([0.0, 0.0, 0.4])
        self.table_top_z = self.table_offset[2]
        self.crate_scale = 0.55
        self.robot_base_pos = np.array(
            [16.5, 0.0, 0.0] if robot_base_pos is None else robot_base_pos,
            dtype=float,
        )
        self.robot_base_ori = np.array(
            [0.0, 0.0, np.pi] if robot_base_ori is None else robot_base_ori,
            dtype=float,
        )
        self.central_conveyor_x_positions = [-3.5, -1.5, 0.5]
        self.central_conveyor_y_limits = (-1.8, 1.8)

        station_y = [-3.0, -1.0, 1.0, 3.0]
        output_x = 4.6
        input_station_types = list(PORT_COLORS.keys())
        output_station_types = ["table", "bin", "conveyor", "shelf"]
        self.input_ports = OrderedDict()
        self.output_ports = OrderedDict()
        for idx, station_type in enumerate(input_station_types):
            if idx == 0:
                self.input_ports[f"input_{idx + 1}_{station_type}"] = {
                    "kind": station_type,
                    "center": np.array([-2.0, -3.8, 0.0]),
                    "side": "input",
                    "index": idx,
                    "table_color_index": idx,
                }
            elif idx == 1:
                self.input_ports[f"input_{idx + 1}_{station_type}"] = {
                    "kind": station_type,
                    "center": np.array([0, -3.8, 0.0]),
                    "side": "input",
                    "index": idx,
                    "table_color_index": idx,
                }
            elif idx == 2:
                self.input_ports[f"input_{idx + 1}_{station_type}"] = {
                    "kind": station_type,
                    "center": np.array([-2.0, 3.8, 0.0]),
                    "side": "input",
                    "index": idx,
                    "table_color_index": idx,
                }
            elif idx == 3:
                self.input_ports[f"input_{idx + 1}_{station_type}"] = {
                    "kind": station_type,
                    "center": np.array([0, 3.8, 0.0]),
                    "side": "input",
                    "index": idx,
                    "table_color_index": idx,
                }
        for idx, (station_type, y_pos) in enumerate(zip(output_station_types, station_y)):
            self.output_ports[f"output_{idx + 1}_{station_type}"] = {
                "kind": station_type,
                "center": np.array([output_x, y_pos, 0.0]),
                "side": "output",
                "index": idx,
                "table_color_index": len(self.input_ports) + idx,
            }

        self.static_scene_objects = []
        self.material_objects = []
        self.material_metadata = OrderedDict()
        self.obj_body_id = {}
        self.has_judge_collision = False
        self._judge_last_collision_pair = None
        self._judge_robot_geom_names = None

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            base_types=base_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    def reward(self, action=None):
        return 0.0

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)

        judge_collision_pairs = self._judge_collision_pairs()
        if judge_collision_pairs:
            self.has_judge_collision = True
            self._judge_last_collision_pair = judge_collision_pairs[0]
            self._print_judge_collision_info(judge_collision_pairs)

        info["has_judge_collision"] = self.has_judge_collision
        return reward, done, info

    def _detect_judge_collision(self):
        judge_collision_pairs = self._judge_collision_pairs()
        if judge_collision_pairs:
            self._judge_last_collision_pair = judge_collision_pairs[0]
            return True
        return False

    def _judge_collision_pairs(self):
        robot_geoms = self._get_judge_robot_geom_names()
        pairs = set()

        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            geom1 = self.sim.model.geom_id2name(contact.geom1)
            geom2 = self.sim.model.geom_id2name(contact.geom2)
            geom1_is_proxy = self._is_judge_proxy_geom(geom1)
            geom2_is_proxy = self._is_judge_proxy_geom(geom2)

            if geom1_is_proxy == geom2_is_proxy:
                continue

            if geom1_is_proxy and geom2 in robot_geoms:
                pairs.add((geom2, geom1))
            elif geom2_is_proxy and geom1 in robot_geoms:
                pairs.add((geom1, geom2))

        return sorted(pairs)

    def _print_judge_collision_info(self, collision_pairs):
        base_xy, base_yaw = self._judge_robot_base_pose()

        for robot_geom, proxy_geom in collision_pairs:
            print("judge_collision_detected:")
            print(f"robot_base_x: {base_xy[0]:.6f}")
            print(f"robot_base_y: {base_xy[1]:.6f}")
            print(f"robot_base_yaw: {base_yaw:.6f}")
            print(f"robot_geom: {robot_geom}")
            print(f"proxy_geom: {proxy_geom}")

    def _judge_robot_base_pose(self):
        try:
            robot = self.robots[0]
            site_name = robot.robot_model.base.correct_naming("center")
            pos = np.array(self.sim.data.site_xpos[self.sim.model.site_name2id(site_name)])
            mat = self.sim.data.get_site_xmat(site_name)
            return pos[:2], float(mat2euler(mat)[2])
        except Exception:
            return np.array([np.nan, np.nan], dtype=float), float("nan")

    def _is_judge_proxy_geom(self, geom_name):
        return geom_name is not None and geom_name.startswith(SCENE_AABB_COLLISION_PREFIX)

    def _get_judge_robot_geom_names(self):
        if self._judge_robot_geom_names is not None:
            return self._judge_robot_geom_names

        names = set()
        for robot in self.robots:
            names.update(getattr(robot.robot_model, "contact_geoms", []))
            prefix = getattr(robot.robot_model, "naming_prefix", "")
            if prefix:
                names.update(
                    geom_name
                    for geom_name in self._all_geom_names()
                    if geom_name.startswith(prefix)
                )

        names.update(
            geom_name
            for geom_name in self._all_geom_names()
            if geom_name.startswith(JUDGE_ROBOT_GEOM_PREFIXES)
        )
        self._judge_robot_geom_names = names
        return names

    def _all_geom_names(self):
        for geom_id in range(self.sim.model.ngeom):
            geom_name = self.sim.model.geom_id2name(geom_id)
            if geom_name is not None:
                yield geom_name

    def _default_siemens_line_object_poses(self):
        # One line-end table per imported Siemens production-line actor.
        # Lines 1-4 are centered on the matched dark-teal tabletop meshes.
        # Lines 5-6 are the original front-view placements that were already correct.
        return [
            np.array([-14.544, 5.010, 0.395 + SIEMENS_UNLOWERED_TABLE_Z_OFFSET]),
            np.array([-9.761, 5.010, 0.395 + SIEMENS_UNLOWERED_TABLE_Z_OFFSET]),
            np.array([-4.316, 5.010, 0.395 + SIEMENS_UNLOWERED_TABLE_Z_OFFSET]),
            np.array([1.487, 5.010, 0.395 + SIEMENS_UNLOWERED_TABLE_Z_OFFSET]),
            np.array([7.186, 3.938, 0.365 + SIEMENS_UNLOWERED_TABLE_Z_OFFSET]),
            np.array([11.937, 3.932, 0.365 + SIEMENS_UNLOWERED_TABLE_Z_OFFSET]),
        ]

    def _infer_siemens_table_poses(self):
        try:
            scene_path = Path(xml_path_completion(SIEMENS_3_3FO3ERRPH7X9_SCENE_XML))
            root = ET.parse(scene_path).getroot()
            asset = root.find("asset")
            if asset is None:
                return []

            materials = {}
            for material in asset.findall("material"):
                rgba = material.get("rgba")
                if rgba is None:
                    continue
                materials[material.get("name")] = np.array([float(v) for v in rgba.split()], dtype=float)

            mesh_files = {
                mesh.get("name"): scene_path.parent / mesh.get("file")
                for mesh in asset.findall("mesh")
                if mesh.get("name") is not None and mesh.get("file") is not None
            }

            bounds_cache = {}
            candidates = []
            for geom in root.findall(".//worldbody//geom"):
                if geom.get("type") != "mesh":
                    continue
                rgba = materials.get(geom.get("material"))
                if rgba is None or not self._is_siemens_table_material(rgba):
                    continue
                mesh_path = mesh_files.get(geom.get("mesh"))
                if mesh_path is None:
                    continue
                if mesh_path not in bounds_cache:
                    bounds_cache[mesh_path] = self._obj_vertex_bounds(mesh_path)
                bounds = bounds_cache[mesh_path]
                if bounds is None:
                    continue

                lower, upper = bounds
                dims = upper - lower
                area = dims[0] * dims[1]
                if dims[2] > 0.18 or area < 0.12 or area > 8.0:
                    continue
                if upper[2] < 0.35 or upper[2] > 1.8:
                    continue
                if dims[0] < 0.20 or dims[1] < 0.20 or max(dims[0], dims[1]) > 4.0:
                    continue

                center = (lower + upper) / 2.0
                candidates.append(
                    {
                        "pose": np.array([center[0], center[1], upper[2] + 0.005], dtype=float),
                        "area": area,
                    }
                )

            return self._select_six_siemens_table_poses(candidates)
        except Exception:
            return []

    @staticmethod
    def _select_six_siemens_table_poses(candidates):
        if len(candidates) < 6:
            return []

        candidates = sorted(candidates, key=lambda item: item["area"], reverse=True)
        unique = []
        for candidate in candidates:
            pose = candidate["pose"]
            if any(np.linalg.norm(pose[:2] - previous["pose"][:2]) < 2.2 for previous in unique):
                continue
            unique.append(candidate)

        if len(unique) < 6:
            return []

        # The required work tables are the repeated line-end tables, not small
        # screens or one-off colored details. Keep the right-side repeated
        # column, then select one center per production-line row.
        x_values = np.array([item["pose"][0] for item in unique], dtype=float)
        right_cutoff = np.percentile(x_values, 55)
        right_side = [item for item in unique if item["pose"][0] >= right_cutoff]
        if len(right_side) < 6:
            right_side = unique

        right_side.sort(key=lambda item: item["pose"][1])
        groups = []
        for item in right_side:
            y = item["pose"][1]
            if not groups or abs(y - np.mean([entry["pose"][1] for entry in groups[-1]])) > 2.4:
                groups.append([item])
            else:
                groups[-1].append(item)

        selected = []
        for group in groups:
            selected.append(max(group, key=lambda item: (item["area"], item["pose"][0]))["pose"])

        if len(selected) > 6:
            selected = sorted(selected, key=lambda pose: pose[1])
            keep_indexes = np.linspace(0, len(selected) - 1, 6).round().astype(int)
            selected = [selected[idx] for idx in keep_indexes]

        if len(selected) != 6:
            return []

        return sorted(selected, key=lambda pose: pose[1], reverse=True)

    @staticmethod
    def _is_siemens_table_material(rgba):
        r, g, b = rgba[:3]
        return g > 0.45 and r < 0.45 and (b > 0.25 or r < 0.20)

    @staticmethod
    def _obj_vertex_bounds(mesh_path):
        vertices = []
        with open(mesh_path, "r", encoding="utf-8", errors="ignore") as mesh_file:
            for line in mesh_file:
                if not line.startswith("v "):
                    continue
                values = line.split()
                if len(values) < 4:
                    continue
                vertices.append([float(values[1]), float(values[2]), float(values[3])])
        if not vertices:
            return None
        vertices = np.asarray(vertices, dtype=float)
        return vertices.min(axis=0), vertices.max(axis=0)

    def _siemens_line_object_specs(self):
        spec_by_name = {spec["name"]: spec for spec in SIEMENS_OBJECT_SPECS}
        return [
            spec_by_name["container_h01"],
            spec_by_name["container_h10"],
            spec_by_name["tote_b01"],
        ]

    def _siemens_line_object_offsets(self):
        return [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, -0.32, 0.0]),
            np.array([0.0, 0.32, 0.0]),
        ]

    def _siemens_static_table_support_surfaces(self):
        return [
            (
                np.array([-16.198, -7.290, 0.393 + SIEMENS_UNLOWERED_TABLE_Z_OFFSET]),
                np.array([0.251, 0.505, 0.010]),
            ),
            (
                np.array([-11.414, -7.135, 0.392 + SIEMENS_UNLOWERED_TABLE_Z_OFFSET]),
                np.array([0.251, 0.505, 0.010]),
            ),
            (
                np.array([-5.969, -7.077, 0.392 + SIEMENS_UNLOWERED_TABLE_Z_OFFSET]),
                np.array([0.251, 0.505, 0.010]),
            ),
            (
                np.array([-0.166, -7.290, 0.393 + SIEMENS_UNLOWERED_TABLE_Z_OFFSET]),
                np.array([0.251, 0.505, 0.010]),
            ),
            (
                np.array([4.872, -7.261, 0.393 + SIEMENS_UNLOWERED_TABLE_Z_OFFSET]),
                np.array([0.254, 0.459, 0.010]),
            ),
            (
                np.array([10.032, -7.267, 0.393 + SIEMENS_UNLOWERED_TABLE_Z_OFFSET]),
                np.array([0.254, 0.459, 0.010]),
            ),
        ]

    def _make_static_box(self, name, half_size, pos, rgba, obj_type="all", density=1000.0, friction=None):
        obj = BoxObject(
            name=name,
            size=half_size,
            rgba=rgba,
            density=density,
            friction=friction,
            joints=None,
            obj_type=obj_type,
        )
        obj.get_obj().set("pos", array_to_string(pos))
        return obj

    def _add_box(self, objects, name, center, half_size, rgba, obj_type="all", density=1000.0, friction=None):
        objects.append(
            self._make_static_box(
                name=name,
                half_size=half_size,
                pos=center,
                rgba=rgba,
                obj_type=obj_type,
                density=density,
                friction=friction,
            )
        )

    def _add_station(self, name, info):
        center = info["center"]
        color = LIGHT_TABLE_COLORS[info["table_color_index"] % len(LIGHT_TABLE_COLORS)]
        table_half_size = self.table_full_size / 2.0
        table_bottom_z = self.table_top_z - self.table_full_size[2]
        leg_half_height = table_bottom_z / 2.0
        leg_rgba = [0.78, 0.80, 0.78, 1.0]

        self._add_box(
            self.static_scene_objects,
            f"{name}_table_top",
            center + np.array([0.0, 0.0, self.table_top_z - table_half_size[2]]),
            table_half_size,
            color,
            friction=self.table_friction,
        )

        leg_offsets = [
            (-table_half_size[0] + 0.10, -table_half_size[1] + 0.10),
            (-table_half_size[0] + 0.10, table_half_size[1] - 0.10),
            (table_half_size[0] - 0.10, -table_half_size[1] + 0.10),
            (table_half_size[0] - 0.10, table_half_size[1] - 0.10),
        ]
        for leg_idx, (x_offset, y_offset) in enumerate(leg_offsets):
            self._add_box(
                self.static_scene_objects,
                f"{name}_table_leg_{leg_idx}",
                center + np.array([x_offset, y_offset, leg_half_height]),
                [0.025, 0.025, leg_half_height],
                leg_rgba,
            )

    def _add_central_conveyor(self, name, x_pos, y_min, y_max):
        # Keep the visible roller surface flush with the table surface.
        top = self.table_top_z - 0.022
        center = np.array([x_pos, (y_min + y_max) / 2.0, 0.0])
        half_width = 0.32
        half_length = (y_max - y_min) / 2.0
        color = PORT_COLORS["conveyor"]
        station_objects = self.static_scene_objects
        leg_top = top - 0.080
        leg_half_height = leg_top / 2.0
        leg_x_offsets = [-half_width + 0.070, half_width - 0.070]
        leg_y_positions = [y_min + 0.350, y_max - 0.350]

        self._add_box(
            station_objects,
            f"{name}_base",
            center + np.array([0.0, 0.0, top - 0.040]),
            [half_width, half_length, 0.040],
            [0.36, 0.38, 0.40, 1.0],
        )
        self._add_box(
            station_objects,
            f"{name}_belt_visual",
            center + np.array([0.0, 0.0, top + 0.004]),
            [half_width - 0.055, half_length - 0.080, 0.006],
            [0.03, 0.035, 0.04, 1.0],
            obj_type="visual",
        )

        roller_offsets = np.linspace(-half_length + 0.25, half_length - 0.25, 15)
        for roller_idx, y_offset in enumerate(roller_offsets):
            self._add_box(
                station_objects,
                f"{name}_roller_{roller_idx}",
                center + np.array([0.0, y_offset, top + 0.014]),
                [half_width - 0.075, 0.012, 0.008],
                [0.80, 0.82, 0.82, 1.0],
                obj_type="visual",
            )

        for rail_idx, x_offset in enumerate([-half_width + 0.010, half_width - 0.010]):
            self._add_box(
                station_objects,
                f"{name}_rail_{rail_idx}",
                center + np.array([x_offset, 0.0, top + 0.035]),
                [0.012, half_length, 0.045],
                color,
            )

        for leg_x_idx, x_offset in enumerate(leg_x_offsets):
            for leg_y_idx, y_pos in enumerate(leg_y_positions):
                self._add_box(
                    station_objects,
                    f"{name}_leg_{leg_x_idx}_{leg_y_idx}",
                    np.array([x_pos + x_offset, y_pos, leg_half_height]),
                    [0.025, 0.025, leg_half_height],
                    [0.42, 0.42, 0.42, 1.0],
                )

    def _construct_static_scene_objects(self):
        self.static_scene_objects = []
        for name, info in self.input_ports.items():
            self._add_station(name, info)
        for name, info in self.output_ports.items():
            self._add_station(name, info)
        for lane_idx, x_pos in enumerate(self.central_conveyor_x_positions):
            self._add_central_conveyor(
                f"central_conveyor_{lane_idx + 1}",
                x_pos,
                self.central_conveyor_y_limits[0],
                self.central_conveyor_y_limits[1],
            )

        # Floor markers keep the two work zones visually distinct.
        self._add_box(
            self.static_scene_objects,
            "input_lane_marker",
            [-4.0, 0.0, 0.011],
            [0.08, 3.65, 0.010],
            [0.95, 0.80, 0.05, 1.0],
            obj_type="visual",
        )
        self._add_box(
            self.static_scene_objects,
            "output_lane_marker",
            [4.0, 0.0, 0.011],
            [0.08, 3.65, 0.010],
            [0.95, 0.80, 0.05, 1.0],
            obj_type="visual",
        )

    def _construct_material_objects(self):
        self.material_objects = []
        self.material_metadata = OrderedDict()
        for idx, (port_name, port_info) in enumerate(self.input_ports.items()):
            if idx == 0:
                obj_name = f"{port_name}_plastic_crate"
                crate = PlasticCrateObject(name=obj_name, scale=self.crate_scale)
                self._add_plastic_crate_grasp_sites(crate, obj_name)
                self.material_objects.append(obj_name)
                self.material_metadata[obj_name] = {
                    "kind": "plastic_crate",
                    "port_name": port_name,
                    "port_info": port_info,
                    "slot": 0,
                    "model": crate,
                    "joint_name": crate.joints[0],
                }
                continue

            spec = SIEMENS_OBJECT_SPECS[idx % len(SIEMENS_OBJECT_SPECS)]
            obj_name = f"{port_name}_{spec['name']}"
            xml_object_class = SIEMENS_XML_OBJECT_CLASSES.get(spec["name"])
            if xml_object_class is not None:
                xml_object = xml_object_class(name=obj_name)
                self._align_siemens_xml_visual(xml_object, obj_name, spec)
                self._add_xml_container_grasp_sites(
                    xml_object,
                    obj_name,
                    **SIEMENS_XML_GRASP_SITE_PARAMS[spec["name"]],
                )
                self.material_objects.append(obj_name)
                self.material_metadata[obj_name] = {
                    "kind": f"{spec['name']}_xml",
                    "port_name": port_name,
                    "port_info": port_info,
                    "slot": 0,
                    "model": xml_object,
                    "joint_name": xml_object.joints[0],
                    "spec": spec,
                }
                continue

            self.material_objects.append(obj_name)
            self.material_metadata[obj_name] = {
                "kind": "siemens",
                "port_name": port_name,
                "port_info": port_info,
                "slot": 0,
                "spec": spec,
            }

    def _infer_siemens_table_poses(self):
        """Infer the six output tables from the compiled Siemens scene."""
        try:
            import mujoco

            scene_path = Path(xml_path_completion(SIEMENS_3_3FO3ERRPH7X9_SCENE_XML))
            model = mujoco.MjModel.from_xml_path(str(scene_path))
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)

            candidates = []
            mesh_geom_type = int(mujoco.mjtGeom.mjGEOM_MESH)
            box_geom_type = int(mujoco.mjtGeom.mjGEOM_BOX)

            for geom_id in range(model.ngeom):
                rgba = np.asarray(model.geom_rgba[geom_id], dtype=float)
                if not self._is_siemens_target_tabletop_rgba(rgba):
                    continue

                geom_type = int(model.geom_type[geom_id])
                if geom_type == mesh_geom_type:
                    mesh_id = int(model.geom_dataid[geom_id])
                    vert_adr = int(model.mesh_vertadr[mesh_id])
                    vert_num = int(model.mesh_vertnum[mesh_id])
                    if vert_num <= 0:
                        continue

                    local_vertices = np.asarray(model.mesh_vert[vert_adr : vert_adr + vert_num], dtype=float)
                    geom_xmat = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
                    geom_xpos = np.asarray(data.geom_xpos[geom_id], dtype=float)
                    world_vertices = local_vertices @ geom_xmat.T + geom_xpos
                    lower = world_vertices.min(axis=0)
                    upper = world_vertices.max(axis=0)
                elif geom_type == box_geom_type:
                    geom_xpos = np.asarray(data.geom_xpos[geom_id], dtype=float)
                    half_size = np.asarray(model.geom_size[geom_id][:3], dtype=float)
                    lower = geom_xpos - half_size
                    upper = geom_xpos + half_size
                else:
                    continue

                dims = upper - lower
                long_side, short_side = sorted(dims[:2], reverse=True)
                area = float(dims[0] * dims[1])

                if dims[2] > 0.18:
                    continue
                if not 0.55 <= upper[2] <= 1.35:
                    continue
                if not 0.35 <= long_side <= 2.40:
                    continue
                if not 0.25 <= short_side <= 1.30:
                    continue
                if not 0.12 <= area <= 2.20:
                    continue

                center = (lower + upper) * 0.5
                candidates.append(
                    {
                        "pose": np.array([center[0], center[1], upper[2] + 0.06], dtype=float),
                        "area": area,
                        "x": float(center[0]),
                        "y": float(center[1]),
                    }
                )

            return self._select_six_output_table_poses(candidates)
        except Exception:
            return []

    @staticmethod
    def _is_siemens_target_tabletop_rgba(rgba):
        r, g, b = rgba[:3]
        return (g > 0.55 and b > 0.45 and r < 0.25) or (g > 0.60 and r < 0.35 and b < 0.35)

    @staticmethod
    def _select_six_output_table_poses(candidates):
        if len(candidates) < 6:
            return []

        candidates = sorted(candidates, key=lambda item: item["area"], reverse=True)
        unique = []
        for candidate in candidates:
            pose = candidate["pose"]
            if any(np.linalg.norm(pose[:2] - previous["pose"][:2]) < 1.10 for previous in unique):
                continue
            unique.append(candidate)

        if len(unique) < 6:
            return []

        # The six output tables are the repeated table surfaces on the same
        # side of the Siemens production lines. Prefer the side with more
        # table-like cyan / green tops, then keep the six ordered line outputs.
        xs = np.asarray([item["x"] for item in unique], dtype=float)
        x_mid = float(np.median(xs))
        right_side = [item for item in unique if item["x"] >= x_mid]
        left_side = [item for item in unique if item["x"] < x_mid]
        side = right_side if len(right_side) >= 6 else left_side
        if len(side) < 6:
            side = unique

        side = sorted(side, key=lambda item: item["area"], reverse=True)
        line_groups = []
        for item in sorted(side, key=lambda entry: entry["y"]):
            if not line_groups:
                line_groups.append([item])
                continue

            group_y = np.mean([entry["y"] for entry in line_groups[-1]])
            if abs(item["y"] - group_y) > 1.35:
                line_groups.append([item])
            else:
                line_groups[-1].append(item)

        selected = [max(group, key=lambda entry: entry["area"])["pose"] for group in line_groups]
        if len(selected) > 6:
            selected = sorted(selected, key=lambda pose: pose[1])
            keep_indexes = np.linspace(0, len(selected) - 1, 6).round().astype(int)
            selected = [selected[index] for index in keep_indexes]

        if len(selected) != 6:
            return []

        return sorted(selected, key=lambda pose: pose[1], reverse=True)

    def _construct_siemens_line_objects(self):
        if len(self.siemens_line_object_poses) != 6:
            raise ValueError("siemens_line_object_poses must contain exactly 6 poses, one for each production line.")

        object_specs = self._siemens_line_object_specs()
        support_surfaces = [
            (
                np.array([-14.694, 5.010, 0.393 + SIEMENS_UNLOWERED_TABLE_Z_OFFSET]),
                np.array([0.418, 0.842, 0.010]),
            ),
            (
                np.array([-9.911, 5.010, 0.392 + SIEMENS_UNLOWERED_TABLE_Z_OFFSET]),
                np.array([0.418, 0.842, 0.010]),
            ),
            (
                np.array([-4.466, 5.010, 0.392 + SIEMENS_UNLOWERED_TABLE_Z_OFFSET]),
                np.array([0.418, 0.842, 0.010]),
            ),
            (
                np.array([1.337, 5.010, 0.393 + SIEMENS_UNLOWERED_TABLE_Z_OFFSET]),
                np.array([0.418, 0.842, 0.010]),
            ),
            (
                np.array([7.036, 3.938, 0.360 + SIEMENS_UNLOWERED_TABLE_Z_OFFSET]),
                np.array([0.250, 0.375, 0.010]),
            ),
            (
                np.array([11.787, 3.932, 0.360 + SIEMENS_UNLOWERED_TABLE_Z_OFFSET]),
                np.array([0.250, 0.375, 0.010]),
            ),
        ]

        for line_idx, line_pose in enumerate(self.siemens_line_object_poses, start=1):
            line_pose = np.asarray(line_pose, dtype=float)
            if line_pose.shape[0] == 2:
                line_pose = np.array([line_pose[0], line_pose[1], 1.08], dtype=float)

            spec = object_specs[(line_idx - 1) % len(object_specs)]
            obj_name = f"line_{line_idx}_{spec['name']}"
            support_pose, support_size = support_surfaces[line_idx - 1]
            xml_object_class = SIEMENS_XML_OBJECT_CLASSES.get(spec["name"])

            self.material_objects.append(obj_name)
            if xml_object_class is not None:
                xml_object = xml_object_class(name=obj_name)
                self._align_siemens_xml_visual(xml_object, obj_name, spec)
                self._add_xml_container_grasp_sites(
                    xml_object,
                    obj_name,
                    **SIEMENS_XML_GRASP_SITE_PARAMS[spec["name"]],
                )
                self.material_metadata[obj_name] = {
                    "kind": f"{spec['name']}_xml",
                    "model": xml_object,
                    "joint_name": xml_object.joints[0],
                    "spec": spec,
                    "fixed_pose": line_pose - np.asarray(xml_object.bottom_offset, dtype=float),
                    "fixed_quat": np.array([0.707106781, 0.0, 0.0, 0.707106781]),
                    "support_pose": support_pose,
                    "support_size": support_size,
                }
                continue

            self.material_metadata[obj_name] = {
                "kind": "siemens",
                "model": None,
                "joint_name": f"{obj_name}_free",
                "spec": spec,
                "fixed_pose": line_pose,
                "fixed_quat": np.array([0.707106781, 0.0, 0.0, 0.707106781]),
                "support_pose": support_pose,
                "support_size": support_size,
            }

    def _nearest_input_port_for_pose(self, pose_xy):
        """Return the input port name (e.g. 'input_3') nearest to *pose_xy*."""
        line_poses = self._default_siemens_line_object_poses()
        best_idx = None
        best_dist = float("inf")
        for idx, lp in enumerate(line_poses):
            dist = float(np.linalg.norm(np.asarray(lp, dtype=float)[:2] - np.asarray(pose_xy, dtype=float)[:2]))
            if dist < best_dist:
                best_dist = dist
                best_idx = idx + 1  # 1-based
        return f"input_{best_idx}" if best_idx is not None else None

    def _construct_hidden_green_tote_replacements(self):
        spec = dict(next(item for item in SIEMENS_OBJECT_SPECS if item["name"] == "tote_b01"))
        spec["rgba"] = SIEMENS_3_3FO3ERRPH7X9_GREEN_TOTE_RGBA

        for obj_name, original_center, original_size in SIEMENS_3_3FO3ERRPH7X9_GREEN_TOTE_REPLACEMENTS:
            xml_object = ToteB01Object(name=obj_name)
            self._align_siemens_xml_visual(xml_object, obj_name, spec)
            self._add_xml_container_grasp_sites(
                xml_object,
                obj_name,
                **SIEMENS_XML_GRASP_SITE_PARAMS["tote_b01"],
            )

            original_bottom_z = float(original_center[2] - original_size[2] * 0.5)
            support_pose = np.array([original_center[0], original_center[1], original_bottom_z], dtype=float)
            fixed_pose = support_pose - np.asarray(xml_object.bottom_offset, dtype=float)
            port_name = self._nearest_input_port_for_pose(fixed_pose[:2])

            self.material_objects.append(obj_name)
            self.material_metadata[obj_name] = {
                "kind": "tote_b01_xml",
                "model": xml_object,
                "joint_name": xml_object.joints[0],
                "spec": spec,
                "port_name": port_name,
                "fixed_pose": fixed_pose,
                "fixed_quat": np.array([1.0, 0.0, 0.0, 0.0]),
                "support_pose": support_pose,
                "support_size": SIEMENS_3_3FO3ERRPH7X9_GREEN_TOTE_SUPPORT_SIZE.copy(),
            }

    def _align_siemens_xml_visual(self, xml_object, obj_name, spec):
        visual_geom = xml_object.get_obj().find(f".//geom[@name='{obj_name}_visual']")
        if visual_geom is None:
            return

        visual_pos = np.asarray(spec.get("visual_pos", [0.0, 0.0, 0.0]), dtype=float).copy()
        visual_pos += np.asarray(xml_object.bottom_offset, dtype=float)
        visual_geom.set("pos", array_to_string(visual_pos))
        visual_geom.set("rgba", array_to_string(spec["rgba"]))
        if "material" in visual_geom.attrib:
            del visual_geom.attrib["material"]

    def _add_plastic_crate_grasp_sites(self, crate, obj_name):
        # Object yaw is pi / 2, so local -Y maps to the world +X side facing the robot.
        near_side_y = -0.235
        upper_rim_z = 0.175
        for site_name, site_pos, rgba in (
            (
                f"{obj_name}_right_grasp_site",
                np.array([0.160, near_side_y, upper_rim_z]) * self.crate_scale,
                "1 0.05 0.05 0",
            ),
            (
                f"{obj_name}_left_grasp_site",
                np.array([-0.160, near_side_y, upper_rim_z]) * self.crate_scale,
                "0.05 0.25 1 0",
            ),
            (
                f"{obj_name}_center_site",
                np.zeros(3),
                "1 1 0 0",
            ),
        ):
            ET.SubElement(
                crate.get_obj(),
                "site",
                name=site_name,
                pos=array_to_string(site_pos),
                size="0.025",
                rgba=rgba,
                group="2",
            )

    def _add_xml_container_grasp_sites(self, container, obj_name, grasp_x, near_side_y, upper_rim_z):
        # Object yaw is pi / 2, so local -Y maps to the world +X side facing the robot.
        top_offset = np.asarray(getattr(container, "top_offset", [0.0, 0.0, upper_rim_z]), dtype=float)
        upper_rim_z = float(top_offset[2])
        for site_name, site_pos, rgba in (
            (
                f"{obj_name}_right_grasp_site",
                [grasp_x, near_side_y, upper_rim_z],
                "1 0 0 0",
            ),
            (
                f"{obj_name}_left_grasp_site",
                [-grasp_x, near_side_y, upper_rim_z],
                "0 1 0 0",
            ),
            (
                f"{obj_name}_center_site",
                [0.0, 0.0, upper_rim_z],
                "1 1 0 0",
            ),
        ):
            ET.SubElement(
                container.get_obj(),
                "site",
                name=site_name,
                pos=array_to_string(site_pos),
                size="0.025",
                rgba=rgba,
                group="2",
            )

    def _station_surface_top(self, port_info, slot):
        return self.table_top_z

    def _sample_item_pose(self, metadata):
        if "fixed_pose" in metadata:
            return (
                np.asarray(metadata["fixed_pose"], dtype=float),
                np.asarray(metadata.get("fixed_quat", [1.0, 0.0, 0.0, 0.0]), dtype=float),
            )

        port_info = metadata["port_info"]
        center = port_info["center"].copy()
        surface_z = self._station_surface_top(port_info, metadata["slot"])
        if metadata.get("model") is not None:
            surface_z -= float(metadata["model"].bottom_offset[2])
        pos = center + np.array(
            [
                0.0,
                0.0,
                surface_z,
            ]
        )
        yaw = np.pi / 2.0
        quat = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
        return pos, quat

    def _load_model(self):
        super()._load_model()

        self.robots[0].init_qpos = TIAGO_GRIPPER_Z_DOWN_INIT_QPOS.copy()
        self.robots[0].init_torso_qpos = np.array([TIAGO_GRIPPER_Z_DOWN_INIT_QPOS[0]])
        self.robots[0].robot_model.set_base_xpos(self.robot_base_pos)
        self.robots[0].robot_model.set_base_ori(self.robot_base_ori)

        mujoco_arena = Siemens3_3FO3ERRPH7X9Arena() if self.use_siemens_arena else EmptyArena()
        mujoco_arena.set_origin([0, 0, 0])
        mujoco_arena.set_camera(
            camera_name="frontview",
            pos=[5.8, -5.0, 3.2],
            quat=[0.508, 0.320, 0.424, 0.680],
        )
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[2.5, -3.2, 2.2],
            quat=[0.621, 0.364, 0.386, 0.577],
        )
        mujoco_arena.set_camera(
            camera_name="birdview",
            pos=[-2.5, 2.0, 32.0],
            quat=[0.7071, 0.0, 0.0, 0.7071],
            camera_attribs={"fovy": "58"},
        )
        mujoco_arena.set_camera(
            camera_name="bird",
            pos=[-2.5, 2.0, 32.0],
            quat=[0.7071, 0.0, 0.0, 0.7071],
            camera_attribs={"fovy": "58"},
        )

        self.static_scene_objects = []
        if self.include_legacy_static_scene:
            self._construct_static_scene_objects()

        self.material_objects = []
        self.material_metadata = OrderedDict()
        if self.include_material_objects:
            self._construct_material_objects()
        if self.include_siemens_line_objects:
            self._construct_siemens_line_objects()
        self._construct_hidden_green_tote_replacements()
        xml_material_objects = [
            metadata["model"]
            for metadata in self.material_metadata.values()
            if metadata.get("model") is not None
        ]

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.static_scene_objects + xml_material_objects,
        )
        if self.material_objects:
            self._add_siemens_material_objects()

    def _add_siemens_material_objects(self):
        for idx, (support_pose, support_size) in enumerate(self._siemens_static_table_support_surfaces(), start=1):
            support_pos = np.asarray(support_pose, dtype=float).copy()
            support_pos[2] -= np.asarray(support_size, dtype=float)[2]
            ET.SubElement(
                self.model.worldbody,
                "geom",
                name=f"siemens_static_table_support_{idx}",
                type="box",
                pos=array_to_string(support_pos),
                size=array_to_string(support_size),
                rgba="0 0 0 0",
                contype="1",
                conaffinity="1",
                friction="1 0.005 0.0001",
                group="3",
            )

        added_assets = set()
        for obj_name in self.material_objects:
            metadata = self.material_metadata[obj_name]
            support_size = metadata.get("support_size")
            if support_size is not None and metadata.get("fixed_pose") is not None:
                support_pos = np.asarray(metadata.get("support_pose", metadata["fixed_pose"]), dtype=float).copy()
                support_pos[2] -= np.asarray(support_size, dtype=float)[2]
                ET.SubElement(
                    self.model.worldbody,
                    "geom",
                    name=f"{obj_name}_support",
                    type="box",
                    pos=array_to_string(support_pos),
                    size=array_to_string(support_size),
                    rgba="0 0 0 0",
                    contype="1",
                    conaffinity="1",
                    group="3",
                )

            if metadata.get("kind") != "siemens":
                continue
            spec = metadata["spec"]
            mesh_path = SIEMENS_MESH_DIR / spec["file"]
            if not mesh_path.exists():
                raise FileNotFoundError(f"Missing Siemens mesh asset: {mesh_path}")

            if spec["material"] not in added_assets:
                ET.SubElement(
                    self.model.asset,
                    "material",
                    name=spec["material"],
                    rgba=array_to_string(spec["rgba"]),
                )
                added_assets.add(spec["material"])
            if spec["mesh"] not in added_assets:
                ET.SubElement(
                    self.model.asset,
                    "mesh",
                    name=spec["mesh"],
                    file=mesh_path.as_posix(),
                )
                added_assets.add(spec["mesh"])

            body = ET.SubElement(
                self.model.worldbody,
                "body",
                name=obj_name,
                pos="0 0 0",
                quat="1 0 0 0",
            )
            ET.SubElement(
                body,
                "inertial",
                pos=array_to_string(spec["collision_pos"]),
                mass=str(spec["mass"]),
                diaginertia=array_to_string(box_diagonal_inertia(spec["mass"], spec["half_size"])),
            )
            ET.SubElement(body, "freejoint", name=f"{obj_name}_free")
            ET.SubElement(
                body,
                "geom",
                name=f"{obj_name}_visual",
                type="mesh",
                mesh=spec["mesh"],
                material=spec["material"],
                quat=array_to_string(SIEMENS_MESH_QUAT),
                pos=array_to_string(spec["visual_pos"]),
                contype="0",
                conaffinity="0",
                density="0",
                group="1",
            )
            ET.SubElement(
                body,
                "geom",
                name=f"{obj_name}_collision",
                type="box",
                pos=array_to_string(spec["collision_pos"]),
                size=array_to_string(spec["half_size"]),
                rgba="0.10 0.55 0.95 0.18",
                contype="1",
                conaffinity="1",
                density="0",
                group="3",
            )
            half_size = np.asarray(spec["half_size"], dtype=float)
            collision_pos = np.asarray(spec["collision_pos"], dtype=float)
            grasp_x = half_size[0] * 0.55
            grasp_y = -(half_size[1] + 0.015)
            grasp_z = collision_pos[2] + half_size[2]
            for site_name, site_pos, rgba in (
                (
                    f"{obj_name}_right_grasp_site",
                    [grasp_x, grasp_y, grasp_z],
                    "1 0 0 0",
                ),
                (
                    f"{obj_name}_left_grasp_site",
                    [-grasp_x, grasp_y, grasp_z],
                    "0 1 0 0",
                ),
                (
                    f"{obj_name}_center_site",
                    collision_pos,
                    "1 1 0 0",
                ),
            ):
                ET.SubElement(
                    body,
                    "site",
                    name=site_name,
                    pos=array_to_string(site_pos),
                    size="0.025",
                    rgba=rgba,
                )

    def _setup_references(self):
        super()._setup_references()
        self.obj_body_id = {}
        for obj in self.static_scene_objects:
            self.obj_body_id[obj.name] = self.sim.model.body_name2id(obj.root_body)
        for obj_name in self.material_objects:
            metadata = self.material_metadata[obj_name]
            body_name = metadata["model"].root_body if metadata.get("model") is not None else obj_name
            self.obj_body_id[obj_name] = self.sim.model.body_name2id(body_name)

    def _setup_observables(self):
        observables = super()._setup_observables()

        if self.use_object_obs:
            modality = "object"
            sensors = []
            names = []

            for obj_name in self.material_objects:

                @sensor(modality=modality)
                def obj_pos(obs_cache, obj_name=obj_name):
                    return np.array(self.sim.data.body_xpos[self.obj_body_id[obj_name]])

                @sensor(modality=modality)
                def obj_quat(obs_cache, obj_name=obj_name):
                    return convert_quat(
                        np.array(self.sim.data.body_xquat[self.obj_body_id[obj_name]]),
                        to="xyzw",
                    )

                obj_pos.__name__ = f"{obj_name}_pos"
                obj_quat.__name__ = f"{obj_name}_quat"
                sensors += [obj_pos, obj_quat]
                names += [obj_pos.__name__, obj_quat.__name__]

            for target_name, target_info in self.output_ports.items():
                target_pos = target_info["center"] + np.array([0.0, 0.0, self._station_surface_top(target_info, 0)])

                @sensor(modality=modality)
                def target_sensor(obs_cache, target_pos=target_pos):
                    return np.array(target_pos)

                target_sensor.__name__ = f"{target_name}_pos"
                sensors.append(target_sensor)
                names.append(target_sensor.__name__)

            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        super()._reset_internal()
        self.has_judge_collision = False
        self._judge_last_collision_pair = None

        for obj_name in self.material_objects:
            metadata = self.material_metadata[obj_name]
            pos, quat = self._sample_item_pose(metadata)
            joint_name = metadata.get("joint_name", f"{obj_name}_free")
            self.sim.data.set_joint_qpos(joint_name, np.concatenate([pos, quat]))

    def _check_success(self):
        return False

    def _check_robot_configuration(self, robots):
        robots = [robots] if type(robots) is str else robots
        assert len(robots) == 1, "FactorySorting currently supports exactly one robot."
        assert robots[0] == "Tiago", "FactorySorting is set up for the Tiago robot."
        assert issubclass(
            ROBOT_CLASS_MAPPING[robots[0]], WheeledRobot
        ), f"FactorySorting expects a wheeled robot, got {ROBOT_CLASS_MAPPING[robots[0]]}."
