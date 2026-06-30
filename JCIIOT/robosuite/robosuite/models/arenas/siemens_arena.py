import xml.etree.ElementTree as ET

from robosuite.models.arenas.arena import Arena
from robosuite.models.base import MujocoXML
from robosuite.utils.mjcf_utils import xml_path_completion


ROOM_CENTER = (-9.36, -0.71)
ROOM_HALF_SIZE = (34.0, 22.0)
ROOM_HEIGHT = 4.0


class SiemensArena(Arena):
    """EmptyArena room + Siemens production-line scene."""

    def __init__(self):
        super().__init__(xml_path_completion("arenas/empty_arena.xml"))
        self._hide_builtin_floor()
        self._resize_room()
        self._merge_siemens_scene()
        self._add_factory_lights()

    def _hide_builtin_floor(self):
        floor = self.worldbody.find("./geom[@name='floor']")
        if floor is not None:
            floor.attrib.pop("material", None)
            floor.set("rgba", "0 0 0 0")
            floor.set("group", "3")

    def _merge_siemens_scene(self):
        siemens_scene = MujocoXML(
            xml_path_completion("objects/siemens/mujoco_original/scene_robosuite.xml")
        )
        self.merge(siemens_scene)

    def _resize_room(self):
        cx, cy = ROOM_CENTER
        hx, hy = ROOM_HALF_SIZE
        hz = ROOM_HEIGHT / 2.0

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
        cx, cy = ROOM_CENTER

        ET.SubElement(
            self.worldbody,
            "light",
            name="siemens_overhead_light",
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
            name="siemens_front_fill_light",
            pos=f"{cx + 8.0} {cy - 10.0} 10",
            dir="-0.4 0.5 -1",
            directional="true",
            diffuse="0.7 0.75 0.8",
            ambient="0.25 0.25 0.25",
            specular="0.15 0.15 0.15",
            castshadow="false",
        )
