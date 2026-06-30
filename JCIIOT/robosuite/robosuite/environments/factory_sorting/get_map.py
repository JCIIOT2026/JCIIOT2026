"""
Generate a 2D navigation map from a scene configuration file.

Default usage:
    python robosuite/environments/factory_sorting/get_map.py

Custom scene usage:
    python robosuite/environments/factory_sorting/get_map.py --config path/to/scene.json

The generator expects world-frame object information. A bird-view image can be
used as an annotation source, but the final config should provide metric
coordinates so the output can be used by a planner.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


FREE = 0
OBSTACLE = 1
STATION = 2
APPROACH = 3
ROBOT = 4

CELL_VALUES = {
    "free": FREE,
    "obstacle": OBSTACLE,
    "station": STATION,
    "approach": APPROACH,
    "robot": ROBOT,
}


def load_scene_config(config_path):
    with config_path.open("r", encoding="utf-8") as f:
        scene = json.load(f)
    validate_scene_config(scene, config_path)
    return normalize_scene_config(scene)


def validate_scene_config(scene, config_path):
    required = ["scene_name", "bounds", "robot", "objects"]
    missing = [key for key in required if key not in scene]
    if missing:
        raise ValueError(f"{config_path} is missing required keys: {missing}")

    for key in ["x_min", "x_max", "y_min", "y_max"]:
        if key not in scene["bounds"]:
            raise ValueError(f"{config_path} bounds is missing '{key}'")

    for key in ["name", "start", "radius"]:
        if key not in scene["robot"]:
            raise ValueError(f"{config_path} robot is missing '{key}'")

    for obj in scene["objects"]:
        for key in ["name", "type", "center", "size"]:
            if key not in obj:
                raise ValueError(f"object {obj} is missing '{key}'")


def normalize_scene_config(scene):
    scene = dict(scene)
    scene["map_name"] = scene.get("map_name", scene["scene_name"])
    scene["coordinate_frame"] = scene.get("coordinate_frame", "world_xy")
    scene["cell_values"] = CELL_VALUES

    robot = dict(scene["robot"])
    robot.setdefault("safety_margin", 0.15)
    scene["robot"] = robot

    objects = []
    input_ports = {}
    output_ports = {}
    obstacles = []

    for raw_obj in scene["objects"]:
        obj = dict(raw_obj)
        obj.setdefault("yaw", 0.0)
        obj.setdefault("is_obstacle", obj["type"] in {"station", "obstacle", "wall", "conveyor"})
        obj.setdefault("visual_value", "station" if obj["type"] == "station" else "obstacle")
        obj["half_size"] = [float(obj["size"][0]) / 2.0, float(obj["size"][1]) / 2.0]
        objects.append(obj)

        role = obj.get("role")
        if role == "input":
            input_ports[obj["name"]] = obj
        elif role == "output":
            output_ports[obj["name"]] = obj

        if obj["is_obstacle"]:
            obstacles.append(obj)

    scene["objects"] = objects
    scene["input_ports"] = input_ports
    scene["output_ports"] = output_ports
    scene["obstacles"] = obstacles
    return scene


def world_to_grid(x, y, bounds, resolution):
    row = int(round((x - bounds["x_min"]) / resolution))
    col = int(round((y - bounds["y_min"]) / resolution))
    return row, col


def grid_to_world(row, col, bounds, resolution):
    x = bounds["x_min"] + row * resolution
    y = bounds["y_min"] + col * resolution
    return float(x), float(y)


def clamp_grid_window(grid, row_min, row_max, col_min, col_max):
    row_min = max(row_min, 0)
    row_max = min(row_max, grid.shape[0] - 1)
    col_min = max(col_min, 0)
    col_max = min(col_max, grid.shape[1] - 1)
    return row_min, row_max, col_min, col_max


def mark_disk(grid, center, radius, bounds, resolution, value):
    center = np.asarray(center, dtype=float)
    row_min, col_min = world_to_grid(center[0] - radius, center[1] - radius, bounds, resolution)
    row_max, col_max = world_to_grid(center[0] + radius, center[1] + radius, bounds, resolution)
    row_min, row_max, col_min, col_max = clamp_grid_window(grid, row_min, row_max, col_min, col_max)

    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            x, y = grid_to_world(row, col, bounds, resolution)
            if (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2:
                grid[row, col] = value


def point_inside_rotated_rect(point, center, half_size, yaw):
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    cos_yaw = math.cos(-yaw)
    sin_yaw = math.sin(-yaw)
    local_x = cos_yaw * dx - sin_yaw * dy
    local_y = sin_yaw * dx + cos_yaw * dy
    return abs(local_x) <= half_size[0] and abs(local_y) <= half_size[1]


def mark_rotated_rect(grid, center, half_size, yaw, bounds, resolution, value, inflation=0.0):
    center = np.asarray(center, dtype=float)
    half_size = np.asarray(half_size, dtype=float) + inflation

    radius = float(np.linalg.norm(half_size))
    row_min, col_min = world_to_grid(center[0] - radius, center[1] - radius, bounds, resolution)
    row_max, col_max = world_to_grid(center[0] + radius, center[1] + radius, bounds, resolution)
    row_min, row_max, col_min, col_max = clamp_grid_window(grid, row_min, row_max, col_min, col_max)

    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            point = grid_to_world(row, col, bounds, resolution)
            if point_inside_rotated_rect(point, center, half_size, yaw):
                grid[row, col] = value


def build_occupancy_grid(scene, resolution):
    bounds = scene["bounds"]
    rows = int(np.ceil((bounds["x_max"] - bounds["x_min"]) / resolution)) + 1
    cols = int(np.ceil((bounds["y_max"] - bounds["y_min"]) / resolution)) + 1
    grid = np.zeros((rows, cols), dtype=np.uint8)

    robot = scene["robot"]
    inflation = float(robot["radius"]) + float(robot.get("safety_margin", 0.0))

    for obj in scene["obstacles"]:
        mark_rotated_rect(
            grid,
            obj["center"],
            obj["half_size"],
            float(obj.get("yaw", 0.0)),
            bounds,
            resolution,
            OBSTACLE,
            inflation=inflation,
        )

    for obj in scene["objects"]:
        if obj.get("visual_value") == "station":
            mark_rotated_rect(
                grid,
                obj["center"],
                obj["half_size"],
                float(obj.get("yaw", 0.0)),
                bounds,
                resolution,
                STATION,
            )

    for obj in scene["objects"]:
        if "approach" in obj:
            mark_disk(grid, obj["approach"], float(obj.get("approach_radius", 0.12)), bounds, resolution, APPROACH)

    mark_disk(grid, robot["start"], float(robot["radius"]), bounds, resolution, ROBOT)
    return grid


def world_to_image_xy(world_xy, image_view):
    """Convert world xy to the axes used by the visualization."""
    axis_values = {
        "world_x": world_xy[0],
        "-world_x": -world_xy[0],
        "world_y": world_xy[1],
        "-world_y": -world_xy[1],
    }
    horizontal = image_view.get("image_horizontal_axis", "world_y")
    vertical = image_view.get("image_vertical_axis", "-world_x")
    return axis_values[horizontal], axis_values[vertical]


def visualization_extent(bounds, image_view):
    corners = [
        (bounds["x_min"], bounds["y_min"]),
        (bounds["x_min"], bounds["y_max"]),
        (bounds["x_max"], bounds["y_min"]),
        (bounds["x_max"], bounds["y_max"]),
    ]
    xs, ys = zip(*(world_to_image_xy(corner, image_view) for corner in corners))
    return [min(xs), max(xs), min(ys), max(ys)]


def display_grid_for_image(grid, image_view):
    horizontal = image_view.get("image_horizontal_axis", "world_y")
    vertical = image_view.get("image_vertical_axis", "-world_x")

    if horizontal.endswith("world_y"):
        image = grid.copy()
    else:
        image = grid.T

    if horizontal.startswith("-"):
        image = np.fliplr(image)

    origin = "upper" if vertical.startswith("-") else "lower"
    return image, origin


def save_visualization(scene, grid, output_path):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError:
        return False

    bounds = scene["bounds"]
    image_view = scene.get(
        "image_view",
        {"image_horizontal_axis": "world_y", "image_vertical_axis": "-world_x"},
    )
    cmap = ListedColormap(["#f2efe7", "#2f3437", "#8ca9d6", "#f1c232", "#d94b4b"])
    image, origin = display_grid_for_image(grid, image_view)
    extent = visualization_extent(bounds, image_view)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap=cmap, interpolation="nearest", origin=origin, extent=extent, aspect="equal")

    for obj in scene["objects"]:
        if obj.get("role") not in {"input", "output"}:
            continue
        x, y = world_to_image_xy(obj["center"], image_view)
        ax.text(
            x,
            y,
            obj["name"],
            ha="center",
            va="center",
            fontsize=8,
            color="black",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.5},
        )
        if "approach" in obj:
            approach_x, approach_y = world_to_image_xy(obj["approach"], image_view)
            ax.plot(approach_x, approach_y, "ko", markersize=3)

    robot_x, robot_y = world_to_image_xy(scene["robot"]["start"], image_view)
    ax.plot(robot_x, robot_y, "ro", markersize=5)
    ax.set_xlabel(f"image horizontal axis: {image_view.get('image_horizontal_axis', 'world_y')}")
    ax.set_ylabel(f"image vertical axis: {image_view.get('image_vertical_axis', '-world_x')}")
    ax.set_title(f"{scene['map_name']} navigation map")
    ax.grid(color="#ffffff", linewidth=0.4, alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def write_outputs(scene, grid, output_dir, resolution):
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = scene["map_name"]

    semantic_path = output_dir / f"{stem}_semantic_map.json"
    grid_path = output_dir / f"{stem}_occupancy_grid.npy"
    png_path = output_dir / f"{stem}_map.png"

    output_scene = dict(scene)
    output_scene["resolution"] = resolution
    output_scene["grid_shape"] = [int(grid.shape[0]), int(grid.shape[1])]
    output_scene["files"] = {
        "semantic_map": semantic_path.name,
        "occupancy_grid": grid_path.name,
        "visualization": png_path.name,
    }

    semantic_path.write_text(json.dumps(output_scene, indent=2), encoding="utf-8")
    np.save(grid_path, grid)
    has_png = save_visualization(output_scene, grid, png_path)
    return semantic_path, grid_path, png_path if has_png else None


def parse_args():
    default_config = Path(__file__).resolve().parent / "maps" / "factory_sorting_3_3fo3errph7x9_scene_regenerated.json"
    parser = argparse.ArgumentParser(description="Generate a navigation map from a scene config.")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help="Scene JSON with bounds, robot info, and object geometry.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "generated_maps",
        help="Directory where map files will be written.",
    )
    parser.add_argument("--resolution", type=float, default=0.05, help="Grid resolution in meters per cell.")
    return parser.parse_args()


def main():
    args = parse_args()
    scene = load_scene_config(args.config)
    grid = build_occupancy_grid(scene, args.resolution)
    semantic_path, grid_path, png_path = write_outputs(scene, grid, args.output_dir, args.resolution)

    print(f"config: {args.config}")
    print(f"semantic map: {semantic_path}")
    print(f"occupancy grid: {grid_path}")
    if png_path is None:
        print("visualization: skipped because matplotlib is not installed")
    else:
        print(f"visualization: {png_path}")


if __name__ == "__main__":
    main()
