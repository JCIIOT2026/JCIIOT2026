<!-- COMPETITION LOCKED — DO NOT MODIFY -->

# L4 Task — Large Object Cross-line Transport

Level: L4 (max 25 points)
Scene: factory_sorting_7

## Task

Transport a large blue container from Pick Station 5 to Place Station 2

## Station Mapping

- Pick Station 5 = input_2, center (-9.76, 5.01)
- Place Station 2 = output_5, center (4.87, -7.26)
- Robot start: (13.5, 0.0)
- Target object: blue_container_h01_back_upper

## Grasp Pose (BC Policy)

- Robot stop point at input_2: (8.56, -3.92, 0.0), yaw=-3.14

## Object Inventory (L4 Scene)

Every input port and its assigned graspable object:

- input_1 → green_tote_b01_left_center, green_tote_b01_left_front, green_tote_b01_left_back
- input_2 → blue_container_h01_back_upper, blue_container_h01_back_lower
- input_3 → blue_container_h10_center, blue_container_h10_left, blue_container_h10_right
- input_4 → line_4_container_h01
- input_5 → line_5_container_h10
- input_6 → line_6_tote_b01

CRITICAL: When calling pick_up, you MUST provide the exact object_name from the inventory above.
