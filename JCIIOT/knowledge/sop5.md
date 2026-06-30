<!-- COMPETITION LOCKED — DO NOT MODIFY -->

# L5 Task — Extreme Distance Transport

Level: L5 (max 30 points)
Scene: factory_sorting_9

## Task

Transport three large white tote boxes from Pick Station 6 to Place Station 1.

## Station Mapping

- Pick Station 6 = input_1, center (-14.54, 5.01)
- Place Station 1 = output_6, center (10.03, -7.27)
- Robot start: (13.5, 0.0)
- Target objects:
  - white_tote_b01_left_center
  - white_tote_b01_left_front
  - white_tote_b01_left_back

## Grasp Pose (BC Policy)

- Robot stop point at input_1: (5.03, -3.84, 0.0), yaw=-3.14

## Object Inventory (L5 Scene)

Every input port and its assigned graspable object:

- input_1: white_tote_b01_left_center, white_tote_b01_left_front, white_tote_b01_left_back
- input_2 → blue_container_h01_back_center, blue_container_h01_back_left, blue_container_h01_back_right
- input_3 → dark_tote_b01_upper, dark_tote_b01_lower
- input_4 → line_4_container_h01
- input_5 → line_5_container_h10
- input_6 → line_6_tote_b01

CRITICAL: When calling pick_up, you MUST provide the exact object_name from the inventory above.
