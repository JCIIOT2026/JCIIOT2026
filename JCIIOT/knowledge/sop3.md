<!-- COMPETITION LOCKED — DO NOT MODIFY -->

# L3 Task — Cross-line Transport + Obstacle + Interference

Level: L3 (max 20 points)
Scene: factory_sorting_5

## Task

Transport an orange tote box from Pick Station 1 to Place Station 2

## Station Mapping

- Pick Station 1 = input_6, center (11.94, 3.93)
- Place Station 2 = output_5, center (4.87, -7.26)
- Robot start: (13.5, 0.0)
- Target object: orange_tote_b01_upper

## Grasp Pose (BC Policy)

- Robot stop point at input_6: (6.00, 4.80, 0.0), yaw=3.14

## Object Inventory (L3 Scene)

Every input port and its assigned graspable object:

- input_1 → line_1_container_h01
- input_2 → line_2_container_h10
- input_3 → blue_container_h10_back
- input_4 → orange_tote_b01_upper, orange_tote_b01_lower
- input_5 → line_5_container_h10
- input_6 → line_6_tote_b01

CRITICAL: When calling pick_up, you MUST provide the exact object_name from the inventory above.
