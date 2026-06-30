<!-- COMPETITION LOCKED — DO NOT MODIFY -->

# L2 Task — Cross-line Transport + Obstacle Avoidance

Level: L2 (max 15 points)
Scene: factory_sorting_3

## Task

Transport a green tote box from Pick Station 1 to Place Station 3

## Station Mapping

- Pick Station 1 = input_6, center (11.937, 3.932)
- Place Station 3 = output_4, center (-0.17, -7.29)
- Robot start: (13.5, 0.0)
- Target object: green_tote_b01_upper

## Object Inventory (L2 Scene)

Every input port and its assigned graspable object:

- input_1 → line_1_container_h01
- input_2 → line_2_container_h10
- input_3 → line_3_tote_b01
- input_4 → line_4_container_h01
- input_5 → line_5_container_h10
- input_6 → green_tote_b01_upper, green_tote_b01_lower

CRITICAL: When calling pick_up, you MUST provide the exact object_name from the inventory above. Do NOT guess or derive object names — use only the names listed here.
