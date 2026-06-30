<!-- COMPETITION LOCKED — DO NOT MODIFY -->

# L1 Task — Single-line Transport

Level: L1 (max 10 points)
Scene: factory_sorting_1

## Task

Transport a blue hollow plastic box from Pick Station 2 to Place Station 3

## Station Mapping

- Pick Station 2 = input_5, center (7.99, 3.94)
- Place Station 3 = output_4, center (-0.17, -7.29)
- Robot stop point at input_5: (8.000006, 4.599997, 0.0), yaw=-3.139
- Robot start: (13.5, 0.0)
- Target object: line_5_container_h01_near

## Grasp Pose (BC Policy)

- This is the precise pose used during BC policy training
- **CRITICAL**: yaw MUST be -3.139. Using different yaw will face the robot away from the object and cause grasp failure. The object sits on the positive-Y side of the production line; positive yaw faces the robot toward positive-Y.
