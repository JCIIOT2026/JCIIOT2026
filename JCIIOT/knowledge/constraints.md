<!-- COMPETITION LOCKED — DO NOT MODIFY -->

# Constraints & Safety Rules

## Transport Constraints
- Only one object may be carried at a time
- Confirm gripper is empty before picking
- Confirm target area is clear before placing
- Per-step timeout: 300s, no automatic retry

## Navigation Constraints
- Avoid obstacles (production line equipment, belts, crossbars)
- Stop distance: chassis ~0.8m from table
- Arm stays in holding posture during navigation

## Collision Constraints
- Collision detection logs to trajectory JSON but does not halt navigation
- Tasks with collision: -5 point penalty
- Collision recorded in has_collision field

## Safety Rules
- Do not move chassis while arm is in motion
- Do not pick while gripper is holding an object
- All operations must verify safety before proceeding

## Auto-planning Rules
- When only a destination is specified, auto-select the nearest available source
- Station names MUST use those from the knowledge base
- Each task level maps to a different factory scene and map file
