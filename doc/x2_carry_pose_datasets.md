# X2 carry-pose datasets

## Current training dataset

The current training dataset was regenerated using hand-body centers of mass as
endpoints. Its adjacent `carry_poses_x2.json` records the generator settings,
including seed 42 and the following geometry parameters:

```bash
--num-poses 512 \
--position-tolerance 0.01 \
--max-end-effector-position-error 0.02 \
--max-parallel-axis-error 0.02 \
--endpoint-offset-lower -0.15 -0.15 -0.15 \
--endpoint-offset-upper 0.15 0.15 0.15
```

Position values are in meters. Endpoint offsets and parallel-axis checks use
the torso frame.

## Ground-parallel test dataset

The COM-based test set uses 512 poses and seed 43, written separately to
`src/assets/data/x2/bones_seed/carry_poses_x2_test_ground_parallel.pkl`.
Its adjacent JSON file records the complete generator configuration.

```bash
conda run -n unitree_rl_mjlab python scripts/generate_x2_carry_pose_dataset.py \
  --ground-parallel --num-poses 512 --seed 43 \
  --position-tolerance 0.01 --orientation-tolerance-deg 1 \
  --max-end-effector-position-error 0.02 \
  --max-parallel-axis-error 0.002 \
  --endpoint-offset-lower -0.15 -0.15 -0.15 \
  --endpoint-offset-upper 0.15 0.15 0.15 \
  --output src/assets/data/x2/bones_seed/carry_poses_x2_test_ground_parallel.pkl
```

In this mode, both endpoint targets share world height and forward position
in a horizontal frame aligned to the torso's heading. Both grasp orientations
use a shared yaw, with the wrist roll links' local +X axes pointing vertically
up and their palm-facing -Z axes horizontal. Pitch and roll are not sampled.
The tilted reference pose is used only to initialize IK and is not saved.

Every saved float32 pose must meet the 2 mm endpoint alignment limit, the
1 degree maximum wrist tilt from world up, joint limits, and the existing
self-collision check. These are finite numerical tolerances, not exact
mathematical equality. Training data is not overwritten.

Validation of the regenerated 512 poses found a maximum hand-COM height
difference of 0.575 mm, maximum forward-alignment error of 0.554 mm, and maximum
wrist tilt of 0.967 degrees. All poses passed joint-limit and self-collision
checks, and none exactly duplicated a pose in the current training file.
With the default physical-box playback size ranges, 100 poses passed the
centered-grasp and torso-clearance filter. The `.validation.json` sidecar
records these results.

![Ground-parallel test-set geometry and wrist tilt](x2_ground_parallel_test_set.png)

To inspect the test set:

```bash
conda run -n unitree_rl_mjlab python scripts/generate_x2_carry_pose_dataset.py \
  --playback-file src/assets/data/x2/bones_seed/carry_poses_x2_test_ground_parallel.pkl \
  --playback-once
```
