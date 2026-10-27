# X2 carrying payload

`SlidingBoxForceEvent` in `src/tasks/locomanipulation/mdp/payload.py` models a
hand-carried box with one aggregate mass sliding on its floor. It is enabled
only in `Agibot-X2-Locomanipulation-Carry-Finetune-Flat`. Basic flat/rough tasks
retain their existing force events and general manipulation data.

The event uses batched Torch operations once per 20 ms policy step. It adds no
MuJoCo bodies, contacts, constraints, physics substeps or Jacobian calculations.
Observation/reward functions and dimensions, symmetry, the force curriculum,
unloaded-episode mask and carry checkpoint progress remain unchanged.

## Mechanics and settings

The midpoint follows the wrist-roll COMs. The box lateral axis follows the hand
span; its up direction comes from the two wrist links' local +X directions,
projected perpendicular to the span. This matches the X2 carry dataset's side-
grasp convention. Averaging those directions approximates a shared rigid grasp.

The contents feel gravity and the box's translational, Euler, centripetal and
Coriolis accelerations. Resting contents stick up to the static-friction limit;
sliding uses lower kinetic friction. Friction depends on estimated normal load.
Damping is implicit, with no spring pulling contents toward the center. Wall
impacts reverse outward speed according to restitution and transfer the full
velocity-change impulse into the hand reactions.

Shell/contents reactions, their lever-arm moments, and shell rotational inertia
produce one total wrench. The two hand forces and residual hand torques reproduce
that wrench. Hand load sharing is an explicit approximation, not a solved grasp.
Inertia and visualization use the same box dimensions, including narrow grasps.

Settings are plain values from `default_payload_params()`, saved in environment
YAML and restored by train/play/eval before constructing the environment:

| Setting | Default |
| --- | --- |
| Total mass / moving contents fraction | Uniform 2–6 kg / 75% |
| Contents travel | +/-10 cm forward, +/-8 cm lateral |
| Static / kinetic friction | 0.15 / 0.10 |
| Wall restitution / damping | 0.10 / 1 per second |
| Initial contents offset / relative velocity | Uniform +/-1 cm / zero |
| Acceleration filter time constant | 0.10 s |
| Unloaded training episodes | 30% |

The shell center is 5 cm below the hand midpoint along box up. The box is 12 cm
high, 25 cm deep by default, and wide enough for the hand span and contents
travel. The contents marker has a 2.5 cm radius and rests on the bottom, 3.5 cm
below the shell center. Geometry and friction are exploratory assumptions.

`force_scale` scales both mass components. Zero scale and unloaded episodes emit
zero wrenches. Existing constant-force presets bypass payload dynamics. The
physical-box playback option disables the reduced load to prevent double loading.
Partial resets clear only the selected environments' state and wrenches; the
first active sample initializes derivative history without an acceleration spike.

Only the current tilting model is retained. Settings contain physical parameters
without a version tag; restore discards the obsolete tag in previously saved
settings. Missing or unexpected parameter fields fail explicitly. Older policy
checkpoints without payload metadata use current carry settings.

## Visualization

```bash
cd /home/ubuntu/unitree_rl_mjlab
PYTHONPATH='' .venv/bin/python scripts/play.py \
  Agibot-X2-Locomanipulation-Carry-Finetune-Flat \
  --checkpoint-file logs/rsl_rl/x2_locomanipulation/2026-09-28_13-06-08_carry-finetune-w-default-pose-0.2-2/model_20000.pt \
  --sliding-payload True --num-envs 1 --viewer native --device cpu
```

Restart playback after code changes. Without `--sliding-payload True`, playback
keeps its zero-force default. The option samples carrying poses and holds each
until reset; it cannot be combined with `--box-mass`. Use `--viewer viser` for a
browser viewer.

The translucent brown box follows inferred tilt, the orange sphere shows the
contents, the cyan marker and raised indicator show combined COM, and magenta
arrows show hand forces. These are display-only overlays at true scale.

Native controls: `R` toggles overlays, Space pauses, Enter samples another pose
and load. Numpad 8/2 changes forward/backward velocity, 4/6 changes sideways
velocity, 7/9 changes yaw velocity, and 5 stops. Contents can legitimately stick
or settle against a wall when the applied acceleration and tilt remain steady.

Evaluation accepts `--force-conditions none payload --body-poses x2_carry`.

## Verification and limits

```bash
env PYTHONPATH='' MPLCONFIGDIR=/tmp/mjlab-review-mpl \
  WARP_CACHE_PATH=/tmp/mjlab-payload-warp PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest tests -q -p no:launch_testing
```

The tests cover friction and tilt, rotation/inertia, wall impulses, wrench
conservation, reset isolation, configuration restore, basic-task isolation,
visualization and headless environment integration. Review also checks invalid
restore atomicity and consistent inertia/display geometry for narrow grasps.
Validation: **252 passed, 15 skipped**. Ruff passed for the changed Python files.

The cleanup reproduces the previous tilting model's entire recorded eight-second
checkpoint trace exactly: contents positions, hand vertical forces, inferred tilt,
mass and activation flags. That trial uses seed 123, a 3.81 kg payload and a
0.3 m/s walking command, with no episode resets.

![Current payload response with the supplied checkpoint](x2_sliding_payload.png)

This remains a reduced floor-contact model: no vertical flight, rolling/spin,
multiple-object collisions, finger friction, dropping or deformation. Degenerate,
too-small or near-inverted grasps disable loading through the existing unloaded
mask. The 20 ms coupling and finite-difference acceleration estimates are
approximations. CPU regression checks and a finite policy trial do not establish
full GPU training throughput or general carrying stability.
