# Locomanipulation Task

The locomanipulation task trains the G1 humanoid to walk while carrying objects. The policy controls 12 lower-body joints (hip, knee, ankle) while the upper body is driven by motion playback from the ACCAD dataset. This split design lets the policy focus on balance and velocity tracking without learning arm coordination from scratch.

Supports two robot variants:
- **G1 29-DOF**: 17 upper-body DOFs (waist, arms, wrists)
- **G1 23-DOF**: 11 upper-body DOFs (no waist roll/pitch, no wrist pitch/yaw)

## Architecture

```
src/tasks/locomanipulation/
├── locomanipulation_env_cfg.py          # Base environment config factory
├── config/
│   ├── g1/
│   │   ├── __init__.py                  # Task registration (29-DOF)
│   │   ├── env_cfgs.py                  # G1 robot-specific overrides
│   │   └── rl_cfg.py                    # PPO config with symmetry
│   └── g1_23dof/
│       ├── __init__.py                  # Task registration (23-DOF)
│       ├── env_cfgs.py                  # G1-23DOF robot-specific overrides
│       └── rl_cfg.py                    # PPO config with 23-DOF symmetry
├── mdp/
│   ├── rewards.py                       # Velocity, height, posture, gait rewards
│   ├── observations.py                  # Foot, wrist, phase observations
│   ├── events.py                        # HandForceEvent, MaxForceEstimator
│   ├── terminations.py                  # Illegal contact termination
│   ├── curriculums.py                   # Force, pose, height, velocity curricula
│   ├── symmetry.py                      # Left-right sagittal plane mirroring
│   ├── upper_body_action.py             # ACCAD motion playback action
│   ├── velocity_command.py              # Velocity command term
│   └── height_command.py                # Base height command term
└── rl/
    ├── __init__.py
    └── runner.py                        # OnPolicy runners with ONNX export
```

### Registered Tasks

| Task Name | Robot | Terrain | Runner |
|-----------|-------|---------|--------|
| `Unitree-G1-Locomanipulation-Rough` | G1 29-DOF | Rough | `LocomanipulationOnPolicyRunner` |
| `Unitree-G1-Locomanipulation-Flat` | G1 29-DOF | Flat | `LocomanipulationOnPolicyRunner` |
| `Unitree-G1-23Dof-Locomanipulation-Rough` | G1 23-DOF | Rough | `G1_23DOF_LocomanipulationOnPolicyRunner` |
| `Unitree-G1-23Dof-Locomanipulation-Flat` | G1 23-DOF | Flat | `G1_23DOF_LocomanipulationOnPolicyRunner` |

## Key Design Decisions

### Split Action Space

The policy outputs actions for 12 lower-body joints only. Upper-body joints are controlled by `UpperBodyMotionAction`, which reads from ACCAD motion data:

- **Training (`pose_only=True`)**: Samples a random frame at episode reset and holds it. Reduces training complexity.
- **Deployment (`pose_only=False`)**: Plays back motion clips frame-by-frame for dynamic upper-body motion.

The `default_pose_ratio` curriculum gradually transitions from HOME_KEYFRAME to diverse motion poses during training.

### External Force Curriculum

`HandForceEvent` simulates carrying heavy objects by applying random wrenches to end-effectors:

- **Jacobian-based bounds**: `MaxForceEstimator` computes physically plausible force limits using `F_max = min(effort_limit / |J|)` per axis.
- **Impulse lifecycle**: Forces applied for 8-12s, then 2-4s cooldown. A fraction (`no_force_ratio=0.05`) of envs stay force-free.
- **Dirichlet axis scaling**: Per-env random scaling across x/y/z axes for diversity.
- **Curriculum**: `force_scale_staged` ramps force from 0 to max over 15k iterations.

### Height Command

`BaseHeightCommand` commands an absolute z-height (range 0.50m–0.785m for G1):

- **Height-dependent posture**: `variable_posture` and `stand_still` rewards look up target joint angles from a 7-entry table (0.50m to 0.785m at 0.05m intervals).
- **Curriculum**: `height_scale_staged` expands the height range over training.
- **Standing weight**: `track_base_height` applies 2x weight when stationary (`|twist_cmd| < 0.1`).

### Symmetry Augmentation

Left-right sagittal plane mirroring doubles effective training data:

- Mirrors observations (base_ang_vel, gravity, commands, phase, joint states)
- Swaps left/right joint pairs and negates roll/yaw joints
- Doubles mini-batches in PPO updates
- Enabled by default, disable with `--agent.algorithm.symmetry_cfg=False`

### Standing Stability

When velocity command magnitude < 0.1, three rewards reinforce stable stance:

| Reward | Weight | Role |
|--------|--------|------|
| `stand_still` | -1.0 | Hold target joint positions |
| `leg_joint_vel_penalty` | -1e-3 | Damp lower-body joint velocities |
| `body_orientation_l2` | -1.0 | Stay upright |

### Curricula

| Curriculum | What it controls | Schedule |
|------------|------------------|----------|
| `force_scale_staged` | External force magnitude | Step-based ramp over 15k iterations |
| `default_pose_ratio_staged` | HOME_KEYFRAME fraction | Starts at 1.0, decreases over training |
| `height_scale_staged` | Height command range | Expands from nominal-only to full range |
| `command_vel` | Velocity command ranges | Staged expansion |
| `terrain_levels` | Terrain difficulty | Velocity-based advancement |

## Training

### Basic Commands

```bash
# G1 29-DOF, flat terrain
python scripts/train.py Unitree-G1-Locomanipulation-Flat --env.scene.num-envs=4096

# G1 29-DOF, rough terrain
python scripts/train.py Unitree-G1-Locomanipulation-Rough --env.scene.num-envs=4096

# G1 23-DOF, flat terrain
python scripts/train.py Unitree-G1-23Dof-Locomanipulation-Flat --env.scene.num-envs=4096

# G1 23-DOF, rough terrain
python scripts/train.py Unitree-G1-23Dof-Locomanipulation-Rough --env.scene.num-envs=4096
```

### Multi-GPU Training

```bash
python scripts/train.py Unitree-G1-Locomanipulation-Flat \
    --gpu-ids 0 1 --env.scene.num-envs=4096
```

### Resume from Checkpoint

```bash
python scripts/train.py Unitree-G1-Locomanipulation-Flat \
    --checkpoint-file logs/rsl_rl/g1_locomanipulation/<date>/model_10000.pt \
    --env.scene.num-envs=4096
```

### Key CLI Overrides

Any config field can be overridden via `--env.<path>.<field>=<value>`:

```bash
# More envs for faster training
--env.scene.num-envs=4096

# Disable symmetry augmentation
--agent.algorithm.symmetry_cfg=False

# Override PPO learning rate
--agent.algorithm.learning_rate=5e-4
```

## Play / Visualization

### Trained Policy

```bash
python scripts/play.py Unitree-G1-Locomanipulation-Flat \
    --checkpoint-file logs/rsl_rl/g1_locomanipulation/<date>/model_20000.pt
```

### Dummy Agents (No Checkpoint)

```bash
# Zero actions
python scripts/play.py Unitree-G1-Locomanipulation-Flat --agent zero

# Random actions
python scripts/play.py Unitree-G1-Locomanipulation-Flat --agent random
```

### Headless Viewer

```bash
python scripts/play.py Unitree-G1-Locomanipulation-Flat \
    --checkpoint-file ... --viewer viser
```

### Record Video

```bash
python scripts/play.py Unitree-G1-Locomanipulation-Flat \
    --checkpoint-file ... --video --video-length 200
```

### Keyboard Controls (Native Viewer)

| Key | Action | Step |
|-----|--------|------|
| KP_8 / KP_2 | lin_vel_x +/- | 0.1 m/s |
| KP_4 / KP_6 | lin_vel_y +/- | 0.1 m/s |
| KP_7 / KP_9 | ang_vel_z +/- | 0.1 rad/s |
| KP_ADD / KP_SUBTRACT | height +/- | 0.02 m |
| KP_5 | zero all velocity | exponential decay |
| KP_0 | reset height to nominal | instant |

### Play-Mode Testing Flags

Set these in the config to test specific conditions:

- **`fixed_upper_body_pose`**: Pin all envs to specific upper-body joint angles
- **`constant_force`**: Apply fixed force every step (bypasses impulse lifecycle)
- **`fixed_command`**: Pin velocity command (skips random resampling)
- **`fixed_height`**: Pin commanded height

### Disable Terminations

```bash
python scripts/play.py Unitree-G1-Locomanipulation-Flat \
    --checkpoint-file ... --no-terminations
```

Useful for viewing motions without episode resets.

## Evaluation

Computes velocity tracking MAE per (model, force_level), averaged across all (pose, velocity) combos. Outputs CSV + JSON + comparison plot.

### Single Checkpoint

```bash
python scripts/eval.py --task Unitree-G1-Locomanipulation-Flat \
    --checkpoint-file logs/.../model_20000.pt
```

### Multi-Model Comparison

```bash
python scripts/eval.py --task Unitree-G1-Locomanipulation-Flat \
    --eval-config eval_config.yaml
```

**Config file format** (`eval_config.yaml`):
```yaml
models:
  - name: "baseline"
    checkpoint: "logs/.../model_20000.pt"
  - name: "experiment-1"
    checkpoint: "logs/.../model_20000.pt"
```

### Mixed 23-DOF + 29-DOF Evaluation

```bash
python scripts/eval.py --eval-config mixed_eval_config.yaml
```

**Config with per-model task** (`mixed_eval_config.yaml`):
```yaml
models:
  - name: "g1-29dof-20k"
    task: "Unitree-G1-Locomanipulation-Flat"
    checkpoint: "logs/.../model_20000.pt"
  - name: "g1-23dof-20k"
    task: "Unitree-G1-23Dof-Locomanipulation-Flat"
    checkpoint: "logs/.../model_20000.pt"
```

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--task <name>` | — | Registered task name. Required unless all models specify `task`. |
| `--eval-config <yaml>` | — | Multi-checkpoint config |
| `--checkpoint-file <pt>` | — | Single checkpoint (ignored if `--eval-config` set) |
| `--force-conditions` | `"none" "medium" "large"` | Force presets to test |
| `--body-poses` | `"zero"` | Pose presets (`"neutral"`, `"zero"`, `"23dof_zero"`) |
| `--vel-x` | `(-0.5, 0.0, 0.5)` | Velocity x sweep (Python tuple syntax) |
| `--vel-y` | `(0.0,)` | Velocity y sweep |
| `--ang-z` | `(-0.5, 0.0, 0.5)` | Angular velocity z sweep |
| `--episode-steps` | 1000 | Steps per combo |
| `--fixed-height` | auto | Override base height. Auto-detected from checkpoint params. |
| `--metric` | `"combined"` | `"combined"`, `"linear"`, or `"angular"` |
| `--viewer` | `"none"` | `"none"` or `"native"` (visual verification) |

### Viewer Mode

```bash
python scripts/eval.py --task Unitree-G1-Locomanipulation-Flat \
    --eval-config eval_config.yaml --viewer native
```

Switch between velocity combos with `,`/`.` keys. Visual only — no metrics saved.

## ONNX Export

### Standalone Export

```bash
# G1 29-DOF
python scripts/export_onnx.py Unitree-G1-Locomanipulation-Flat \
    --checkpoint-file logs/.../model_20000.pt

# G1 23-DOF
python scripts/export_onnx.py Unitree-G1-23Dof-Locomanipulation-Flat \
    --checkpoint-file logs/.../model_20000.pt

# Custom output directory
python scripts/export_onnx.py Unitree-G1-Locomanipulation-Flat \
    --checkpoint-file ... --output-dir /tmp/export
```

### Auto-Export During Training

The runner automatically exports `policy.onnx` on every `save()` call (every 500 iterations by default). Output goes to the checkpoint directory.

### Metadata

Exported ONNX files include:
- Joint names
- PD gains
- Action scales
- Observation normalizer stats (mean, std) baked into the graph

## Utilities

### Compute Height Postures

```bash
python scripts/compute_height_postures.py
```

Computes IK-based joint postures for G1 at different standing heights (0.50m–0.785m). Used by `variable_posture` and `stand_still` rewards to look up target joint angles from commanded height.

### Check Motion Collisions

```bash
# 23-DOF headless scan (default)
python scripts/check_motion_collisions.py

# 29-DOF headless scan
python scripts/check_motion_collisions.py --robot g1

# Visual playback
python scripts/check_motion_collisions.py --show

# Remove collision frames and save cleaned data
python scripts/check_motion_collisions.py --robot g1_23dof --clean
python scripts/check_motion_collisions.py --robot g1 --clean
```

Checks self-collision statistics for ACCAD motion data. The `--clean` flag removes collision frames and saves a cleaned pkl file. Use `--show` for visual playback with MuJoCo viewer (enable contacts via viewer menu → Rendering → Contacts).

## 23-DOF Specifics

### Motion Data Remapping

The 23-DOF variant uses `motion_dof_indices` to extract 11 upper-body DOFs from 29-DOF motion data:

```python
MOTION_DOF_INDICES_23DOF = (12, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26)
```

### Key Differences from 29-DOF

| Aspect | 29-DOF | 23-DOF |
|--------|--------|--------|
| Upper-body DOFs | 17 | 11 |
| Wrist bodies | `wrist_yaw_link` | `wrist_roll_rubber_hand` |
| Motion data | `accad_all.pkl` | `accad_all_g1_23dof_clean.pkl` |
| Gain presets | `G1_GAIN_PRESETS` | `G1_23DOF_GAIN_PRESETS` |
| Symmetry | `G1Symmetry` | `G1_23DOFSymmetry` |
| Runner | `LocomanipulationOnPolicyRunner` | `G1_23DOF_LocomanipulationOnPolicyRunner` |

### Gain Presets

```python
G1_23DOF_GAIN_PRESETS = {
    "default": {...},
    "unitree": {...},
    "unitree_stiff": {...},  # Used by default in configs
}
```

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/tasks/locomanipulation/locomanipulation_env_cfg.py` | Base config factory |
| `src/tasks/locomanipulation/config/g1/env_cfgs.py` | G1 29-DOF config |
| `src/tasks/locomanipulation/config/g1_23dof/env_cfgs.py` | G1 23-DOF config |
| `src/tasks/locomanipulation/mdp/events.py` | HandForceEvent + MaxForceEstimator |
| `src/tasks/locomanipulation/mdp/rewards.py` | All reward functions |
| `src/tasks/locomanipulation/mdp/symmetry.py` | Symmetry augmentation |
| `src/tasks/locomanipulation/mdp/upper_body_action.py` | ACCAD motion playback |
| `src/tasks/locomanipulation/mdp/height_command.py` | Base height command |
| `src/tasks/locomanipulation/rl/runner.py` | OnPolicy runners with ONNX export |
| `scripts/train.py` | Training script |
| `scripts/play.py` | Visualization script |
| `scripts/eval.py` | Evaluation script |
| `scripts/export_onnx.py` | ONNX export script |
| `scripts/compute_height_postures.py` | IK posture computation |
| `scripts/check_motion_collisions.py` | Collision checking/cleaning |
