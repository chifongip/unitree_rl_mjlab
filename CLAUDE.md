# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Unitree RL Mjlab is a reinforcement learning project for Unitree robots (Go2, G1, G1-23DOF, H1_2, A2, AS2, R1, H2) built on the [mjlab](https://github.com/mujocolab/mjlab) framework with MuJoCo as the physics backend. The workflow is: **Train** → **Play** → **Sim2Real Deploy**.

## Common Commands

### Setup
```bash
conda create -n unitree_rl_mjlab python=3.11
conda activate unitree_rl_mjlab
pip install -e .
```

### Training (velocity tracking)
```bash
# Single GPU
python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096

# Multi-GPU
python scripts/train.py Unitree-G1-Flat --gpu-ids 0 1 --env.scene.num-envs=4096

# Motion imitation (tracking)
python scripts/train.py Unitree-G1-Tracking-No-State-Estimation --motion_file=src/assets/motions/g1/dance1_subject2.npz --env.scene.num-envs=4096

# Locomanipulation (lower-body policy + upper-body motion playback)
python scripts/train.py Unitree-G1-Locomanipulation-Flat --env.scene.num-envs=4096

# Locomanipulation G1-23DOF
python scripts/train.py Unitree-G1-23Dof-Locomanipulation-Flat --env.scene.num-envs=4096
```

### Play / Visualization
```bash
# Trained policy
python scripts/play.py Unitree-G1-Flat --checkpoint_file=logs/rsl_rl/g1_velocity/<date>/model_<iter>.pt

# Dummy agents (no checkpoint needed)
python scripts/play.py Unitree-G1-Flat --agent zero
python scripts/play.py Unitree-G1-Flat --agent random

# Record video
python scripts/play.py Unitree-G1-Flat --checkpoint_file=... --video --video-length 200

# Headless viewer (no display)
python scripts/play.py Unitree-G1-Flat --checkpoint_file=... --viewer viser
```

**Keyboard controls** (native viewer only, numpad):

| Key | Action | Step |
|-----|--------|------|
| KP_8 / KP_2 | lin_vel_x +/- | 0.1 m/s |
| KP_4 / KP_6 | lin_vel_y +/- | 0.1 m/s |
| KP_7 / KP_9 | ang_vel_z +/- | 0.1 rad/s |
| KP_ADD / KP_SUBTRACT | height +/- | 0.02 m |
| KP_5 | zero all velocity | — |
| KP_0 | reset height to nominal | — |

### Evaluation
```bash
# Single checkpoint
python scripts/eval.py --task Unitree-G1-Locomanipulation-Flat \
    --checkpoint-file logs/.../model_20000.pt

# Multi-model comparison via YAML config
python scripts/eval.py --task Unitree-G1-Locomanipulation-Flat \
    --eval-config eval_config.yaml

# Mixed 23-DOF + 29-DOF evaluation (no --task needed if all models specify task)
python scripts/eval.py --eval-config mixed_eval_config.yaml

# Visual verification with viewer
python scripts/eval.py --task Unitree-G1-Locomanipulation-Flat \
    --eval-config eval_config.yaml --viewer native
```

Computes velocity tracking MAE per (model, force_level), averaged across all (pose, velocity) combos. Outputs CSV + JSON + comparison plot (force vs error, one curve per model).

**Key flags:**
- `--task <name>`: Registered task name. Required unless all models in `--eval-config` specify their own `task`.
- `--eval-config <yaml>`: Multi-checkpoint config (see format below)
- `--checkpoint-file <pt>`: Single checkpoint (ignored if `--eval-config` set)
- `--force-conditions`: Force presets to test (`"none"`, `"medium"` (-15N), `"large"` (-30N))
- `--body-poses`: Pose presets (`"neutral"` = default joint pos, `"zero"` = explicit zeros for 29-DOF, `"23dof_zero"` for 23-DOF)
- `--vel-x`, `--vel-y`, `--ang-z`: Velocity command sweeps (Python tuple syntax)
- `--episode-steps`: Steps per combo (default 1000)
- `--fixed-height`: Override base height for all models. If omitted, auto-detected from each checkpoint's saved training params.
- `--metric`: `"combined"` (default), `"linear"`, or `"angular"`
- `--viewer`: `"none"` (default) or `"native"` (opens GLFW window)

**Config file format** (`eval_config.yaml`):
```yaml
models:
  - name: "29dof-20k"
    task: "Unitree-G1-Locomanipulation-Flat"     # optional, defaults to --task
    checkpoint: "logs/.../model_20000.pt"
  - name: "23dof-20k"
    task: "Unitree-G1-23Dof-Locomanipulation-Flat"
    checkpoint: "logs/.../model_20000.pt"
```

**Multi-task support**: Models from different tasks (e.g. 23-DOF and 29-DOF) can be evaluated in a single run. Each model gets its own env built from its task's config (action space, runner class, robot assets). The `task` column in results distinguishes them. Plot labels show short task names when multiple tasks are present.

**Base height**: `fixed_height` is auto-loaded from each checkpoint's `env_overrides.yaml`/`env.yaml` (reads `commands.base_height.fixed_height`, falls back to `nominal_height`). CLI `--fixed-height` overrides all.

**Performance**: Velocity combos are parallelized — envs are partitioned into groups (one per combo) with per-env commands on `vel_command_b`, so all combos run in a single episode batch. `num_envs` is auto-adjusted to be divisible by the number of combos (rounded down, minimum 1 env/combo). Force/pose remain sequential (global config mutation). Viewer mode uses the same partitioning; switch between combos with `,`/`.` keys. Progress bar via tqdm.

### Export ONNX
```bash
# Velocity task
python scripts/export_onnx.py Unitree-G1-Flat \
    --checkpoint-file logs/rsl_rl/g1_velocity/<date>/model_<iter>.pt

# Locomanipulation task
python scripts/export_onnx.py Unitree-G1-Locomanipulation-Flat \
    --checkpoint-file logs/.../model_20000.pt

# Tracking task (also exports motion-bundled ONNX)
python scripts/export_onnx.py Unitree-G1-Tracking-No-State-Estimation \
    --checkpoint-file logs/.../model_<iter>.pt \
    --motion-file src/assets/motions/g1/dance1_subject2.npz

# Custom output directory
python scripts/export_onnx.py Unitree-G1-Flat \
    --checkpoint-file logs/.../model_<iter>.pt --output-dir /tmp/export
```

Exports a trained checkpoint to `policy.onnx` with metadata (joint names, PD gains, action scales, obs normalizer stats baked in). Output goes to the checkpoint's directory by default. For tracking tasks, also exports a motion-bundled ONNX with reference data. Uses `play=True` env config and restores training-time params from the checkpoint's `params/` dir.

**Key flags:**
- `--checkpoint-file <pt>`: Path to `.pt` checkpoint (required)
- `--output-dir <dir>`: Output directory (default: checkpoint's directory)
- `--motion-file <npz>`: Motion file for tracking tasks
- `--device <str>`: `cpu` (default) or `cuda:0`

### Compute Height Postures (Locomanipulation)
```bash
python scripts/compute_height_postures.py
```
Computes IK-based joint postures for G1 at different standing heights (0.50m–0.785m). Used by `variable_posture` and `stand_still` rewards to look up target joint angles from commanded height.

### Convert motion CSV to NPZ
```bash
python scripts/csv_to_npz.py --input-file src/assets/motions/g1/dance1.csv --output-name dance1.npz --input-fps 30 --output-fps 50 --robot g1
```

### Check Motion Collisions (Locomanipulation)
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

# Verify cleaned data
python scripts/check_motion_collisions.py --motion-file src/assets/data/g1/accad_all_g1_23dof_clean.pkl
```

Checks self-collision statistics for ACCAD motion data on the G1 robot. Supports both 29-DOF and 23-DOF variants. The `--clean` flag removes collision frames and saves a cleaned pkl file. Use `--show` for visual playback with MuJoCo viewer (enable contacts via viewer menu → Rendering → Contacts).

**Key flags:**
- `--robot`: `g1_23dof` (default) or `g1`
- `--motion-file`: Path to motion pkl (default: `accad_all.pkl`)
- `--show`: Open MuJoCo viewer
- `--clip`: Filter to clips matching substring
- `--collision-only`: Only show collision frames (with `--show`)
- `--clean`: Save cleaned motion data

### Deploy (C++ on real robot or unitree_mujoco simulator)
```bash
# Install C++ build dependencies (Ubuntu)
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev

# Build unitree_mujoco simulator
cd simulate && mkdir build && cd build && cmake .. && make -j8

# Build robot deploy binary
cd deploy/robots/g1 && mkdir build && cd build && cmake .. && make

# Run in simulation
./g1_ctrl --network=lo
# Run on real robot
./g1_ctrl --network=enp5s0
```

### List Registered Environments
```bash
python scripts/list_envs.py
```

## Architecture

### Task System
Tasks are registered via `register_mjlab_task()` in `src/tasks/<type>/config/<robot>/__init__.py`. Three task families:

- **Velocity** (`src/tasks/velocity/`): Velocity tracking with flat/rough terrain.
- **Tracking** (`src/tasks/tracking/`): Motion imitation from NPZ reference motions.
- **Locomanipulation** (`src/tasks/locomanipulation/`): Lower-body policy (12 DOF) with upper-body motion playback from ACCAD dataset. G1 only.

Robot-specific configs live in `config/<robot>/env_cfgs.py` — they call the base factory and customize scene entities, sensors, reward params, and terminations.

### Configuration Pattern (dict-based, no @configclass)
All manager configs (rewards, observations, actions, commands, terminations, events, curriculum) are plain dicts of `TermCfg` objects. Robot configs override specific entries by mutating the dict returned from the factory. This is a hard requirement from mjlab — do not use `@configclass` or bridge helpers.

### MDP Modules
Custom terms are in `src/tasks/<type>/mdp/` which re-exports from `mjlab.envs.mdp` and adds project-specific:
- `rewards.py`, `observations.py`, `terminations.py`, `curriculums.py` (velocity)
- `rewards.py`, `observations.py`, `terminations.py`, `commands.py`, `metrics.py` (tracking)
- `rewards.py`, `observations.py`, `terminations.py`, `curriculums.py`, `events.py`, `velocity_command.py`, `height_command.py`, `upper_body_action.py` (locomanipulation)

### Locomanipulation Summary

Policy controls 12 lower-body joints only; upper body driven by ACCAD motion data via `UpperBodyMotionAction`. Supports both G1 29-DOF (17 upper-body DOFs) and G1 23-DOF (11 upper-body DOFs). Two modes: pose-only (sample frame at reset, hold) and clip playback (frame-by-frame). `waist_yaw_only=True` zeros out waist_roll/pitch from motion data. `default_pose_ratio` curriculum (`default_pose_ratio_staged`) gradually introduces diverse upper-body poses during training. `fixed_upper_body_pose` (play mode) pins all envs to a specific upper-body joint configuration.

**G1-23DOF specifics**: Uses `motion_dof_indices` to remap 29-DOF motion data columns to 23-DOF joint layout. Wrist body names are `wrist_roll_rubber_hand` (not `wrist_yaw_link`). Symmetry uses `G1_23DOFSymmetry` with 23-joint swap/flip mappings. Cleaned motion data (`accad_all_g1_23dof_clean.pkl`) removes frames with self-collisions. Gain presets (`G1_23DOF_GAIN_PRESETS`) support "default", "unitree", "unitree_stiff".

**Rewards restricted to lower-body joints** (matching policy control): `pose` (variable_posture), `stand_still`, `joint_acc_l2`, `joint_pos_limits`, `leg_joint_vel_penalty`. Full-body rewards (policy compensates via hips): `body_orientation_l2`, `body_ang_vel`, `angular_momentum`.

**`variable_posture`**: Penalizes deviation from default pose with per-joint std that varies by speed regime: `std_standing` (speed < 0.1), `std_walking` (0.1–1.5), `std_running` (>= 1.5). Hard thresholds — no blending. When `height_postures` is set, the desired posture is looked up from a `{height: {joint: radians}}` table based on the commanded height.

**`stand_still`**: Penalizes joint deviation from target pose only when velocity command magnitude < `command_threshold`. Also supports `height_postures` lookup. Works with `leg_joint_vel_penalty` (damps joint velocities when standing) and `body_orientation_l2` for standing stability.

**`body_orientation_l2`**: L2 penalty on projected gravity xy (upright orientation). Optionally applies different weights for standing vs walking, gated by twist command magnitude via `standing_command_name`/`standing_threshold`/`standing_weight`/`walking_weight` (same pattern as `track_base_height`). Default: 3x penalty when stationary, 1x when walking.

**`track_base_height`**: Gaussian reward `exp(-(cmd_z - actual_z)^2 / std^2)` multiplied by per-env weight: `standing_weight` (default 1.0) when `|twist_cmd| < 0.1`, `walking_weight` (default 0.5) otherwise.

**`self_collision_cost`**: Penalizes pelvis self-collisions using a `ContactSensor` with force history (`history_length=4`). Counts substeps where any contact force exceeds 10N.

**`foot_swing_height`** (class-based): Tracks peak foot height during each swing phase and penalizes deviation from `target_height` at landing (`first_contact`). Unlike `feet_clearance` (velocity-weighted, negligible at low speeds), this provides a speed-independent signal that fires at every landing. Stateful — maintains `peak_heights` tensor, cleared via `reset()` at episode boundaries. Monitor `Metrics/peak_height_mean` during training.

External force curriculum (`HandForceEvent`) applies random wrenches to end-effectors to simulate carrying objects. Force bounds computed via Jacobian transpose (`MaxForceEstimator`). Two curriculum options: step-based (`force_scale_staged`) and adaptive (`force_curriculum_adaptive`). Per-env Dirichlet axis scaling for force diversity. Config in `cfg.events["hand_force"]` and `cfg.curriculum["force_curriculum"]`.

Base height command (`BaseHeightCommand`) controls absolute z-height with a height-dependent posture table (7 entries, 0.50m–0.785m) computed via `scripts/compute_height_postures.py` (IK solver + scipy optimization). Curriculum: `height_scale_staged` ramps `height_scale` from 0 to 1. Both `variable_posture` and `stand_still` look up target joint angles from this table.

Symmetric data augmentation doubles mini-batches by mirroring across the sagittal plane. Enabled via `SymmetryPpoAlgorithmCfg.symmetry_cfg=True` (default). Disable with `--agent.algorithm.symmetry_cfg=False`. `LocomanipulationOnPolicyRunner.__init__` pops `symmetry_cfg` before PPO init to avoid kwarg conflict. Runner also auto-exports `policy.onnx` on save.

**Play-mode config** (`play=True` in `unitree_g1_locomanipulation_flat_env_cfg`):
- Infinite episode length, disables observation corruption
- Removes `push_robot` event, clears all curricula
- Sets `hand_force` to `no_force_ratio=1.0` with zero force range (disables random impulses; `constant_force` can still be set for testing)
- Adds `randomize_terrain` on reset
- Sets `fixed_upper_body_pose` (HOME_KEYFRAME), `fixed_command=(0,0,0)`, `fixed_height=0.785`
- Flat variant: narrows command ranges, removes terrain_scan/height_scan
- `--no_terminations` flag disables all termination conditions (useful for viewing motions with dummy agents)

**Play-mode testing flags** (set in config, uncomment to activate):
- `fixed_upper_body_pose` (action cfg): pin upper body to specific joint angles
- `constant_force` (event params): apply fixed force every step, bypasses impulse lifecycle
- `fixed_command` (command cfg): pin velocity command
- `fixed_height` (command cfg): pin commanded height

**Keyboard controls** (native viewer, locomanipulation): `KeyboardCommandOverride` in `play.py` provides numpad 8/2=vel_x, 4/6=vel_y, 7/9=yaw, +/-=height, 5=zero vel (with exponential decay), 0=reset height. Velocity adjustments are instant; zeroing uses decay toward zero (~1s time constant at 50Hz). Monkey-patches `compute()` on twist and base_height command terms.

### Play Mode Config Restoration

When playing a trained policy, `scripts/play.py` restores training-time env config from saved params so the policy sees the same observation distribution it was trained on.

**`_extract_scalar_overrides`** (duplicated in `train.py` and `tune.py`): Recursively extracts only scalar values (int/float/bool/str/None/Enum) plus simple dicts/lists from `asdict(env_cfg)`. No depth limit — reaches observation term params nested 5+ levels deep. Saved as `env_overrides.yaml` alongside `env.yaml` and `agent.yaml`.

**`_load_params_overrides`** (`play.py`): Prefers `env_overrides.yaml` (clean YAML, `safe_load`-able). Falls back to `env.yaml` with Python tag stripping. Auto-detected from checkpoint `log_dir/params/`, overridable via `--params_dir`. Calls `_supplement_obs_terms` after loading `env_overrides.yaml` to fill in missing observation term params from `env.yaml` (needed for training runs saved with the old `max_depth=3` extraction which omitted terms 5+ levels deep).

**`_apply_env_overrides`** (`play.py`): Restores saved scalar params into play-time `env_cfg`, printing diffs for each changed field. Covers: top-level scalars (decimation, seed), observation group/term `history_length`, observation term `params` (scalar-only — structured objects like `SceneEntityCfg` are skipped since they lose their type when round-tripped through YAML), sim timestep, and action scale dicts. Uses `_check_and_set` pattern: compares current vs saved, prints diff, only sets if different.

**`applied_params` dedup**: Because `critic_terms = {**actor_terms, ...}` creates shallow copies sharing the same `ObservationTermCfg` objects, the restore loop tracks `(id(term_obj), param_key)` tuples to avoid re-processing the same term twice (which would cause the critic group to overwrite the actor group's changes).

### Robot Assets
`src/assets/robots/<robot>/` — each exports a `get_<robot>_robot_cfg()` function and a constants module with joint names, body names, default poses.

### Custom Runners
`src/tasks/<type>/rl/runner.py` — `VelocityOnPolicyRunner` / `MotionTrackingOnPolicyRunner` / `LocomanipulationOnPolicyRunner` extend `MjlabOnPolicyRunner` to auto-export `policy.onnx` on save for deployment. `G1_23DOF_LocomanipulationOnPolicyRunner` overrides the symmetry function for 23-DOF.

### ONNX Export
Training runners auto-export `policy.onnx` on every `save()` call. `scripts/export_onnx.py` provides standalone export for any checkpoint. Both paths call `runner.export_policy_to_onnx()` (opset 18, `dynamo=False`) which wraps the actor in `_OnnxMLPModel` — this bakes the obs normalizer (`_mean`, `_std`) and deterministic action output into the graph as constants. Metadata (joint names, PD gains, action scales) is attached via `get_base_metadata()` + `attach_metadata_to_onnx()`. For tracking tasks, `MotionTrackingOnPolicyRunner` also exports a motion-bundled ONNX with reference data as registered buffers.

### Deploy (C++)
`deploy/robots/<robot>/` — C++ control binaries using ONNX Runtime for inference, communicating via CycloneDDS/unitree_sdk2. FSM states in `deploy/include/FSM/`.

### Simulation (unitree_mujoco)
`simulate/` — C++ unitree_mujoco integration for sim-testing deployed policies before real-robot use.

## CLI / Config Override
Both `train.py` and `play.py` use [tyro](https://github.com/brentyi/tyro) for CLI parsing. Any env/agent config field can be overridden via `--env.<path>.<field>=<value>` or `--agent.<field>=<value>`.

## Key mjlab API Mappings
- `asset_name` → `entity_name`
- `body_pos_w` → `body_link_pos_w`, `body_quat_w` → `body_link_quat_w`
- `AdditiveUniformNoiseCfg` → `UniformNoiseCfg`
- Sensors: `ContactSensorCfg`, `RayCastSensorCfg` from `mjlab.sensor`
- Terrain: `TerrainEntityCfg`, `TerrainGeneratorCfg` from `mjlab.terrains`
- Training: `MjlabOnPolicyRunner`, `RslRlVecEnvWrapper` from `mjlab.rl`

## Testing / Linting
No CI or linting configuration exists. Mock tests verify correctness:
- `tests/test_symmetry.py` — symmetry augmentation (joint swaps, sign flips, batch doubling, history-aware mirroring) for both 29-DOF and 23-DOF. Run with `PYTHONPATH="" python -m pytest tests/test_symmetry.py -v`.
- `tests/test_max_force_estimator.py` — Jacobian-based force estimation (mock + G1 integration). Run with `python tests/test_max_force_estimator.py`.

pytest has ROS plugin conflicts — run tests directly or use `python -m pytest -p no:launch_testing`.
