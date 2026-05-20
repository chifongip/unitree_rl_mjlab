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

### Convert motion CSV to NPZ
```bash
python scripts/csv_to_npz.py --input-file src/assets/motions/g1/dance1.csv --output-name dance1.npz --input-fps 30 --output-fps 50 --robot g1
```

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

- **Velocity** (`src/tasks/velocity/`): Velocity tracking with flat/rough terrain. Base config in `velocity_env_cfg.py` via `make_velocity_env_cfg()`.
- **Tracking** (`src/tasks/tracking/`): Motion imitation from NPZ reference motions. Base config in `tracking_env_cfg.py` via `make_tracking_env_cfg()`.
- **Locomanipulation** (`src/tasks/locomanipulation/`): Lower-body policy (12 DOF) with upper-body motion playback from ACCAD dataset. Base config in `locomanipulation_env_cfg.py` via `make_locomanipulation_env_cfg()`. G1 only.

Robot-specific configs live in `config/<robot>/env_cfgs.py` — they call the base factory and customize scene entities, sensors, reward params, and terminations.

### Configuration Pattern (dict-based, no @configclass)
All manager configs (rewards, observations, actions, commands, terminations, events, curriculum) are plain dicts of `TermCfg` objects. Robot configs override specific entries by mutating the dict returned from the factory. This is a hard requirement from mjlab — do not use `@configclass` or bridge helpers.

### MDP Modules
Custom terms are in `src/tasks/<type>/mdp/` which re-exports from `mjlab.envs.mdp` and adds project-specific:
- `rewards.py`, `observations.py`, `terminations.py`, `curriculums.py` (velocity)
- `rewards.py`, `observations.py`, `terminations.py`, `commands.py`, `metrics.py` (tracking)
- `rewards.py`, `observations.py`, `terminations.py`, `curriculums.py`, `events.py`, `velocity_command.py`, `height_command.py`, `upper_body_action.py` (locomanipulation)

### Locomanipulation Upper-Body Modes

`UpperBodyMotionAction` (`src/tasks/locomanipulation/mdp/upper_body_action.py`) drives upper-body joints (17 DOF) from ACCAD motion data. Two modes:

- **Clip playback** (`pose_only=False`): plays back motion clips frame-by-frame, wrapping cyclically.
- **Pose only** (`pose_only=True`): samples a single random frame at reset and holds it for the full episode. Default during training.

`default_pose_ratio` controls the fraction of envs holding HOME_KEYFRAME vs a motion-derived pose. A step-based curriculum (`default_pose_ratio_staged` in `curriculums.py`) can lower this ratio over training to gradually introduce more diverse poses.

### Locomanipulation Force Curriculum

`HandForceEvent` (`src/tasks/locomanipulation/mdp/events.py`) applies random external wrenches to end-effector bodies to simulate carrying heavy objects. Forces are specified in the **world frame** via `write_external_wrench_to_sim`.

**Lifecycle:** Each env cycles through cooldown → trigger → sustain (duration) → expire.
- `force_range_max * force_scale` defines the per-axis force sampling range, where `force_scale ∈ [0, 1]` is learned by the curriculum.
- `no_force_ratio` (0.05): fraction of envs receiving zero force for baseline episodes.
- `zero_force_prob`: per-axis independent probability of zeroing that force component.
- `body_point_offset_range`: local-frame offset rotated to world frame; torque = `cross(offset_w, force)`.

**Curriculum** — two options in `curriculums.py`:
- `force_scale_staged`: step-based schedule (e.g., `step:0→scale:0, step:48000→scale:0.2`). Default.
- `force_curriculum_adaptive`: adjusts `force_scale` based on completed episode length mean.

**Config:** `cfg.events["hand_force"]` and `cfg.curriculum["force_curriculum"]` in `config/g1/env_cfgs.py`. Play mode removes `hand_force`.

### Locomanipulation Max Force Estimation

`MaxForceEstimator` (`src/tasks/locomanipulation/mdp/events.py`) computes per-EE force bounds dynamically using the Jacobian transpose relationship `τ = J^T · F`. This ensures external forces are physically plausible for the current arm configuration.

**Algorithm** (ported from FALCON `_calculate_max_ee_forces()`):
1. Compute translational Jacobian for each wrist body via `mjwarp.jac()`
2. Slice columns for constraint DOFs (configurable via `constraint_joint_names`)
3. Compute per-joint, per-axis force limits: `F_max[axis, i] = effort_limit[i] / (|J[axis, i]| + ε)`
4. Take most restrictive across joints: `F_max[axis] = min_i(F_max[axis, i])`
5. Clip to hard bounds (`force_range_max`) and scale by `force_scale` curriculum
6. Apply per-env Dirichlet `force_xyz_scale` for axis-wise diversity

**Config params** in `cfg.events["hand_force"]`:
- `max_force_estimation: True` — enable Jacobian-based bounds
- `constraint_joint_names`: tuple of regex patterns for joints to include in the constraint (default: arms only, 7 per side)

**Example force bounds** (G1, arms-only constraint, `eps=1e-2`):
- All joints at 0: x=±118N, y=±119N, z=±124N
- HOME_KEYFRAME (bent elbows): x=±66N, y=±75N, z=±318N

**Tests:** `tests/test_max_force_estimator.py` — 5 mock tests + 1 integration test with real G1 MuJoCo model. Run with `python -c "import sys; sys.path.insert(0,'.'); from tests.test_max_force_estimator import *; [t() for t in [test_single_joint_force_bounds, test_multiple_joints_most_restrictive_wins, test_symmetric_effort_gives_symmetric_bounds, test_two_end_effectors_independent, test_zero_jacobian_gives_large_finite_bound]]"` (pytest has ROS plugin conflicts).

### Locomanipulation Play-Mode Testing

Four config options enable controlled model comparison in play mode:

- **`fixed_upper_body_pose`** (`UpperBodyMotionActionCfg`): `dict[str, float] | None` — pin all envs to a specific pose (joint name → radians). Set in `env_cfgs.py` or via `--env.actions.upper_body_motion.fixed_upper_body_pose='{"left_shoulder_pitch_joint": -1.57}'`.
- **`constant_force`** (`HandForceEvent` params): `dict[str, float] | None` — apply a fixed force every step (axis → Newtons). Set via `--env.events.hand_force.params.constant_force='{"z": -10.0}'`.
- **`fixed_command`** (`UniformVelocityCommandCfg`): `tuple[float, float, float] | None` — pin velocity to `(lin_vel_x, lin_vel_y, ang_vel_z)`. Set via `--env.commands.twist.fixed_command='(0.5,0.0,0.0)'`.
- **`fixed_height`** (`BaseHeightCommandCfg`): `float | None` — pin commanded height (meters). Set via `--env.commands.base_height.fixed_height=0.7`.

All default to `None` (existing behavior). Play mode keeps `hand_force` event with random forces disabled; set `constant_force` to activate.

### Locomanipulation Symmetric Data Augmentation

Uses rsl_rl's built-in `symmetry_cfg` in PPO to double mini-batches by mirroring observations and actions across the sagittal plane (x-z plane, flip y-axis). Implementation in `src/tasks/locomanipulation/mdp/symmetry.py`.

**Mirror rules** (MuJoCo convention: x=forward, y=left, z=up):
- `base_ang_vel`: negate [0], [2] (pseudovector)
- `projected_gravity`: negate [1]
- `command`: negate [1] (lin_vel_y), [2] (ang_vel_z)
- `phase`: negate [0], [1] (half-period shift)
- `joint_pos/vel` (29 DOF): swap L/R pairs + negate roll/yaw joints
- `actions` (12 DOF lower body): same swap+negate
- Foot/wrist force terms: swap L/R + negate y-component

**Sign-flip joints**: all roll and yaw joints. Pitch joints do NOT flip.

**Config**: `SymmetryPpoAlgorithmCfg` (local subclass of `RslRlPpoAlgorithmCfg`) in `rl_cfg.py` with `symmetry_cfg: bool` field. Enabled by default. Disable via `--agent.algorithm.symmetry_cfg=False`.

**Runner lifecycle**: `LocomanipulationOnPolicyRunner.__init__` pops `symmetry_cfg` before `super().__init__()` to avoid kwarg conflict with PPO, then sets `self.alg.symmetry` directly (without mutating `self.cfg`, which `train.py` later dumps to YAML).

**Tests**: `tests/test_symmetry.py` — 25 mock tests covering joint swaps, sign flips, per-term mirror rules, batch doubling, and double-mirror identity.

### Locomanipulation Base Height Command

`BaseHeightCommand` (`src/tasks/locomanipulation/mdp/height_command.py`) commands an absolute world-frame z-height for the robot root. The robot must maintain the specified height whether standing or walking.

**Command:** `BaseHeightCommandCfg` with `ranges=(min_z, max_z)` and optional `fixed_height` for play mode. Default range for G1: (0.5, 0.785) meters.

**Reward:** `track_base_height` in `rewards.py` — Gaussian kernel `exp(-(cmd_z - actual_z)^2 / std^2)` with `std = sqrt(0.05)`.

**Observation:** `base_height_command` in both actor and critic groups, exposed via `generated_commands` with `command_name="base_height"`. Shape: `[num_envs, 1]`.

**Height-dependent posture:** `variable_posture` reward accepts `base_height_command_name` and `height_postures` params. When set, the desired posture is looked up from a `{height: {joint_name: radians}}` table based on the commanded height, instead of using the fixed `default_joint_pos`. This ensures the pose reference adapts (e.g., bent knees when crouching).

**Height posture table (G1):** 7 entries from 0.50m to 0.785m at 0.05m intervals. Computed offline via `scripts/compute_height_postures.py` (IK solver using MuJoCo forward kinematics + scipy optimization). The script constrains foot capsule geoms to ground (z=0), pelvis above foot centroid, and legs in the sagittal plane. Run with `--show` to visualize in MuJoCo viewer.

**Play mode:** `fixed_height=0.785` (nominal G1 height). Set via `--env.commands.base_height.fixed_height=0.7`.

**Symmetry:** Height command is a scalar z-value, invariant under sagittal mirror. No special mirror transform needed.

### Robot Assets
`src/assets/robots/<robot>/` — each exports a `get_<robot>_robot_cfg()` function and a constants module with joint names, body names, default poses.

### Custom Runners
`src/tasks/<type>/rl/runner.py` — `VelocityOnPolicyRunner` / `MotionTrackingOnPolicyRunner` / `LocomanipulationOnPolicyRunner` extend `MjlabOnPolicyRunner` to auto-export `policy.onnx` on save for deployment.

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
- `tests/test_symmetry.py` — symmetry augmentation (joint swaps, sign flips, batch doubling). Run with `python tests/test_symmetry.py`.
- `tests/test_max_force_estimator.py` — Jacobian-based force estimation (mock + G1 integration). Run with `python -c "import sys; sys.path.insert(0,'.'); from tests.test_max_force_estimator import *; [t() for t in [test_single_joint_force_bounds, test_multiple_joints_most_restrictive_wins, test_symmetric_effort_gives_symmetric_bounds, test_two_end_effectors_independent, test_zero_jacobian_gives_large_finite_bound]]"`.

pytest has ROS plugin conflicts — run tests directly or use `python -m pytest -p no:launch_testing`.