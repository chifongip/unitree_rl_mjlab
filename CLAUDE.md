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

**Keyboard controls** (native viewer only, numpad): 8/2=forward/back, 4/6=left/right, 7/9=yaw, +/-=height, 5=zero velocity, 0=reset height. See "Play-Mode Keyboard Overrides" below.

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

`default_pose_ratio` controls the fraction of envs holding HOME_KEYFRAME vs a motion-derived pose. A step-based curriculum (`default_pose_ratio_staged` in `curriculums.py`) lowers this ratio from 1.0 to 0.05 over training, keeping 5% of envs at the default pose as a baseline.

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

### Play-Mode Keyboard Overrides

`scripts/play.py` provides numpad-driven command overrides for the native MuJoCo viewer. No config changes needed — overrides activate on first keypress and persist across resets.

**How it works:** `KeyboardCommandOverride` holds velocity/height floats as shared state between the GLFW key callback thread and the main physics thread. `_patch_command_compute()` wraps each command term's `compute()` to apply overrides after the base computation (after resampling). Thread-safe by design (GIL-atomic attribute access). Only active with `--viewer native` (default when display is available); viser viewer is unaffected.

**Key mapping** (numpad — avoids MuJoCo camera control conflicts):

| Key | Action | Step |
|-----|--------|------|
| KP_8 / KP_2 | lin_vel_x +/- | 0.1 m/s |
| KP_4 / KP_6 | lin_vel_y +/- | 0.1 m/s |
| KP_7 / KP_9 | ang_vel_z +/- | 0.1 rad/s |
| KP_ADD / KP_SUBTRACT | height +/- | 0.02 m |
| KP_5 | zero all velocity | — |
| KP_0 | reset height to nominal | — |

Current values print to terminal on each keypress: `[KB] vel=(+1.0, +0.0, +0.0) h=0.785`. Gracefully skips missing command terms (e.g., velocity-only tasks without `base_height`).

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

**History-aware mirroring**: All mirror classes accept `history_length` and correctly handle `history_length > 1` with `flatten_history_dim=True` (the default). When history is present, mjlab flattens segments in term-major order (`[t0_feat, t1_feat, ..., tN_feat]`). Mirror classes reshape the flattened segment to `(batch, history_length, feature_dim)`, apply the transform independently per frame, and write back. With `history_length=1` (current locomanipulation default), the fast path is taken with no reshape overhead.

**Config**: `SymmetryPpoAlgorithmCfg` (local subclass of `RslRlPpoAlgorithmCfg`) in `rl_cfg.py` with `symmetry_cfg: bool` field. Enabled by default. Disable via `--agent.algorithm.symmetry_cfg=False`.

**Runner lifecycle**: `LocomanipulationOnPolicyRunner.__init__` pops `symmetry_cfg` before `super().__init__()` to avoid kwarg conflict with PPO, then sets `self.alg.symmetry` directly (without mutating `self.cfg`, which `train.py` later dumps to YAML).

**Tests**: `tests/test_symmetry.py` — 35 mock tests covering joint swaps, sign flips, per-term mirror rules, batch doubling, double-mirror identity, and history-aware mirroring (per-frame correctness with `history_length > 1`).

### Locomanipulation Angular Velocity Tuning

The G1 locomanipulation policy controls only 12 lower-body joints; upper body is motion-playback controlled. Only `hip_yaw` joints (Z-axis) can produce yaw rotation via differential actuation. This makes low-speed in-place rotation mechanically difficult.

**Key tuning in `config/g1/env_cfgs.py`:**
- `track_angular_velocity` std: `sqrt(0.05)` (tightened from base `sqrt(0.5)` to steepen the reward gradient for small errors)
- `ang_vel_xy_weight`: 0.1 (up from default 0.05, penalizes roll/pitch angular velocity during turning)
- `stand_still` command_threshold: 0.05 (lowered from 0.1 so rotation commands 0.05–0.1 rad/s aren't penalized as standing)
- Pose reward hip_yaw std: standing=0.08, walking=0.25, running=0.35 (relaxed from 0.05/0.15/0.25 to allow differential hip_yaw for turning)

**Reward landscape comparison** (std=0.224 vs old std=0.707):
| Error (rad/s) | Old reward | New reward |
|---|---|---|
| 0.1 | 0.980 | 0.819 |
| 0.2 | 0.923 | 0.449 |
| 0.3 | 0.839 | 0.169 |

**Play-mode testing:**
```bash
python scripts/play.py Unitree-G1-Locomanipulation-Flat --checkpoint_file=<path> --env.commands.twist.fixed_command='(0.0,0.0,0.3)'
```

### Locomanipulation Walk-to-Stand Transition Smoothing

The policy exhibited instability when transitioning from walking to standing: forward lean on sudden stops, body sway, and amplified instability under external forces. Root causes and fixes:

**Root causes:**
- **10x reward discontinuity**: `variable_posture` used hard masks to switch std between standing (0.05) and walking (0.5) at 0.1 m/s. This created a cliff in the reward landscape during deceleration.
- **Instantaneous command zeroing**: `_update_command` set velocity to exactly zero in one step for standing envs. No deceleration ramp.
- **Resample spike**: `_resample_command` overwrote `vel_command_b` with random values even for standing envs, which the decay then fought against.

**Fix 1 — Sigmoid blend in `variable_posture`** (`rewards.py:433-447`): Replaced hard standing/walking/running masks with sigmoid blends. `blend_scale` (default 30.0, configurable via config params) gives a ~0.13 m/s transition zone centered at `walking_threshold=0.1`. Speed below ~0.03 m/s is fully standing, above ~0.17 m/s fully walking, with smooth interpolation between.

Formula: `std = std_standing * (1-α_s) + std_walking * α_s`, where `α_s = σ((speed - threshold) * blend_scale)`. Same pattern for walking→running boundary.

**Fix 2 — Exponential command decay** (`velocity_command.py:117-147`): Added `_stand_decay_active` tracker (init at line 45). When an env becomes standing, instead of zeroing instantly, the command decays via `vel *= 0.98` each step (~1s time constant at 50Hz). Snaps to zero when `|cmd| < 0.01`. Decay state auto-clears when envs leave standing mode. Skips activation for envs whose command is already zero to avoid oscillating `_stand_decay_active`.

**Fix 3 — Preserve pre-resample velocity** (`velocity_command.py:81-100`): `_resample_command` saves `old_vel` before overwriting, then restores it for envs selected as standing. This ensures standing envs decay from whatever they were actually tracking, not a random new command.

**Parameter guidance:**
- `decay = exp(-dt / τ)` where `dt = 0.02s` (4× decimation × 0.005s physics). τ = 1.0s → decay = 0.980. Match τ to the robot's physical deceleration capability.
- `blend_scale`: transition width ≈ 4/k. k=30 → 0.13 m/s blend zone. Wider (k=20) = smoother but less precise regime separation. Narrower (k=50) = sharper, closer to original behavior. Configurable via `cfg.rewards["pose"].params["blend_scale"]`.

**Standing stability rewards** (added to address persistent sway after the above fixes):

| Reward | Weight | Scope | Role |
|---|---|---|---|
| `stand_still` | -1.0 | Lower-body joints | "Be at the target pose" (position) |
| `leg_joint_vel_penalty` | -0.5 | Lower-body joints | "And stay there" (velocity damping) |
| `base_lin_vel_penalty` | -1.0 | Base xy velocity | "Don't drift" (horizontal velocity) |
| `body_orientation_l2` | -1.0 | Torso link | "Stay upright" (torso tilt) |

`base_lin_vel_penalty` (`rewards.py:138-153`): Penalizes `||v_xy||^2` when command < 0.1. Fills the gap left by `track_linear_velocity`'s broad Gaussian (std=0.5), which has a weak gradient for small drift.

`leg_joint_vel_penalty` (`rewards.py:154-172`): Penalizes `sum(joint_vel^2)` for lower-body joints when command < 0.1. Acts as a damper on corrective leg motions to prevent self-excited oscillations during walk-to-stand transitions. Complements `stand_still` (position) with a velocity constraint.

**Verification:** Short training run (5 iterations, 64 envs) — no NaN, stable losses, all 19 reward terms reporting normally.

### Locomanipulation Base Height Command

`BaseHeightCommand` (`src/tasks/locomanipulation/mdp/height_command.py`) commands an absolute world-frame z-height for the robot root. The robot must maintain the specified height whether standing or walking.

**Command:** `BaseHeightCommandCfg` with `ranges=(min_z, max_z)`, `nominal_height`, `max_deviation`, `height_scale ∈ [0, 1]`, and `nominal_height_ratio`. The sampling range is `(nominal - max_deviation * scale, nominal)`. A fraction `nominal_height_ratio` of envs always command `nominal_height` (baseline episodes). Optional `fixed_height` overrides random sampling. Default range for G1: (0.5, 0.785) meters, `nominal_height_ratio=0.05`.

**Curriculum:** `height_scale_staged` in `curriculums.py` — same step-based pattern as `force_scale_staged`. Ramps `height_scale` from 0.0 (nominal only) to 1.0 (full range) over training. Config: `cfg.curriculum["height_scale"]` in `config/g1/env_cfgs.py`. Play mode clears all curricula.

**Reward:** `track_base_height` in `rewards.py` — Gaussian kernel `exp(-(cmd_z - actual_z)^2 / std^2)` with `std = sqrt(0.05)`.

**Observation:** `base_height_command` in both actor and critic groups, exposed via `generated_commands` with `command_name="base_height"`. Shape: `[num_envs, 1]`. Additionally, `root_height` (actual z-position) is in the **critic only** as privileged info via `mdp.root_height` in `observations.py`.

**Height-dependent posture:** `variable_posture` reward accepts `base_height_command_name` and `height_postures` params. When set, the desired posture is looked up from a `{height: {joint_name: radians}}` table based on the commanded height, instead of using the fixed `default_joint_pos`. This ensures the pose reference adapts (e.g., bent knees when crouching).

**Height-aware stand_still:** `stand_still` is a class (not a plain function) that accepts the same `height_postures` and `base_height_command_name` params. When set, the target posture is looked up from the height table based on commanded height, rather than always using `default_joint_pos`. This prevents `stand_still` from fighting the crouched posture needed for lower heights. Without height_postures, falls back to `default_joint_pos` (backward compatible).

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