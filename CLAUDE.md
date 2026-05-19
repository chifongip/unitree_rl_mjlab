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
- `rewards.py`, `observations.py`, `terminations.py`, `curriculums.py`, `events.py`, `velocity_command.py`, `upper_body_action.py` (locomanipulation)

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

### Locomanipulation Play-Mode Testing

Three config options enable controlled model comparison in play mode:

- **`fixed_upper_body_pose`** (`UpperBodyMotionActionCfg`): `dict[str, float] | None` — pin all envs to a specific pose (joint name → radians). Set in `env_cfgs.py` or via `--env.actions.upper_body_motion.fixed_upper_body_pose='{"left_shoulder_pitch_joint": -1.57}'`.
- **`constant_force`** (`HandForceEvent` params): `dict[str, float] | None` — apply a fixed force every step (axis → Newtons). Set via `--env.events.hand_force.params.constant_force='{"z": -10.0}'`.
- **`fixed_command`** (`UniformVelocityCommandCfg`): `tuple[float, float, float] | None` — pin velocity to `(lin_vel_x, lin_vel_y, ang_vel_z)`. Set via `--env.commands.twist.fixed_command='(0.5,0.0,0.0)'`.

All default to `None` (existing behavior). Play mode keeps `hand_force` event with random forces disabled; set `constant_force` to activate.

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
No test suite, CI, or linting configuration exists in this repository.