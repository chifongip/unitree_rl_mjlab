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

**Keyboard controls** (native viewer only, numpad):

| Key | Action | Step |
|-----|--------|------|
| KP_8 / KP_2 | lin_vel_x +/- | 0.1 m/s |
| KP_4 / KP_6 | lin_vel_y +/- | 0.1 m/s |
| KP_7 / KP_9 | ang_vel_z +/- | 0.1 rad/s |
| KP_ADD / KP_SUBTRACT | height +/- | 0.02 m |
| KP_5 | zero all velocity | — |
| KP_0 | reset height to nominal | — |

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

Policy controls 12 lower-body joints only; upper body (17 DOF) driven by ACCAD motion data via `UpperBodyMotionAction`. Two modes: pose-only (sample frame at reset, hold) and clip playback (frame-by-frame). `default_pose_ratio` curriculum gradually introduces diverse upper-body poses during training.

External force curriculum (`HandForceEvent`) applies random wrenches to end-effectors to simulate carrying objects. Force bounds computed via Jacobian transpose (`MaxForceEstimator`). Two curriculum options: step-based (`force_scale_staged`) and adaptive (`force_curriculum_adaptive`).

Base height command (`BaseHeightCommand`) controls absolute z-height with a height-dependent posture table computed via IK solver. Symmetric data augmentation doubles mini-batches by mirroring across the sagittal plane.

Play-mode testing: `fixed_upper_body_pose`, `constant_force`, `fixed_command`, `fixed_height` for controlled model comparison.

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
- `tests/test_symmetry.py` — symmetry augmentation (joint swaps, sign flips, batch doubling). Run with `python -m pytest tests/test_symmetry.py -p no:launch_testing`.
- `tests/test_max_force_estimator.py` — Jacobian-based force estimation (mock + G1 integration). Run with `python tests/test_max_force_estimator.py`.

pytest has ROS plugin conflicts — run tests directly or use `python -m pytest -p no:launch_testing`.
