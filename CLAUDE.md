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
python scripts/play.py Unitree-G1-Flat --checkpoint_file=logs/rsl_rl/g1_velocity/<date>/model_<iter>.pt
```

### Convert motion CSV to NPZ
```bash
python scripts/csv_to_npz.py --input-file src/assets/motions/g1/dance1.csv --output-name dance1.npz --input-fps 30 --output-fps 50 --robot g1
```

### Deploy (C++ on real robot or unitree_mujoco simulator)
```bash
# Build unitree_mujoco simulator
cd simulate && mkdir build && cd build && cmake .. && make -j8

# Build robot deploy binary
cd deploy/robots/g1 && mkdir build && cd build && cmake .. && make

# Run in simulation
./g1_ctrl --network=lo
# Run on real robot
./g1_ctrl --network=enp5s0
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
- `rewards.py`, `observations.py`, `terminations.py`, `curriculums.py`, `velocity_command.py`, `upper_body_action.py` (locomanipulation)

### Robot Assets
`src/assets/robots/<robot>/` — each exports a `get_<robot>_robot_cfg()` function and a constants module with joint names, body names, default poses.

### Custom Runners
`src/tasks/<type>/rl/runner.py` — `VelocityOnPolicyRunner` / `TrackingOnPolicyRunner` / `LocomanipulationOnPolicyRunner` extend `MjlabOnPolicyRunner` to auto-export `policy.onnx` on save for deployment.

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

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.