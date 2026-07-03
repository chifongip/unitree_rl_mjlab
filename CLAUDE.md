# CLAUDE.md

## Project Overview

Locomanipulation RL project for G1 (29-DOF, 23-DOF) and Agibot X2 humanoids on [mjlab](https://github.com/mujocolab/mjlab) + MuJoCo. Policy controls lower body + waist; upper body driven by motion playback. Workflow: **Train → Play → Deploy**.

## Registered Tasks
- `Unitree-G1-Locomanipulation-Rough/Flat`
- `Unitree-G1-23Dof-Locomanipulation-Rough/Flat`
- `Agibot-X2-Locomanipulation-Rough/Flat`

## Commands

### Setup
```bash
conda create -n unitree_rl_mjlab python=3.11 && conda activate unitree_rl_mjlab
pip install -e .
```

### Train / Play / Eval / Export
```bash
python scripts/train.py <TaskName> --env.scene.num-envs=4096 --agent.logger=tensorboard
python scripts/play.py <TaskName> --agent zero --no-terminations True
python scripts/play.py <TaskName> --checkpoint_file=logs/.../model_<iter>.pt
python scripts/eval.py --task <TaskName> --eval-config eval_config.yaml
python scripts/export_onnx.py <TaskName> --checkpoint-file logs/.../model_20000.pt
```

**Keyboard** (numpad): 8/2=vel_x, 4/6=vel_y, 7/9=ang_z, +/-=height, 5=zero, 1/3=waist_yaw, 0=reset. F8-F11: predefined upper body poses.

### Check Motion Collisions
```bash
python scripts/check_motion_collisions.py --robot g1 --clean
python scripts/check_motion_collisions.py --robot x2 --motion-file src/assets/data/x2/amass/amass_all.pkl --clean
```

### Convert BONES-SEED Motion Data
```bash
# Default: Gestures, Communication, Baseline; dedup by description; downsample 120→30 FPS
python scripts/convert_bones_seed.py --dedup --output src/assets/data/g1/bones_seed/bones_seed.pkl

# Object Manipulation only
python scripts/convert_bones_seed.py --categories "Object Manipulation" --dedup --output src/assets/data/g1/bones_seed/bones_seed.pkl

# Multiple categories
python scripts/convert_bones_seed.py --categories "Gestures,Communication,Baseline,Object Manipulation" --dedup --output src/assets/data/g1/bones_seed/bones_seed.pkl
```

## Architecture

### Locomanipulation Task
Policy controls lower-body + waist DOFs. Upper body driven by `UpperBodyMotionAction` with `exclude_waist=True`. Config: `src/tasks/locomanipulation/config/<robot>/env_cfgs.py`. Base factory: `src/tasks/locomanipulation/locomanipulation_env_cfg.py`.

**Dict-based configs, no `@configclass`**. All manager configs are plain dicts of `TermCfg` objects. Hard requirement from mjlab.

### G1 vs X2 Comparison

| Aspect | G1 29-DOF | G1 23-DOF | Agibot X2 |
|---|---|---|---|
| Actuated joints | 29 | 23 | 29 |
| Actuator model | `UnitreeActuatorCfg` (motor model) | Same | `BuiltinPositionActuatorCfg` (generic) |
| Gain presets | default/unitree/unitree_stiff | Same | default/agibot_stiff |
| Waist order | yaw/roll/pitch | yaw only | yaw/pitch/roll |
| Wrist order | roll/pitch/yaw | roll only | yaw/pitch/roll |
| Arm joints | 14 (7/arm) | 10 (5/arm) | 14 (7/arm) |
| Symmetry class | `G1Symmetry` (29) | `G1_23DOFSymmetry` (23) | `X2Symmetry` (29) |
| Motion data | `accad/accad_all.pkl` | `accad/accad_all.pkl` | `amass/amass_all.pkl` |
| `motion_dof_indices` | (15,16,17,18,19,20,21,22,23,24,25,26,27,28) | (15,16,17,18,19,22,23,24,25,26) | (15,16,17,18,19,20,21,22,23,24,25,26,27,28) |
| Nominal height | 0.76m | 0.76m | 0.66m |
| Foot geoms | 7/foot | 7/foot | 7/foot |
| Height postures | `scripts/postures.py` | Same | `scripts/postures_x2.py` |

### Waist Regulation
`waist_regulation` penalizes waist roll/pitch deviation from default pose (weight=-1.0). Uses penalty kernel `1 - exp(-mean(sq(diff)) / std²)`. Applies to G1 29-DOF and X2; removed for G1 23-DOF (no roll/pitch joints). Standing weight 2x, walking weight 1x.

### Key Gotchas
- **Joint order matters for symmetry**: Symmetry is index-based. G1 waist = yaw/roll/pitch, X2 waist = yaw/pitch/roll. Wrong indices = wrong mirroring.
- **`preserve_order=False`**: `SceneEntityCfg.resolve()` returns joints in MJCF natural order, not pattern order. `actuator_names` pattern order is irrelevant.
- **Motion data columns**: G1 ACCAD data is in G1 joint order. X2 AMASS data is in X2 joint order. `motion_dof_indices` must match the data's column layout.
- **X2 MJCF**: `x2_ultra_no_head.xml` — head body kept (visual + collision), joints removed. No uncontrolled DOFs.

## Testing
```bash
PYTHONPATH="" python -m pytest tests/test_symmetry.py -v -p no:launch_testing
PYTHONPATH="" python -m pytest tests/test_max_force_estimator.py -v -p no:launch_testing
```
