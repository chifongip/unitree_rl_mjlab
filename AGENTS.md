# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives under `src/`. Robot definitions, meshes, and motion data are in `src/assets/`; reinforcement-learning environments are grouped by task in `src/tasks/{velocity,tracking,locomanipulation}` with robot-specific configuration below each task's `config/` directory. Shared environment and RL utilities live in `src/instinct_mj/`. Use `scripts/` for training, playback, evaluation, export, and data-conversion entry points. Native deployment controllers are under `deploy/robots/<robot>/`, while `simulate/` contains the C++ MuJoCo simulator. Tests belong in `tests/`; documentation and media belong in `doc/`.

## Build, Test, and Development Commands

- `uv sync --dev` installs the Python 3.11 package and development tools from `uv.lock`. Alternatively, use `pip install -e .` in a Python 3.11 environment.
- `python scripts/list_envs.py` lists registered task names.
- `python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096` starts a representative training run.
- `python scripts/play.py Unitree-G1-Flat --checkpoint_file=logs/.../model_<iter>.pt` visualizes a checkpoint.
- `PYTHONPATH="" python -m pytest tests -v -p no:launch_testing` runs the unit suite without ROS's `launch_testing` plugin.
- `ruff check .` runs Python lint checks.
- `cmake -S simulate -B simulate/build && cmake --build simulate/build -j8` builds the simulator. Robot controllers use the same pattern from `deploy/robots/<robot>`.

## Coding Style & Naming Conventions

Follow the existing Python style: two-space indentation, type hints, concise docstrings, `snake_case` for functions/modules, and `PascalCase` for classes. Keep manager configurations as plain dictionaries of `TermCfg` objects; do not introduce `@configclass` in task configs. Preserve robot joint ordering—symmetry and motion-data mappings are index-based. C++ targets use C++17; match the formatting of the surrounding file.

## Testing Guidelines

Tests use `pytest` and follow `tests/test_<feature>.py`; test functions start with `test_`. Add focused regression coverage for changes to rewards, observations, symmetry, force estimation, or configuration factories. No numeric coverage threshold is defined, but all affected tests should pass before review.

## Commit & Pull Request Guidelines

Recent history favors Conventional Commit subjects such as `feat(x2): ...`, `fix(g1): ...`, and `chore: ...`. Keep commits imperative, scoped, and narrowly focused. Pull requests should explain behavioral/configuration changes, identify affected robots and tasks, list validation commands, and link relevant issues. Include videos, plots, or screenshots when simulation behavior or visualization changes; do not commit generated `logs/`, checkpoints, or local build artifacts.
