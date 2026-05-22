"""Script to play RL agent with RSL-RL."""

import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro
import yaml

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


class KeyboardCommandOverride:
    """Keyboard-driven command overrides for the native viewer.

    Shared state between GLFW thread (key callback) and main physics thread
    (monkey-patched compute). Thread-safe: callback only sets Python floats
    which are GIL-atomic; compute reads them on the main thread.

    Keyboard presses set *target* values. The patched compute blends the
    current command toward the target each step using exponential decay
    (default factor 0.95, ~1s time constant at 50Hz), producing smooth
    transitions instead of instant jumps.
    """

    def __init__(self, nominal_height: float = 0.785, decay: float = 0.95):
        self.target_vel_x: float = 0.0
        self.target_vel_y: float = 0.0
        self.target_ang_vel_z: float = 0.0
        self.vel_active: bool = False

        self.target_height: float = nominal_height
        self.height_active: bool = False
        self.nominal_height: float = nominal_height

        self.decay: float = decay

        self.vel_step: float = 0.1
        self.ang_step: float = 0.1
        self.height_step: float = 0.02

        self._last_key_time: float = 0.0
        self._debounce_s: float = 0.05

    def __call__(self, key: int) -> None:
        """Key callback — runs on GLFW thread, must not touch env/sim."""
        now = time.monotonic()
        if now - self._last_key_time < self._debounce_s:
            return
        self._last_key_time = now

        from mjlab.viewer.native.keys import (
            KEY_KP_0,
            KEY_KP_2,
            KEY_KP_4,
            KEY_KP_5,
            KEY_KP_6,
            KEY_KP_7,
            KEY_KP_8,
            KEY_KP_9,
            KEY_KP_ADD,
            KEY_KP_SUBTRACT,
        )

        handled = True
        if key == KEY_KP_8:
            self.target_vel_x += self.vel_step
        elif key == KEY_KP_2:
            self.target_vel_x -= self.vel_step
        elif key == KEY_KP_6:
            self.target_vel_y -= self.vel_step
        elif key == KEY_KP_4:
            self.target_vel_y += self.vel_step
        elif key == KEY_KP_9:
            self.target_ang_vel_z -= self.ang_step
        elif key == KEY_KP_7:
            self.target_ang_vel_z += self.ang_step
        elif key == KEY_KP_ADD:
            self.target_height += self.height_step
        elif key == KEY_KP_SUBTRACT:
            self.target_height -= self.height_step
        elif key == KEY_KP_5:
            self.target_vel_x = 0.0
            self.target_vel_y = 0.0
            self.target_ang_vel_z = 0.0
        elif key == KEY_KP_0:
            self.target_height = self.nominal_height
        else:
            handled = False

        if handled:
            self.vel_active = True
            self.height_active = True
            print(
                f"\r[KB] vel=({self.target_vel_x:+.1f}, {self.target_vel_y:+.1f}, "
                f"{self.target_ang_vel_z:+.1f}) h={self.target_height:.3f}  ",
                end="",
                flush=True,
            )


def _patch_command_compute(term, override: KeyboardCommandOverride, term_type: str):
    """Monkey-patch a command term's compute() to apply keyboard overrides.

    Velocity adjustments (numpad 2/4/6/8, 7/9) are applied instantly for
    responsive control.  Zeroing all commands (KP_5) uses exponential decay
    toward zero (~1s time constant at 50Hz) to smooth the walk-to-stand
    transition.  Height commands are always instant.
    """
    original_compute = term.compute

    if term_type == "twist":
        def patched_compute(dt):
            original_compute(dt)
            if override.vel_active:
                if (override.target_vel_x == 0.0 and override.target_vel_y == 0.0
                        and override.target_ang_vel_z == 0.0):
                    d = override.decay
                    term.vel_command_b[:, 0] *= d
                    term.vel_command_b[:, 1] *= d
                    term.vel_command_b[:, 2] *= d
                    small = torch.norm(term.vel_command_b, dim=-1) < 0.1
                    if small.any():
                        term.vel_command_b[small] = 0.0
                else:
                    term.vel_command_b[:, 0] = override.target_vel_x
                    term.vel_command_b[:, 1] = override.target_vel_y
                    term.vel_command_b[:, 2] = override.target_ang_vel_z

        term.compute = patched_compute

    elif term_type == "base_height":
        def patched_compute(dt):
            original_compute(dt)
            if override.height_active:
                term._height_command[:, 0] = override.target_height

        term.compute = patched_compute


_PYTHON_TAG_RE = re.compile(r"!!python/\S+")


def _load_params_overrides(params_dir: Path) -> dict | None:
  """Load safe scalar/dict overrides from training params directory.

  Prefers env_overrides.yaml (clean, safe-loadable) if it exists.
  Falls back to env.yaml with Python tag stripping.
  Returns None if no loadable file is found.
  """
  clean_path = params_dir / "env_overrides.yaml"
  if clean_path.exists():
    try:
      data = yaml.safe_load(clean_path.read_text())
      return data if isinstance(data, dict) else None
    except Exception as e:
      print(f"[WARN] Could not load {clean_path}: {e}")

  full_path = params_dir / "env.yaml"
  if not full_path.exists():
    return None
  try:
    text = full_path.read_text()
    cleaned = _PYTHON_TAG_RE.sub("", text)
    data = yaml.safe_load(cleaned)
    return data if isinstance(data, dict) else None
  except Exception as e:
    print(f"[WARN] Could not load params from {full_path}: {e}")
    return None


def _apply_env_overrides(env_cfg, overrides: dict) -> None:
  """Apply scalar/dict overrides from saved training params to env_cfg.

  Only overrides fields that actually differ from the current config.
  Prints each differing field as: field: current_value -> saved_value.
  """
  diffs: list[str] = []

  def _check_and_set(path: str, current, saved) -> bool:
    """Return True and append to diffs if values differ."""
    if current != saved:
      diffs.append(f"  {path}: {current!r} -> {saved!r}")
      return True
    return False

  # Top-level scalar fields.
  for key in ("decimation", "is_finite_horizon", "scale_rewards_by_dt", "seed"):
    if key in overrides and isinstance(overrides[key], (int, float, bool)):
      current = getattr(env_cfg, key)
      if _check_and_set(key, current, overrides[key]):
        setattr(env_cfg, key, overrides[key])

  # Observation group history_length and per-term overrides.
  # Track (id(term_obj), param_key) to avoid re-applying overrides to shared term
  # objects.  In the config factory, critic_terms = {**actor_terms, ...} creates
  # shallow copies — the term objects are the same Python objects in both groups.
  applied_params: set[tuple[int, str]] = set()
  if "observations" in overrides and isinstance(overrides["observations"], dict):
    for group_name, group_data in overrides["observations"].items():
      if not isinstance(group_data, dict) or group_name not in env_cfg.observations:
        continue
      group_cfg = env_cfg.observations[group_name]

      if "history_length" in group_data and isinstance(
          group_data["history_length"], (int, type(None))):
        path = f"observations.{group_name}.history_length"
        if _check_and_set(path, group_cfg.history_length, group_data["history_length"]):
          group_cfg.history_length = group_data["history_length"]

      if "terms" in group_data and isinstance(group_data["terms"], dict):
        for term_name, term_data in group_data["terms"].items():
          if (not isinstance(term_data, dict)
              or term_name not in group_cfg.terms):
            continue
          if "history_length" in term_data and isinstance(
              term_data["history_length"], int):
            path = f"observations.{group_name}.terms.{term_name}.history_length"
            current = group_cfg.terms[term_name].history_length
            if _check_and_set(path, current, term_data["history_length"]):
              group_cfg.terms[term_name].history_length = term_data["history_length"]

          if "params" in term_data and isinstance(term_data["params"], dict):
            for param_key, param_value in term_data["params"].items():
              if param_key not in group_cfg.terms[term_name].params:
                continue
              # Only restore scalar params. Structured objects like
              # SceneEntityCfg lose their type when round-tripped through
              # YAML and would break downstream code expecting .name etc.
              if not isinstance(param_value, (int, float, bool, str, type(None))):
                continue
              term_obj = group_cfg.terms[term_name]
              if (id(term_obj), param_key) in applied_params:
                continue
              current = group_cfg.terms[term_name].params[param_key]
              path = (
                f"observations.{group_name}.terms.{term_name}.params.{param_key}"
              )
              if _check_and_set(path, current, param_value):
                group_cfg.terms[term_name].params[param_key] = param_value
              applied_params.add((id(term_obj), param_key))

  # Simulation timestep.
  if "sim" in overrides and isinstance(overrides["sim"], dict):
    mujoco_data = overrides["sim"].get("mujoco")
    if isinstance(mujoco_data, dict) and "timestep" in mujoco_data:
      if isinstance(mujoco_data["timestep"], (int, float)):
        if _check_and_set("sim.mujoco.timestep",
                           env_cfg.sim.mujoco.timestep, mujoco_data["timestep"]):
          env_cfg.sim.mujoco.timestep = mujoco_data["timestep"]

  # Action scale dicts.
  if "actions" in overrides and isinstance(overrides["actions"], dict):
    for action_name, action_data in overrides["actions"].items():
      if not isinstance(action_data, dict) or action_name not in env_cfg.actions:
        continue
      if "scale" in action_data and isinstance(action_data["scale"], dict):
        path = f"actions.{action_name}.scale"
        if _check_and_set(path, env_cfg.actions[action_name].scale, action_data["scale"]):
          env_cfg.actions[action_name].scale = action_data["scale"]

  if diffs:
    print("[INFO]: Params differ from saved training config:")
    print("\n".join(diffs))
  else:
    print("[INFO]: All saved params match current config")


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""
  params_dir: str | None = None
  """Path to params directory containing env.yaml. Auto-detected from checkpoint
  location when not specified. Use for wandb checkpoints where params/ is not
  co-located."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  # Auto-restore env config from training params.
  if TRAINED_MODE and log_dir is not None:
    if cfg.params_dir is not None:
      params_path = Path(cfg.params_dir)
    else:
      params_path = log_dir / "params"
    overrides = _load_params_overrides(params_path)
    if overrides is not None:
      _apply_env_overrides(env_cfg, overrides)
      print(f"[INFO]: Restored env config from {params_path}")

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  # Set up keyboard command override for native viewer.
  override = None
  if resolved_viewer == "native":
    nominal_height = 0.785
    cmd_mgr = env.unwrapped.command_manager
    if hasattr(cmd_mgr, "_terms") and "base_height" in cmd_mgr._terms:
      bh_cfg = cmd_mgr._terms["base_height"].cfg
      if hasattr(bh_cfg, "nominal_height"):
        nominal_height = bh_cfg.nominal_height

    override = KeyboardCommandOverride(nominal_height=nominal_height)

    for term_name in ("twist", "base_height"):
      if term_name in cmd_mgr._terms:
        _patch_command_compute(cmd_mgr._terms[term_name], override, term_name)
      else:
        print(f"[INFO] Keyboard override: '{term_name}' term not found, skipping")

    print(
      "[INFO] Keyboard overrides: numpad 8/2=vel_x, 4/6=vel_y, "
      "7/9=yaw, +/-=height, 5=zero vel, 0=reset height"
    )

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy, key_callback=override).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
