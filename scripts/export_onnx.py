"""Export a trained checkpoint to ONNX format with metadata."""

import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
import tyro
import yaml

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends

_PYTHON_TAG_RE = re.compile(r"!!python/\S+")


def _load_params_overrides(params_dir: Path) -> dict | None:
  """Load safe scalar/dict overrides from training params directory."""
  clean_path = params_dir / "env_overrides.yaml"
  if clean_path.exists():
    try:
      data = yaml.safe_load(clean_path.read_text())
      if isinstance(data, dict):
        return data
      return None
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
  """Apply scalar/dict overrides from saved training params to env_cfg."""
  diffs: list[str] = []

  def _check_and_set(path: str, current, saved) -> bool:
    if current != saved:
      diffs.append(f"  {path}: {current!r} -> {saved!r}")
      return True
    return False

  for key in ("decimation", "is_finite_horizon", "scale_rewards_by_dt", "seed"):
    if key in overrides and isinstance(overrides[key], (int, float, bool)):
      current = getattr(env_cfg, key)
      if _check_and_set(key, current, overrides[key]):
        setattr(env_cfg, key, overrides[key])

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

  if "sim" in overrides and isinstance(overrides["sim"], dict):
    mujoco_data = overrides["sim"].get("mujoco")
    if isinstance(mujoco_data, dict) and "timestep" in mujoco_data:
      if isinstance(mujoco_data["timestep"], (int, float)):
        if _check_and_set("sim.mujoco.timestep",
                           env_cfg.sim.mujoco.timestep, mujoco_data["timestep"]):
          env_cfg.sim.mujoco.timestep = mujoco_data["timestep"]

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
class ExportConfig:
  checkpoint_file: str
  """Path to .pt checkpoint file."""
  output_dir: str | None = None
  """Output directory for ONNX file. Defaults to checkpoint's directory."""
  motion_file: str | None = None
  """Motion file for tracking tasks (required for tracking task export)."""
  device: str = "cpu"


def run_export(task_id: str, cfg: ExportConfig) -> None:
  configure_torch_backends()

  checkpoint_path = Path(cfg.checkpoint_file)
  if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

  output_dir = Path(cfg.output_dir) if cfg.output_dir else checkpoint_path.parent
  output_dir.mkdir(parents=True, exist_ok=True)

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  # Detect tracking task and set motion file if needed.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )
  if is_tracking_task and cfg.motion_file:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = cfg.motion_file

  # Restore training-time params so observation dimensions match the checkpoint.
  params_path = checkpoint_path.parent / "params"
  if params_path.exists():
    overrides = _load_params_overrides(params_path)
    if overrides is not None:
      _apply_env_overrides(env_cfg, overrides)
      print(f"[INFO]: Restored env config from {params_path}")

  env_cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=cfg.device)
  runner.load(str(checkpoint_path), load_cfg={"actor": True}, strict=True,
              map_location=cfg.device)

  # Export policy.onnx.
  runner.export_policy_to_onnx(str(output_dir), "policy.onnx")
  onnx_path = output_dir / "policy.onnx"
  metadata = get_base_metadata(env.unwrapped, "local")
  attach_metadata_to_onnx(str(onnx_path), metadata)
  print(f"[INFO]: Exported {onnx_path}")

  # For tracking tasks, also export motion-bundled ONNX.
  if is_tracking_task:
    from src.tasks.tracking.rl.runner import MotionTrackingOnPolicyRunner
    assert isinstance(runner, MotionTrackingOnPolicyRunner)
    motion_filename = f"{task_id}.onnx"
    runner.export_motion_policy_to_onnx(str(output_dir), motion_filename)
    motion_onnx_path = output_dir / motion_filename
    from mjlab.tasks.tracking.mdp import MotionCommand
    motion_term = cast(MotionCommand, env.unwrapped.command_manager.get_term("motion"))
    motion_metadata = get_base_metadata(env.unwrapped, "local")
    motion_metadata.update({
      "anchor_body_name": motion_term.cfg.anchor_body_name,
      "body_names": list(motion_term.cfg.body_names),
    })
    attach_metadata_to_onnx(str(motion_onnx_path), motion_metadata)
    print(f"[INFO]: Exported {motion_onnx_path}")

  env.close()


def main():
  import mjlab.tasks  # noqa: F401
  import src.tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    ExportConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )

  run_export(chosen_task, args)


if __name__ == "__main__":
  main()
