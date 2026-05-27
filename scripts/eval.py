"""Evaluate trained locomanipulation policies — velocity tracking error per force level.

Computes MAE per (model, force_level), averaged across all (pose, velocity) combos.
Plots force vs error with one curve per model for comparison.

Usage:
    # Single checkpoint
    python scripts/eval.py Unitree-G1-Locomanipulation-Flat \\
        --checkpoint-file logs/.../model_20000.pt

    # Multi-model comparison via config file
    python scripts/eval.py Unitree-G1-Locomanipulation-Flat \\
        --eval-config eval_config.yaml

    # Visual verification with viewer
    python scripts/eval.py Unitree-G1-Locomanipulation-Flat \\
        --eval-config eval_config.yaml --viewer native
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Literal

import torch
import tyro
import yaml
from tqdm import tqdm

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends


# ---------------------------------------------------------------------------
# Upper-body pose presets (17 joints, values in radians).
# ---------------------------------------------------------------------------

POSE_PRESETS: dict[str, dict[str, float]] = {
    "neutral": {},
    "zero": {
        "waist_yaw_joint": 0.0,
        "waist_roll_joint": 0.0,
        "waist_pitch_joint": 0.0,
        "left_shoulder_pitch_joint": 0.0,
        "left_shoulder_roll_joint": 0.0,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 0.0,
        "left_wrist_roll_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
        "right_shoulder_pitch_joint": 0.0,
        "right_shoulder_roll_joint": 0.0,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": 0.0,
        "right_wrist_roll_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
    }
}

# ---------------------------------------------------------------------------
# Force condition presets.
# ---------------------------------------------------------------------------

_FORCE_PRESETS: dict[str, dict[str, float] | None] = {
    "none": None,
    "medium": {"x": 0.0, "y": 0.0, "z": -15.0},
    "large": {"x": 0.0, "y": 0.0, "z": -30.0},
}


# ---------------------------------------------------------------------------
# Param override helpers (duplicated from play.py).
# ---------------------------------------------------------------------------

_PYTHON_TAG_RE = re.compile(r"!!python/\S+")


def _load_params_overrides(params_dir: Path) -> dict | None:
    """Load safe scalar/dict overrides from training params directory."""
    clean_path = params_dir / "env_overrides.yaml"
    if clean_path.exists():
        try:
            data = yaml.safe_load(clean_path.read_text())
            if isinstance(data, dict):
                _supplement_obs_terms(data, params_dir)
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


def _supplement_obs_terms(data: dict, params_dir: Path) -> None:
    """Fill missing observation term params from env.yaml."""
    obs_data = data.get("observations")
    if not isinstance(obs_data, dict):
        return
    for group_data in obs_data.values():
        if isinstance(group_data, dict) and "terms" in group_data:
            return

    full_path = params_dir / "env.yaml"
    if not full_path.exists():
        return
    try:
        text = full_path.read_text()
        cleaned = _PYTHON_TAG_RE.sub("", text)
        full_data = yaml.safe_load(cleaned)
    except Exception:
        return

    if not isinstance(full_data, dict):
        return
    full_obs = full_data.get("observations")
    if not isinstance(full_obs, dict):
        return
    for group_name, group_data in full_obs.items():
        if not isinstance(group_data, dict):
            continue
        if "terms" in group_data and group_name in obs_data:
            obs_data[group_name]["terms"] = group_data["terms"]


def _apply_env_overrides(env_cfg, overrides: dict) -> None:
    """Apply scalar/dict overrides from saved training params."""
    applied_params: set[tuple[int, str]] = set()

    for key in ("decimation", "is_finite_horizon", "scale_rewards_by_dt", "seed"):
        if key in overrides and isinstance(overrides[key], (int, float, bool)):
            setattr(env_cfg, key, overrides[key])

    if "observations" in overrides and isinstance(overrides["observations"], dict):
        for group_name, group_data in overrides["observations"].items():
            if not isinstance(group_data, dict) or group_name not in env_cfg.observations:
                continue
            group_cfg = env_cfg.observations[group_name]
            if "history_length" in group_data and isinstance(
                group_data["history_length"], (int, type(None))
            ):
                group_cfg.history_length = group_data["history_length"]
            if "terms" in group_data and isinstance(group_data["terms"], dict):
                for term_name, term_data in group_data["terms"].items():
                    if not isinstance(term_data, dict) or term_name not in group_cfg.terms:
                        continue
                    if "history_length" in term_data and isinstance(
                        term_data["history_length"], int
                    ):
                        group_cfg.terms[term_name].history_length = term_data["history_length"]
                    if "params" in term_data and isinstance(term_data["params"], dict):
                        for param_key, param_value in term_data["params"].items():
                            if param_key not in group_cfg.terms[term_name].params:
                                continue
                            if not isinstance(
                                param_value, (int, float, bool, str, type(None))
                            ):
                                continue
                            term_obj = group_cfg.terms[term_name]
                            if (id(term_obj), param_key) in applied_params:
                                continue
                            group_cfg.terms[term_name].params[param_key] = param_value
                            applied_params.add((id(term_obj), param_key))

    if "sim" in overrides and isinstance(overrides["sim"], dict):
        mujoco_data = overrides["sim"].get("mujoco")
        if isinstance(mujoco_data, dict) and "timestep" in mujoco_data:
            if isinstance(mujoco_data["timestep"], (int, float)):
                env_cfg.sim.mujoco.timestep = mujoco_data["timestep"]

    if "actions" in overrides and isinstance(overrides["actions"], dict):
        for action_name, action_data in overrides["actions"].items():
            if not isinstance(action_data, dict) or action_name not in env_cfg.actions:
                continue
            if "scale" in action_data and isinstance(action_data["scale"], dict):
                env_cfg.actions[action_name].scale = action_data["scale"]


# ---------------------------------------------------------------------------
# Combo dataclass.
# ---------------------------------------------------------------------------


@dataclass
class EvalCombo:
    vel_x: float
    vel_y: float
    ang_z: float
    force_name: str
    force_dict: dict[str, float] | None
    pose_name: str
    pose_dict: dict[str, float]


@dataclass
class EpisodeData:
    """Per-step recorded state for one evaluation batch."""
    vel_b: torch.Tensor       # (N, T, 3)
    ang_vel_b: torch.Tensor   # (N, T, 3)
    ep_len: torch.Tensor      # (N, T)
    num_steps: int
    combo: EvalCombo


# ---------------------------------------------------------------------------
# Metric computation.
# ---------------------------------------------------------------------------


def compute_velocity_metrics(ep: EpisodeData) -> dict[str, float]:
    """Compute MAE for linear, angular, and combined velocity tracking error."""
    cmd = torch.tensor(
        [ep.combo.vel_x, ep.combo.vel_y, ep.combo.ang_z],
        device=ep.vel_b.device,
    )
    actual = torch.cat([ep.vel_b[:, :, :2], ep.ang_vel_b[:, :, 2:3]], dim=-1)  # (N, T, 3)
    error = actual - cmd  # (N, T, 3)

    # Mask out post-reset steps (ep_len == 0 means just reset).
    valid = ep.ep_len > 0  # (N, T)
    if not valid.any():
        return {"mae_vel_xy": float("nan"), "mae_ang_z": float("nan"), "mae_combined": float("nan")}

    error_valid = error[valid]  # (M, 3)
    mae_vel_xy = torch.norm(error_valid[:, :2], dim=-1).mean().item()
    mae_ang_z = error_valid[:, 2].abs().mean().item()
    mae_combined = torch.norm(error_valid, dim=-1).mean().item()

    return {
        "mae_vel_xy": mae_vel_xy,
        "mae_ang_z": mae_ang_z,
        "mae_combined": mae_combined,
    }


# ---------------------------------------------------------------------------
# Env configuration and episode running.
# ---------------------------------------------------------------------------


def _configure_env_base(
    env_cfg,
    num_envs: int,
    fixed_height: float,
    episode_steps: int,
) -> None:
    """Apply base env settings. Force/pose/velocity are set at runtime."""
    env_cfg.scene.num_envs = num_envs

    # Height command.
    env_cfg.commands["base_height"].fixed_height = fixed_height

    # Default pose placeholder (ensures _fixed_pose tensor is always created).
    env_cfg.actions["upper_body_motion"].fixed_upper_body_pose = {
        joint: 0.0 for joint in (
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint", "left_elbow_joint",
            "left_wrist_roll_joint", "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint", "right_elbow_joint",
            "right_wrist_roll_joint", "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        )
    }

    # Default force: no force (will be overridden at runtime).
    force_params = env_cfg.events["hand_force"].params
    force_params["constant_force"] = None
    force_params["no_force_ratio"] = 1.0
    force_params["force_range_max"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}

    # Disable mid-episode command resampling (velocity is set via reset only).
    long_time = float(episode_steps + 100)
    env_cfg.commands["twist"].resampling_time_range = (long_time, long_time)


def _configure_force_and_pose(env, force_name: str, pose_name: str) -> None:
    """Mutate force and pose on a live env."""
    # Force: modify event params.
    force_cfg = env.unwrapped.event_manager.get_term_cfg("hand_force")
    force_dict = _FORCE_PRESETS[force_name]
    if force_dict is not None:
        force_cfg.params["constant_force"] = force_dict
        force_cfg.params["no_force_ratio"] = 0.0
    else:
        force_cfg.params["constant_force"] = None
        force_cfg.params["no_force_ratio"] = 1.0

    # Pose: modify action term's internal tensor.
    pose_dict = POSE_PRESETS[pose_name]
    action_term = env.unwrapped.action_manager.get_term("upper_body_motion")
    fixed = action_term._default_joint_pos[0].clone()
    all_names = action_term._entity.joint_names
    for name, value in pose_dict.items():
        matches = [i for i, jid in enumerate(action_term._joint_ids) if all_names[jid] == name]
        if matches:
            fixed[matches[0]] = value
    if action_term._waist_zero_cols:
        fixed[action_term._waist_zero_cols] = 0.0
    if action_term._fixed_pose is not None:
        action_term._fixed_pose[:] = fixed
    else:
        action_term._fixed_pose = fixed



def run_episode_batch_multi_vel(
    env,
    policy,
    vel_combos: list[tuple[float, float, float]],
    num_steps: int,
    device: str,
) -> list[EpisodeData]:
    """Run one episode batch with all velocity combos in parallel.

    Partitions envs into equal groups (one per combo), sets per-env
    velocity commands on vel_command_b directly, runs one shared
    episode, then splits results by group.
    """
    num_envs = env.unwrapped.num_envs
    n_combos = len(vel_combos)
    envs_per_combo = num_envs // n_combos

    group_boundaries = [i * envs_per_combo for i in range(n_combos + 1)]
    group_env_ids = [
        torch.arange(group_boundaries[i], group_boundaries[i + 1], device=device)
        for i in range(n_combos)
    ]

    desired_commands = torch.zeros(num_envs, 3, device=device)
    for i, (vx, vy, wz) in enumerate(vel_combos):
        ids = group_env_ids[i]
        desired_commands[ids, 0] = vx
        desired_commands[ids, 1] = vy
        desired_commands[ids, 2] = wz

    term = env.unwrapped.command_manager.get_term("twist")
    term.cfg.fixed_command = None

    original_resample_cmd = term._resample_command

    def _multi_vel_resample(env_ids: torch.Tensor) -> None:
        term.vel_command_b[env_ids] = desired_commands[env_ids]
        term.is_standing_env[env_ids] = False
        if hasattr(term, "is_heading_env"):
            term.is_heading_env[env_ids] = False

    term._resample_command = _multi_vel_resample

    try:
        obs, _ = env.reset()

        vel_b = torch.zeros(num_envs, num_steps, 3, device=device)
        ang_vel_b = torch.zeros(num_envs, num_steps, 3, device=device)
        ep_len = torch.zeros(num_envs, num_steps, dtype=torch.long, device=device)

        for t in range(num_steps):
            with torch.no_grad():
                action = policy(obs)
            obs, _, _, _ = env.step(action)

            robot = env.unwrapped.scene["robot"]
            vel_b[:, t] = robot.data.root_link_lin_vel_b
            ang_vel_b[:, t] = robot.data.root_link_ang_vel_b
            ep_len[:, t] = env.unwrapped.episode_length_buf
    finally:
        term._resample_command = original_resample_cmd

    results: list[EpisodeData] = []
    for i, (vx, vy, wz) in enumerate(vel_combos):
        ids = group_env_ids[i]
        combo = EvalCombo(
            vel_x=vx, vel_y=vy, ang_z=wz,
            force_name="", force_dict=None,
            pose_name="", pose_dict={},
        )
        results.append(EpisodeData(
            vel_b=vel_b[ids],
            ang_vel_b=ang_vel_b[ids],
            ep_len=ep_len[ids],
            num_steps=num_steps,
            combo=combo,
        ))

    return results



def _run_viewer_combo_multi_vel(
    env,
    policy,
    vel_combos: list[tuple[float, float, float]],
    force_name: str,
    pose_name: str,
    num_steps: int,
    device: str,
) -> None:
    """Run all velocity combos in parallel with the native viewer."""
    from mjlab.viewer import NativeMujocoViewer

    num_envs = env.unwrapped.num_envs
    n_combos = len(vel_combos)
    envs_per_combo = num_envs // n_combos

    group_boundaries = [i * envs_per_combo for i in range(n_combos + 1)]
    group_env_ids = [
        torch.arange(group_boundaries[i], group_boundaries[i + 1], device=device)
        for i in range(n_combos)
    ]

    desired_commands = torch.zeros(num_envs, 3, device=device)
    for i, (vx, vy, wz) in enumerate(vel_combos):
        ids = group_env_ids[i]
        desired_commands[ids, 0] = vx
        desired_commands[ids, 1] = vy
        desired_commands[ids, 2] = wz

    _configure_force_and_pose(env, force_name, pose_name)

    term = env.unwrapped.command_manager.get_term("twist")
    term.cfg.fixed_command = None

    original_resample_cmd = term._resample_command

    def _multi_vel_resample(env_ids: torch.Tensor) -> None:
        term.vel_command_b[env_ids] = desired_commands[env_ids]
        term.is_standing_env[env_ids] = False
        if hasattr(term, "is_heading_env"):
            term.is_heading_env[env_ids] = False

    term._resample_command = _multi_vel_resample

    try:
        env.reset()

        print(f"  {n_combos} velocity combos across {num_envs} envs "
              f"({envs_per_combo} envs/combo). Use , and . to switch envs.")
        for i, (vx, vy, wz) in enumerate(vel_combos):
            lo, hi = group_boundaries[i], group_boundaries[i + 1]
            print(f"    envs {lo:3d}-{hi - 1:3d}: vel=({vx:.1f},{vy:.1f},{wz:.1f})")

        NativeMujocoViewer(env, policy).run(num_steps=num_steps)
    finally:
        term._resample_command = original_resample_cmd


def _build_env(
    task_id: str,
    cfg: EvalConfig,
    device: str,
    agent_cfg,
    resume_path: Path | None,
    log_dir: Path | None,
    is_trained: bool,
    num_envs_override: int | None = None,
):
    """Create env + load policy once. Force/pose/velocity set at runtime."""
    env_cfg = load_env_cfg(task_id, play=True)

    if is_trained and log_dir is not None:
        params_path = Path(cfg.params_dir) if cfg.params_dir else log_dir / "params"
        overrides = _load_params_overrides(params_path)
        if overrides is not None:
            _apply_env_overrides(env_cfg, overrides)

    _configure_env_base(env_cfg, num_envs_override or cfg.num_envs, cfg.fixed_height, cfg.episode_steps)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    clip_val = agent_cfg.clip_actions if agent_cfg is not None else 0.0
    env = RslRlVecEnvWrapper(env, clip_actions=clip_val)

    # Load policy.
    if is_trained and agent_cfg is not None and resume_path is not None:
        runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=device)
        runner.load(str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device)
        policy = runner.get_inference_policy(device=device)
    else:
        action_shape = env.unwrapped.action_space.shape
        if cfg.agent == "zero":
            policy = lambda obs: torch.zeros(action_shape, device=device)  # noqa: E731
        else:
            policy = lambda obs: 2 * torch.rand(action_shape, device=device) - 1  # noqa: E731

    return env, policy


# ---------------------------------------------------------------------------
# Main evaluation loop.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation configuration."""
    eval_config: str | None = None
    """Path to YAML file defining models to compare. Format:
    models:
      - name: "20k"
        checkpoint: "logs/.../model_20000.pt"
      - name: "30k"
        checkpoint: "logs/.../model_30000.pt"
    """
    checkpoint_file: str | None = None
    """Single checkpoint (ignored if --eval-config is set)."""
    params_dir: str | None = None
    device: str | None = None
    num_envs: int = 512
    episode_steps: int = 1000
    vel_x: tuple[float, ...] = (-0.5, 0.0, 0.5)
    vel_y: tuple[float, ...] = (0.0,)
    ang_z: tuple[float, ...] = (-0.5, 0.0, 0.5)
    force_conditions: tuple[str, ...] = ("none", "medium", "large")
    body_poses: tuple[str, ...] = ("zero",)
    fixed_height: float = 0.785
    output_dir: str = "eval_results"
    metric: Literal["combined", "linear", "angular"] = "combined"
    agent: Literal["zero", "random", "trained"] = "trained"
    viewer: Literal["none", "native"] = "none"



def run_eval(task_id: str, cfg: EvalConfig):
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    is_trained = cfg.agent == "trained"
    agent_cfg = load_rl_cfg(task_id) if is_trained else None

    # Resolve checkpoints from config file or single checkpoint.
    checkpoints: list[tuple[str, Path | None, Path | None]] = []
    if is_trained:
        if cfg.eval_config is not None:
            with open(cfg.eval_config) as f:
                eval_cfg_data = yaml.safe_load(f)
            for entry in eval_cfg_data.get("models", []):
                name = entry.get("name", "")
                ckpt_path = Path(entry["checkpoint"])
                if not ckpt_path.exists():
                    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
                if not name:
                    name = ckpt_path.stem
                checkpoints.append((name, ckpt_path, ckpt_path.parent))
        elif cfg.checkpoint_file is not None:
            p = Path(cfg.checkpoint_file)
            if not p.exists():
                raise FileNotFoundError(f"Checkpoint not found: {p}")
            checkpoints.append((p.stem, p, p.parent))
        else:
            raise ValueError("Provide --eval-config or --checkpoint-file for trained agent")
    else:
        checkpoints.append((cfg.agent, None, None))

    vel_combos = list(product(cfg.vel_x, cfg.vel_y, cfg.ang_z))

    # Auto-adjust num_envs for even partitioning across velocity combos.
    n_combos = len(vel_combos)
    if n_combos > 1 and cfg.num_envs % n_combos != 0:
        effective_num_envs = (cfg.num_envs // n_combos) * n_combos
        effective_num_envs = max(effective_num_envs, n_combos)
        print(
            f"[INFO] Adjusted num_envs {cfg.num_envs} -> {effective_num_envs} "
            f"({effective_num_envs // n_combos} envs/combo, {n_combos} combos)"
        )
    else:
        effective_num_envs = cfg.num_envs

    # --- Viewer mode: visual verification, no metrics ---
    if cfg.viewer == "native":
        for model_name, resume_path, log_dir in checkpoints:
            env, policy = _build_env(
                task_id, cfg, device,
                agent_cfg, resume_path, log_dir, is_trained,
                num_envs_override=effective_num_envs,
            )
            try:
                n_total = len(cfg.force_conditions) * len(cfg.body_poses)
                combo_idx = 0
                for force_name in cfg.force_conditions:
                    if force_name not in _FORCE_PRESETS:
                        raise ValueError(f"Unknown force condition: {force_name}")
                    for pose_name in cfg.body_poses:
                        if pose_name not in POSE_PRESETS:
                            raise ValueError(f"Unknown body pose: {pose_name}")
                        combo_idx += 1
                        print(
                            f"[{combo_idx}/{n_total}] {model_name} | "
                            f"force={force_name} pose={pose_name}"
                        )
                        _run_viewer_combo_multi_vel(
                            env, policy, vel_combos,
                            force_name, pose_name, cfg.episode_steps, device,
                        )
            finally:
                env.close()
        return

    # --- Headless mode: metric collection ---
    n_vel = len(vel_combos)
    n_poses = len(cfg.body_poses)
    n_forces = len(cfg.force_conditions)
    total_combos = len(checkpoints) * n_forces * n_poses * n_vel

    all_results: list[dict] = []

    pbar = tqdm(total=total_combos, desc="Evaluating", unit="combo")

    for model_name, resume_path, log_dir in checkpoints:
        # Build env + policy ONCE per model.
        env, policy = _build_env(
            task_id, cfg, device,
            agent_cfg, resume_path, log_dir, is_trained,
            num_envs_override=effective_num_envs,
        )

        try:
            for force_name in cfg.force_conditions:
                if force_name not in _FORCE_PRESETS:
                    raise ValueError(f"Unknown force condition: {force_name}")

                combo_metrics: list[dict[str, float]] = []

                for pose_name in cfg.body_poses:
                    if pose_name not in POSE_PRESETS:
                        raise ValueError(f"Unknown body pose: {pose_name}")

                    _configure_force_and_pose(env, force_name, pose_name)

                    ep_datas = run_episode_batch_multi_vel(
                        env, policy, vel_combos, cfg.episode_steps, device,
                    )
                    for (vx, vy, wz), ep_data in zip(vel_combos, ep_datas):
                        combo = EvalCombo(
                            vel_x=vx, vel_y=vy, ang_z=wz,
                            force_name=force_name,
                            force_dict=_FORCE_PRESETS[force_name],
                            pose_name=pose_name,
                            pose_dict=POSE_PRESETS[pose_name],
                        )
                        ep_data.combo = combo
                        metrics = compute_velocity_metrics(ep_data)

                        valid = ep_data.ep_len > 0
                        mean_ep_len = ep_data.ep_len.float().mean().item() if valid.any() else 0.0
                        has_reset = ep_data.ep_len[:, 1:] < ep_data.ep_len[:, :-1]
                        survival = (~has_reset.any(dim=1)).float().mean().item()

                        combo_metrics.append({
                            **metrics,
                            "mean_ep_length": mean_ep_len,
                            "survival_rate": survival,
                        })
                        pbar.update(1)

                # Aggregate across all combos for this (model, force).
                agg: dict[str, float] = {}
                for key in ("mae_vel_xy", "mae_ang_z", "mae_combined", "mean_ep_length", "survival_rate"):
                    vals = [m[key] for m in combo_metrics if m[key] == m[key]]
                    agg[key] = sum(vals) / len(vals) if vals else float("nan")

                metric_key = f"mae_{cfg.metric}" if cfg.metric != "combined" else "mae_combined"
                pbar.set_postfix_str(
                    f"{model_name}/{force_name}: {metric_key}={agg[metric_key]:.4f}"
                )

                all_results.append({
                    "model": model_name,
                    "force": force_name,
                    **agg,
                })
        finally:
            env.close()

    pbar.close()
    _save_results(all_results, cfg)


# ---------------------------------------------------------------------------
# Output.
# ---------------------------------------------------------------------------

_ALL_RESULT_KEYS = [
    "model", "force",
    "mae_vel_xy", "mae_ang_z", "mae_combined",
    "mean_ep_length", "survival_rate",
]


def _save_results(results: list[dict], cfg: EvalConfig):
    """Write CSV and plots."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.output_dir) / f"{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV.
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_ALL_RESULT_KEYS)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[INFO] CSV saved to {csv_path}")

    # JSON.
    json_path = out_dir / "config.json"
    with open(json_path, "w") as f:
        json.dump({"config": asdict(cfg), "results": results}, f, indent=2, default=str)
    print(f"[INFO] JSON saved to {json_path}")

    _plot_results(results, cfg, out_dir)


def _plot_results(results: list[dict], cfg: EvalConfig, out_dir: Path):
    """Generate comparison plot: force vs error, one curve per model."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed, skipping plots")
        return

    _, ax = plt.subplots(figsize=(8, 5))

    # Determine metric key and label.
    metric_map = {
        "combined": ("mae_combined", "MAE Combined (m/s + rad/s)"),
        "linear": ("mae_vel_xy", "MAE Linear Velocity (m/s)"),
        "angular": ("mae_ang_z", "MAE Angular Velocity (rad/s)"),
    }
    metric_key, metric_label = metric_map[cfg.metric]

    # Collect force levels in order.
    force_levels = list(cfg.force_conditions)

    # Group by model.
    models: dict[str, list[dict]] = {}
    for r in results:
        models.setdefault(r["model"], []).append(r)

    for model_name, rows in models.items():
        # Order by force_levels.
        row_by_force = {r["force"]: r for r in rows}
        x_labels = [f for f in force_levels if f in row_by_force]
        y_vals = [row_by_force[f][metric_key] for f in x_labels]
        ax.plot(x_labels, y_vals, "o-", label=model_name, linewidth=2, markersize=8)

    ax.set_xlabel("Force Level")
    ax.set_ylabel(metric_label)
    ax.set_title(f"Velocity Tracking Error vs Force Level\n({cfg.metric})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = out_dir / "plots.png"
    plt.savefig(fig_path, dpi=150)
    print(f"[INFO] Plots saved to {fig_path}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI entry point.
# ---------------------------------------------------------------------------


def main():
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )

    args = tyro.cli(
        EvalConfig,
        args=remaining_args,
        default=EvalConfig(),
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )

    run_eval(chosen_task, args)


if __name__ == "__main__":
    main()
