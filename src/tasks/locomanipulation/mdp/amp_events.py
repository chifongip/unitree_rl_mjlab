"""AMP motion reset events.

Provides startup and reset event callbacks that load expert motion data
from NPZ files and reset environments to random motion frames.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class MotionLoader:
  """Load motion NPZ files for env resets (world-frame data)."""

  def __init__(self, motion_dir: str, device: str = "cpu"):
    self.motion_data: list[dict] = []
    self.motion_data = self._load_dir(motion_dir, device)
    assert len(self.motion_data) > 0, f"No npz files found in: {motion_dir}"
    self.motion_names = [m["motion_name"] for m in self.motion_data]

  @staticmethod
  def _load_dir(dir_path: str, device: str) -> list[dict]:
    assert os.path.isdir(dir_path), f"Not a directory: {dir_path}"
    result = []
    for filename in sorted(os.listdir(dir_path)):
      if not filename.endswith(".npz"):
        continue
      motion_name = os.path.splitext(filename)[0]
      data = np.load(os.path.join(dir_path, filename))
      result.append({
        "motion_name": motion_name,
        "fps": data["fps"],
        "dof_pos": torch.tensor(data["joint_pos"], dtype=torch.float32, device=device),
        "dof_vel": torch.tensor(data["joint_vel"], dtype=torch.float32, device=device),
        "body_pos_w": torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device),
        "body_quat_w": torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device),
        "body_lin_vel_w": torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device),
        "body_ang_vel_w": torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device),
      })
    return result


class MotionResetManager:
  """Singleton that manages motion frame data for AMP environment resets."""

  _instance: MotionResetManager | None = None

  def __init__(self) -> None:
    self.frames: dict[str, dict[str, torch.Tensor]] = {}

  @classmethod
  def get(cls) -> MotionResetManager:
    if cls._instance is None:
      cls._instance = cls()
    return cls._instance

  def init(self, env: ManagerBasedRlEnv, motion_dir: str) -> None:
    if motion_dir in self.frames:
      return

    loader = MotionLoader(motion_dir=motion_dir, device=str(env.device))
    self.frames[motion_dir] = self._concat_frames(loader.motion_data)
    frame_count = self.frames[motion_dir]["root_pos"].shape[0]
    print(
      f"[MotionResetManager] Loaded {len(loader.motion_data)} clips, "
      f"{frame_count} frames from {motion_dir}"
    )

  def reset(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    motion_dir: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> None:
    if env_ids is None:
      env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    if len(env_ids) == 0:
      return

    frames = self.frames[motion_dir]
    total_frames = frames["root_pos"].shape[0]
    num_reset = len(env_ids)
    idx = torch.randint(0, total_frames, (num_reset,), device=env.device)

    asset: Entity = env.scene[asset_cfg.name]

    # Root pose — preserve env origin XY, take Z from motion.
    root_pos = frames["root_pos"][idx]
    root_quat = frames["root_quat"][idx]
    positions = env.scene.env_origins[env_ids].clone()
    positions[:, 2] = root_pos[:, 2]

    root_pose = torch.cat([positions, root_quat], dim=-1)
    asset.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)

    # Root velocity.
    root_vel = torch.cat(
      [frames["root_lin_vel"][idx], frames["root_ang_vel"][idx]], dim=-1
    )
    asset.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)

    # Joint state — clamp to limits.
    joint_pos = frames["joint_pos"][idx]
    joint_vel = frames["joint_vel"][idx]

    soft_joint_pos_limits = asset.data.soft_joint_pos_limits
    assert soft_joint_pos_limits is not None
    joint_pos_limits = soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
    joint_pos_clamped = joint_pos[:, asset_cfg.joint_ids].clamp_(
      joint_pos_limits[..., 0], joint_pos_limits[..., 1]
    )

    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, list):
      joint_ids = torch.tensor(joint_ids, device=env.device)

    asset.write_joint_state_to_sim(
      joint_pos_clamped,
      joint_vel[:, asset_cfg.joint_ids],
      env_ids=env_ids,
      joint_ids=joint_ids,
    )

  @staticmethod
  def _concat_frames(motions: list[dict]) -> dict[str, torch.Tensor]:
    root_pos_list, root_quat_list = [], []
    root_lin_vel_list, root_ang_vel_list = [], []
    joint_pos_list, joint_vel_list = [], []
    for motion in motions:
      root_pos_list.append(motion["body_pos_w"][:, 0, :])
      root_quat_list.append(motion["body_quat_w"][:, 0, :])
      root_lin_vel_list.append(motion["body_lin_vel_w"][:, 0, :])
      root_ang_vel_list.append(motion["body_ang_vel_w"][:, 0, :])
      joint_pos_list.append(motion["dof_pos"])
      joint_vel_list.append(motion["dof_vel"])
    return {
      "root_pos": torch.cat(root_pos_list, dim=0),
      "root_quat": torch.cat(root_quat_list, dim=0),
      "root_lin_vel": torch.cat(root_lin_vel_list, dim=0),
      "root_ang_vel": torch.cat(root_ang_vel_list, dim=0),
      "joint_pos": torch.cat(joint_pos_list, dim=0),
      "joint_vel": torch.cat(joint_vel_list, dim=0),
    }


# ------------------------------------------------------------------
# Event callback wrappers (thin delegates to singleton)
# ------------------------------------------------------------------


def init_motion_loader(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  motion_dir: str,
) -> None:
  """Startup event: load motion data from NPZ files."""
  MotionResetManager.get().init(env=env, motion_dir=motion_dir)


def reset_from_motion_data(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  motion_dir: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reset event: reset envs from random motion frames."""
  MotionResetManager.get().reset(
    env=env, env_ids=env_ids, motion_dir=motion_dir, asset_cfg=asset_cfg
  )
