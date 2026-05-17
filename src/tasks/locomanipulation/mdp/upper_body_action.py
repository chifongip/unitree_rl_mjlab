"""Upper-body motion playback action term.

Loads whole-body motion data from a pkl file, extracts upper-body DOF,
and plays back random clips frame-by-frame during each episode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import joblib
import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

UPPER_BODY_JOINT_NAMES = (
  r"waist_.*_joint",
  r".*_shoulder_.*_joint",
  r".*_elbow_joint",
  r".*_wrist_.*_joint",
)


class UpperBodyMotionAction(ActionTerm):
  """Play back upper-body motion clips from a dataset.

  At each reset, a random motion clip and start frame are selected.
  During the episode, the frame index is computed from elapsed time
  (step count × step_dt) and the clip's fps, with wrapping.
  """

  cfg: UpperBodyMotionActionCfg

  def __init__(self, cfg: UpperBodyMotionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    # Resolve upper-body joint indices on the robot.
    asset_cfg = SceneEntityCfg(self.cfg.entity_name, joint_names=self.cfg.joint_names)
    asset_cfg.resolve(self._env.scene)
    self._joint_ids = asset_cfg.joint_ids

    # Load motion data and extract upper-body DOF (indices 12-28 = 17 joints).
    data = joblib.load(cfg.motion_file)
    raw_clips: list[torch.Tensor] = []
    self._fps: int = 0
    for v in data.values():
      dof = torch.tensor(v["dof"], dtype=torch.float32)
      raw_clips.append(dof[:, 12:29])
      self._fps = v["fps"]

    self._num_clips = len(raw_clips)
    self._step_dt = env.step_dt

    # Pad all clips into a single tensor (num_clips, max_len, 17).
    max_len = max(len(c) for c in raw_clips)
    self._clip_lengths = torch.tensor(
      [len(c) for c in raw_clips], dtype=torch.long, device=self.device
    )
    self._clips = torch.zeros(self._num_clips, max_len, 17, device=self.device)
    for i, clip in enumerate(raw_clips):
      self._clips[i, : len(clip)] = clip.to(self.device)

    # Default joint positions for envs that don't play back motion.
    self._default_joint_pos = self._entity.data.default_joint_pos[:, self._joint_ids]

    # Per-env state: which clip and start frame. -1 means use default pose.
    self._clip_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._start_frame = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

  @property
  def action_dim(self) -> int:
    return 0

  @property
  def raw_action(self) -> torch.Tensor:
    return torch.empty(0, device=self.device)

  def process_actions(self, actions: torch.Tensor) -> None:
    del actions

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)

    if isinstance(env_ids, slice):
      count = (
        self.num_envs
        if env_ids == slice(None)
        else len(range(*env_ids.indices(self.num_envs)))
      )
    else:
      count = len(env_ids)

    # Sample random clips and start frames (vectorized).
    rand_clips = torch.randint(0, self._num_clips, (count,), device=self.device)
    clip_lens = self._clip_lengths[rand_clips]
    rand_starts = (torch.rand(count, device=self.device) * clip_lens.float()).long()

    # For a fraction of envs, use default pose instead of motion playback.
    use_default = torch.rand(count, device=self.device) < self.cfg.default_pose_ratio
    rand_clips[use_default] = -1  # sentinel for default pose

    if isinstance(env_ids, slice):
      self._clip_idx[:] = rand_clips
      self._start_frame[:] = rand_starts
    else:
      self._clip_idx[env_ids] = rand_clips
      self._start_frame[env_ids] = rand_starts

    # Write first frame to sim state and set actuator targets.
    first_frame = self._gather_targets(env_ids)
    self._entity.write_joint_state_to_sim(
      first_frame,
      torch.zeros_like(first_frame),
      env_ids=env_ids,
      joint_ids=self._joint_ids,
    )
    # Direct indexing to avoid broadcast issues with tensor env_ids + joint_ids.
    if isinstance(env_ids, slice):
      self._entity._data.joint_pos_target[:, self._joint_ids] = first_frame
    else:
      self._entity._data.joint_pos_target[env_ids.unsqueeze(1), self._joint_ids] = (
        first_frame
      )

  def _gather_targets(
    self, env_ids: torch.Tensor | slice | None = None
  ) -> torch.Tensor:
    """Compute upper-body targets for the given envs from elapsed time."""
    if env_ids is None:
      env_ids = slice(None)

    if isinstance(env_ids, slice):
      clip_idx = self._clip_idx
      start_frame = self._start_frame
      steps = self._env.episode_length_buf
      default_pos = self._default_joint_pos
    else:
      clip_idx = self._clip_idx[env_ids]
      start_frame = self._start_frame[env_ids]
      steps = self._env.episode_length_buf[env_ids]
      default_pos = self._default_joint_pos[env_ids]

    # Mask: which envs use motion playback vs default pose.
    use_motion = clip_idx >= 0

    # Compute frame indices (vectorized).
    elapsed_s = steps.float() * self._step_dt
    frame_offset = (elapsed_s * self._fps).long()
    safe_clip_idx = clip_idx.clamp(min=0)  # replace -1 with 0 for indexing
    clip_lens = self._clip_lengths[safe_clip_idx]
    frame_idx = (start_frame + frame_offset) % clip_lens

    # Look up targets from padded clip tensor: (num_clips, max_len, 17).
    targets = self._clips[safe_clip_idx, frame_idx]  # (n, 17)

    # Substitute default pose for non-motion envs.
    targets = torch.where(use_motion.unsqueeze(1), targets, default_pos)

    return targets

  def apply_actions(self) -> None:
    target = self._gather_targets()
    self._entity.set_joint_position_target(target, joint_ids=self._joint_ids)


@dataclass(kw_only=True)
class UpperBodyMotionActionCfg(ActionTermCfg):
  """Configuration for upper-body motion playback."""

  motion_file: str = ""
  """Path to the pkl motion file."""

  default_pose_ratio: float = 0.5
  """Fraction of envs that hold the default upper-body pose instead of playing motion."""

  joint_names: tuple[str, ...] = UPPER_BODY_JOINT_NAMES
  """Regex patterns for upper-body joint names."""

  def build(self, env: ManagerBasedRlEnv) -> UpperBodyMotionAction:
    return UpperBodyMotionAction(self, env)
