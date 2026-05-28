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

  When cfg.pose_only is True, a single random frame is sampled at reset
  and held for the full episode (no time-varying playback).
  """

  cfg: UpperBodyMotionActionCfg

  def __init__(self, cfg: UpperBodyMotionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    # Resolve upper-body joint indices on the robot.
    asset_cfg = SceneEntityCfg(self.cfg.entity_name, joint_names=self.cfg.joint_names)
    asset_cfg.resolve(self._env.scene)
    self._joint_ids = asset_cfg.joint_ids

    # Find positions of waist_roll and waist_pitch to zero out when yaw_only is set.
    if cfg.waist_yaw_only:
      all_names = self._entity.joint_names
      zero_names = {"waist_roll_joint", "waist_pitch_joint"}
      self._waist_zero_cols = [
        i for i, jid in enumerate(self._joint_ids) if all_names[jid] in zero_names
      ]
    else:
      self._waist_zero_cols = []

    # Load motion data and extract upper-body DOF.
    data = joblib.load(cfg.motion_file)
    raw_clips: list[torch.Tensor] = []
    self._fps: int = 0
    dof_indices = list(cfg.motion_dof_indices) if cfg.motion_dof_indices is not None else None
    for v in data.values():
      dof = torch.tensor(v["dof"], dtype=torch.float32)
      raw_clips.append(dof[:, dof_indices] if dof_indices is not None else dof[:, 12:29])
      self._fps = v["fps"]

    self._num_upper_dofs = len(self._joint_ids)

    self._num_clips = len(raw_clips)
    self._step_dt = env.step_dt

    # Pad all clips into a single tensor (num_clips, max_len, num_upper_dofs).
    max_len = max(len(c) for c in raw_clips)
    self._clip_lengths = torch.tensor(
      [len(c) for c in raw_clips], dtype=torch.long, device=self.device
    )
    self._clips = torch.zeros(self._num_clips, max_len, self._num_upper_dofs, device=self.device)
    for i, clip in enumerate(raw_clips):
      self._clips[i, : len(clip)] = clip.to(self.device)

    # Default joint positions for envs that don't play back motion.
    self._default_joint_pos = self._entity.data.default_joint_pos[:, self._joint_ids]

    # Per-env state: which clip and start frame. -1 means use default pose.
    self._clip_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._start_frame = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    # Cached single-frame target for pose_only mode.
    self._pose_target = torch.zeros(self.num_envs, self._num_upper_dofs, device=self.device)

    # Build fixed pose target if specified.
    self._fixed_pose: torch.Tensor | None = None
    if cfg.fixed_upper_body_pose is not None:
      all_names = self._entity.joint_names
      fixed = self._default_joint_pos[0].clone()  # start from default pose
      for name, value in cfg.fixed_upper_body_pose.items():
        # Find this joint's position within the upper-body joint list.
        matches = [
          i for i, jid in enumerate(self._joint_ids) if all_names[jid] == name
        ]
        if not matches:
          raise ValueError(f"Joint '{name}' not found in upper-body joints")
        fixed[matches[0]] = value
      if self._waist_zero_cols:
        fixed[self._waist_zero_cols] = 0.0
      self._fixed_pose = fixed  # (num_upper_dofs,)

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

    # Fixed pose mode: write the fixed pose and skip random sampling.
    if self._fixed_pose is not None:
      targets = self._fixed_pose.unsqueeze(0).expand(count, -1)
      self._entity.write_joint_state_to_sim(
        targets, torch.zeros_like(targets), env_ids=env_ids, joint_ids=self._joint_ids
      )
      if isinstance(env_ids, slice):
        self._entity._data.joint_pos_target[:, self._joint_ids] = targets
        self._pose_target[:] = targets
      else:
        self._entity._data.joint_pos_target[env_ids.unsqueeze(1), self._joint_ids] = targets
        self._pose_target[env_ids] = targets
      return

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

    # In pose_only mode, cache the sampled frame as a fixed target for the episode.
    if self.cfg.pose_only:
      default_pos = (
        self._default_joint_pos
        if isinstance(env_ids, slice)
        else self._default_joint_pos[env_ids]
      )
      targets = self._compute_frame_targets(rand_clips, rand_starts, use_default, default_pos)
      if isinstance(env_ids, slice):
        self._pose_target[:] = targets
      else:
        self._pose_target[env_ids] = targets

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

  def _compute_frame_targets(
    self,
    clip_idx: torch.Tensor,
    start_frame: torch.Tensor,
    use_default: torch.Tensor,
    default_pos: torch.Tensor,
  ) -> torch.Tensor:
    """Look up a single frame from the motion data for each env.

    Used by reset() to cache pose targets in pose_only mode.
    """
    safe_clip_idx = clip_idx.clamp(min=0)
    targets = self._clips[safe_clip_idx, start_frame]  # (n, num_upper_dofs)
    targets = torch.where((~use_default).unsqueeze(1), targets, default_pos)
    if self._waist_zero_cols:
      targets[:, self._waist_zero_cols] = 0.0
    return targets

  def _gather_targets(
    self, env_ids: torch.Tensor | slice | None = None
  ) -> torch.Tensor:
    """Compute upper-body targets for the given envs."""
    if env_ids is None:
      env_ids = slice(None)

    # Fixed pose or pose_only mode: return the cached target (already includes
    # default pose substitution and waist zeroing from reset).
    if self._fixed_pose is not None or self.cfg.pose_only:
      if isinstance(env_ids, slice):
        return self._pose_target
      return self._pose_target[env_ids]

    # Clip playback mode: compute frame index from elapsed time.
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

    # Look up targets from padded clip tensor: (num_clips, max_len, num_upper_dofs).
    targets = self._clips[safe_clip_idx, frame_idx]  # (n, num_upper_dofs)

    # Substitute default pose for non-motion envs.
    targets = torch.where(use_motion.unsqueeze(1), targets, default_pos)

    if self._waist_zero_cols:
      targets[:, self._waist_zero_cols] = 0.0

    return targets

  def apply_actions(self) -> None:
    target = self._gather_targets()
    self._entity.set_joint_position_target(target, joint_ids=self._joint_ids)


@dataclass(kw_only=True)
class UpperBodyMotionActionCfg(ActionTermCfg):
  """Configuration for upper-body motion playback."""

  motion_file: str = ""
  """Path to the pkl motion file."""

  motion_dof_indices: tuple[int, ...] | None = None
  """Indices into the motion data's DOF array to extract upper-body joints.
  If None, uses slice(12, 29) for 29-DOF backward compatibility."""

  default_pose_ratio: float = 0.5
  """Fraction of envs that hold the default upper-body pose instead of playing motion."""

  joint_names: tuple[str, ...] = UPPER_BODY_JOINT_NAMES
  """Regex patterns for upper-body joint names."""

  waist_yaw_only: bool = False
  """If True, only apply waist_yaw_joint from motion data; zero out waist_roll and waist_pitch."""

  pose_only: bool = False
  """If True, sample a single random frame at reset and hold it for the full episode
  instead of playing back the clip frame-by-frame."""

  fixed_upper_body_pose: dict[str, float] | None = None
  """If set, all envs use this exact pose instead of random sampling.
  Keys are joint names (e.g. "left_shoulder_pitch_joint"), values in radians.
  Overrides default_pose_ratio and pose_only when set."""

  def build(self, env: ManagerBasedRlEnv) -> UpperBodyMotionAction:
    return UpperBodyMotionAction(self, env)
