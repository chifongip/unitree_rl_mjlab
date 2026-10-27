"""Upper-body motion playback action term.

Loads whole-body motion data from a pkl file, extracts upper-body DOF,
and plays back random clips frame-by-frame during each episode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import joblib
import torch

from mjlab.actuator.actuator import TransmissionType
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.instinct_mj.controllers import BatchedJointObserver, JointObserverCfg

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

    # Exclude all waist joints from the upper-body set when policy controls them.
    if cfg.exclude_waist:
      all_names = self._entity.joint_names
      waist_names = {"waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"}
      non_waist = [
        i for i, jid in enumerate(self._joint_ids) if all_names[jid] not in waist_names
      ]
      self._joint_ids = [self._joint_ids[i] for i in non_waist]

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
    dof_indices = list(cfg.motion_dof_indices) if cfg.motion_dof_indices is not None else None

    manip_clips, self._fps = self._load_motion_file(cfg.motion_file, dof_indices)
    self._num_manip_clips = len(manip_clips)

    # Optional second pool for locomotion clips (command-driven mixing).
    self._has_loco_dataset = cfg.locomotion_motion_file is not None
    if self._has_loco_dataset:
      loco_clips, loco_fps = self._load_motion_file(cfg.locomotion_motion_file, dof_indices)
      if loco_fps != self._fps:
        raise ValueError(
          f"locomotion_motion_file FPS ({loco_fps}) must match motion_file FPS ({self._fps})"
        )
      self._num_loco_clips = len(loco_clips)
      raw_clips = manip_clips + loco_clips
    else:
      self._num_loco_clips = 0
      raw_clips = manip_clips

    self._num_upper_dofs = len(self._joint_ids)

    self._num_clips = len(raw_clips)
    indices = cfg.sample_clip_indices
    if indices is not None and (not indices or min(indices) < 0 or max(indices) >= self._num_clips):
      raise ValueError("sample_clip_indices must select valid motion clips")
    self._sample_pool = torch.tensor(
      indices if indices is not None else tuple(range(self._num_clips)),
      device=self.device, dtype=torch.long,
    )
    self._step_dt = env.step_dt

    self._observer: BatchedJointObserver | None = None
    self._observer_update_pending = True
    if cfg.observer is not None:
      self._initialize_observer(cfg.observer)

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

    # Deferred selection flag for dual-dataset mode.
    self._pending_selection = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

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

  @staticmethod
  def _load_motion_file(path: str, dof_indices: list[int] | None) -> tuple[list[torch.Tensor], int]:
    """Load a pkl motion file and return (clips_list, fps)."""
    data = joblib.load(path)
    clips: list[torch.Tensor] = []
    fps: int = 0
    for v in data.values():
      dof = torch.tensor(v["dof"], dtype=torch.float32)
      clips.append(dof[:, dof_indices] if dof_indices is not None else dof[:, 12:29])
      fps = v["fps"]
    return clips, fps

  def _select_clips(self, env_ids: torch.Tensor) -> None:
    """Select motion clips for the given envs based on current velocity command.

    Standing envs (vel norm < threshold) get manipulation clips (indices
    0.._num_manip_clips-1). Walking envs get locomotion clips (indices
    _num_manip_clips.._num_clips-1).
    """
    twist_term = self._env.command_manager.get_term("twist")
    vel = twist_term.vel_command_b[env_ids]
    standing_flag = twist_term.is_standing_env[env_ids]

    vel_norm = torch.norm(vel, dim=1)
    is_standing = (vel_norm < self.cfg.command_threshold) | standing_flag

    count = len(env_ids)
    rand_clips = torch.empty(count, dtype=torch.long, device=self.device)

    standing_mask = is_standing.nonzero(as_tuple=False).flatten()
    walking_mask = (~is_standing).nonzero(as_tuple=False).flatten()

    if len(standing_mask) > 0:
      rand_clips[standing_mask] = torch.randint(
        0, self._num_manip_clips, (len(standing_mask),), device=self.device
      )
    if len(walking_mask) > 0:
      rand_clips[walking_mask] = torch.randint(
        0, self._num_loco_clips, (len(walking_mask),), device=self.device
      ) + self._num_manip_clips

    clip_lens = self._clip_lengths[rand_clips]
    rand_starts = (torch.rand(count, device=self.device) * clip_lens.float()).long()

    # Apply default_pose_ratio: substitute default pose for some envs.
    use_default = torch.rand(count, device=self.device) < self.cfg.default_pose_ratio
    rand_clips[use_default] = -1  # sentinel for default pose

    self._clip_idx[env_ids] = rand_clips
    self._start_frame[env_ids] = rand_starts

  @property
  def action_dim(self) -> int:
    return 0

  @property
  def raw_action(self) -> torch.Tensor:
    return torch.empty(0, device=self.device)

  def process_actions(self, actions: torch.Tensor) -> None:
    del actions
    self._observer_update_pending = True

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._reset_observer(env_ids)
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

    # Dual-dataset mode: defer clip selection to first _gather_targets
    # call so the velocity command (sampled by command_manager.reset after
    # action_manager.reset) is available for the standing/walking decision.
    if self._has_loco_dataset:
      if isinstance(env_ids, slice):
        self._pending_selection[:] = True
      else:
        self._pending_selection[env_ids] = True
      # use default pose as placeholder until _gather_targets selects real clips.
      return

    # Sample random clips and start frames (vectorized).
    rand_clips = self._sample_pool[
      torch.randint(0, len(self._sample_pool), (count,), device=self.device)
    ]
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

    # Deferred clip selection for newly-reset envs in dual-dataset mode.
    # Runs before target computation so the correct dataset is used.
    if self._has_loco_dataset:
      if isinstance(env_ids, slice):
        pending_mask = self._pending_selection
      else:
        pending_mask = self._pending_selection[env_ids]
      if pending_mask.any():
        pending_rel = pending_mask.nonzero(as_tuple=False).flatten()
        if isinstance(env_ids, slice):
          actual_ids = pending_rel
        else:
          actual_ids = env_ids[pending_rel]
        self._select_clips(actual_ids)
        self._pending_selection[actual_ids] = False

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
    if self._observer is not None:
      if self._observer_update_pending:
        # These are encoder/velocity feedback channels, also available on the
        # robot. Policy observations and their corruption are not involved.
        q = self._entity.data.joint_pos_biased[:, self._observer_joint_ids]
        qd = self._entity.data.joint_vel[:, self._observer_joint_ids]
        settings = self._observer.cfg
        if settings.position_noise_std:
          q = q + torch.randn_like(q) * settings.position_noise_std
        if settings.velocity_noise_std:
          qd = qd + torch.randn_like(qd) * settings.velocity_noise_std
        self._observer.step(target[:, self._observer_target_cols], q, qd, self._step_dt)
        self._observer_update_pending = False
      self._entity.set_joint_effort_target(
        self._observer.effort, joint_ids=self._observer_joint_ids
      )

  def _initialize_observer(self, cfg: JointObserverCfg) -> None:
    """Resolve explicit joint-name order and controller-known actuator settings."""
    from src.assets.robots.agibot_x2.compensated_actuator import CompensatedPositionActuator

    joint_ids = [self._entity.joint_names.index(name) for name in cfg.joint_names]
    if any(j not in self._joint_ids for j in joint_ids):
      raise ValueError("Observer joints must be owned by upper-body motion playback")
    if any(name.startswith("waist_") for name in cfg.joint_names):
      raise ValueError("Torque compensation is restricted to arm joints")
    self._observer_target_cols = [self._joint_ids.index(j) for j in joint_ids]
    # Entity setters require tensor joint IDs for outer indexing when env_ids
    # is also a tensor (as it is during partial and automatic resets).
    self._observer_joint_ids = torch.tensor(
      joint_ids, device=self.device, dtype=torch.long
    )
    gains: list[float] = []
    damping: list[float] = []
    limits: list[float] = []
    for name in cfg.joint_names:
      matching = [
        a for a in self._entity.actuators
        if a.transmission_type == TransmissionType.JOINT and name in a.target_names
      ]
      if len(matching) != 1 or not isinstance(matching[0], CompensatedPositionActuator):
        raise ValueError(f"Observer joint {name} requires a compensated position actuator")
      actuator = matching[0].cfg
      if actuator.effort_limit is None:
        raise ValueError(f"Observer joint {name} requires an actuator effort limit")
      if actuator.delay_max_lag:
        raise ValueError("Synchronous observer requires zero command transport delay")
      gains.append(actuator.stiffness)
      damping.append(actuator.damping)
      limits.append(actuator.effort_limit)
    self._observer = BatchedJointObserver(
      cfg, tuple(gains), tuple(damping), tuple(limits),
      num_envs=self.num_envs, device=self.device,
      dtype=self._entity.data.joint_pos.dtype,
    )

  def _reset_observer(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if self._observer is not None:
      self._observer.reset(env_ids)
      ids = slice(None) if env_ids is None else env_ids
      self._entity.set_joint_effort_target(
        self._observer.effort[ids], joint_ids=self._observer_joint_ids, env_ids=ids
      )

  def set_fixed_pose(self, pose_dict: dict[str, float] | None) -> None:
    """Set a fixed upper body pose for all environments (runtime override).

    Args:
      pose_dict: Joint name to radians mapping. None or empty dict resets
        to the model's default joint positions.
    """
    all_names = self._entity.joint_names
    fixed = self._default_joint_pos[0].clone()
    if pose_dict:
      for name, value in pose_dict.items():
        matches = [
          i for i, jid in enumerate(self._joint_ids) if all_names[jid] == name
        ]
        if not matches:
          raise ValueError(f"Joint '{name}' not found in upper-body joints")
        fixed[matches[0]] = value
    if self._waist_zero_cols:
      fixed[self._waist_zero_cols] = 0.0

    self._fixed_pose = fixed
    self._pose_target[:] = fixed.unsqueeze(0).expand_as(self._pose_target)
    # This method teleports the simulated joint state. Discard pre-teleport
    # predictor history and queued correction in the same call.
    self._reset_observer()
    self._observer_update_pending = True
    self._entity.write_joint_state_to_sim(
      self._pose_target,
      torch.zeros_like(self._pose_target),
      env_ids=slice(None),
      joint_ids=self._joint_ids,
    )
    self._entity._data.joint_pos_target[:, self._joint_ids] = self._pose_target

  def get_fixed_pose(self) -> dict[str, float]:
    """Return the current fixed pose as {joint_name: radians}."""
    all_names = self._entity.joint_names
    return {
      all_names[self._joint_ids[i]]: self._pose_target[0, i].item()
      for i in range(self._num_upper_dofs)
    }


@dataclass(kw_only=True)
class UpperBodyMotionActionCfg(ActionTermCfg):
  """Configuration for upper-body motion playback.

  When ``locomotion_motion_file`` is set, the action maintains two motion
  pools and selects clips based on the current velocity command: standing
  envs use ``motion_file`` (manipulation gestures), walking envs use
  ``locomotion_motion_file`` (coordinated locomotion arm swing).
  """

  motion_file: str = ""
  """Path to the pkl motion file (standing / manipulation data when
  locomotion_motion_file is set)."""

  observer: JointObserverCfg | None = None
  """Optional internal arm controller. Adds no policy actions or observations."""

  restore_observer_from_checkpoint: bool = True
  """Restore saved arm control; False keeps this environment's controller for fine-tuning."""

  locomotion_motion_file: str | None = None
  """Optional path to a second motion pkl used when the velocity command
  norm exceeds ``command_threshold``. When None, only ``motion_file`` is
  used (backward compatible)."""

  command_threshold: float = 0.1
  """Velocity command L2 norm threshold. Below → standing (motion_file).
  At or above → walking (locomotion_motion_file)."""

  motion_dof_indices: tuple[int, ...] | None = None
  """Indices into the motion data's DOF array to extract upper-body joints.
  If None, uses slice(12, 29) for 29-DOF backward compatibility."""

  default_pose_ratio: float = 0.5
  """Fraction of envs that hold the default upper-body pose instead of playing motion."""

  joint_names: tuple[str, ...] = UPPER_BODY_JOINT_NAMES
  """Regex patterns for upper-body joint names."""

  waist_yaw_only: bool = False
  """If True, only apply waist_yaw_joint from motion data; zero out waist_roll and waist_pitch."""

  exclude_waist: bool = False
  """If True, exclude all waist_*_joint from the upper-body set (policy controls them)."""

  pose_only: bool = False
  """If True, sample a single random frame at reset and hold it for the full episode
  instead of playing back the clip frame-by-frame."""

  sample_clip_indices: tuple[int, ...] | None = None
  """Optional subset of motion clips eligible for single-dataset reset sampling."""

  fixed_upper_body_pose: dict[str, float] | None = None
  """If set, all envs use this exact pose instead of random sampling.
  Keys are joint names (e.g. "left_shoulder_pitch_joint"), values in radians.
  Overrides default_pose_ratio and pose_only when set."""

  def build(self, env: ManagerBasedRlEnv) -> UpperBodyMotionAction:
    return UpperBodyMotionAction(self, env)
