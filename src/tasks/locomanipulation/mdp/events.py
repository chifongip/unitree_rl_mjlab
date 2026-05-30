"""Locomanipulation MDP event terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco_warp as mjwarp
import torch
import warp as wp

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class MaxForceEstimator:
  """Estimate maximum allowable end-effector forces via Jacobian transpose.

  For each end-effector, computes the max Cartesian force (per axis) that
  keeps all arm/waist joint torques within their effort limits:
    F_max[axis] = min_i( effort_limit[i] / |J[axis, i]| )
  """

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    asset: Entity,
    ee_body_names: tuple[str, ...],
    constraint_joint_names: tuple[str, ...],
  ):
    self._num_envs = env.num_envs
    self._device = env.device
    self._asset = asset

    # Resolve EE body IDs: entity-local for data access, global for mjwarp.jac.
    self._ee_local_body_ids: list[int] = []
    self._ee_global_body_ids: list[int] = []
    for name in ee_body_names:
      ids, _ = asset.find_bodies(name)
      local_id = ids[0]
      self._ee_local_body_ids.append(local_id)
      self._ee_global_body_ids.append(int(asset.indexing.body_ids[local_id].item()))

    # Resolve constraint joint entity-local indices and model DOF addresses.
    joint_ids, _ = asset.find_joints(constraint_joint_names)
    joint_ids_t = torch.tensor(joint_ids, device=self._device, dtype=torch.long)
    self._constraint_dof_adr = asset.indexing.joint_v_adr[joint_ids_t]

    # Build per-DOF effort limit tensor for constraint DOFs.
    num_dofs = len(self._constraint_dof_adr)
    effort = torch.zeros(num_dofs, device=self._device)
    for act in asset.actuators:
      act_dof_adr = asset.indexing.joint_v_adr[act.target_ids]
      fl = act.cfg.effort_limit
      for adr in act_dof_adr:
        mask = self._constraint_dof_adr == adr
        if mask.any():
          effort[mask] = fl
    self._effort_limit = effort  # (num_constraint_dofs,)

    # Allocate warp tensors for mjwarp.jac.
    nworld = self._num_envs
    nv = env.sim.mj_model.nv
    with wp.ScopedDevice(env.sim.wp_device):
      self._jacp_wp = wp.zeros((nworld, 3, nv), dtype=float)
      self._point_wp = wp.zeros(nworld, dtype=wp.vec3)
      self._body_wp = wp.zeros(nworld, dtype=wp.int32)

  def estimate(
    self, env: ManagerBasedRlEnv
  ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Compute per-EE force bounds from current configuration.

    Returns:
      (force_mins, force_maxes) — lists of (nworld, 3) tensors, one per EE.
    """
    force_mins: list[torch.Tensor] = []
    force_maxes: list[torch.Tensor] = []

    for local_id, global_id in zip(self._ee_local_body_ids, self._ee_global_body_ids):
      # Set point to body CoM position (entity-local index).
      com_pos = self._asset.data.body_com_pos_w[:, local_id]  # (nworld, 3)
      self._point_wp.assign(wp.from_torch(com_pos, dtype=wp.vec3))
      self._body_wp.fill_(global_id)

      # Compute translational Jacobian.
      with wp.ScopedDevice(env.sim.wp_device):
        mjwarp.jac(
          env.sim.wp_model, env.sim.wp_data,
          self._jacp_wp, None, self._point_wp, self._body_wp,
        )

      # Slice constraint DOF columns: (nworld, 3, n_constraint_dofs).
      jacp = wp.to_torch(self._jacp_wp)[:, :, self._constraint_dof_adr]

      # Per-joint, per-axis force limits: effort / (|J| + eps).
      eps = 1e-2
      inv_jac = 1.0 / (jacp.abs() + eps)
      f_max_all = inv_jac * self._effort_limit
      f_min_all = -inv_jac * self._effort_limit

      # Most restrictive across joints.
      f_max = f_max_all.min(dim=2).values
      f_min = f_min_all.max(dim=2).values

      force_mins.append(f_min)
      force_maxes.append(f_max)

    return force_mins, force_maxes


class HandForceEvent:
  """Apply per-axis random forces to end-effector bodies with impulse lifecycle.

  Simulates carrying heavy objects. Forces are applied for a sampled duration,
  followed by a cooldown. A fraction of envs (no_force_ratio) receive no force,
  maintaining baseline locomotion skills.

  Use with mode="step".
  """

  @property
  def viz_cfg(self):
    class VizCfg:
      rgba: tuple[float, float, float, float] = (0.9, 0.2, 0.8, 0.9)
      scale: float = 0.02
      width: float = 0.005
      min_force: float = 1.0
    return VizCfg()

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    self._body_ids = cfg.params["asset_cfg"].body_ids
    self._num_bodies = (
      len(self._body_ids)
      if isinstance(self._body_ids, list)
      else self._asset.num_bodies
    )
    self._num_envs = env.num_envs
    self._device = env.device
    self._step_dt = env.step_dt

    # Max force estimation via Jacobian.
    self._max_force_estimation = cfg.params.get("max_force_estimation", False)
    self._estimator: MaxForceEstimator | None = None
    if self._max_force_estimation:
      ee_body_names = cfg.params["asset_cfg"].body_names
      self._estimator = MaxForceEstimator(
        env=env,
        asset=self._asset,
        ee_body_names=ee_body_names,
        constraint_joint_names=cfg.params["constraint_joint_names"],
      )

    # Per-env Dirichlet scaling for axis-wise force diversity.
    self._force_xyz_scale = torch.distributions.Dirichlet(
      torch.tensor([1.0, 1.0, 1.0], device=self._device)
    ).sample((self._num_envs,))

    self._time_remaining = torch.zeros(self._num_envs, device=self._device)
    self._interval_time_left = torch.zeros(self._num_envs, device=self._device)
    self._active = torch.zeros(
      self._num_envs, device=self._device, dtype=torch.bool
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    torque_range: tuple[float, float],
    duration_s: tuple[float, float],
    cooldown_s: tuple[float, float],
    asset_cfg: SceneEntityCfg,
    no_force_ratio: float = 0.0,
    body_point_offset_range: dict[str, tuple[float, float]] | None = None,
    force_range_max: dict[str, tuple[float, float]] | None = None,
    force_scale: float = 0.0,
    zero_force_prob: dict[str, float] | None = None,
    force_range: dict[str, tuple[float, float]] | None = None,
    constant_force: dict[str, float] | None = None,
    max_force_estimation: bool = False,
    **kwargs: object,
  ) -> None:
    del env_ids, asset_cfg

    # Constant force mode: apply a fixed force every step, bypass impulse lifecycle.
    if constant_force is not None:
      n = self._num_envs
      forces = torch.zeros(n, self._num_bodies, 3, device=self._device)
      for axis, key in enumerate(("x", "y", "z")):
        forces[:, :, axis] = constant_force.get(key, 0.0)
      self._asset.write_external_wrench_to_sim(
        forces, torch.zeros_like(forces), body_ids=self._body_ids
      )
      self._active[:] = True
      return

    dt = self._step_dt
    effective_scale = min(max(force_scale, 0.0), 1.0)

    # Determine force bounds: Jacobian-based or static.
    if max_force_estimation and self._estimator is not None:
      # Dynamic bounds from Jacobian, scaled by curriculum.
      ee_force_mins, ee_force_maxes = self._estimator.estimate(env)
      # Clip to hard bounds and apply force_scale.
      keys = ("x", "y", "z")
      for ee_idx in range(len(ee_force_mins)):
        for axis, key in enumerate(keys):
          if force_range_max is not None:
            hard_lo, hard_hi = force_range_max[key]
          else:
            hard_lo, hard_hi = -float("inf"), float("inf")
          ee_force_mins[ee_idx][:, axis] = ee_force_mins[ee_idx][:, axis].clamp(
            hard_lo, hard_hi
          ) * effective_scale
          ee_force_maxes[ee_idx][:, axis] = ee_force_maxes[ee_idx][:, axis].clamp(
            hard_lo, hard_hi
          ) * effective_scale
      # Per-env Dirichlet axis scaling for training diversity.
      for ee_idx in range(len(ee_force_mins)):
        ee_force_mins[ee_idx] *= self._force_xyz_scale
        ee_force_maxes[ee_idx] *= self._force_xyz_scale
      use_per_ee_bounds = True
    else:
      # Static bounds from config.
      if force_range_max is not None:
        force_range = {
          key: (lo * effective_scale, hi * effective_scale)
          for key, (lo, hi) in force_range_max.items()
        }
      assert force_range is not None, "Must provide force_range or force_range_max"
      use_per_ee_bounds = False

    self._time_remaining[self._active] -= dt

    # Clear expired impulses and resample cooldown timers.
    expired = self._active & (self._time_remaining <= 0)
    if expired.any():
      expired_ids = expired.nonzero(as_tuple=False).squeeze(-1)
      zeros = torch.zeros(
        (len(expired_ids), self._num_bodies, 3), device=self._device
      )
      self._asset.write_external_wrench_to_sim(
        zeros, zeros, env_ids=expired_ids, body_ids=self._body_ids
      )
      self._active[expired_ids] = False
      self._time_remaining[expired_ids] = 0.0
      int_low, int_high = cooldown_s
      self._interval_time_left[expired_ids] = (
        torch.rand(len(expired_ids), device=self._device) * (int_high - int_low)
        + int_low
      )

    self._interval_time_left -= dt

    # Trigger new impulses for eligible envs.
    eligible = (~self._active) & (self._interval_time_left <= 0)
    if not eligible.any():
      return

    trigger_ids = eligible.nonzero(as_tuple=False).squeeze(-1)
    n = len(trigger_ids)

    # Apply no_force_ratio: skip a fraction of envs.
    if no_force_ratio > 0:
      mask = torch.rand(n, device=self._device) >= no_force_ratio
      trigger_ids = trigger_ids[mask]
      n = len(trigger_ids)
      if n == 0:
        return

    size = (n, self._num_bodies, 3)

    # Sample per-axis forces.
    forces = torch.zeros(size, device=self._device)
    if use_per_ee_bounds:
      # Per-EE force sampling using Jacobian-derived bounds.
      for ee_idx in range(self._num_bodies):
        f_min = ee_force_mins[ee_idx][trigger_ids]  # (n, 3)
        f_max = ee_force_maxes[ee_idx][trigger_ids]  # (n, 3)
        u = torch.rand(n, 3, device=self._device)
        forces[:, ee_idx, :] = f_min + (f_max - f_min) * u
    else:
      for axis, key in enumerate(("x", "y", "z")):
        lo, hi = force_range.get(key, (0.0, 0.0))
        forces[:, :, axis] = torch.empty(
          n, self._num_bodies, device=self._device
        ).uniform_(lo, hi)

    # Zero individual axes based on zero_force_prob.
    if zero_force_prob is not None:
      for axis, key in enumerate(("x", "y", "z")):
        prob = zero_force_prob.get(key, 0.0)
        if prob > 0:
          mask = torch.rand(n, self._num_bodies, device=self._device) < prob
          forces[:, :, axis][mask] = 0.0

    # Sample independent torque (scaled by per-env scale).
    torques = torch.zeros(size, device=self._device)
    tor_lo, tor_hi = torque_range
    if tor_lo != 0.0 or tor_hi != 0.0:
      torques = torch.empty(size, device=self._device).uniform_(tor_lo, tor_hi)

    # Randomize body_point_offset per-episode and compute torque.
    if body_point_offset_range is not None:
      offset = torch.zeros(n, 3, device=self._device)
      for axis, key in enumerate(("x", "y", "z")):
        lo, hi = body_point_offset_range.get(key, (0.0, 0.0))
        offset[:, axis] = torch.empty(n, device=self._device).uniform_(lo, hi)

      body_quat = self._asset.data.body_com_quat_w[trigger_ids][:, self._body_ids]
      offset_w = quat_apply(
        body_quat.reshape(-1, 4),
        offset.unsqueeze(1).expand(n, self._num_bodies, 3).reshape(-1, 3),
      ).reshape(n, self._num_bodies, 3)
      torques = torques + torch.cross(offset_w, forces, dim=-1)

    self._asset.write_external_wrench_to_sim(
      forces, torques, env_ids=trigger_ids, body_ids=self._body_ids
    )

    # Set duration timer and mark active.
    dur_low, dur_high = duration_s
    self._time_remaining[trigger_ids] = (
      torch.rand(n, device=self._device) * (dur_high - dur_low) + dur_low
    )
    self._active[trigger_ids] = True

    # Resample interval timers.
    int_low, int_high = cooldown_s
    self._interval_time_left[trigger_ids] = (
      torch.rand(n, device=self._device) * (int_high - int_low) + int_low
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)

    if isinstance(env_ids, slice):
      active_ids = self._active.nonzero(as_tuple=False).squeeze(-1)
    else:
      active_ids = env_ids[self._active[env_ids]]

    if len(active_ids) > 0:
      zeros = torch.zeros(
        (len(active_ids), self._num_bodies, 3), device=self._device
      )
      self._asset.write_external_wrench_to_sim(
        zeros, zeros, env_ids=active_ids, body_ids=self._body_ids
      )

    self._time_remaining[env_ids] = 0.0
    self._interval_time_left[env_ids] = 0.0
    self._active[env_ids] = False

    # Resample per-env Dirichlet axis scaling.
    n = len(env_ids) if isinstance(env_ids, torch.Tensor) else self._num_envs
    self._force_xyz_scale[env_ids] = torch.distributions.Dirichlet(
      torch.tensor([1.0, 1.0, 1.0], device=self._device)
    ).sample((n,))

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    if not self._active.any():
      return
    viz = self.viz_cfg
    min_sq = viz.min_force * viz.min_force
    wrench = self._asset.data.body_external_wrench
    com_pos = self._asset.data.body_com_pos_w
    for env_idx in visualizer.get_env_indices(self._num_envs):
      if not self._active[env_idx]:
        continue
      for i in (self._body_ids if isinstance(self._body_ids, list) else range(self._num_bodies)):
        force = wrench[env_idx, i, :3]
        if (force * force).sum().item() < min_sq:
          continue
        force_np = force.cpu().numpy()
        start_np = com_pos[env_idx, i].cpu().numpy()
        end_np = start_np + force_np * viz.scale
        visualizer.add_arrow(
          start=start_np,
          end=end_np,
          color=viz.rgba,
          width=viz.width,
        )


class TriangleWaveForceEvent:
  """Apply per-axis forces to end-effector bodies with mode-dependent behavior.

  Standing envs (|v_cmd| < threshold): Force oscillates via triangle wave between
  force_min and force_max, producing smooth continuous disturbance.

  Walking envs (|v_cmd| >= threshold): Force XY components are projected to oppose
  the walking direction (resistance), Z unchanged. Simulates dragging weight.

  Use with mode="step".
  """

  @property
  def viz_cfg(self):
    class VizCfg:
      rgba: tuple[float, float, float, float] = (0.9, 0.2, 0.8, 0.9)
      scale: float = 0.02
      width: float = 0.005
      min_force: float = 1.0
    return VizCfg()

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    self._body_ids = cfg.params["asset_cfg"].body_ids
    self._num_bodies = (
      len(self._body_ids)
      if isinstance(self._body_ids, list)
      else self._asset.num_bodies
    )
    self._num_envs = env.num_envs
    self._device = env.device
    self._step_dt = env.step_dt

    # Max force estimation via Jacobian.
    self._max_force_estimation = cfg.params.get("max_force_estimation", False)
    self._estimator: MaxForceEstimator | None = None
    if self._max_force_estimation:
      ee_body_names = cfg.params["asset_cfg"].body_names
      self._estimator = MaxForceEstimator(
        env=env,
        asset=self._asset,
        ee_body_names=ee_body_names,
        constraint_joint_names=cfg.params["constraint_joint_names"],
      )

    # Per-env Dirichlet scaling for axis-wise force diversity.
    self._force_xyz_scale = torch.distributions.Dirichlet(
      torch.tensor([1.0, 1.0, 1.0], device=self._device)
    ).sample((self._num_envs,))

    # Triangle wave state.
    self._force_phase_ts = torch.rand(self._num_envs, 1, device=self._device)
    self._force_phase = torch.abs(
      torch.remainder(self._force_phase_ts, 2.0) - 1.0
    )
    dur_lo, dur_hi = cfg.params["duration_s"]
    self._dur_lo_steps = int(dur_lo / self._step_dt)
    self._dur_hi_steps = int(dur_hi / self._step_dt)
    self._force_duration = torch.randint(
      self._dur_lo_steps, self._dur_hi_steps + 1,
      (self._num_envs, 1), device=self._device,
    ).float()

    # Standing/walking gate.
    self._command_name = cfg.params.get("command_name", "twist")
    self._command_threshold = cfg.params.get("command_threshold", 0.1)

    # No-force mask (per-episode, resampled at reset).
    self._no_force_ratio = cfg.params.get("no_force_ratio", 0.0)
    self._no_force_mask = (
      torch.rand(self._num_envs, device=self._device) < self._no_force_ratio
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    torque_range: tuple[float, float],
    duration_s: tuple[float, float],
    asset_cfg: SceneEntityCfg,
    force_range_max: dict[str, tuple[float, float]] | None = None,
    force_scale: float = 0.0,
    force_range: dict[str, tuple[float, float]] | None = None,
    constant_force: dict[str, float] | None = None,
    max_force_estimation: bool = False,
    no_force_ratio: float = 0.0,
    body_point_offset_range: dict[str, tuple[float, float]] | None = None,
    command_name: str = "twist",
    command_threshold: float = 0.1,
    cooldown_s: tuple[float, float] = (0.0, 0.0),
    **kwargs: object,
  ) -> None:
    del env_ids, asset_cfg, cooldown_s

    # --- Constant force mode: bypass everything ---
    if constant_force is not None:
      n = self._num_envs
      forces = torch.zeros(n, self._num_bodies, 3, device=self._device)
      for axis, key in enumerate(("x", "y", "z")):
        forces[:, :, axis] = constant_force.get(key, 0.0)
      self._asset.write_external_wrench_to_sim(
        forces, torch.zeros_like(forces), body_ids=self._body_ids
      )
      return

    effective_scale = min(max(force_scale, 0.0), 1.0)

    # --- Compute force bounds (same logic as HandForceEvent) ---
    if max_force_estimation and self._estimator is not None:
      ee_force_mins, ee_force_maxes = self._estimator.estimate(env)
      keys = ("x", "y", "z")
      for ee_idx in range(len(ee_force_mins)):
        for axis, key in enumerate(keys):
          if force_range_max is not None:
            hard_lo, hard_hi = force_range_max[key]
          else:
            hard_lo, hard_hi = -float("inf"), float("inf")
          ee_force_mins[ee_idx][:, axis] = (
            ee_force_mins[ee_idx][:, axis].clamp(hard_lo, hard_hi) * effective_scale
          )
          ee_force_maxes[ee_idx][:, axis] = (
            ee_force_maxes[ee_idx][:, axis].clamp(hard_lo, hard_hi) * effective_scale
          )
      for ee_idx in range(len(ee_force_mins)):
        ee_force_mins[ee_idx] *= self._force_xyz_scale
        ee_force_maxes[ee_idx] *= self._force_xyz_scale
      use_per_ee_bounds = True
    else:
      if force_range_max is not None:
        force_range = {
          key: (lo * effective_scale, hi * effective_scale)
          for key, (lo, hi) in force_range_max.items()
        }
      assert force_range is not None
      use_per_ee_bounds = False

    # --- Standing/walking gate ---
    twist_cmd = env.command_manager.get_command(self._command_name)
    total_cmd = torch.norm(twist_cmd[:, :2], dim=1) + torch.abs(twist_cmd[:, 2])
    is_standing = total_cmd < self._command_threshold

    # --- Update phase for standing envs only ---
    active = is_standing & ~self._no_force_mask
    self._force_phase_ts[active] = torch.remainder(
      self._force_phase_ts[active] + 1.0 / self._force_duration[active], 2.0
    )
    self._force_phase[active] = torch.abs(
      self._force_phase_ts[active] - 1.0
    )

    # --- Compute raw force from phase ---
    forces = torch.zeros(self._num_envs, self._num_bodies, 3, device=self._device)
    if use_per_ee_bounds:
      for ee_idx in range(self._num_bodies):
        f_min = ee_force_mins[ee_idx]
        f_max = ee_force_maxes[ee_idx]
        forces[:, ee_idx, :] = f_min + (f_max - f_min) * self._force_phase
    else:
      for axis, key in enumerate(("x", "y", "z")):
        lo, hi = force_range.get(key, (0.0, 0.0))
        forces[:, :, axis] = lo + (hi - lo) * self._force_phase.squeeze(1)

    # --- Walking resistance projection ---
    walking = ~is_standing & ~self._no_force_mask
    if walking.any():
      base_quat = self._asset.data.body_link_quat_w[:, 0, :]
      cmd_xy = twist_cmd[:, :2]
      cmd_3d = torch.cat(
        [cmd_xy, torch.zeros(self._num_envs, 1, device=self._device)], dim=-1
      )
      walk_dir = quat_apply(base_quat, cmd_3d)[:, :2]
      walk_dir_norm = torch.norm(walk_dir, dim=-1, keepdim=True) + 1e-6
      resist_unit = torch.zeros_like(walk_dir)
      resist_unit[walking] = -walk_dir[walking] / walk_dir_norm[walking]

      for ee_idx in range(self._num_bodies):
        force_xy = forces[:, ee_idx, :2]
        proj = torch.sum(
          force_xy[walking] * resist_unit[walking], dim=-1, keepdim=True
        )
        force_xy[walking] = torch.abs(proj) * resist_unit[walking]
        forces[:, ee_idx, :2] = force_xy

    # --- Zero out no-force envs ---
    forces[self._no_force_mask] = 0.0

    # --- Compute torque via body_point_offset ---
    torques = torch.zeros_like(forces)
    tor_lo, tor_hi = torque_range
    if tor_lo != 0.0 or tor_hi != 0.0:
      torques = torch.empty_like(forces).uniform_(tor_lo, tor_hi)

    if body_point_offset_range is not None:
      n = self._num_envs
      offset = torch.zeros(n, 3, device=self._device)
      for axis, key in enumerate(("x", "y", "z")):
        lo, hi = body_point_offset_range.get(key, (0.0, 0.0))
        offset[:, axis] = torch.empty(n, device=self._device).uniform_(lo, hi)
      body_quat = self._asset.data.body_com_quat_w[:, self._body_ids]
      offset_w = quat_apply(
        body_quat.reshape(-1, 4),
        offset.unsqueeze(1).expand(n, self._num_bodies, 3).reshape(-1, 3),
      ).reshape(n, self._num_bodies, 3)
      torques = torques + torch.cross(offset_w, forces, dim=-1)

    self._asset.write_external_wrench_to_sim(
      forces, torques, body_ids=self._body_ids
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)

    n = len(env_ids) if isinstance(env_ids, torch.Tensor) else self._num_envs
    self._force_phase_ts[env_ids] = torch.rand(n, 1, device=self._device)
    self._force_phase[env_ids] = torch.abs(
      torch.remainder(self._force_phase_ts[env_ids], 2.0) - 1.0
    )
    self._force_duration[env_ids] = torch.randint(
      self._dur_lo_steps, self._dur_hi_steps + 1,
      (n, 1), device=self._device,
    ).float()

    self._force_xyz_scale[env_ids] = torch.distributions.Dirichlet(
      torch.tensor([1.0, 1.0, 1.0], device=self._device)
    ).sample((n,))

    self._no_force_mask[env_ids] = (
      torch.rand(n, device=self._device) < self._no_force_ratio
    )

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    viz = self.viz_cfg
    min_sq = viz.min_force * viz.min_force
    wrench = self._asset.data.body_external_wrench
    com_pos = self._asset.data.body_com_pos_w
    for env_idx in visualizer.get_env_indices(self._num_envs):
      for i in (
        self._body_ids
        if isinstance(self._body_ids, list)
        else range(self._num_bodies)
      ):
        force = wrench[env_idx, i, :3]
        if (force * force).sum().item() < min_sq:
          continue
        force_np = force.cpu().numpy()
        start_np = com_pos[env_idx, i].cpu().numpy()
        end_np = start_np + force_np * viz.scale
        visualizer.add_arrow(
          start=start_np, end=end_np, color=viz.rgba, width=viz.width,
        )
