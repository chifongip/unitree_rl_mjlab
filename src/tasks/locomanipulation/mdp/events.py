"""Locomanipulation MDP event terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


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
  ) -> None:
    del env, env_ids, asset_cfg

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

    # Compute force_range from force_range_max × effective_scale.
    if force_range_max is not None:
      effective_scale = min(max(force_scale, 0.0), 1.0)
      force_range = {
        key: (lo * effective_scale, hi * effective_scale)
        for key, (lo, hi) in force_range_max.items()
      }
    assert force_range is not None, "Must provide force_range or force_range_max"

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
