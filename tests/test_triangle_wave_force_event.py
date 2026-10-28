"""Unit tests for triangle-wave hand-force state and sampling."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.tasks.locomanipulation.mdp.events import (
  HandForceEvent,
  TriangleWaveForceEvent,
  _advance_triangle_phase,
  _enabled_force_axes,
  _resolve_triangle_period_s,
  _sample_enabled_axis_scales,
)


def test_z_only_force_uses_full_axis_scale():
  device = torch.device("cpu")
  enabled = _enabled_force_axes(
    {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (-40.0, 0.0)},
    device,
  )

  scales = _sample_enabled_axis_scales(enabled, (32, 2), device)

  assert torch.all(scales[..., :2] == 0.0)
  assert torch.all(scales[..., 2] == 1.0)


def test_enabled_axis_scales_exclude_disabled_axes_and_sum_to_one():
  device = torch.device("cpu")
  enabled = _enabled_force_axes(
    {"x": (-10.0, 10.0), "y": (0.0, 0.0), "z": (-40.0, 0.0)},
    device,
  )

  scales = _sample_enabled_axis_scales(enabled, (64, 2), device)

  assert torch.all(scales[..., 1] == 0.0)
  assert torch.allclose(scales.sum(dim=-1), torch.ones(64, 2))
  assert torch.all(scales[..., enabled] > 0.0)


def test_all_zero_force_bounds_produce_zero_scales():
  device = torch.device("cpu")
  enabled = _enabled_force_axes(
    {axis: (0.0, 0.0) for axis in ("x", "y", "z")}, device
  )

  scales = _sample_enabled_axis_scales(enabled, (4, 2), device)

  assert torch.all(scales == 0.0)


def test_triangle_phase_completes_one_cycle_in_period_steps():
  phase_ts = torch.zeros(1, 1, 1)
  period_steps = torch.full_like(phase_ts, 4.0)
  active = torch.tensor([True])

  phases = []
  for _ in range(4):
    phases.append(
      _advance_triangle_phase(phase_ts, period_steps, active).item()
    )

  assert phases == pytest.approx([0.5, 0.0, 0.5, 1.0])
  assert phase_ts.item() == pytest.approx(0.0)


def test_legacy_duration_is_converted_from_half_period():
  with pytest.warns(DeprecationWarning, match="half-period"):
    period = _resolve_triangle_period_s({"duration_s": (3.0, 5.0)})

  assert period == (6.0, 10.0)
  assert _resolve_triangle_period_s({"period_s": (6.0, 10.0)}) == period


def test_reset_only_resamples_selected_environment_state():
  event = TriangleWaveForceEvent.__new__(TriangleWaveForceEvent)
  event._num_envs = 3
  event._num_bodies = 2
  event._device = torch.device("cpu")
  event._period_lo_steps = 300
  event._period_hi_steps = 500
  event._force_phase_ts = torch.tensor([
    [[0.1], [0.2]],
    [[0.3], [0.4]],
    [[0.5], [0.6]],
  ])
  event._force_phase = torch.abs(event._force_phase_ts - 1.0)
  event._force_period = torch.full((3, 2, 1), 400.0)
  event._enabled_force_axes = torch.tensor([False, False, True])
  event._force_xyz_scale = torch.zeros(3, 2, 3)
  event._force_xyz_scale[..., 2] = 1.0
  event._body_point_offset_range = {
    "x": (0.01, 0.01),
    "y": (0.02, 0.02),
    "z": (0.03, 0.03),
  }
  event._body_point_offset_b = torch.tensor([
    [[0.11, 0.12, 0.13], [0.14, 0.15, 0.16]],
    [[0.21, 0.22, 0.23], [0.24, 0.25, 0.26]],
    [[0.31, 0.32, 0.33], [0.34, 0.35, 0.36]],
  ])
  event._no_force_ratio = 0.0
  event._no_force_mask = torch.zeros(3, dtype=torch.bool)
  event._no_projection_ratio = 0.0
  event._no_projection_mask = torch.zeros(3, dtype=torch.bool)

  phase_before = event._force_phase_ts.clone()
  offsets_before = event._body_point_offset_b.clone()
  event.reset(torch.tensor([1]))

  assert torch.equal(event._force_phase_ts[[0, 2]], phase_before[[0, 2]])
  assert torch.equal(event._body_point_offset_b[[0, 2]], offsets_before[[0, 2]])
  assert torch.allclose(
    event._body_point_offset_b[1],
    torch.tensor([[0.01, 0.02, 0.03], [0.01, 0.02, 0.03]]),
  )
  assert torch.all(event._force_xyz_scale[1, :, :2] == 0.0)
  assert torch.all(event._force_xyz_scale[1, :, 2] == 1.0)
  assert torch.all((event._force_period[1] >= 300) & (event._force_period[1] <= 500))


def test_expiring_impulse_preserves_previous_load_for_immediate_replacement():
  event = HandForceEvent.__new__(HandForceEvent)
  event._num_envs = event._num_bodies = 1
  event._body_ids = [0]
  event._device = torch.device("cpu")
  event._step_dt = 0.02
  event._active = torch.tensor([True])
  event._time_remaining = torch.tensor([0.01])
  event._interval_time_left = torch.zeros(1)
  event._no_force_mask = torch.tensor([False])
  event._body_point_offset_range = None
  event._body_point_offset_b = torch.zeros(1, 1, 3)
  event._force_xyz_scale = torch.tensor([[0., 0., 1.]])
  event._sample_body_point_offsets = lambda ids: None
  wrench = torch.tensor([[[0., 0., -2., 0., 0., 0.]]])
  old = wrench.clone()

  def write(forces, torques, **kwargs):
    wrench[..., :3] = forces
    wrench[..., 3:] = torques

  def estimate(env, application_offsets_b, previous_wrench):
    assert torch.count_nonzero(wrench) == 0  # The old impulse was cleared.
    torch.testing.assert_close(previous_wrench, old)
    return [torch.tensor([[-9., -9., -1.]])], [torch.full((1, 3), 9.)]

  event._asset = SimpleNamespace(
    data=SimpleNamespace(body_external_wrench=wrench),
    write_external_wrench_to_sim=write,
  )
  event._estimator = SimpleNamespace(estimate=estimate)
  event(
    None, None, torque_range=(0., 0.), duration_s=(1., 1.),
    cooldown_s=(0., 0.), asset_cfg=None, force_scale=1.,
    force_range_max={"z": (-40., 0.)}, max_force_estimation=True,
  )
  assert event._active[0]
  assert -1. <= wrench[0, 0, 2] <= 0.


def test_x2_triangle_force_estimation_runs_with_real_rotational_jacobians():
  from mjlab.envs import ManagerBasedRlEnv
  from src.tasks.locomanipulation.config.x2.env_cfgs import (
    agibot_x2_locomanipulation_flat_env_cfg,
  )

  cfg = agibot_x2_locomanipulation_flat_env_cfg(play=True)
  cfg.scene.num_envs = 2
  cfg.seed = 123
  params = cfg.events["hand_force"].params
  params.update(constant_force=None, force_scale=0.5, no_force_ratio=0.)
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    env.reset()
    for _ in range(8):
      obs, reward, *_ = env.step(torch.zeros(2, 15))
      assert torch.isfinite(obs["actor"]).all()
      assert torch.isfinite(reward).all()
      event = env.event_manager.get_term_cfg("hand_force").func
      forces = env.scene["robot"].data.body_external_wrench[:, event._body_ids, :3]
      assert torch.isfinite(forces).all()
      assert (forces[..., :2] == 0).all()
      assert ((forces[..., 2] >= -20.) & (forces[..., 2] <= 0.)).all()
    assert torch.count_nonzero(forces) > 0
  finally:
    env.close()
