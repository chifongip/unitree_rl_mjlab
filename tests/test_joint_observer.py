"""Batched observer parity, limiting and independent-world lifecycle tests."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from src.instinct_mj.controllers import BatchedJointObserver, JointObserverCfg


def _observer(*, algorithm="nonlinear", mode="ACTIVE", dtype=torch.float64, num_envs=2):
  cfg = JointObserverCfg(
    joint_names=("a", "b"), algorithm=algorithm, mode=mode,
    inertia=(0.1, 0.03), correction_limits_nm=(1.0, 0.8),
    friction_limit=0.3, friction_viscosity=(40.0, 30.0),
    release_slew_nm_s=16.0 if algorithm == "nonlinear" else 2.0,
  )
  return BatchedJointObserver(
    cfg, (50.0, 20.0), (3.0, 2.0), (2.0, 1.0),
    num_envs=num_envs, dtype=dtype,
  )


@pytest.mark.parametrize("device", [
  "cpu:0",
  pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable",
  )),
])
def test_feedback_uses_resolved_device(device):
  observer = BatchedJointObserver(
    _observer().cfg, (50, 20), (3, 2), (2, 1), num_envs=2, device=device,
  )
  feedback = torch.zeros(observer.shape, device=device)
  output = observer.step(feedback, feedback, feedback, 0.02)
  assert output.device == feedback.device == observer.device
  assert observer.initialized.all()


@pytest.mark.parametrize("algorithm", ["linearized", "nonlinear"])
@pytest.mark.parametrize("mode", ["ACTIVE", "SHADOW"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", [
  "cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable",
  )),
])
def test_matches_frozen_robojudo_synchronous_reference(algorithm, mode, dtype, device):
  # Generated from deployment sources, not from the Torch implementation.
  with np.load(Path(__file__).parent / "fixtures" / "x2_observer_reference.npz") as trace:
    cfg = JointObserverCfg(
      joint_names=tuple(str(n) for n in trace["joint_names"]),
      algorithm=algorithm, mode=mode, inertia=tuple(trace["inertia"]),
      friction_limit=0.3, friction_viscosity=tuple(trace["viscosity"]),
      correction_limits_nm=tuple(trace["correction_limits"]),
      release_slew_nm_s=16.0 if algorithm == "nonlinear" else 2.0,
    )
    observer = BatchedJointObserver(
      cfg, tuple(trace["kp"]), tuple(trace["kd"]), tuple(trace["torque_limits"]),
      num_envs=2, dtype=dtype, device=device,
    )
    key = f"{algorithm}_{mode}"
    for step, (target, q, qd) in enumerate(zip(trace["target"], trace["position"], trace["velocity"])):
      output = observer.step(*[torch.from_numpy(x).to(device=device, dtype=dtype) for x in (target, q, qd)], dt=0.02)
      tolerance = 1e-4 if dtype == torch.float32 else 1e-10
      np.testing.assert_allclose(output.cpu().numpy(), trace[f"{key}_effort"][step], atol=tolerance, rtol=tolerance)
      np.testing.assert_allclose(observer.estimate.cpu().numpy(), trace[f"{key}_estimate"][step], atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("algorithm", ["linearized", "nonlinear"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_first_sample_bounds_and_slew(algorithm, dtype):
  observer = _observer(algorithm=algorithm, dtype=dtype)
  q = torch.zeros(observer.shape, dtype=dtype)
  target = q.clone()
  velocity = torch.full_like(q, -0.1)
  assert torch.count_nonzero(observer.step(target, q, velocity, 0.02)) == 0
  previous = observer.effort.clone()
  for _ in range(30):
    output = observer.step(target, q, velocity, 0.02)
    assert torch.all(output.abs() <= observer.limits)
    rate = 16.0 if algorithm == "nonlinear" else 2.0
    assert torch.all((output - previous).abs() <= rate * 0.02 + 1e-6)
    previous = output.clone()
  assert torch.count_nonzero(output) > 0


def test_release_crosses_zero_before_opposite_growth():
  observer = _observer()
  observer.initialized[:] = True
  observer.effort[:] = 0.5
  observer.estimate[:] = 1.0  # Requests opposite correction.
  # Preserve that estimate: feedback exactly matches its previous prediction.
  q = observer.previous_position.clone()
  velocity, _ = observer._predict(observer.estimate, 0.02)
  output = observer.step(q, q, velocity, 0.02)
  torch.testing.assert_close(output, torch.full_like(output, 0.18))
  velocity, _ = observer._predict(observer.estimate, 0.02)
  output = observer.step(q, q, velocity, 0.02)
  assert torch.count_nonzero(output) == 0


@pytest.mark.parametrize("mode", ["OFF", "SHADOW"])
def test_inactive_modes_always_issue_zero(mode):
  observer = _observer(mode=mode)
  q = torch.zeros(observer.shape, dtype=observer.dtype)
  velocity = torch.full_like(q, -0.1)
  for _ in range(5):
    assert torch.count_nonzero(observer.step(q, q, velocity, 0.02)) == 0
  if mode == "SHADOW":
    assert torch.count_nonzero(observer.estimate) > 0


def test_headroom_removal_reconciles_issued_effort():
  observer = _observer()
  q = torch.zeros(observer.shape, dtype=observer.dtype)
  observer.step(q, q, q, 0.02)
  observer.effort[:] = 0.5
  observer.estimate[:] = -0.5
  target = torch.full_like(q, 1.0)  # PD is far beyond the torque limit.
  output = observer.step(target, q, q, 0.02)
  assert torch.count_nonzero(output) == 0
  assert observer.antiwindup.all()
  assert torch.count_nonzero(observer.effort) == 0


def test_partial_reset_and_invalid_world_do_not_affect_other_worlds():
  observer = _observer()
  q = torch.zeros(observer.shape, dtype=observer.dtype)
  velocity = torch.full_like(q, -0.1)
  for _ in range(4):
    observer.step(q, q, velocity, 0.02)
  survivor = observer.estimate[1].clone()
  observer.reset(torch.tensor([0]))
  assert not observer.initialized[0]
  assert torch.count_nonzero(observer.effort[0]) == 0
  torch.testing.assert_close(observer.estimate[1], survivor)
  velocity[0, 0] = float("nan")
  output = observer.step(q, q, velocity, 0.02)
  assert torch.isfinite(output).all()
  assert torch.count_nonzero(output[0]) == 0
  assert not observer.initialized[0]
  assert observer.initialized[1]


@pytest.mark.parametrize("change", [
  {"inertia": 0.0}, {"correction_limits_nm": 4.0},
  {"bandwidth_rad_s": float("nan")}, {"position_noise_std": -1.0},
  {"algorithm": "linearized", "release_slew_nm_s": 16.0},
])
def test_invalid_configuration_is_rejected(change):
  cfg = replace(_observer().cfg, **change)
  with pytest.raises(ValueError):
    BatchedJointObserver(cfg, (50.0, 20.0), (3.0, 2.0), (2.0, 1.0), num_envs=2)


@pytest.mark.parametrize("dt", [0, -0.02, float("nan"), float("inf")])
def test_invalid_timestep_clears_issued_effort_and_history(dt):
  observer = _observer()
  feedback = torch.zeros(observer.shape, dtype=observer.dtype)
  for _ in range(3):
    observer.step(feedback, feedback, feedback - 0.1, 0.02)
  assert torch.count_nonzero(observer.effort) > 0
  with pytest.raises(ValueError, match="dt must be positive and finite"):
    observer.step(feedback, feedback, feedback, dt)
  assert not observer.initialized.any()
  assert torch.count_nonzero(observer.effort) == 0
  assert torch.count_nonzero(observer.estimate) == 0
