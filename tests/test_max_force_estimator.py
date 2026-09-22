"""Regression tests for torque-limit-aware end-effector force estimation."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from unittest.mock import patch

import pytest
import torch

from src.tasks.locomanipulation.mdp.events import (
  MaxForceEstimator,
  _intersect_force_bounds,
)


@dataclass
class _ActuatorCfg:
  effort_limit: float | None


@dataclass
class _Actuator:
  target_ids: torch.Tensor
  cfg: _ActuatorCfg


class _Indexing:
  def __init__(self, num_bodies: int, joint_v_adr: torch.Tensor):
    self.body_ids = torch.arange(10, 10 + num_bodies)
    self.joint_v_adr = joint_v_adr


class _Data:
  def __init__(self, num_envs: int, num_bodies: int, baseline: torch.Tensor):
    self.body_com_pos_w = torch.zeros(num_envs, num_bodies, 3)
    self.body_com_quat_w = torch.zeros(num_envs, num_bodies, 4)
    self.body_com_quat_w[..., 0] = 1.0
    self.qfrc_actuator = baseline


class _Entity:
  def __init__(
    self,
    body_names: tuple[str, ...],
    joint_names: tuple[str, ...],
    effort_limits: tuple[float | None, ...],
    baseline: torch.Tensor,
    duplicate_actuator_joint: int | None = None,
  ):
    self._body_names = body_names
    self._joint_names = joint_names
    self.indexing = _Indexing(
      len(body_names), torch.arange(6, 6 + len(joint_names))
    )
    self.actuators = [
      _Actuator(torch.tensor([idx]), _ActuatorCfg(limit))
      for idx, limit in enumerate(effort_limits)
    ]
    if duplicate_actuator_joint is not None:
      idx = duplicate_actuator_joint
      self.actuators.append(
        _Actuator(torch.tensor([idx]), _ActuatorCfg(effort_limits[idx]))
      )
    self.data = _Data(baseline.shape[0], len(body_names), baseline)

  def find_bodies(self, name: str):
    if name not in self._body_names:
      return [], []
    idx = self._body_names.index(name)
    return [idx], [name]

  def find_joints(self, names: tuple[str, ...]):
    matches = [(self._joint_names.index(name), name) for name in names
               if name in self._joint_names]
    return [match[0] for match in matches], [match[1] for match in matches]


class _Model:
  def __init__(self, nv: int):
    self.nv = nv


class _Sim:
  def __init__(self, nv: int):
    self.mj_model = _Model(nv)
    self.wp_device = "cpu"
    self.wp_model = None
    self.wp_data = None


class _Env:
  def __init__(self, num_envs: int, nv: int):
    self.num_envs = num_envs
    self.device = torch.device("cpu")
    self.sim = _Sim(nv)


class _WarpArray(torch.Tensor):
  @staticmethod
  def __new__(cls, value):
    return torch.Tensor._make_subclass(cls, value, False)

  def assign(self, source):
    self.copy_(source)


@contextmanager
def _scoped_device(_device):
  yield


def _wp_zeros(shape, dtype=float):
  if dtype is torch.int32:
    return _WarpArray(torch.zeros(shape, dtype=torch.int32))
  if isinstance(shape, int):
    return _WarpArray(torch.zeros(shape, 3))
  return _WarpArray(torch.zeros(shape))


def _wp_from_torch(value, dtype=None):
  del dtype
  return value


def _warp_patches(jacobians, captured_points=None):
  call_idx = 0

  def jac(_model, _data, jacp, _jacr, point, _body):
    nonlocal call_idx
    jacp.copy_(jacobians[call_idx])
    if captured_points is not None:
      captured_points.append(point.clone())
    call_idx += 1

  import mujoco_warp
  import warp

  return (
    patch.object(mujoco_warp, "jac", jac),
    patch.object(warp, "zeros", _wp_zeros),
    patch.object(warp, "from_torch", _wp_from_torch),
    patch.object(warp, "to_torch", lambda value: value),
    patch.object(warp, "ScopedDevice", _scoped_device),
    patch.object(warp, "vec3", torch.float32),
    patch.object(warp, "int32", torch.int32),
  )


def _build_estimator(
  jacobians: list[torch.Tensor],
  body_names: tuple[str, ...] = ("hand",),
  joint_names: tuple[str, ...] = ("arm",),
  mappings: dict[str, tuple[str, ...]] | None = None,
  effort_limits: tuple[float | None, ...] = (10.0,),
  baseline: torch.Tensor | None = None,
  duplicate_actuator_joint: int | None = None,
):
  num_envs = jacobians[0].shape[0]
  nv = jacobians[0].shape[2]
  if baseline is None:
    baseline = torch.zeros(num_envs, len(joint_names))
  asset = _Entity(
    body_names, joint_names, effort_limits, baseline, duplicate_actuator_joint
  )
  env = _Env(num_envs, nv)
  if mappings is None:
    mappings = {body_names[0]: joint_names}
  patches = _warp_patches(jacobians)
  with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
    estimator = MaxForceEstimator(
      env, asset, body_names,
      constraint_joint_names_by_body=mappings,
      effort_limit_scale=0.9,
    )
  return estimator, env, asset


def _estimate(estimator, env, jacobians, offsets=None, captured_points=None):
  patches = _warp_patches(jacobians, captured_points)
  with patches[0], patches[2], patches[3], patches[4], patches[5]:
    return estimator.estimate(env, offsets)


def test_signed_bounds_include_baseline_effort_and_reserve():
  jac = torch.zeros(1, 3, 7)
  jac[0, :, 6] = torch.tensor([2.0, -1.0, 0.0])
  estimator, env, _ = _build_estimator([jac], baseline=torch.tensor([[3.0]]))

  force_mins, force_maxes = _estimate(estimator, env, [jac])

  torch.testing.assert_close(force_mins[0][0, :2], torch.tensor([-6.0, -6.0]))
  torch.testing.assert_close(force_maxes[0][0, :2], torch.tensor([3.0, 12.0]))
  assert force_mins[0][0, 2].isneginf()
  assert force_maxes[0][0, 2].isposinf()


def test_dirichlet_scaled_force_respects_all_joint_limits():
  jac = torch.zeros(1, 3, 8)
  jac[0, :, 6:] = torch.tensor([[2.0, -1.0], [1.0, 3.0], [-2.0, 2.0]])
  baseline = torch.tensor([[2.0, -1.0]])
  estimator, env, _ = _build_estimator(
    [jac], joint_names=("a", "b"), effort_limits=(10.0, 20.0), baseline=baseline
  )
  force_mins, force_maxes = _estimate(estimator, env, [jac])
  alpha = torch.tensor([0.2, 0.3, 0.5])
  force = alpha * 0.8 * force_maxes[0][0]

  resulting_effort = baseline[0] + force @ jac[0, :, 6:]

  assert torch.all(resulting_effort.abs() <= torch.tensor([9.0, 18.0]) + 1e-6)


def test_each_end_effector_uses_only_its_arm():
  left_jac = torch.zeros(1, 3, 9)
  right_jac = torch.zeros(1, 3, 9)
  left_jac[0, 0, 6] = 1.0
  left_jac[0, 0, 8] = 100.0
  right_jac[0, 0, 7] = 2.0
  right_jac[0, 0, 8] = 100.0
  mappings = {"left_hand": ("left_arm",), "right_hand": ("right_arm",)}
  estimator, env, _ = _build_estimator(
    [left_jac, right_jac],
    body_names=("left_hand", "right_hand"),
    joint_names=("left_arm", "right_arm", "weak_torso"),
    mappings=mappings,
    effort_limits=(10.0, 20.0, 0.1),
  )

  mins, maxes = _estimate(estimator, env, [left_jac, right_jac])

  torch.testing.assert_close(mins[0][0, 0], torch.tensor(-9.0))
  torch.testing.assert_close(maxes[0][0, 0], torch.tensor(9.0))
  torch.testing.assert_close(mins[1][0, 0], torch.tensor(-9.0))
  torch.testing.assert_close(maxes[1][0, 0], torch.tensor(9.0))


def test_infeasible_baseline_disables_force_for_environment():
  jac = torch.ones(2, 3, 7)
  estimator, env, _ = _build_estimator(
    [jac], baseline=torch.tensor([[9.1], [8.9]])
  )

  mins, maxes = _estimate(estimator, env, [jac])

  torch.testing.assert_close(mins[0][0], torch.zeros(3))
  torch.testing.assert_close(maxes[0][0], torch.zeros(3))
  assert (maxes[0][1] > 0.0).all()


def test_jacobian_uses_world_point_from_body_offset():
  jac = torch.zeros(1, 3, 7)
  jac[0, 0, 6] = 1.0
  estimator, env, asset = _build_estimator([jac])
  asset.data.body_com_pos_w[0, 0] = torch.tensor([1.0, 2.0, 3.0])
  offsets = torch.tensor([[[0.2, -0.1, 0.3]]])
  captured_points = []

  _estimate(estimator, env, [jac], offsets, captured_points)

  torch.testing.assert_close(
    captured_points[0][0], torch.tensor([1.2, 1.9, 3.3])
  )


def test_hard_caps_are_intersected_instead_of_clamped_outside_interval():
  mins = [torch.tensor([[-2.0, -float("inf"), -1.0]])]
  maxes = [torch.tensor([[3.0, float("inf"), 1.0]])]

  _intersect_force_bounds(
    mins, maxes,
    {"x": (5.0, 6.0), "y": (-4.0, 4.0), "z": (-0.5, 0.5)},
    0.5,
  )

  torch.testing.assert_close(mins[0], torch.tensor([[0.0, -2.0, -0.25]]))
  torch.testing.assert_close(maxes[0], torch.tensor([[0.0, 2.0, 0.25]]))


def test_multi_ee_rejects_legacy_flat_joint_list():
  jac = torch.zeros(1, 3, 7)
  asset = _Entity(("left", "right"), ("arm",), (10.0,), torch.zeros(1, 1))
  env = _Env(1, 7)
  patches = _warp_patches([jac, jac])
  with patches[1], patches[4], patches[5], patches[6], pytest.raises(
    ValueError, match="only supported for one end effector"
  ):
    MaxForceEstimator(env, asset, ("left", "right"), constraint_joint_names=("arm",))


@pytest.mark.parametrize("duplicate", [False, True])
def test_invalid_actuator_mapping_is_rejected(duplicate):
  jac = torch.zeros(1, 3, 7)
  effort_limits = () if not duplicate else (10.0,)
  if not duplicate:
    asset = _Entity(("hand",), ("arm",), (10.0,), torch.zeros(1, 1))
    asset.actuators.clear()
  else:
    asset = _Entity(
      ("hand",), ("arm",), effort_limits, torch.zeros(1, 1),
      duplicate_actuator_joint=0,
    )
  env = _Env(1, 7)
  patches = _warp_patches([jac])
  with patches[1], patches[4], patches[5], patches[6], pytest.raises(
    ValueError, match="exactly one actuator"
  ):
    MaxForceEstimator(
      env, asset, ("hand",), constraint_joint_names_by_body={"hand": ("arm",)}
    )


def test_offset_shape_is_validated():
  jac = torch.zeros(1, 3, 7)
  estimator, env, _ = _build_estimator([jac])
  with pytest.raises(ValueError, match="must have shape"):
    _estimate(estimator, env, [jac], torch.zeros(1, 3))
