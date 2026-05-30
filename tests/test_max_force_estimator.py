"""Tests for MaxForceEstimator Jacobian-based force estimation.

Mocks warp/mjwarp GPU dependencies to test the algorithm on CPU with
known Jacobian values.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.tasks.locomanipulation.mdp.events import MaxForceEstimator


# ── Mock objects ──────────────────────────────────────────────────────────────


@dataclass
class _MockActuatorCfg:
  effort_limit: float


@dataclass
class _MockActuator:
  target_ids: torch.Tensor
  cfg: _MockActuatorCfg


class _MockBodyComPosW:
  """Allows slice indexing like asset.data.body_com_pos_w[:, local_id]."""

  def __init__(self, tensor: torch.Tensor):
    self._t = tensor

  def __getitem__(self, idx):
    return self._t[idx]


class _MockEntityData:
  def __init__(self, body_com_pos_w: torch.Tensor):
    self.body_com_pos_w = _MockBodyComPosW(body_com_pos_w)


class _MockIndexing:
  def __init__(self, body_ids: torch.Tensor, joint_v_adr: torch.Tensor):
    self.body_ids = body_ids
    self.joint_v_adr = joint_v_adr


class _MockEntity:
  def __init__(
    self,
    body_ids: torch.Tensor,
    joint_v_adr: torch.Tensor,
    actuators: list[_MockActuator],
    body_com_pos_w: torch.Tensor,
    joint_names: list[str],
    body_names: list[str],
  ):
    self._body_names = body_names
    self._joint_names = joint_names
    self.indexing = _MockIndexing(body_ids, joint_v_adr)
    self.actuators = actuators
    self.data = _MockEntityData(body_com_pos_w)

  def find_bodies(self, name: str):
    idx = self._body_names.index(name)
    return [idx], [name]

  def find_joints(self, names):
    if isinstance(names, str):
      names = [names]
    ids = []
    matched = []
    for name in names:
      for i, jn in enumerate(self._joint_names):
        if jn == name:
          ids.append(i)
          matched.append(jn)
    return ids, matched


class _MockMjModel:
  def __init__(self, nv: int):
    self.nv = nv


class _MockSim:
  def __init__(self, nv: int, wp_device: str = "cpu"):
    self.mj_model = _MockMjModel(nv)
    self.wp_device = wp_device
    self.wp_model = None
    self.wp_data = None


class _MockEnv:
  def __init__(self, num_envs: int, nv: int, device: str = "cpu"):
    self.num_envs = num_envs
    self.device = torch.device(device)
    self.sim = _MockSim(nv)


# ── Warp/mjwarp patches ──────────────────────────────────────────────────────


@contextmanager
def _noop_scoped_device(device: str):
  yield


def _make_jac_stub(jacobian: torch.Tensor):
  """Return a mjwarp.jac stub that writes `jacobian` into `jacp`.

  Args:
    jacobian: shape (nworld, 3, nv) — the Jacobian to inject.
  """

  def _jac(model, data, jacp, jacr, point, body):
    # jacp is a "warp tensor" (actually torch tensor from _wp_zeros).
    jacp[:] = jacobian

  return _jac


class _WarpArray(torch.Tensor):
  """Mock warp array that adds assign on top of torch.Tensor."""

  @staticmethod
  def __new__(cls, *args, **kwargs):
    return super().__new__(cls, *args, **kwargs)

  def assign(self, src):
    self.data.copy_(src)
    return self


def _wp_zeros(shape, dtype=float):
  if dtype is torch.int32:
    return _WarpArray(torch.zeros(shape, dtype=torch.int32))
  # Single int shape with float dtype → vec3 array → trailing dim of 3.
  if isinstance(shape, int):
    return _WarpArray(torch.zeros(shape, 3, dtype=torch.float32))
  # Tuple shape → regular multi-dim tensor.
  return _WarpArray(torch.zeros(shape, dtype=torch.float32))


def _wp_from_torch(tensor, dtype=None):
  return _WarpArray(tensor)


def _wp_to_torch(tensor):
  return tensor


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_estimator(
  num_envs: int,
  nv: int,
  body_names: list[str],
  joint_names: list[str],
  constraint_joint_names: tuple[str, ...],
  body_ids_tensor: torch.Tensor,
  joint_v_adr_tensor: torch.Tensor,
  actuators: list[_MockActuator],
  body_com_pos_w: torch.Tensor,
  jacobian: torch.Tensor,
) -> MaxForceEstimator:
  """Build a MaxForceEstimator with mocked dependencies."""
  env = _MockEnv(num_envs, nv)
  asset = _MockEntity(
    body_ids=body_ids_tensor,
    joint_v_adr=joint_v_adr_tensor,
    actuators=actuators,
    body_com_pos_w=body_com_pos_w,
    joint_names=joint_names,
    body_names=body_names,
  )

  import mujoco_warp as mjwarp_mod
  import warp as wp_mod

  with (
    patch.object(mjwarp_mod, "jac", _make_jac_stub(jacobian)),
    patch.object(wp_mod, "zeros", _wp_zeros),
    patch.object(wp_mod, "from_torch", _wp_from_torch),
    patch.object(wp_mod, "to_torch", _wp_to_torch),
    patch.object(wp_mod, "ScopedDevice", _noop_scoped_device),
    patch.object(wp_mod, "vec3", torch.float32),
    patch.object(wp_mod, "int32", torch.int32),
  ):
    estimator = MaxForceEstimator(
      env=env,
      asset=asset,
      ee_body_names=body_names,
      constraint_joint_names=constraint_joint_names,
    )
  # Store jacobian for estimate() calls.
  estimator._test_jacobian = jacobian
  return estimator


def _estimate(estimator: MaxForceEstimator, num_envs: int, nv: int):
  """Call estimate() with mocked mjwarp.jac."""
  env = _MockEnv(num_envs, nv)

  import mujoco_warp as mjwarp_mod
  import warp as wp_mod

  with (
    patch.object(mjwarp_mod, "jac", _make_jac_stub(estimator._test_jacobian)),
    patch.object(wp_mod, "from_torch", _wp_from_torch),
    patch.object(wp_mod, "to_torch", _wp_to_torch),
    patch.object(wp_mod, "ScopedDevice", _noop_scoped_device),
    patch.object(wp_mod, "vec3", torch.float32),
  ):
    return estimator.estimate(env)


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_single_joint_force_bounds():
  """Single arm joint, known Jacobian → verify exact force bounds."""
  num_envs = 1
  nv = 7  # 6 floating base + 1 joint
  eps = 1e-2

  # Joint 0 at DOF address 6 (after floating base).
  body_names = ["left_wrist"]
  joint_names = ["left_shoulder_pitch_joint"]
  constraint_joint_names = ("left_shoulder_pitch_joint",)
  body_ids = torch.tensor([0])
  joint_v_adr = torch.tensor([6])
  effort_limit = 10.0
  actuators = [_MockActuator(
    target_ids=torch.tensor([0]),
    cfg=_MockActuatorCfg(effort_limit=effort_limit),
  )]
  body_com_pos_w = torch.zeros(1, 1, 3)

  # Jacobian: J[xyz, joint] = [1.0, 0.5, 0.0]
  # F_max = effort / (|J| + eps)
  # x: 10 / (1.0 + 0.01) = 9.90099
  # y: 10 / (0.5 + 0.01) = 19.60784
  # z: 10 / (0.0 + 0.01) = 1000.0
  jacobian = torch.zeros(num_envs, 3, nv)
  jacobian[0, 0, 6] = 1.0
  jacobian[0, 1, 6] = 0.5
  jacobian[0, 2, 6] = 0.0

  estimator = _make_estimator(
    num_envs, nv, body_names, joint_names, constraint_joint_names,
    body_ids, joint_v_adr, actuators, body_com_pos_w, jacobian,
  )
  force_mins, force_maxes = _estimate(estimator, num_envs, nv)

  f_max = force_maxes[0]
  f_min = force_mins[0]

  expected_max = torch.tensor([[
    effort_limit / (1.0 + eps),
    effort_limit / (0.5 + eps),
    effort_limit / (0.0 + eps),
  ]])
  expected_min = -expected_max

  assert torch.allclose(f_max, expected_max, atol=1e-4), f"got {f_max}"
  assert torch.allclose(f_min, expected_min, atol=1e-4), f"got {f_min}"


def test_multiple_joints_most_restrictive_wins():
  """Two joints — the tighter constraint dominates per axis."""
  num_envs = 1
  nv = 8  # 6 floating base + 2 joints
  eps = 1e-2

  body_names = ["left_wrist"]
  joint_names = ["joint_a", "joint_b"]
  constraint_joint_names = ("joint_a", "joint_b")
  body_ids = torch.tensor([0])
  joint_v_adr = torch.tensor([6, 7])
  actuators = [
    _MockActuator(
      target_ids=torch.tensor([0]),
      cfg=_MockActuatorCfg(effort_limit=10.0),
    ),
    _MockActuator(
      target_ids=torch.tensor([1]),
      cfg=_MockActuatorCfg(effort_limit=5.0),
    ),
  ]
  body_com_pos_w = torch.zeros(1, 1, 3)

  # J = [[1.0, 0.0],   ← x-axis: joint_a has J=1, joint_b has J=0
  #      [0.0, 1.0]]   ← y-axis: joint_a has J=0, joint_b has J=1
  jacobian = torch.zeros(num_envs, 3, nv)
  jacobian[0, 0, 6] = 1.0  # x, joint_a
  jacobian[0, 1, 7] = 1.0  # y, joint_b

  estimator = _make_estimator(
    num_envs, nv, body_names, joint_names, constraint_joint_names,
    body_ids, joint_v_adr, actuators, body_com_pos_w, jacobian,
  )
  force_mins, force_maxes = _estimate(estimator, num_envs, nv)

  f_max = force_maxes[0]

  # x-axis: min(10/(1+0.01), 5/(0+0.01)) = min(9.9, 500) = 9.9
  # y-axis: min(10/(0+0.01), 5/(1+0.01)) = min(1000, 4.95) = 4.95
  # z-axis: min(10/(0+0.01), 5/(0+0.01)) = min(1000, 500) = 500
  expected = torch.tensor([[
    10.0 / (1.0 + eps),
    5.0 / (1.0 + eps),
    5.0 / (0.0 + eps),
  ]])
  assert torch.allclose(f_max, expected, atol=1e-4), f"got {f_max}"


def test_symmetric_effort_gives_symmetric_bounds():
  """With symmetric effort limits, f_min == -f_max."""
  num_envs = 2
  nv = 7

  body_names = ["ee"]
  joint_names = ["joint_0"]
  constraint_joint_names = ("joint_0",)
  body_ids = torch.tensor([0])
  joint_v_adr = torch.tensor([6])
  effort = 20.0
  actuators = [_MockActuator(
    target_ids=torch.tensor([0]),
    cfg=_MockActuatorCfg(effort_limit=effort),
  )]
  body_com_pos_w = torch.zeros(num_envs, 1, 3)

  jacobian = torch.zeros(num_envs, 3, nv)
  jacobian[:, 0, 6] = 2.0
  jacobian[:, 1, 6] = -3.0
  jacobian[:, 2, 6] = 0.5

  estimator = _make_estimator(
    num_envs, nv, body_names, joint_names, constraint_joint_names,
    body_ids, joint_v_adr, actuators, body_com_pos_w, jacobian,
  )
  force_mins, force_maxes = _estimate(estimator, num_envs, nv)

  for i in range(len(force_mins)):
    assert torch.allclose(force_mins[i], -force_maxes[i], atol=1e-6)


def test_two_end_effectors_independent():
  """Two EEs with different Jacobians → independent force bounds."""
  num_envs = 1
  nv = 8

  body_names = ["left_wrist", "right_wrist"]
  joint_names = ["left_joint", "right_joint"]
  constraint_joint_names = ("left_joint", "right_joint")
  body_ids = torch.tensor([0, 1])
  joint_v_adr = torch.tensor([6, 7])
  actuators = [
    _MockActuator(
      target_ids=torch.tensor([0]),
      cfg=_MockActuatorCfg(effort_limit=10.0),
    ),
    _MockActuator(
      target_ids=torch.tensor([1]),
      cfg=_MockActuatorCfg(effort_limit=20.0),
    ),
  ]
  body_com_pos_w = torch.zeros(1, 2, 3)

  # Left EE: only left_joint matters (J_left=1.0, J_right=0.0)
  # Right EE: only right_joint matters (J_left=0.0, J_right=2.0)
  jacobian_left = torch.zeros(num_envs, 3, nv)
  jacobian_left[0, 0, 6] = 1.0

  jacobian_right = torch.zeros(num_envs, 3, nv)
  jacobian_right[0, 0, 7] = 2.0

  # Concatenate: the estimator calls jac() per EE, so we need to provide
  # the correct jacobian for each call. We'll use a list and pop.
  jacobians = [jacobian_left, jacobian_right]
  call_count = [0]

  def _multi_jac(model, data, jacp, jacr, point, body):
    jacp[:] = jacobians[call_count[0]]
    call_count[0] += 1

  env = _MockEnv(num_envs, nv)
  asset = _MockEntity(
    body_ids=body_ids,
    joint_v_adr=joint_v_adr,
    actuators=actuators,
    body_com_pos_w=body_com_pos_w,
    joint_names=joint_names,
    body_names=body_names,
  )

  import mujoco_warp as mjwarp_mod
  import warp as wp_mod

  with (
    patch.object(mjwarp_mod, "jac", _multi_jac),
    patch.object(wp_mod, "zeros", _wp_zeros),
    patch.object(wp_mod, "from_torch", _wp_from_torch),
    patch.object(wp_mod, "to_torch", _wp_to_torch),
    patch.object(wp_mod, "ScopedDevice", _noop_scoped_device),
    patch.object(wp_mod, "vec3", torch.float32),
    patch.object(wp_mod, "int32", torch.int32),
  ):
    estimator = MaxForceEstimator(
      env=env, asset=asset,
      ee_body_names=body_names, constraint_joint_names=constraint_joint_names,
    )

  with (
    patch.object(mjwarp_mod, "jac", _multi_jac),
    patch.object(wp_mod, "from_torch", _wp_from_torch),
    patch.object(wp_mod, "to_torch", _wp_to_torch),
    patch.object(wp_mod, "ScopedDevice", _noop_scoped_device),
    patch.object(wp_mod, "vec3", torch.float32),
  ):
    call_count[0] = 0
    force_mins, force_maxes = estimator.estimate(env)

  eps = 1e-2
  # Left EE (jacobian_left): constraint DOFs [6,7], J=[1.0, 0.0]
  # f_max = min(10/(|1|+eps), 10/(|0|+eps)) = [9.9, 1000, 1000]
  expected_left = torch.tensor([[
    10.0 / (1.0 + eps), 10.0 / eps, 10.0 / eps,
  ]])
  # Right EE (jacobian_right): constraint DOFs [6,7], J=[0.0, 2.0]
  # f_max = min(10/(|0|+eps), 20/(|2|+eps)) = [9.9, 500, 500]
  expected_right = torch.tensor([[
    min(10.0 / eps, 20.0 / (2.0 + eps)),
    min(10.0 / eps, 20.0 / eps),
    min(10.0 / eps, 20.0 / eps),
  ]])

  assert torch.allclose(force_maxes[0], expected_left, atol=1e-3)
  assert torch.allclose(force_maxes[1], expected_right, atol=1e-3)


def test_zero_jacobian_gives_large_finite_bound():
  """When J[axis, i] ≈ 0, force limit → effort/eps (large but finite)."""
  num_envs = 1
  nv = 7
  eps = 1e-2

  body_names = ["ee"]
  joint_names = ["joint_0"]
  constraint_joint_names = ("joint_0",)
  body_ids = torch.tensor([0])
  joint_v_adr = torch.tensor([6])
  effort = 50.0
  actuators = [_MockActuator(
    target_ids=torch.tensor([0]),
    cfg=_MockActuatorCfg(effort_limit=effort),
  )]
  body_com_pos_w = torch.zeros(1, 1, 3)

  # J = [0, 0, 0] — all zero Jacobian entries.
  jacobian = torch.zeros(num_envs, 3, nv)

  estimator = _make_estimator(
    num_envs, nv, body_names, joint_names, constraint_joint_names,
    body_ids, joint_v_adr, actuators, body_com_pos_w, jacobian,
  )
  force_mins, force_maxes = _estimate(estimator, num_envs, nv)

  expected = effort / eps  # 50 / 0.01 = 5000
  assert torch.allclose(force_maxes[0], torch.full((1, 3), expected), atol=1e-2)
  assert torch.allclose(force_mins[0], torch.full((1, 3), -expected), atol=1e-2)


def test_g1_play_mode_pose_force_bounds():
  """Compute max forces for G1 at the play-mode pose (all upper body joints = 0).

  Uses the real MuJoCo model to compute the Jacobian, then applies the same
  algorithm as MaxForceEstimator. Prints results for inspection.
  """
  import mujoco

  from src.assets.robots.unitree_g1.g1_constants import G1_XML, get_spec

  spec = get_spec()
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)

  # Set all joint positions to zero (play mode pose).
  for i in range(model.njnt):
    jnt = model.joint(i)
    if jnt.type[0] != mujoco.mjtJoint.mjJNT_FREE:
      qadr = jnt.qposadr[0]
      data.qpos[qadr] = 0.0

  # Set floating base height.
  data.qpos[2] = 0.8

  mujoco.mj_forward(model, data)

  # Find wrist body IDs.
  left_wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
  right_wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
  assert left_wrist_id >= 0, "left_wrist_yaw_link not found"
  assert right_wrist_id >= 0, "right_wrist_yaw_link not found"

  # Find arm joint DOF addresses.
  arm_joint_patterns = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
  ]
  arm_dof_adr = []
  for name in arm_joint_patterns:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert jid >= 0, f"Joint {name} not found"
    arm_dof_adr.append(model.jnt_dofadr[jid])

  # Get effort limits from MuJoCo actuator force range.
  effort_limits = {}
  for name in arm_joint_patterns:
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_actuator")
    if aid >= 0:
      effort_limits[name] = float(model.actuatorforcerange[aid, 1])
    else:
      # Try without _actuator suffix.
      effort_limits[name] = 25.0  # Default ACTUATOR_5020

  eps = 1e-2

  results = {}
  for ee_name, body_id in [("left_wrist", left_wrist_id), ("right_wrist", right_wrist_id)]:
    # Compute Jacobian at body CoM.
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    body_com = data.xipos[body_id]
    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)

    # Slice arm DOF columns.
    jacp_arm = jacp[:, arm_dof_adr]

    # Compute force bounds per joint, per axis.
    f_max_all = np.zeros((3, len(arm_dof_adr)))
    f_min_all = np.zeros((3, len(arm_dof_adr)))
    for i, (dof_adr, jname) in enumerate(zip(arm_dof_adr, arm_joint_patterns)):
      el = effort_limits[jname]
      for axis in range(3):
        j_val = abs(jacp_arm[axis, i])
        f_max_all[axis, i] = el / (j_val + eps)
        f_min_all[axis, i] = -el / (j_val + eps)

    # Most restrictive across joints.
    f_max = f_max_all.min(axis=1)
    f_min = f_min_all.max(axis=1)

    results[ee_name] = {
      "f_min": f_min.tolist(),
      "f_max": f_max.tolist(),
      "body_com": body_com.tolist(),
    }

  # Print results.
  import json
  print("\n=== G1 Play Mode Pose — Max Force Bounds ===")
  print(json.dumps(results, indent=2))

  # Basic sanity checks.
  for ee_name, r in results.items():
    f_max = r["f_max"]
    f_min = r["f_min"]
    for axis in range(3):
      assert f_max[axis] > 0, f"{ee_name} axis {axis}: f_max should be positive"
      assert f_min[axis] < 0, f"{ee_name} axis {axis}: f_min should be negative"
      assert abs(f_min[axis]) == pytest.approx(f_max[axis], rel=1e-3), \
        f"{ee_name} axis {axis}: f_min should be -f_max (symmetric effort)"


def test_g1_23dof_play_mode_pose_force_bounds():
  """Compute max forces for G1-23DOF at the play-mode pose (all upper body joints = 0).

  Uses the real MuJoCo model to compute the Jacobian, then applies the same
  algorithm as MaxForceEstimator. Prints results for inspection.
  """
  import mujoco

  from src.assets.robots.unitree_g1.g1_23dof_constants import G1_23DOF_XML, get_spec

  spec = get_spec()
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)

  # Set all joint positions to zero (play mode pose).
  for i in range(model.njnt):
    jnt = model.joint(i)
    if jnt.type[0] != mujoco.mjtJoint.mjJNT_FREE:
      qadr = jnt.qposadr[0]
      data.qpos[qadr] = 0.0

  # Set floating base height.
  data.qpos[2] = 0.8

  mujoco.mj_forward(model, data)

  # Find wrist body IDs (23-DOF uses wrist_roll_rubber_hand, not wrist_yaw_link).
  left_wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_roll_rubber_hand")
  right_wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_roll_rubber_hand")
  assert left_wrist_id >= 0, "left_wrist_roll_rubber_hand not found"
  assert right_wrist_id >= 0, "right_wrist_roll_rubber_hand not found"

  # Find arm joint DOF addresses (23-DOF omits wrist_pitch and wrist_yaw).
  arm_joint_patterns = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint",
  ]
  arm_dof_adr = []
  for name in arm_joint_patterns:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert jid >= 0, f"Joint {name} not found"
    arm_dof_adr.append(model.jnt_dofadr[jid])

  # Get effort limits from MuJoCo actuator force range.
  effort_limits = {}
  for name in arm_joint_patterns:
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_actuator")
    if aid >= 0:
      effort_limits[name] = float(model.actuatorforcerange[aid, 1])
    else:
      effort_limits[name] = 25.0  # Default ACTUATOR_5020

  eps = 1e-2

  results = {}
  for ee_name, body_id in [("left_wrist", left_wrist_id), ("right_wrist", right_wrist_id)]:
    # Compute Jacobian at body CoM.
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    body_com = data.xipos[body_id]
    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)

    # Slice arm DOF columns.
    jacp_arm = jacp[:, arm_dof_adr]

    # Compute force bounds per joint, per axis.
    f_max_all = np.zeros((3, len(arm_dof_adr)))
    f_min_all = np.zeros((3, len(arm_dof_adr)))
    for i, (dof_adr, jname) in enumerate(zip(arm_dof_adr, arm_joint_patterns)):
      el = effort_limits[jname]
      for axis in range(3):
        j_val = abs(jacp_arm[axis, i])
        f_max_all[axis, i] = el / (j_val + eps)
        f_min_all[axis, i] = -el / (j_val + eps)

    # Most restrictive across joints.
    f_max = f_max_all.min(axis=1)
    f_min = f_min_all.max(axis=1)

    results[ee_name] = {
      "f_min": f_min.tolist(),
      "f_max": f_max.tolist(),
      "body_com": body_com.tolist(),
    }

  # Print results.
  import json
  print("\n=== G1-23DOF Play Mode Pose — Max Force Bounds ===")
  print(json.dumps(results, indent=2))

  # Basic sanity checks.
  for ee_name, r in results.items():
    f_max = r["f_max"]
    f_min = r["f_min"]
    for axis in range(3):
      assert f_max[axis] > 0, f"{ee_name} axis {axis}: f_max should be positive"
      assert f_min[axis] < 0, f"{ee_name} axis {axis}: f_min should be negative"
      assert abs(f_min[axis]) == pytest.approx(f_max[axis], rel=1e-3), \
        f"{ee_name} axis {axis}: f_min should be -f_max (symmetric effort)"


if __name__ == "__main__":
  import pytest
  pytest.main([__file__, "-v"])
