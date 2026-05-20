"""Tests for symmetric data augmentation.

Uses mock env/observation_manager to test G1Symmetry and
g1_locomanipulation_symmetry without MuJoCo.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import pytest
import torch
from tensordict import TensorDict

from src.tasks.locomanipulation.mdp.symmetry import (
  G1Symmetry,
  _JOINT_SWAP_PARTNERS,
  _NegateIndices,
  _SIGN_FLIP_JOINTS,
  _SwapAndFlipJoints,
  _SwapFootForces,
  _SwapFootValues,
  _SwapWristForces,
  g1_locomanipulation_symmetry,
)


# ── Mock objects ──────────────────────────────────────────────────────────────

@dataclass
class _MockTermCfg:
  pass


class _MockGroupCfg:
  def __init__(self, terms: dict[str, _MockTermCfg]):
    self.terms = OrderedDict(terms)


class _MockObsManager:
  def __init__(
    self,
    group_term_names: dict[str, list[str]],
    group_term_dims: dict[str, list[int]],
    group_cfgs: dict[str, _MockGroupCfg],
  ):
    self._group_obs_term_names = group_term_names
    self._group_obs_term_dim = {
      g: [(d,) for d in dims] for g, dims in group_term_dims.items()
    }
    self.cfg = group_cfgs


class _MockEnv:
  def __init__(self, obs_manager: _MockObsManager, device: str = "cpu"):
    self.device = torch.device(device)
    self.observation_manager = obs_manager
    self._self = self  # for unwrapped property

  @property
  def unwrapped(self):
    return self


# ── Actor observation layout ─────────────────────────────────────────────────

_ACTOR_TERMS = ["base_ang_vel", "projected_gravity", "command", "phase",
                "joint_pos", "joint_vel", "actions"]
_ACTOR_DIMS = [3, 3, 3, 2, 29, 29, 12]  # total = 81

_CRITIC_EXTRA_TERMS = ["base_lin_vel", "foot_height", "foot_air_time",
                       "foot_contact", "foot_contact_forces", "wrist_force"]
_CRITIC_EXTRA_DIMS = [3, 2, 2, 2, 6, 6]

_CRITIC_TERMS = _ACTOR_TERMS + _CRITIC_EXTRA_TERMS
_CRITIC_DIMS = _ACTOR_DIMS + _CRITIC_EXTRA_DIMS  # total = 102


def _make_mock_env() -> _MockEnv:
  actor_cfg = _MockGroupCfg({t: _MockTermCfg() for t in _ACTOR_TERMS})
  critic_cfg = _MockGroupCfg({t: _MockTermCfg() for t in _CRITIC_TERMS})
  obs_mgr = _MockObsManager(
    group_term_names={"actor": list(_ACTOR_TERMS), "critic": list(_CRITIC_TERMS)},
    group_term_dims={"actor": list(_ACTOR_DIMS), "critic": list(_CRITIC_DIMS)},
    group_cfgs={"actor": actor_cfg, "critic": critic_cfg},
  )
  return _MockEnv(obs_mgr)


def _make_symmetry() -> G1Symmetry:
  return G1Symmetry(_make_mock_env())


def _make_actor_obs(batch: int = 4) -> torch.Tensor:
  """Create actor observation tensor with known per-joint values."""
  return torch.randn(batch, sum(_ACTOR_DIMS))


def _make_critic_obs(batch: int = 4) -> torch.Tensor:
  return torch.randn(batch, sum(_CRITIC_DIMS))


def _make_actions(batch: int = 4) -> torch.Tensor:
  return torch.randn(batch, 12)


def _make_obs_td(batch: int = 4) -> TensorDict:
  return TensorDict(
    {"actor": _make_actor_obs(batch), "critic": _make_critic_obs(batch)},
    batch_size=[batch],
  )


# ── Tests: Joint swap and sign mask ──────────────────────────────────────────

class TestJointSwapAndSign:
  def test_swap_partners_symmetric(self):
    """Every swap pair should be bidirectional."""
    for src, dst in _JOINT_SWAP_PARTNERS.items():
      assert _JOINT_SWAP_PARTNERS[dst] == src, f"({src}, {dst}) not symmetric"

  def test_all_29_joints_covered(self):
    assert set(_JOINT_SWAP_PARTNERS.keys()) == set(range(29))

  def test_midline_identity(self):
    for j in (12, 13, 14):
      assert _JOINT_SWAP_PARTNERS[j] == j

  def test_sign_flip_contains_roll_yaw_only(self):
    """Sign-flip set should include all roll/yaw joints, no pitch joints."""
    roll_yaw_names = {
      "hip_roll", "hip_yaw", "ankle_roll", "waist_roll", "waist_yaw",
      "shoulder_roll", "shoulder_yaw", "wrist_roll", "wrist_yaw",
    }
    # Joint index → name mapping (G1 29-DOF).
    joint_names = [
      "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
      "left_ankle_pitch", "left_ankle_roll",
      "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
      "right_ankle_pitch", "right_ankle_roll",
      "waist_yaw", "waist_roll", "waist_pitch",
      "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
      "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
      "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
      "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
    ]
    for idx, name in enumerate(joint_names):
      suffix = name.split("_", 1)[-1] if "_" in name else name
      # Handle left_/right_ prefix
      parts = name.split("_")
      if parts[0] in ("left", "right"):
        suffix = "_".join(parts[1:])
      should_flip = any(s in suffix for s in ("roll", "yaw"))
      assert (idx in _SIGN_FLIP_JOINTS) == should_flip, (
        f"Joint {idx} ({name}): expected flip={should_flip}, got {idx in _SIGN_FLIP_JOINTS}"
      )


# ── Tests: Action mirroring ──────────────────────────────────────────────────

class TestActionMirror:
  def test_swap_left_right(self):
    """Left leg actions should swap with right leg actions."""
    sym = _make_symmetry()
    actions = torch.zeros(1, 12)
    # Set left leg to values 0-5, right leg to values 6-11.
    actions[0, :6] = torch.arange(6, dtype=torch.float)
    actions[0, 6:] = torch.arange(6, 12, dtype=torch.float)

    mirrored = sym.mirror_actions(actions)

    # Left leg should now have right leg values (and vice versa).
    # But sign-flip joints get negated.
    for i in range(6):
      j = i + 6  # partner
      expected = actions[0, j].item()
      if i in _SIGN_FLIP_JOINTS:
        expected = -expected
      assert mirrored[0, i].item() == pytest.approx(expected), (
        f"Action {i}: expected {expected}, got {mirrored[0, i].item()}"
      )

  def test_double_mirror_identity(self):
    """mirror(mirror(actions)) == actions."""
    sym = _make_symmetry()
    actions = _make_actions(batch=8)
    mirrored = sym.mirror_actions(actions)
    double_mirrored = sym.mirror_actions(mirrored)
    assert torch.allclose(actions, double_mirrored, atol=1e-6)


# ── Tests: Individual observation term mirrors ───────────────────────────────

class TestTermMirrors:
  def test_base_ang_vel(self):
    m = _NegateIndices((0, 2))
    x = torch.tensor([[1.0, 2.0, 3.0]])
    m.apply(x, None)
    assert torch.allclose(x, torch.tensor([[-1.0, 2.0, -3.0]]))

  def test_projected_gravity(self):
    m = _NegateIndices((1,))
    x = torch.tensor([[1.0, 2.0, 3.0]])
    m.apply(x, None)
    assert torch.allclose(x, torch.tensor([[1.0, -2.0, 3.0]]))

  def test_command(self):
    m = _NegateIndices((1, 2))
    x = torch.tensor([[0.5, 1.0, -0.3]])
    m.apply(x, None)
    assert torch.allclose(x, torch.tensor([[0.5, -1.0, 0.3]]))

  def test_phase(self):
    m = _NegateIndices((0, 1))
    x = torch.tensor([[0.707, 0.707]])
    m.apply(x, None)
    assert torch.allclose(x, torch.tensor([[-0.707, -0.707]]))

  def test_base_lin_vel(self):
    m = _NegateIndices((1,))
    x = torch.tensor([[1.0, -2.0, 0.5]])
    m.apply(x, None)
    assert torch.allclose(x, torch.tensor([[1.0, 2.0, 0.5]]))

  def test_swap_foot_values(self):
    m = _SwapFootValues()
    x = torch.tensor([[0.1, 0.2]])
    m.apply(x, None)
    assert torch.allclose(x, torch.tensor([[0.2, 0.1]]))

  def test_swap_foot_forces(self):
    m = _SwapFootForces()
    # [left_fx, left_fy, left_fz, right_fx, right_fy, right_fz]
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    m.apply(x, None)
    # Expected: right goes to left (negate fy), left goes to right (negate fy)
    assert torch.allclose(x, torch.tensor([[4.0, -5.0, 6.0, 1.0, -2.0, 3.0]]))

  def test_swap_wrist_forces(self):
    m = _SwapWristForces()
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    m.apply(x, None)
    assert torch.allclose(x, torch.tensor([[4.0, -5.0, 6.0, 1.0, -2.0, 3.0]]))


# ── Tests: Full observation mirroring ────────────────────────────────────────

class TestObsMirror:
  def test_actor_obs_shape_preserved(self):
    sym = _make_symmetry()
    obs = _make_actor_obs(batch=4)
    mirrored = sym.mirror_obs(obs, "actor")
    assert mirrored.shape == obs.shape

  def test_critic_obs_shape_preserved(self):
    sym = _make_symmetry()
    obs = _make_critic_obs(batch=4)
    mirrored = sym.mirror_obs(obs, "critic")
    assert mirrored.shape == obs.shape

  def test_joint_pos_segment_swap(self):
    """Verify joint_pos segment is correctly swapped."""
    sym = _make_symmetry()
    obs = torch.zeros(1, sum(_ACTOR_DIMS))
    # Find joint_pos offset: base_ang_vel(3) + projected_gravity(3) + command(3) + phase(2) = 11
    jp_offset = 11
    # Set left leg joints to 100+index, right leg to 200+index.
    for i in range(6):
      obs[0, jp_offset + i] = 100 + i       # left
      obs[0, jp_offset + 6 + i] = 200 + i   # right
    # Set waist to 300+index.
    for i in range(3):
      obs[0, jp_offset + 12 + i] = 300 + i

    mirrored = sym.mirror_obs(obs, "actor")

    # Check left-right swap for leg joints.
    for i in range(6):
      j = i + 6
      expected = obs[0, jp_offset + j].item()
      if i in _SIGN_FLIP_JOINTS:
        expected = -expected
      assert mirrored[0, jp_offset + i].item() == pytest.approx(expected), (
        f"joint_pos[{i}]"
      )

    # Check waist stays in place (yaw and roll negated).
    for i in range(3):
      expected = obs[0, jp_offset + 12 + i].item()
      if (12 + i) in _SIGN_FLIP_JOINTS:
        expected = -expected
      assert mirrored[0, jp_offset + 12 + i].item() == pytest.approx(expected), (
        f"waist[{12 + i}]"
      )

  def test_double_mirror_actor_identity(self):
    """mirror(mirror(obs)) == obs for actor group."""
    sym = _make_symmetry()
    obs = _make_actor_obs(batch=8)
    mirrored = sym.mirror_obs(obs, "actor")
    double_mirrored = sym.mirror_obs(mirrored, "actor")
    assert torch.allclose(obs, double_mirrored, atol=1e-6)

  def test_double_mirror_critic_identity(self):
    """mirror(mirror(obs)) == obs for critic group."""
    sym = _make_symmetry()
    obs = _make_critic_obs(batch=8)
    mirrored = sym.mirror_obs(obs, "critic")
    double_mirrored = sym.mirror_obs(mirrored, "critic")
    assert torch.allclose(obs, double_mirrored, atol=1e-6)


# ── Tests: g1_locomanipulation_symmetry function ─────────────────────────────

class TestAugmentationFunction:
  def test_batch_doubling_obs_and_actions(self):
    """With both obs and actions, output batch should be 2× input."""
    env = _make_mock_env()
    batch = 4
    obs = _make_obs_td(batch)
    actions = _make_actions(batch)

    aug_obs, aug_actions = g1_locomanipulation_symmetry(env, obs, actions)

    assert aug_obs is not None
    assert aug_actions is not None
    assert aug_obs.batch_size[0] == batch * 2
    assert aug_actions.shape[0] == batch * 2

  def test_batch_doubling_obs_only(self):
    """With obs only, output obs batch should be 2× input, actions None."""
    env = _make_mock_env()
    batch = 4
    obs = _make_obs_td(batch)

    aug_obs, aug_actions = g1_locomanipulation_symmetry(env, obs, None)

    assert aug_obs is not None
    assert aug_actions is None
    assert aug_obs.batch_size[0] == batch * 2

  def test_batch_doubling_actions_only(self):
    """With actions only, output actions batch should be 2× input, obs None."""
    env = _make_mock_env()
    batch = 4
    actions = _make_actions(batch)

    aug_obs, aug_actions = g1_locomanipulation_symmetry(env, None, actions)

    assert aug_obs is None
    assert aug_actions is not None
    assert aug_actions.shape[0] == batch * 2

  def test_original_preserved_in_first_half(self):
    """First half of augmented output should equal original input."""
    env = _make_mock_env()
    batch = 4
    obs = _make_obs_td(batch)
    actions = _make_actions(batch)

    aug_obs, aug_actions = g1_locomanipulation_symmetry(env, obs, actions)

    # Actions: first half = original.
    assert torch.allclose(aug_actions[:batch], actions, atol=1e-6)

    # Obs: first half = original per group.
    for key in obs.keys():
      assert torch.allclose(aug_obs[key][:batch], obs[key], atol=1e-6)

  def test_mirrored_differs_from_original(self):
    """Second half (mirrored) should differ from first half (original)."""
    env = _make_mock_env()
    batch = 4
    obs = _make_obs_td(batch)
    actions = _make_actions(batch)

    aug_obs, aug_actions = g1_locomanipulation_symmetry(env, obs, actions)

    # Actions should differ (unless all zeros).
    assert not torch.allclose(aug_actions[:batch], aug_actions[batch:], atol=1e-6)

  def test_double_augment_identity(self):
    """Applying augmentation twice should give original in first quarter."""
    env = _make_mock_env()
    batch = 4
    obs = _make_obs_td(batch)
    actions = _make_actions(batch)

    aug_obs, aug_actions = g1_locomanipulation_symmetry(env, obs, actions)
    aug_obs2, aug_actions2 = g1_locomanipulation_symmetry(env, aug_obs, aug_actions)

    # First quarter of second augmentation = first half of first augmentation = original.
    for key in obs.keys():
      assert torch.allclose(aug_obs2[key][:batch], obs[key], atol=1e-6)
    assert torch.allclose(aug_actions2[:batch], actions, atol=1e-6)
