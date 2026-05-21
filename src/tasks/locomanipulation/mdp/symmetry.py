"""Left-right symmetric data augmentation for G1 locomanipulation.

Mirrors observations and actions across the sagittal plane (x-z plane, flip y-axis)
to double effective training data. Uses rsl_rl's built-in symmetry_cfg in PPO.

Coordinate convention (MuJoCo body frame): x = forward, y = left, z = up.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper

# G1 29-DOF joint ordering (from MJCF XML).
# Indices 0-5: left leg, 6-11: right leg, 12-14: waist, 15-21: left arm, 22-28: right arm.
_JOINT_SWAP_PARTNERS = {
  # Left leg <-> Right leg
  0: 6, 1: 7, 2: 8, 3: 9, 4: 10, 5: 11,
  6: 0, 7: 1, 8: 2, 9: 3, 10: 4, 11: 5,
  # Left arm <-> Right arm
  15: 22, 16: 23, 17: 24, 18: 25, 19: 26, 20: 27, 21: 28,
  22: 15, 23: 16, 24: 17, 25: 18, 26: 19, 27: 20, 28: 21,
  # Waist (midline) — identity
  12: 12, 13: 13, 14: 14,
}

# Joints whose sign must be negated after swap (roll/yaw axes reverse under sagittal mirror).
_SIGN_FLIP_JOINTS = {
  # Roll joints
  1, 7,    # hip_roll (L, R)
  5, 11,   # ankle_roll (L, R)
  13,      # waist_roll
  16, 23,  # shoulder_roll (L, R)
  19, 26,  # wrist_roll (L, R)
  # Yaw joints
  2, 8,    # hip_yaw (L, R)
  12,      # waist_yaw
  17, 24,  # shoulder_yaw (L, R)
  21, 28,  # wrist_yaw (L, R)
}

class G1Symmetry:
  """Precomputes mirroring tensors for G1 locomanipulation observations and actions."""

  def __init__(self, env: RslRlVecEnvWrapper) -> None:
    self.device = env.device
    self._obs_manager = env.unwrapped.observation_manager

    # Build joint mirroring tensors (29-DOF full body).
    self._joint_swap_idx = self._build_swap_index(29, _JOINT_SWAP_PARTNERS)
    self._joint_sign_mask = self._build_sign_mask(29, _SIGN_FLIP_JOINTS)

    # Build action mirroring tensors (12-DOF lower body).
    # Lower-body indices 0-11 map to full-body indices 0-11.
    self._action_swap_idx = self._build_swap_index(12, {
      k: v for k, v in _JOINT_SWAP_PARTNERS.items() if k < 12
    })
    self._action_sign_mask = self._build_sign_mask(12, {
      j for j in _SIGN_FLIP_JOINTS if j < 12
    })

    # Build per-group observation mirror plans.
    self._group_mirror_plans: dict[str, list[_TermMirror]] = {}
    for group_name in self._obs_manager.cfg:
      self._group_mirror_plans[group_name] = self._build_group_plan(group_name)

  @staticmethod
  def _build_swap_index(n: int, partners: dict[int, int]) -> torch.Tensor:
    idx = torch.arange(n)
    for src, dst in partners.items():
      idx[src] = dst
    return idx

  @staticmethod
  def _build_sign_mask(n: int, flip_set: set[int]) -> torch.Tensor:
    mask = torch.ones(n)
    for j in flip_set:
      mask[j] = -1.0
    return mask

  def _build_group_plan(self, group_name: str) -> list[_TermMirror]:
    """Build a mirror plan for each term in an observation group."""
    plan: list[_TermMirror] = []
    group_cfg = self._obs_manager.cfg[group_name]
    if group_cfg is None:
      return plan

    for term_name in group_cfg.terms:
      mirror = _get_term_mirror(term_name)
      if mirror is not None:
        plan.append(mirror)
      else:
        plan.append(_IdentityMirror())
    return plan

  def mirror_obs(self, obs: torch.Tensor, group_name: str) -> torch.Tensor:
    """Mirror an observation tensor for a given group."""
    mirrored = obs.clone()
    term_dims = self._obs_manager._group_obs_term_dim[group_name]
    plan = self._group_mirror_plans[group_name]

    idx = 0
    for mirror, dims in zip(plan, term_dims):
      dim = int(torch.tensor(dims).prod().item())
      segment = mirrored[:, idx : idx + dim]
      mirror.apply(segment, self)
      idx += dim
    return mirrored

  def mirror_actions(self, actions: torch.Tensor) -> torch.Tensor:
    """Mirror lower-body actions."""
    return actions[..., self._action_swap_idx.to(actions.device)] * self._action_sign_mask.to(actions.device)


# --- Per-term mirror rules ---

class _TermMirror:
  """Base class for observation term mirroring."""
  def apply(self, segment: torch.Tensor, _ctx: G1Symmetry) -> None:
    raise NotImplementedError


class _IdentityMirror(_TermMirror):
  def apply(self, segment: torch.Tensor, _ctx: G1Symmetry) -> None:
    pass


class _NegateIndices(_TermMirror):
  """Negate specific indices within the segment."""
  def __init__(self, indices: tuple[int, ...]):
    self._indices = indices

  def apply(self, segment: torch.Tensor, _ctx: G1Symmetry) -> None:
    for i in self._indices:
      segment[:, i] = -segment[:, i]


class _SwapAndFlipJoints(_TermMirror):
  """Swap left/right joints and negate roll/yaw joints."""
  def __init__(self, n_joints: int, swap_partners: dict[int, int], flip_set: set[int]):
    self._swap_idx = G1Symmetry._build_swap_index(n_joints, swap_partners)
    self._sign_mask = G1Symmetry._build_sign_mask(n_joints, flip_set)

  def apply(self, segment: torch.Tensor, _ctx: G1Symmetry) -> None:
    swap = self._swap_idx.to(segment.device)
    sign = self._sign_mask.to(segment.device)
    segment[:] = segment[..., swap] * sign


class _SwapFootValues(_TermMirror):
  """Swap left/right foot values (each foot is 1 value)."""
  def apply(self, segment: torch.Tensor, _ctx: G1Symmetry) -> None:
    # Assumes segment has shape (batch, 2): [left, right].
    segment[:] = segment[..., [1, 0]]


class _SwapFootForces(_TermMirror):
  """Swap left/right foot forces and negate y-component."""
  def apply(self, segment: torch.Tensor, _ctx: G1Symmetry) -> None:
    # segment shape: (batch, 6) = [left_fx, left_fy, left_fz, right_fx, right_fy, right_fz]
    left = segment[:, :3].clone()
    right = segment[:, 3:].clone()
    left[:, 1] = -left[:, 1]
    right[:, 1] = -right[:, 1]
    segment[:, :3] = right
    segment[:, 3:] = left


class _SwapWristForces(_TermMirror):
  """Swap left/right wrist forces and negate y-component."""
  def apply(self, segment: torch.Tensor, _ctx: G1Symmetry) -> None:
    # Same layout as foot forces.
    left = segment[:, :3].clone()
    right = segment[:, 3:].clone()
    left[:, 1] = -left[:, 1]
    right[:, 1] = -right[:, 1]
    segment[:, :3] = right
    segment[:, 3:] = left


def _get_term_mirror(term_name: str) -> _TermMirror | None:
  """Return the mirror transform for a known observation term, or None for identity."""
  # Simple sign-flip terms.
  if term_name == "base_ang_vel":
    return _NegateIndices((0, 2))  # x (roll), z (yaw) — pseudovector
  if term_name == "projected_gravity":
    return _NegateIndices((1,))    # y (left)
  if term_name == "command":
    return _NegateIndices((1, 2))  # lin_vel_y, ang_vel_z
  if term_name == "phase":
    return _NegateIndices((0, 1))  # half-period shift: sin→−sin, cos→−cos
  if term_name == "base_lin_vel":
    return _NegateIndices((1,))    # y (left)

  # Joint swap-and-flip terms (full 29-DOF).
  if term_name in ("joint_pos", "joint_vel"):
    return _SwapAndFlipJoints(29, _JOINT_SWAP_PARTNERS, _SIGN_FLIP_JOINTS)

  # Action term (12-DOF lower body).
  if term_name == "actions":
    return _SwapAndFlipJoints(12, {
      k: v for k, v in _JOINT_SWAP_PARTNERS.items() if k < 12
    }, {j for j in _SIGN_FLIP_JOINTS if j < 12})

  # Foot terms (swap left/right).
  if term_name in ("foot_height", "foot_air_time", "foot_contact"):
    return _SwapFootValues()
  if term_name == "foot_contact_forces":
    return _SwapFootForces()
  if term_name == "wrist_force":
    return _SwapWristForces()

  # Height scan — grid dimensions depend on sensor config; skip if unknown.
  if term_name == "height_scan":
    return None  # Handled separately if rough terrain is active.

  # Base height terms — no mirroring (pseudovector with z-up convention, so sign doesn't flip).
  if term_name == "base_height_command":
    return None
  if term_name == "base_height":
    return None

  # Unknown term — no mirroring (safe default).
  return None


# --- Module-level function for rsl_rl symmetry_cfg ---

_g1_symmetry_cache: dict[int, G1Symmetry] = {}


def g1_locomanipulation_symmetry(
  env: RslRlVecEnvWrapper,
  obs: dict[str, torch.Tensor] | None,
  actions: torch.Tensor | None,
) -> tuple[dict[str, torch.Tensor] | None, torch.Tensor | None]:
  """Left-right symmetry augmentation for G1 locomanipulation.

  Called by rsl_rl PPO during update(). Returns (augmented_obs, augmented_actions)
  where the first half is original and the second half is mirrored.

  Handles three call modes:
    1. (env, obs, actions) — both provided
    2. (env, obs, None) — obs only (mirror loss computation)
    3. (env, None, actions) — actions only (mirror loss computation)
  """
  # Cache the G1Symmetry instance per env (keyed by id to handle wrapper layers).
  env_id = id(env)
  if env_id not in _g1_symmetry_cache:
    _g1_symmetry_cache[env_id] = G1Symmetry(env)
  symmetry = _g1_symmetry_cache[env_id]

  aug_obs = None
  aug_actions = None

  if obs is not None:
    mirrored_obs = {}
    for group_name, group_obs in obs.items():
      if group_name in symmetry._group_mirror_plans:
        mirrored_obs[group_name] = symmetry.mirror_obs(group_obs, group_name)
      else:
        mirrored_obs[group_name] = group_obs.clone()
    # torch.cat handles both TensorDict and plain dict.
    aug_obs = torch.cat([obs, type(obs)(mirrored_obs, batch_size=obs.batch_size)], dim=0)

  if actions is not None:
    mirrored_actions = symmetry.mirror_actions(actions)
    aug_actions = torch.cat([actions, mirrored_actions], dim=0)

  return aug_obs, aug_actions
