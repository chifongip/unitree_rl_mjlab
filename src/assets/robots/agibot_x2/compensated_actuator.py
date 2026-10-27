"""Native position PD with additive effort and a combined actuator torque limit."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.actuator import ActuatorCmd, IdealPdActuator, IdealPdActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.utils.spec import create_position_actuator

if TYPE_CHECKING:
  import mujoco

  from mjlab.entity import Entity


class CompensatedPositionActuator(IdealPdActuator["CompensatedPositionActuatorCfg"]):
  """Translate effort into native PD control without changing stored targets.

  Reuse mjlab's PD gain/limit interface so standard randomization events work.
  Native position elements retain implicit PD integration; overriding compute
  keeps this actuator out of both builtin identity batching and ideal-PD fusion.
  """

  def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
    for name in target_names:
      self._mjs_actuators.append(create_position_actuator(
        spec, name, stiffness=self.cfg.stiffness, damping=self.cfg.damping,
        effort_limit=self.cfg.effort_limit, armature=self.cfg.armature,
        frictionloss=self.cfg.frictionloss, viscous_damping=self.cfg.viscous_damping,
        transmission_type=self.cfg.transmission_type,
      ))

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    # Read the actuator gain for the conversion only, never for the observer.
    kp = self.entity.data.model.actuator_gainprm[:, self.global_ctrl_ids, 0]
    return cmd.position_target + cmd.effort_target / kp

  def set_gains(
    self,
    env_ids: torch.Tensor | slice,
    kp: torch.Tensor | None = None,
    kd: torch.Tensor | None = None,
  ) -> None:
    """Synchronize mjlab's randomized parameters with the native PD elements."""
    for value, positive in ((kp, True), (kd, False)):
      if value is not None and (
        not torch.isfinite(value).all() or (value <= 0 if positive else value < 0).any()
      ):
        raise ValueError("Native PD requires finite positive kp and nonnegative kd")
    super().set_gains(env_ids, kp, kd)
    rows = env_ids[:, None] if isinstance(env_ids, torch.Tensor) else env_ids
    model = self.entity.data.model
    if kp is not None:
      model.actuator_gainprm[rows, self.global_ctrl_ids, 0] = self.stiffness[env_ids].to(model.actuator_gainprm)
      model.actuator_biasprm[rows, self.global_ctrl_ids, 1] = -self.stiffness[env_ids].to(model.actuator_biasprm)
    if kd is not None:
      model.actuator_biasprm[rows, self.global_ctrl_ids, 2] = -self.damping[env_ids].to(model.actuator_biasprm)

  def set_effort_limit(
    self, env_ids: torch.Tensor | slice, effort_limit: torch.Tensor,
  ) -> None:
    """Apply standard effort-limit randomization to native total torque limits."""
    if not torch.isfinite(effort_limit).all() or (effort_limit <= 0).any():
      raise ValueError("Native PD requires finite positive effort limits")
    super().set_effort_limit(env_ids, effort_limit)
    rows = env_ids[:, None] if isinstance(env_ids, torch.Tensor) else env_ids
    limits = self.entity.data.model.actuator_forcerange
    limits[rows, self.global_ctrl_ids, 0] = -self.force_limit[env_ids].to(limits)
    limits[rows, self.global_ctrl_ids, 1] = self.force_limit[env_ids].to(limits)


@dataclass(kw_only=True)
class CompensatedPositionActuatorCfg(IdealPdActuatorCfg):
  def __post_init__(self) -> None:
    super().__post_init__()
    if self.transmission_type == TransmissionType.SITE:
      raise ValueError("Compensated position actuators require joint or tendon transmission")
    if not math.isfinite(self.stiffness) or self.stiffness <= 0:
      raise ValueError("Compensated position actuator requires finite positive stiffness")
    if not math.isfinite(self.damping) or self.damping < 0:
      raise ValueError("Compensated position actuator requires finite nonnegative damping")
    if self.effort_limit is None or not math.isfinite(self.effort_limit) or self.effort_limit <= 0:
      raise ValueError("Compensated position actuator requires a finite positive effort limit")

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str],
  ) -> CompensatedPositionActuator:
    return CompensatedPositionActuator(self, entity, target_ids, target_names)
