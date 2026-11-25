"""X2 arm compensation using the frozen RoboJuDo simulation reference."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import asdict, fields
from typing import Literal

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.envs import ManagerBasedRlEnvCfg

from src.assets.robots.agibot_x2.compensated_actuator import CompensatedPositionActuatorCfg
from src.instinct_mj.controllers import JointObserverCfg
from src.tasks.locomanipulation.mdp.upper_body_action import UpperBodyMotionActionCfg

from ..observer_restore import arm_actuator_indices, load_observer, restore_arm_actuators
from .reference import X2_ARM_JOINT_NAMES, X2_OBSERVER_REFERENCE


X2_OBSERVER_JOINT_NAMES = X2_ARM_JOINT_NAMES


def configure_x2_torque_compensation(
  cfg: ManagerBasedRlEnvCfg,
  *,
  algorithm: Literal["linearized", "nonlinear"] = "nonlinear",
  mode: Literal["OFF", "SHADOW", "ACTIVE"] = "ACTIVE",
  correction_limit_fraction: float = 0.8,
  correction_limit_nm: float = 0.0,
) -> None:
  """Configure arm actuators and observer before constructing the environment.

  Observations, policy actions, gains, rewards and other robots are untouched.
  X2 environment factories use the nonlinear ACTIVE preset by default.
  Set correction_limit_nm to zero to use only the fraction of actuator limits.
  Bounds are simulation settings, not hardware-qualified torque ratings.
  """
  action = cfg.actions.get("upper_body_motion")
  if (
    not isinstance(action, UpperBodyMotionActionCfg)
    or not action.exclude_waist
    or action.entity_name != "robot"
  ):
    raise ValueError("X2 compensation requires arm-only upper-body motion playback")
  if algorithm not in ("linearized", "nonlinear") or mode not in ("OFF", "SHADOW", "ACTIVE"):
    raise ValueError("Unknown observer algorithm or mode")
  if not math.isfinite(correction_limit_fraction) or not 0 < correction_limit_fraction <= 1:
    raise ValueError("Correction limit fraction must be finite and lie in (0, 1]")
  if not math.isfinite(correction_limit_nm) or correction_limit_nm < 0:
    raise ValueError("Correction limit Nm cap must be finite and nonnegative (0 disables it)")
  robot = cfg.scene.entities["robot"]
  actuators = robot.articulation.actuators
  limits: list[float] = []
  indices = arm_actuator_indices(actuators, X2_OBSERVER_JOINT_NAMES)
  arm_actuator_ids = set(indices)
  for name, index in zip(X2_OBSERVER_JOINT_NAMES, indices, strict=True):
    actuator = actuators[index]
    if not isinstance(actuator, (BuiltinPositionActuatorCfg, CompensatedPositionActuatorCfg)):
      raise ValueError(f"Expected one position actuator for X2 joint {name}")
    if (
      actuator.effort_limit is None
      or not math.isfinite(actuator.effort_limit)
      or actuator.effort_limit <= 0
    ):
      raise ValueError(f"X2 compensation requires a finite positive effort limit for {name}")
    if actuator.delay_max_lag:
      raise ValueError("Synchronous X2 observer does not model command transport delay")
    limit = correction_limit_fraction * actuator.effort_limit
    limits.append(min(correction_limit_nm, limit) if correction_limit_nm else limit)
  converted = tuple(
    CompensatedPositionActuatorCfg(**{
      f.name: getattr(a, f.name) for f in fields(BuiltinPositionActuatorCfg)
    }) if i in arm_actuator_ids else a
    for i, a in enumerate(actuators)
  )
  observer = JointObserverCfg(
    **asdict(X2_OBSERVER_REFERENCE),
    algorithm=algorithm, mode=mode, correction_limits_nm=tuple(limits),
    slew_nm_s=2.0, release_slew_nm_s=16.0 if algorithm == "nonlinear" else 2.0,
  )
  robot.articulation.actuators = converted
  action.observer = observer


def restore_x2_torque_compensation(
  cfg: ManagerBasedRlEnvCfg, observer_data: dict, actuator_data: list[dict],
) -> None:
  """Restore the complete saved arm controller before environment construction.

  Validate a staged copy so incomplete legacy overrides cannot silently select
  current preset coefficients, gains or limits for a trained policy.
  """
  observer = load_observer(observer_data, X2_OBSERVER_JOINT_NAMES)
  staged = deepcopy(cfg)
  configure_x2_torque_compensation(staged, algorithm=observer.algorithm, mode=observer.mode)
  templates = staged.scene.entities["robot"].articulation.actuators
  restored = restore_arm_actuators(templates, observer, actuator_data)
  cfg.scene.entities["robot"].articulation.actuators = restored
  cfg.actions["upper_body_motion"].observer = observer
