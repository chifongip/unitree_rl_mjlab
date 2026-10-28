"""X2 arm compensation using the frozen RoboJuDo simulation reference."""

from __future__ import annotations

import math
import re
from copy import deepcopy
from dataclasses import fields
from typing import Literal

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.envs import ManagerBasedRlEnvCfg

from src.assets.robots.agibot_x2.compensated_actuator import CompensatedPositionActuatorCfg
from src.instinct_mj.controllers import BatchedJointObserver, JointObserverCfg
from src.tasks.locomanipulation.mdp.upper_body_action import UpperBodyMotionActionCfg


X2_OBSERVER_JOINT_NAMES = tuple(
  f"{side}_{joint}_joint"
  for side in ("left", "right")
  for joint in (
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
    "wrist_yaw", "wrist_pitch", "wrist_roll",
  )
)

# RoboJuDo-Plus/robojudo/config/x2/observer_parameters.py, frozen simulation
# reference. These fitted local model coefficients are not physical inertias
# or measurements of the hardware. They must stay independent of plant DR.
_INERTIA = (
  0.15532404980256578, 0.10078795654951321, 0.07674159569739615,
  0.07182380882978089, 0.03, 0.03, 0.03,
  0.1553018849773343, 0.1012386139709888, 0.07677501375357826,
  0.07193011707176135, 0.03, 0.03, 0.03,
)
_VISCOSITY = (
  106.80829994233635, 228.63155219753165, 29.852550817628547,
  44.32684035243929, 28.85973128906408, 31.97077929396376, 32.10787293174743,
  106.63727466536166, 228.8259861441458, 29.86332093192248,
  44.346794114158094, 28.860217878180926, 31.968164529938882, 32.102352113935815,
)


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
  arm_actuator_ids: set[int] = set()
  for name in X2_OBSERVER_JOINT_NAMES:
    matching = [
      i for i, a in enumerate(actuators)
      if a.transmission_type == TransmissionType.JOINT
      and any(re.fullmatch(p, name) for p in a.target_names_expr)
    ]
    if len(matching) != 1 or not isinstance(
      actuators[matching[0]], (BuiltinPositionActuatorCfg, CompensatedPositionActuatorCfg)
    ):
      raise ValueError(f"Expected one position actuator for X2 joint {name}")
    actuator = actuators[matching[0]]
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
    arm_actuator_ids.add(matching[0])
  converted = tuple(
    CompensatedPositionActuatorCfg(**{
      f.name: getattr(a, f.name) for f in fields(BuiltinPositionActuatorCfg)
    }) if i in arm_actuator_ids else a
    for i, a in enumerate(actuators)
  )
  observer = JointObserverCfg(
    joint_names=X2_OBSERVER_JOINT_NAMES,
    algorithm=algorithm, mode=mode, inertia=_INERTIA,
    friction_limit=0.3, friction_viscosity=_VISCOSITY,
    bandwidth_rad_s=8.0, correction_limits_nm=tuple(limits),
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
  required = {f.name for f in fields(JointObserverCfg)}
  missing = required - observer_data.keys()
  if missing:
    raise ValueError(f"Saved torque compensation is incomplete; missing {sorted(missing)}")
  parameters = dict(observer_data)
  for key in (
    "joint_names", "inertia", "correction_limits_nm", "friction_limit", "friction_viscosity",
  ):
    if isinstance(parameters[key], list):
      parameters[key] = tuple(parameters[key])
  observer = JointObserverCfg(**parameters)
  if set(observer.joint_names) != set(X2_OBSERVER_JOINT_NAMES):
    raise ValueError("Saved X2 observer must select exactly the 14 arm joints")
  staged = deepcopy(cfg)
  configure_x2_torque_compensation(staged, algorithm=observer.algorithm, mode=observer.mode)
  # Stripping Python tags from legacy env.yaml leaves an enum constructor's
  # argument as a one-element list. Normalize it without executing YAML tags.
  actuator_data = [dict(a) for a in actuator_data]
  for saved in actuator_data:
    transmission = saved.get("transmission_type")
    if isinstance(transmission, list) and len(transmission) == 1:
      saved["transmission_type"] = transmission[0]
  restored = []
  for actuator in staged.scene.entities["robot"].articulation.actuators:
    if not isinstance(actuator, CompensatedPositionActuatorCfg):
      restored.append(actuator)
      continue
    matching = [
      saved for saved in actuator_data
      if saved.get("transmission_type") == TransmissionType.JOINT
      and tuple(saved.get("target_names_expr", ())) == actuator.target_names_expr
    ]
    if len(matching) != 1:
      raise ValueError(f"Missing or ambiguous saved arm actuator {actuator.target_names_expr}")
    saved = dict(matching[0])
    missing = {f.name for f in fields(CompensatedPositionActuatorCfg)} - saved.keys()
    if missing:
      raise ValueError(f"Saved arm actuator is incomplete; missing {sorted(missing)}")
    saved["target_names_expr"] = tuple(saved["target_names_expr"])
    saved["transmission_type"] = TransmissionType(saved["transmission_type"])
    restored.append(CompensatedPositionActuatorCfg(**saved))

  # Validate model vectors, scalar settings and correction authority against
  # the restored controller settings, never against simulated plant truth.
  selected = [
    next(a for a in restored if a.transmission_type == TransmissionType.JOINT
         and any(re.fullmatch(p, name) for p in a.target_names_expr))
    for name in observer.joint_names
  ]
  BatchedJointObserver(
    observer, tuple(a.stiffness for a in selected), tuple(a.damping for a in selected),
    tuple(a.effort_limit for a in selected), num_envs=1,
  )
  cfg.scene.entities["robot"].articulation.actuators = tuple(restored)
  cfg.actions["upper_body_motion"].observer = observer
