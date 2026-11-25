"""G1 observer configuration and checkpoint restoration."""

import re
from dataclasses import asdict

from mjlab.envs import ManagerBasedRlEnvCfg

from src.assets.robots.unitree_g1.unitree_actuators import UnitreeActuatorCfg
from src.instinct_mj.controllers import JointObserverCfg

from ..observer_restore import arm_actuator_indices, load_observer, restore_arm_actuators
from .reference import (
  ARM_23,
  ARM_29,
  G1_OBSERVER_REFERENCE_23,
  G1_OBSERVER_REFERENCE_29,
)


def _unitree_arm_actuators(actuators, names: tuple[str, ...]) -> list[UnitreeActuatorCfg]:
  selected = [actuators[index] for index in arm_actuator_indices(actuators, names)]
  if any(not isinstance(actuator, UnitreeActuatorCfg) for actuator in selected):
    raise ValueError("G1 compensation requires Unitree arm actuators")
  return selected


def configure_g1_torque_compensation(cfg: ManagerBasedRlEnvCfg, *, reduced: bool) -> None:
  reference = G1_OBSERVER_REFERENCE_23 if reduced else G1_OBSERVER_REFERENCE_29
  actuators = _unitree_arm_actuators(
    cfg.scene.entities["robot"].articulation.actuators, reference.joint_names,
  )
  cfg.actions["upper_body_motion"].observer = JointObserverCfg(
    **asdict(reference), algorithm="nonlinear", mode="ACTIVE",
    correction_limits_nm=tuple(0.8 * a.effort_limit for a in actuators),
    slew_nm_s=2.0, release_slew_nm_s=16.0,
  )


def restore_g1_torque_compensation(
  cfg: ManagerBasedRlEnvCfg, observer_data: dict, actuator_data: list[dict],
) -> None:
  articulation = cfg.scene.entities["robot"].articulation
  has_wrist_yaw = any(
    re.fullmatch(pattern, "left_wrist_yaw_joint")
    for actuator in articulation.actuators for pattern in actuator.target_names_expr
  )
  observer = load_observer(observer_data, ARM_29 if has_wrist_yaw else ARM_23)
  _unitree_arm_actuators(articulation.actuators, observer.joint_names)
  restored = restore_arm_actuators(articulation.actuators, observer, actuator_data)
  articulation.actuators = restored
  cfg.actions["upper_body_motion"].observer = observer
