"""Shared validation for restoring saved arm controller settings."""

import re
from dataclasses import fields, replace

from mjlab.actuator.actuator import TransmissionType

from src.instinct_mj.controllers import BatchedJointObserver, JointObserverCfg


def load_observer(data: dict, expected_joints: tuple[str, ...]) -> JointObserverCfg:
  missing = {field.name for field in fields(JointObserverCfg)} - data.keys()
  if missing:
    raise ValueError(f"Saved torque compensation is incomplete; missing {sorted(missing)}")
  observer = JointObserverCfg(**{
    key: tuple(value) if isinstance(value, list) else value for key, value in data.items()
  })
  if (len(observer.joint_names) != len(expected_joints)
      or set(observer.joint_names) != set(expected_joints)):
    raise ValueError("Saved observer joints do not match the robot variant")
  return observer


def arm_actuator_indices(actuators, joint_names: tuple[str, ...]) -> list[int]:
  indices = []
  for name in joint_names:
    matches = [
      i for i, actuator in enumerate(actuators)
      if actuator.transmission_type == TransmissionType.JOINT
      and any(re.fullmatch(pattern, name) for pattern in actuator.target_names_expr)
    ]
    if len(matches) != 1:
      raise ValueError(f"Expected one position actuator for joint {name}")
    indices.append(matches[0])
  return indices


def _saved_transmission(data: dict):
  value = data.get("transmission_type")
  # Legacy YAML can represent the enum constructor's argument as a list.
  return value[0] if isinstance(value, list) and len(value) == 1 else value


def restore_arm_actuators(actuators, observer: JointObserverCfg, saved_actuators: list[dict]):
  """Build and validate a replacement tuple while leaving the environment untouched."""
  joint_indices = arm_actuator_indices(actuators, observer.joint_names)
  restored = list(actuators)
  for index in dict.fromkeys(joint_indices):
    template = actuators[index]
    patterns = template.target_names_expr
    matches = [
      saved for saved in saved_actuators
      if _saved_transmission(saved) == template.transmission_type
      and tuple(saved.get("target_names_expr", ())) == patterns
    ]
    if len(matches) != 1:
      raise ValueError(f"Missing or ambiguous saved arm actuator {patterns}")
    values = dict(matches[0])
    missing = {field.name for field in fields(template)} - values.keys()
    if missing:
      raise ValueError(f"Saved arm actuator is incomplete; missing {sorted(missing)}")
    values.update(target_names_expr=patterns, transmission_type=template.transmission_type)
    restored[index] = replace(template, **values)

  selected = [restored[index] for index in joint_indices]
  if any(a.delay_max_lag for a in selected):
    raise ValueError("Synchronous observers require zero command transport delay")
  BatchedJointObserver(
    observer, tuple(a.stiffness for a in selected), tuple(a.damping for a in selected),
    tuple(a.effort_limit for a in selected), num_envs=1,
  )
  return tuple(restored)
