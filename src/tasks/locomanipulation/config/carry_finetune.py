"""Shared carry fine-tuning environment configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import EventTermCfg, SceneEntityCfg

from src.tasks.locomanipulation.mdp.payload import (
  SlidingBoxForceEvent, default_payload_params, default_torque_bounds,
)
from src.tasks.locomanipulation.mdp.upper_body_action import UpperBodyMotionActionCfg

from .carry_reference import CarryPoseReference


def configure_carry_finetune(
  cfg: ManagerBasedRlEnvCfg,
  *,
  play: bool,
  reference: CarryPoseReference,
) -> ManagerBasedRlEnvCfg:
  """Replace independent hand perturbations with a shared carry load."""
  upper_body_motion = cfg.actions["upper_body_motion"]
  assert isinstance(upper_body_motion, UpperBodyMotionActionCfg)
  upper_body_motion.motion_file = str(reference.dataset)
  upper_body_motion.motion_dof_indices = reference.motion_dof_indices
  upper_body_motion.locomotion_motion_file = None
  upper_body_motion.pose_only = True

  # Rehearse unloaded default-posture locomotion throughout fine-tuning. Keep this
  # fraction fixed instead of restarting the base posture curriculum.
  upper_body_motion.default_pose_ratio = 0.0 if play else 0.15
  cfg.curriculum.pop("default_pose_ratio", None)

  # Only carry episodes assume both hands support one shared box. Basic tasks
  # retain their independent forces and wider manipulation-pose distribution.
  cfg.events["hand_force"] = EventTermCfg(
    func=SlidingBoxForceEvent, mode="step",
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=reference.hands, preserve_order=True),
      "upper_body_action_name": "upper_body_motion",
      "payload": default_payload_params(),
      "torque_bounds": default_torque_bounds(reference.arm_names),
      "force_scale": 1.0 if play else 0.0,
      # Total unloaded fraction: 15% default poses plus 15% carry poses.
      "no_force_ratio": 0.0 if play else 0.3,
      "constant_force": {"x": 0.0, "y": 0.0, "z": 0.0} if play else None,
    },
  )
  if reference.hand_up_local is not None:
    cfg.events["hand_force"].params["hand_up_local"] = reference.hand_up_local
  cfg.observations["critic"].terms["wrist_force"].params["asset_cfg"] = SceneEntityCfg(
    "robot", body_names=reference.hands, preserve_order=True,
  )

  # The carry runner sets start_step to the loaded checkpoint's step counter.
  # This gives the new posture distribution its own load ramp on fine-tuning.
  if not play:
    cfg.curriculum["force_curriculum"].params["start_step"] = 0

  return cfg
