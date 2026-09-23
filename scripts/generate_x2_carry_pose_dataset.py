#!/usr/bin/env python3
"""Generate X2 carry-pose clips by sampling arm endpoint poses and solving IK.

The supplied reference pose defines one endpoint pose for each arm. Each
candidate samples a parallel pair of endpoint poses in the ``torso_link``
frame, then solves bounded IK for both arms. The endpoints share forward and
height coordinates, while their lateral positions and orientations can vary
independently. The left endpoint stays on the +Y side and the right stays on
the -Y side; both remain in front of the torso. Unreachable
candidates are skipped. The resulting clips are static: every frame in a clip
is identical, so they work with either ``pose_only=True`` or normal upper-body
motion playback.

The reference pose is not a joint-space sampling bound. Generated poses may
use the complete X2 arm joint ranges; only the model's joint limits and the
final endpoint-target residual decide whether an IK result is retained.

The output format is compatible with ``UpperBodyMotionAction``:
``dict[str, {"dof": (frames, 29) float32, "fps": int}]``.  The action term
selects the 14 arm columns from this 29-DOF data.

Example:
  python scripts/generate_x2_carry_pose_dataset.py --num-poses 256 --output \
    src/assets/data/x2/bones_seed/carry_poses_x2.pkl --playback --playback-max-poses 32

To inspect an existing dataset without generating new poses:
  python scripts/generate_x2_carry_pose_dataset.py --playback-file \
    src/assets/data/x2/bones_seed/carry_poses_x2.pkl --playback-once

To use a measured carry pose instead of the embedded reference, save the 14 arm
joint values as a JSON object and pass ``--base-pose-json path/to/pose.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import joblib
import mujoco
import mujoco.viewer
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_XML = REPO_ROOT / "src/assets/robots/agibot_x2/xmls/x2_ultra_no_head.xml"
DEFAULT_OUTPUT = REPO_ROOT / "src/assets/data/x2/bones_seed/carry_poses_x2.pkl"

ARM_JOINT_NAMES = (
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_yaw_joint",
  "left_wrist_pitch_joint",
  "left_wrist_roll_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_yaw_joint",
  "right_wrist_pitch_joint",
  "right_wrist_roll_joint",
)

END_EFFECTOR_BODY_NAMES = ("left_wrist_roll_link", "right_wrist_roll_link")
REFERENCE_FRAME_BODY_NAME = "torso_link"

# A measured X2 carry-pose reference. Values are radians and can be replaced with
# --base-pose-json for another object or grasp.
DEFAULT_CARRY_POSE = {
  "left_shoulder_pitch_joint": -0.081,
  "left_shoulder_roll_joint": 0.026,
  "left_shoulder_yaw_joint": -0.049,
  "left_elbow_joint": -2.188,
  "left_wrist_yaw_joint": 0.238,
  "left_wrist_pitch_joint": 0.326,
  "left_wrist_roll_joint": 0.215,
  "right_shoulder_pitch_joint": -0.081,
  "right_shoulder_roll_joint": -0.026,
  "right_shoulder_yaw_joint": 0.049,
  "right_elbow_joint": -2.188,
  "right_wrist_yaw_joint": -0.238,
  "right_wrist_pitch_joint": 0.326,
  "right_wrist_roll_joint": -0.215,
}

# Reference terminal-link poses from X2 forward kinematics, in torso_link frame.
# XYZ is in metres; quaternion order is (w, x, y, z). Rounded to 3 decimals.
# These are wrist_roll_link origins, not fingertip or grasp-point positions.
# Left:  xyz=(0.198,  0.202, 0.170), quat=(0.574, -0.004, -0.819,  0.009)
# Right: xyz=(0.198, -0.202, 0.169), quat=(0.576,  0.004, -0.817, -0.009)
# Endpoint midpoint: xyz=(0.198, 0.000, 0.170)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument(
    "--base-pose-json",
    type=Path,
    help="JSON object containing exactly the 14 X2 arm-joint values.",
  )
  parser.add_argument(
    "--num-poses",
    type=int,
    default=256,
    help="Maximum number of valid clips to write, including the reference pose.",
  )
  parser.add_argument(
    "--num-candidates",
    type=int,
    default=None,
    help="Endpoint-pose candidates to test; defaults to 20 times --num-poses.",
  )
  parser.add_argument(
    "--frames-per-clip",
    type=int,
    default=60,
    help="Repeated static frames per clip (default: 60, or two seconds at 30 FPS).",
  )
  parser.add_argument("--fps", type=int, default=30)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument(
    "--translation-range",
    type=float,
    default=0.5,
    help=(
      "Symmetric fallback for endpoint offsets when --endpoint-offset-lower "
      "and --endpoint-offset-upper are omitted (default: 0.5)."
    ),
  )
  parser.add_argument(
    "--endpoint-offset-lower",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    help=(
      "Inclusive torso-frame XYZ lower bound, in metres, for each endpoint "
      "offset from its reference pose. Must be used with "
      "--endpoint-offset-upper."
    ),
  )
  parser.add_argument(
    "--endpoint-offset-upper",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    help=(
      "Inclusive torso-frame XYZ upper bound, in metres, for each endpoint "
      "offset from its reference pose. Must be used with "
      "--endpoint-offset-lower."
    ),
  )
  parser.add_argument(
    "--min-end-effector-forward",
    type=float,
    default=0.05,
    help=(
      "Minimum torso-frame +X coordinate of either arm endpoint in metres; "
      "prevents back-side carries (default: 0.05)."
    ),
  )
  parser.add_argument(
    "--min-end-effector-side-clearance",
    type=float,
    default=0.0,
    help=(
      "Minimum torso-frame distance of each endpoint from the Y=0 plane, "
      "on its own side (default: 0.0)."
    ),
  )
  parser.add_argument(
    "--rotation-range-deg",
    type=float,
    default=35.0,
    help=(
      "Uniform independent per-endpoint, per-axis rotation interval [-x, x] "
      "in degrees (default: 35)."
    ),
  )
  parser.add_argument(
    "--wide-sample-ratio",
    type=float,
    default=1.0,
    help=(
      "Fraction of samples drawn from the full ranges; values below 1 use "
      "15%% ranges for the remainder (default: 1.0)."
    ),
  )
  parser.add_argument(
    "--position-tolerance",
    type=float,
    default=0.003,
    help="IK position residual scale in metres (default: 0.003).",
  )
  parser.add_argument(
    "--orientation-tolerance-deg",
    type=float,
    default=5.0,
    help="IK orientation residual scale in degrees (default: 5).",
  )
  parser.add_argument(
    "--joint-regularization",
    type=float,
    default=0.0,
    help="Optional IK initial-pose regularization scale in radians; 0 disables it.",
  )
  parser.add_argument(
    "--max-end-effector-position-error",
    type=float,
    default=0.006,
    help="Reject a sample if either endpoint misses its target by more than this many metres.",
  )
  parser.add_argument(
    "--max-end-effector-orientation-error-deg",
    type=float,
    default=10.0,
    help="Reject a sample if either endpoint misses its target orientation by more than this angle.",
  )
  parser.add_argument(
    "--max-parallel-axis-error",
    type=float,
    default=0.002,
    help=(
      "Reject a sample if its endpoints differ by more than this many metres "
      "along torso-frame X or Z (default: 0.002)."
    ),
  )
  parser.add_argument(
    "--max-self-penetration",
    type=float,
    default=0.001,
    help=(
      "Reject poses with robot self-contact deeper than this many metres "
      "(default: 0.001)."
    ),
  )
  parser.add_argument(
    "--playback",
    action="store_true",
    help="Open a MuJoCo viewer and cycle through the generated poses.",
  )
  parser.add_argument(
    "--playback-file",
    type=Path,
    help="Load a saved carry-pose .pkl and play it without generating poses.",
  )
  parser.add_argument(
    "--playback-duration",
    type=float,
    default=0.75,
    help="Seconds to show each pose in the viewer (default: 0.75).",
  )
  parser.add_argument(
    "--playback-max-poses",
    type=int,
    default=0,
    help="Maximum poses to show; 0 shows all poses (default: 0).",
  )
  parser.add_argument(
    "--playback-once",
    action="store_true",
    help="Exit after one playback pass instead of looping.",
  )
  return parser.parse_args()


def load_base_pose(path: Path | None) -> dict[str, float]:
  """Return a validated carry pose with exactly the X2 arm joints."""
  if path is None:
    pose = DEFAULT_CARRY_POSE
  else:
    with path.open(encoding="utf-8") as input_file:
      pose = json.load(input_file)
    if not isinstance(pose, dict):
      raise ValueError(f"{path} must contain a JSON object")

  names = set(pose)
  expected = set(ARM_JOINT_NAMES)
  if names != expected:
    raise ValueError(
      "carry-pose joints must exactly match the X2 arm joints; "
      f"missing={sorted(expected - names)}, additional={sorted(names - expected)}"
    )
  validated = {name: float(pose[name]) for name in ARM_JOINT_NAMES}
  if not np.isfinite(list(validated.values())).all():
    raise ValueError("carry pose contains a non-finite joint value")
  return validated


def scalar_joint_info(
  model: mujoco.MjModel,
) -> tuple[list[str], dict[str, int], dict[str, int]]:
  """Return model-order scalar joints and their qpos/joint IDs."""
  names: list[str] = []
  qpos_addresses: dict[str, int] = {}
  joint_ids: dict[str, int] = {}
  for joint_id in range(model.njnt):
    if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
      continue
    if model.jnt_type[joint_id] not in (
      mujoco.mjtJoint.mjJNT_HINGE,
      mujoco.mjtJoint.mjJNT_SLIDE,
    ):
      raise ValueError(f"joint {joint_id} is not scalar")
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    if name is None:
      raise ValueError(f"unnamed scalar joint {joint_id}")
    names.append(name)
    qpos_addresses[name] = int(model.jnt_qposadr[joint_id])
    joint_ids[name] = joint_id
  return names, qpos_addresses, joint_ids


def rotation_matrix_from_rotvec(rotation_vector: np.ndarray) -> np.ndarray:
  """Return a 3x3 rotation matrix for an axis-angle rotation vector."""
  angle = float(np.linalg.norm(rotation_vector))
  if angle < 1e-12:
    return np.eye(3)
  axis = rotation_vector / angle
  cross = np.array(
    [
      [0.0, -axis[2], axis[1]],
      [axis[2], 0.0, -axis[0]],
      [-axis[1], axis[0], 0.0],
    ]
  )
  return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)


def rotation_error_vector(current: np.ndarray, target: np.ndarray) -> np.ndarray:
  """Return the full axis-angle rotation taking current to target."""
  return Rotation.from_matrix(target @ current.T).as_rotvec()


def set_arm_qpos(
  data: mujoco.MjData,
  qpos_addresses: dict[str, int],
  arm_qpos: np.ndarray,
) -> None:
  for index, name in enumerate(ARM_JOINT_NAMES):
    data.qpos[qpos_addresses[name]] = arm_qpos[index]


def end_effector_transforms(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  end_effector_body_ids: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
  """Return terminal arm-link origins and rotations in the world frame."""
  positions = data.xpos[list(end_effector_body_ids)].copy()
  rotations = data.xmat[list(end_effector_body_ids)].reshape(2, 3, 3).copy()
  return positions, rotations


def has_self_collision(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  max_penetration: float,
) -> bool:
  """Detect penetrating contacts between robot geoms, excluding the world."""
  for contact in data.contact[: data.ncon]:
    if contact.dist >= -max_penetration:
      continue
    if model.geom_bodyid[contact.geom1] and model.geom_bodyid[contact.geom2]:
      return True
  return False


def solve_arm_ik(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  qpos_addresses: dict[str, int],
  end_effector_body_ids: tuple[int, int],
  initial_arm_qpos: np.ndarray,
  lower: np.ndarray,
  upper: np.ndarray,
  target_positions: np.ndarray,
  target_rotations: np.ndarray,
  position_tolerance: float,
  orientation_tolerance_rad: float,
  reference_frame_position: np.ndarray,
  reference_frame_rotation: np.ndarray,
  parallel_axis_tolerance: float,
  joint_regularization: float,
) -> tuple[np.ndarray, float, float, float]:
  """Solve bounded bilateral endpoint-pose IK and return maximum residuals."""
  qpos_template = data.qpos.copy()

  def residual(arm_qpos: np.ndarray) -> np.ndarray:
    data.qpos[:] = qpos_template
    set_arm_qpos(data, qpos_addresses, arm_qpos)
    mujoco.mj_forward(model, data)
    positions, rotations = end_effector_transforms(model, data, end_effector_body_ids)
    position_residual = ((positions - target_positions) / position_tolerance).ravel()
    orientation_residual = np.concatenate(
      [
        rotation_error_vector(rotations[index], target_rotations[index])
        / orientation_tolerance_rad
        for index in range(2)
      ]
    )
    positions_frame = (
      reference_frame_rotation.T @ (positions - reference_frame_position).T
    ).T
    parallel_residual = (
      positions_frame[0, (0, 2)] - positions_frame[1, (0, 2)]
    ) / parallel_axis_tolerance
    if joint_regularization <= 0.0:
      return np.concatenate((position_residual, orientation_residual, parallel_residual))
    posture_residual = (arm_qpos - initial_arm_qpos) / joint_regularization
    return np.concatenate(
      (position_residual, orientation_residual, parallel_residual, posture_residual)
    )

  result = least_squares(
    residual,
    x0=initial_arm_qpos,
    bounds=(lower, upper),
    method="trf",
    max_nfev=300,
    ftol=1e-10,
    xtol=1e-10,
    gtol=1e-10,
  )
  if not result.success:
    raise RuntimeError(result.message)

  data.qpos[:] = qpos_template
  set_arm_qpos(data, qpos_addresses, result.x)
  mujoco.mj_forward(model, data)
  positions, rotations = end_effector_transforms(model, data, end_effector_body_ids)
  max_position_error = float(np.linalg.norm(positions - target_positions, axis=1).max())
  max_orientation_error = float(
    max(
      np.linalg.norm(rotation_error_vector(rotations[index], target_rotations[index]))
      for index in range(2)
    )
  )
  positions_frame = (
    reference_frame_rotation.T @ (positions - reference_frame_position).T
  ).T
  max_parallel_axis_error = float(
    np.abs(positions_frame[0, (0, 2)] - positions_frame[1, (0, 2)]).max()
  )
  return result.x, max_position_error, max_orientation_error, max_parallel_axis_error


def sample_endpoint_targets(
  reference_positions_frame: np.ndarray,
  reference_rotations_frame: np.ndarray,
  rng: np.random.Generator,
  endpoint_offset_lower: np.ndarray,
  endpoint_offset_upper: np.ndarray,
  rotation_range_rad: float,
  wide_sample_ratio: float,
  min_end_effector_forward: float,
  min_end_effector_side_clearance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Sample parallel endpoint positions with independent lateral and rotation offsets."""
  range_scale = 1.0 if rng.random() < wide_sample_ratio else 0.15
  scaled_offset_lower = endpoint_offset_lower * range_scale
  scaled_offset_upper = endpoint_offset_upper * range_scale
  reference_center = reference_positions_frame.mean(axis=0)

  # Both endpoint targets must have equal X and Z.  Intersect the per-endpoint
  # offset bounds after accounting for the small nominal left/right offsets.
  shared_x_lower = max(
    np.max(scaled_offset_lower[0] + reference_positions_frame[:, 0] - reference_center[0]),
    min_end_effector_forward - reference_center[0],
  )
  shared_x_upper = np.min(
    scaled_offset_upper[0] + reference_positions_frame[:, 0] - reference_center[0]
  )
  shared_z_lower = np.max(
    scaled_offset_lower[2] + reference_positions_frame[:, 2] - reference_center[2]
  )
  shared_z_upper = np.min(
    scaled_offset_upper[2] + reference_positions_frame[:, 2] - reference_center[2]
  )
  if shared_x_lower > shared_x_upper or shared_z_lower > shared_z_upper:
    raise ValueError(
      "the endpoint-offset bounds cannot produce parallel targets that satisfy "
      "--min-end-effector-forward"
    )

  target_positions = reference_positions_frame.copy()
  target_positions[:, 0] = reference_center[0] + rng.uniform(
    shared_x_lower, shared_x_upper
  )
  target_positions[:, 2] = reference_center[2] + rng.uniform(
    shared_z_lower, shared_z_upper
  )
  left_y_lower = max(
    reference_positions_frame[0, 1] + scaled_offset_lower[1],
    min_end_effector_side_clearance,
  )
  left_y_upper = reference_positions_frame[0, 1] + scaled_offset_upper[1]
  right_y_lower = reference_positions_frame[1, 1] + scaled_offset_lower[1]
  right_y_upper = min(
    reference_positions_frame[1, 1] + scaled_offset_upper[1],
    -min_end_effector_side_clearance,
  )
  if left_y_lower > left_y_upper or right_y_lower > right_y_upper:
    raise ValueError(
      "the endpoint-offset bounds cannot keep both arms on their respective "
      "sides of the torso"
    )
  target_positions[0, 1] = rng.uniform(left_y_lower, left_y_upper)
  target_positions[1, 1] = rng.uniform(right_y_lower, right_y_upper)

  rotation_vectors = rng.uniform(
    -rotation_range_rad, rotation_range_rad, size=(2, 3)
  ) * range_scale
  rotations = np.stack(
    [rotation_matrix_from_rotvec(vector) for vector in rotation_vectors]
  )
  target_rotations = np.einsum("bij,bjk->bik", rotations, reference_rotations_frame)
  translation = target_positions - reference_positions_frame
  return target_positions, target_rotations, translation, rotation_vectors


def safe_float32_dof(
  dof: np.ndarray,
  dof_names: list[str],
  model: mujoco.MjModel,
  joint_ids: dict[str, int],
) -> np.ndarray:
  """Convert to float32 without rounding a limited joint beyond its range."""
  output = dof.astype(np.float32)
  for index, name in enumerate(dof_names):
    joint_id = joint_ids[name]
    if not model.jnt_limited[joint_id]:
      continue
    lower = np.nextafter(
      np.float32(model.jnt_range[joint_id, 0]), np.float32(np.inf)
    )
    upper = np.nextafter(
      np.float32(model.jnt_range[joint_id, 1]), np.float32(-np.inf)
    )
    output[index] = np.clip(output[index], lower, upper)
  return output


def set_dof_qpos(
  data: mujoco.MjData,
  dof: np.ndarray,
  dof_names: list[str],
  qpos_addresses: dict[str, int],
) -> None:
  """Set all scalar model joints from one 29-DOF dataset frame."""
  data.qpos[:] = data.model.qpos0
  data.qvel[:] = 0.0
  for index, name in enumerate(dof_names):
    data.qpos[qpos_addresses[name]] = dof[index]


def load_motion_data(
  path: Path, num_dofs: int
) -> dict[str, dict[str, np.ndarray | int]]:
  """Load a saved carry-pose dataset and check its playback shape."""
  motion_data = joblib.load(path)
  if not isinstance(motion_data, dict) or not motion_data:
    raise ValueError(f"{path} must contain a nonempty dictionary of clips")
  for name, clip in motion_data.items():
    if not isinstance(name, str) or not isinstance(clip, dict):
      raise ValueError(f"{path} contains an invalid clip entry")
    dof = clip.get("dof")
    fps = clip.get("fps")
    if (
      not isinstance(dof, np.ndarray)
      or dof.ndim != 2
      or dof.shape[0] < 1
      or dof.shape[1] != num_dofs
      or not np.issubdtype(dof.dtype, np.floating)
      or not np.isfinite(dof).all()
      or not isinstance(fps, (int, np.integer))
      or fps < 1
    ):
      raise ValueError(
        f"clip {name!r} in {path} must have finite dof frames of shape "
        f"(frames, {num_dofs}) and a positive integer fps"
      )
  return motion_data


def playback(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  motion_data: dict[str, dict[str, np.ndarray | int]],
  dof_names: list[str],
  qpos_addresses: dict[str, int],
  end_effector_body_ids: tuple[int, int],
  duration: float,
  max_poses: int,
  once: bool,
) -> None:
  """Cycle through static clips in a passive MuJoCo viewer."""
  items = list(motion_data.items())
  if max_poses > 0:
    items = items[:max_poses]
  print(f"Opening viewer with {len(items)} carry poses. Press Ctrl+C to stop.")
  with mujoco.viewer.launch_passive(model, data) as viewer:
    try:
      while viewer.is_running():
        for index, (name, clip) in enumerate(items, start=1):
          if not viewer.is_running():
            return
          dof = clip["dof"]
          assert isinstance(dof, np.ndarray)
          set_dof_qpos(data, dof[0], dof_names, qpos_addresses)
          mujoco.mj_forward(model, data)
          end_effector_positions, _ = end_effector_transforms(
            model, data, end_effector_body_ids
          )
          for end_effector_index, position in enumerate(end_effector_positions):
            mujoco.mjv_initGeom(
              viewer.user_scn.geoms[end_effector_index],
              mujoco.mjtGeom.mjGEOM_SPHERE,
              np.array([0.025, 0.0, 0.0]),
              position,
              np.eye(3).flatten(),
              np.array([0.2, 0.9, 0.3, 0.8], dtype=np.float32),
            )
          viewer.user_scn.ngeom = len(end_effector_positions)
          viewer.sync()
          print(f"Pose {index:03d}/{len(items)}: {name}")
          deadline = time.monotonic() + duration
          while viewer.is_running() and time.monotonic() < deadline:
            time.sleep(min(0.02, deadline - time.monotonic()))
        if once:
          return
    except KeyboardInterrupt:
      print("\nViewer closed.")


def main() -> None:
  args = parse_args()
  if args.playback_duration <= 0.0:
    raise ValueError("--playback-duration must be positive")
  if args.playback_max_poses < 0:
    raise ValueError("--playback-max-poses cannot be negative")
  if args.playback_file is not None:
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    dof_names, qpos_addresses, _ = scalar_joint_info(model)
    end_effector_body_ids = tuple(
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
      for name in END_EFFECTOR_BODY_NAMES
    )
    if any(body_id < 0 for body_id in end_effector_body_ids):
      raise ValueError(f"model is missing endpoint bodies {END_EFFECTOR_BODY_NAMES}")
    motion_data = load_motion_data(args.playback_file, len(dof_names))
    playback(
      model, data, motion_data, dof_names, qpos_addresses,
      end_effector_body_ids, args.playback_duration,
      args.playback_max_poses, args.playback_once,
    )
    return

  if args.num_poses < 1:
    raise ValueError("--num-poses must be positive")
  if args.num_candidates is not None and args.num_candidates < 1:
    raise ValueError("--num-candidates must be positive")
  if args.frames_per_clip < 1 or args.fps < 1:
    raise ValueError("--frames-per-clip and --fps must be positive")
  if not 0.0 <= args.wide_sample_ratio <= 1.0:
    raise ValueError("--wide-sample-ratio must be in [0, 1]")
  if (args.endpoint_offset_lower is None) != (args.endpoint_offset_upper is None):
    raise ValueError(
      "--endpoint-offset-lower and --endpoint-offset-upper must be set together"
    )
  if args.min_end_effector_forward < 0.0:
    raise ValueError("--min-end-effector-forward cannot be negative")
  if args.min_end_effector_side_clearance < 0.0:
    raise ValueError("--min-end-effector-side-clearance cannot be negative")
  if args.joint_regularization < 0.0:
    raise ValueError("--joint-regularization cannot be negative")
  if args.max_self_penetration < 0.0:
    raise ValueError("--max-self-penetration cannot be negative")
  positive_values = [
    args.rotation_range_deg,
    args.position_tolerance,
    args.orientation_tolerance_deg,
    args.max_end_effector_position_error,
    args.max_end_effector_orientation_error_deg,
    args.max_parallel_axis_error,
  ]
  if args.endpoint_offset_lower is None:
    endpoint_offset_lower = np.full(3, -args.translation_range)
    endpoint_offset_upper = np.full(3, args.translation_range)
    positive_values.append(args.translation_range)
  else:
    endpoint_offset_lower = np.asarray(args.endpoint_offset_lower, dtype=float)
    endpoint_offset_upper = np.asarray(args.endpoint_offset_upper, dtype=float)
    if np.any(endpoint_offset_lower >= endpoint_offset_upper):
      raise ValueError(
        "each --endpoint-offset-lower value must be less than its upper bound"
      )
  if min(positive_values) <= 0.0:
    raise ValueError("all tolerances, bounds, and standard deviations must be positive")

  model = mujoco.MjModel.from_xml_path(str(args.xml))
  data = mujoco.MjData(model)
  dof_names, qpos_addresses, joint_ids = scalar_joint_info(model)
  if len(dof_names) != 29:
    raise ValueError(f"expected 29 scalar X2 joints, found {len(dof_names)}")
  if not set(ARM_JOINT_NAMES).issubset(qpos_addresses):
    raise ValueError("X2 model is missing one or more required arm joints")

  base_pose = load_base_pose(args.base_pose_json)
  seed_arm_qpos = np.array([base_pose[name] for name in ARM_JOINT_NAMES])
  arm_joint_ids = np.array([joint_ids[name] for name in ARM_JOINT_NAMES])
  model_lower = model.jnt_range[arm_joint_ids, 0]
  model_upper = model.jnt_range[arm_joint_ids, 1]
  lower = model_lower
  upper = model_upper
  if np.any(seed_arm_qpos < lower) or np.any(seed_arm_qpos > upper):
    raise ValueError("base carry pose violates X2 joint limits")
  # This is an optimizer starting point only. With the default zero
  # regularization it neither adds a residual nor narrows the feasible set.
  initial_arm_qpos = seed_arm_qpos.copy()

  end_effector_body_ids = tuple(
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    for name in END_EFFECTOR_BODY_NAMES
  )
  if any(body_id < 0 for body_id in end_effector_body_ids):
    raise ValueError(f"model is missing endpoint bodies {END_EFFECTOR_BODY_NAMES}")
  reference_frame_body_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, REFERENCE_FRAME_BODY_NAME
  )
  if reference_frame_body_id < 0:
    raise ValueError(f"model is missing reference body {REFERENCE_FRAME_BODY_NAME!r}")
  data.qpos[:] = model.qpos0
  data.qvel[:] = 0.0
  set_arm_qpos(data, qpos_addresses, seed_arm_qpos)
  mujoco.mj_forward(model, data)
  reference_positions, reference_rotations = end_effector_transforms(
    model, data, end_effector_body_ids
  )
  reference_frame_position = data.xpos[reference_frame_body_id].copy()
  reference_frame_rotation = data.xmat[reference_frame_body_id].reshape(3, 3).copy()
  reference_positions_frame = (
    reference_frame_rotation.T @ (reference_positions - reference_frame_position).T
  ).T
  reference_rotations_frame = np.einsum(
    "ij,bjk->bik", reference_frame_rotation.T, reference_rotations
  )
  if np.any(reference_positions_frame[:, 0] < args.min_end_effector_forward):
    raise ValueError(
      "a reference arm endpoint is behind --min-end-effector-forward; choose "
      "a smaller value or a reference pose farther forward"
    )
  if (
    reference_positions_frame[0, 1] < args.min_end_effector_side_clearance
    or reference_positions_frame[1, 1] > -args.min_end_effector_side_clearance
  ):
    raise ValueError(
      "the reference endpoints do not satisfy --min-end-effector-side-clearance"
    )

  rng = np.random.default_rng(args.seed)
  orientation_tolerance_rad = math.radians(args.orientation_tolerance_deg)
  max_orientation_error_rad = math.radians(
    args.max_end_effector_orientation_error_deg
  )
  rotation_range_rad = math.radians(args.rotation_range_deg)
  num_candidates = args.num_candidates or args.num_poses * 20
  motion_data: dict[str, dict[str, np.ndarray | int]] = {}
  data.qpos[:] = model.qpos0
  set_arm_qpos(data, qpos_addresses, seed_arm_qpos)
  seed_dof = np.array(
    [data.qpos[qpos_addresses[name]] for name in dof_names], dtype=np.float32
  )
  seed_dof = safe_float32_dof(seed_dof, dof_names, model, joint_ids)
  set_dof_qpos(data, seed_dof, dof_names, qpos_addresses)
  mujoco.mj_forward(model, data)
  if has_self_collision(model, data, args.max_self_penetration):
    raise ValueError("the reference pose has a penetrating robot self-contact")
  seed_positions, _ = end_effector_transforms(model, data, end_effector_body_ids)
  seed_positions_frame = (
    reference_frame_rotation.T @ (seed_positions - reference_frame_position).T
  ).T
  if (
    np.any(seed_positions_frame[:, 0] < args.min_end_effector_forward)
    or seed_positions_frame[0, 1] < args.min_end_effector_side_clearance
    or seed_positions_frame[1, 1] > -args.min_end_effector_side_clearance
    or np.abs(
      seed_positions_frame[0, (0, 2)] - seed_positions_frame[1, (0, 2)]
    ).max() > args.max_parallel_axis_error
  ):
    raise ValueError("the reference pose does not satisfy endpoint geometry limits")
  motion_data["carry_pose_0000"] = {
    "dof": np.repeat(seed_dof[None, :], args.frames_per_clip, axis=0),
    "fps": args.fps,
  }
  print("[carry_pose_0000] reference pose")
  print(
    "Endpoint offset bounds in torso frame: "
    f"lower={np.round(endpoint_offset_lower, 4)}, "
    f"upper={np.round(endpoint_offset_upper, 4)}"
  )

  ik_skips = 0
  residual_skips = 0
  collision_skips = 0
  for _ in range(num_candidates):
    if len(motion_data) >= args.num_poses:
      break
    (
      target_positions_frame,
      target_rotations_frame,
      translation,
      rotation_vectors,
    ) = sample_endpoint_targets(
      reference_positions_frame,
      reference_rotations_frame,
      rng,
      endpoint_offset_lower,
      endpoint_offset_upper,
      rotation_range_rad,
      args.wide_sample_ratio,
      args.min_end_effector_forward,
      args.min_end_effector_side_clearance,
    )
    target_positions = reference_frame_position + (
      reference_frame_rotation @ target_positions_frame.T
    ).T
    target_rotations = np.einsum(
      "ij,bjk->bik", reference_frame_rotation, target_rotations_frame
    )
    try:
      (
        arm_qpos,
        position_error,
        orientation_error,
        parallel_axis_error,
      ) = solve_arm_ik(
        model,
        data,
        qpos_addresses,
        end_effector_body_ids,
        initial_arm_qpos,
        lower,
        upper,
        target_positions,
        target_rotations,
        args.position_tolerance,
        orientation_tolerance_rad,
        reference_frame_position,
        reference_frame_rotation,
        args.max_parallel_axis_error,
        args.joint_regularization,
      )
    except RuntimeError:
      ik_skips += 1
      continue
    solved_positions, _ = end_effector_transforms(model, data, end_effector_body_ids)
    solved_positions_frame = (
      reference_frame_rotation.T @ (solved_positions - reference_frame_position).T
    ).T
    if (
      position_error > args.max_end_effector_position_error
      or orientation_error > max_orientation_error_rad
      or parallel_axis_error > args.max_parallel_axis_error
      or solved_positions_frame[0, 1] < args.min_end_effector_side_clearance
      or solved_positions_frame[1, 1] > -args.min_end_effector_side_clearance
    ):
      residual_skips += 1
      continue

    data.qpos[:] = model.qpos0
    set_arm_qpos(data, qpos_addresses, arm_qpos)
    dof = np.array([data.qpos[qpos_addresses[name]] for name in dof_names], dtype=np.float32)
    dof = safe_float32_dof(dof, dof_names, model, joint_ids)
    set_dof_qpos(data, dof, dof_names, qpos_addresses)
    mujoco.mj_forward(model, data)
    if has_self_collision(model, data, args.max_self_penetration):
      collision_skips += 1
      continue
    saved_positions, _ = end_effector_transforms(model, data, end_effector_body_ids)
    saved_positions_frame = (
      reference_frame_rotation.T @ (saved_positions - reference_frame_position).T
    ).T
    if (
      np.any(saved_positions_frame[:, 0] < args.min_end_effector_forward)
      or saved_positions_frame[0, 1] < args.min_end_effector_side_clearance
      or saved_positions_frame[1, 1] > -args.min_end_effector_side_clearance
      or np.abs(
        saved_positions_frame[0, (0, 2)] - saved_positions_frame[1, (0, 2)]
      ).max() > args.max_parallel_axis_error
    ):
      residual_skips += 1
      continue
    clip = np.repeat(dof[None, :], args.frames_per_clip, axis=0)
    key = f"carry_pose_{len(motion_data):04d}"
    motion_data[key] = {"dof": clip, "fps": args.fps}
    print(
      f"[{key}] position_error={position_error * 1000.0:.2f} mm, "
      f"orientation_error={math.degrees(orientation_error):.2f} deg, "
      f"parallel_xz_error={parallel_axis_error * 1000.0:.2f} mm, "
      f"endpoint_translation_torso={np.round(translation, 4)}, "
      f"endpoint_position_torso={np.round(target_positions_frame, 4)}, "
      f"endpoint_rotation_deg={np.round(np.degrees(rotation_vectors), 2)}"
    )

  print(
    f"Accepted {len(motion_data) - 1}/{num_candidates} sampled candidates; "
    f"skipped ik={ik_skips}, residual={residual_skips}, collision={collision_skips}."
  )
  if len(motion_data) < args.num_poses:
    print(
      f"[warn] requested up to {args.num_poses} clips but wrote only "
      f"{len(motion_data)} valid poses; increase --num-candidates to sample more."
    )

  args.output.parent.mkdir(parents=True, exist_ok=True)
  joblib.dump(motion_data, args.output)
  print(
    f"Wrote {args.output}: {len(motion_data)} static clips, "
    f"{args.frames_per_clip} frames/clip at {args.fps} FPS"
  )
  if args.playback:
    playback(
      model,
      data,
      motion_data,
      dof_names,
      qpos_addresses,
      end_effector_body_ids,
      args.playback_duration,
      args.playback_max_poses,
      args.playback_once,
    )


if __name__ == "__main__":
  main()
