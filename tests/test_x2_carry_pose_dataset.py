"""Regression tests for X2 carry-pose generation and validation."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import joblib
import mujoco
import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/generate_x2_carry_pose_dataset.py"
SPEC = importlib.util.spec_from_file_location("generate_x2_carry_pose_dataset", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


def test_rotation_error_handles_half_turn():
  target = generator.rotation_matrix_from_rotvec(np.array([math.pi, 0.0, 0.0]))

  error = generator.rotation_error_vector(np.eye(3), target)

  assert np.linalg.norm(error) == pytest.approx(math.pi)


def test_self_collision_detects_penetrating_robot_geoms():
  model = mujoco.MjModel.from_xml_string(
    "<mujoco><worldbody>"
    '<body name="left"><freejoint/><geom type="sphere" size="0.1"/></body>'
    '<body name="right" pos="0.1 0 0">'
    '<freejoint/><geom type="sphere" size="0.1"/></body>'
    "</worldbody></mujoco>"
  )
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)

  assert generator.has_self_collision(model, data, max_penetration=0.001)
  assert not generator.has_self_collision(model, data, max_penetration=0.11)


def test_playback_file_skips_generation(tmp_path, monkeypatch):
  input_path = tmp_path / "carry_poses.pkl"
  output_path = tmp_path / "not_generated.pkl"
  joblib.dump({"carry_pose_0000": {"dof": np.zeros((1, 29), dtype=np.float32), "fps": 30}}, input_path)
  calls = []
  monkeypatch.setattr(generator, "playback", lambda *args: calls.append(args))
  monkeypatch.setattr(
    sys,
    "argv",
    [
      str(SCRIPT_PATH),
      "--playback-file", str(input_path),
      "--output", str(output_path),
      "--num-poses", "0",
      "--playback-max-poses", "1",
      "--playback-once",
    ],
  )

  generator.main()

  assert not output_path.exists()
  assert len(calls) == 1
  assert list(calls[0][2]) == ["carry_pose_0000"]
  assert calls[0][-2:] == (1, True)


def test_playback_file_rejects_wrong_joint_count(tmp_path):
  input_path = tmp_path / "invalid.pkl"
  joblib.dump({"bad": {"dof": np.zeros((1, 14), dtype=np.float32), "fps": 30}}, input_path)

  with pytest.raises(ValueError, match=r"shape \(frames, 29\)"):
    generator.load_motion_data(input_path, 29)


def test_saved_carry_poses_obey_geometry_and_collision_limits(tmp_path, monkeypatch):
  output = tmp_path / "carry_poses.pkl"
  monkeypatch.setattr(
    sys,
    "argv",
    [
      str(SCRIPT_PATH),
      "--output", str(output),
      "--num-poses", "4",
      "--num-candidates", "30",
      "--frames-per-clip", "1",
      "--endpoint-offset-lower", "-0.01", "-0.01", "-0.01",
      "--endpoint-offset-upper", "0.02", "0.01", "0.02",
      "--rotation-range-deg", "2",
      "--min-end-effector-forward", "0.19",
    ],
  )
  generator.main()

  clips = joblib.load(output)
  assert len(clips) == 4
  model = mujoco.MjModel.from_xml_path(str(generator.DEFAULT_XML))
  data = mujoco.MjData(model)
  dof_names, addresses, _ = generator.scalar_joint_info(model)
  torso_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, generator.REFERENCE_FRAME_BODY_NAME
  )
  endpoint_ids = tuple(
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    for name in generator.END_EFFECTOR_BODY_NAMES
  )
  for clip in clips.values():
    generator.set_dof_qpos(data, clip["dof"][0], dof_names, addresses)
    mujoco.mj_forward(model, data)
    positions, _ = generator.end_effector_transforms(model, data, endpoint_ids)
    torso_rotation = data.xmat[torso_id].reshape(3, 3)
    local = (torso_rotation.T @ (positions - data.xpos[torso_id]).T).T

    assert np.all(local[:, 0] >= 0.19)
    assert local[0, 1] >= 0.0
    assert local[1, 1] <= 0.0
    assert np.abs(local[0, (0, 2)] - local[1, (0, 2)]).max() <= 0.002
    assert not generator.has_self_collision(model, data, max_penetration=0.001)
