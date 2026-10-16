"""Regression coverage for physical X2 playback payloads."""

import mujoco
import numpy as np
import pytest
import torch
import joblib
from types import SimpleNamespace

from mjlab.scene import Scene
from scripts.play import PREDEFINED_POSES_X2, _patch_command_compute
from src.tasks.locomanipulation.config.x2.carry_box import (
  add_carry_box, add_sampled_carry_box, _reset_box, _sampled_box_geometry,
)
from src.tasks.locomanipulation.config.x2.env_cfgs import (
  agibot_x2_locomanipulation_flat_env_cfg,
)

CENTERED_TEST_POSE = dict(zip(PREDEFINED_POSES_X2[3], (
  -0.681775, 0.372695, -0.166413, -1.506897, 0.706496, 0.516424, 0.448444,
  -0.867395, -0.010435, 0.459897, -1.453243, -0.646434, 0.518135, -0.519556,
)))


@pytest.mark.parametrize("mass", [1.0, 10.0])
@pytest.mark.parametrize("depth", [0.3, 0.6])
def test_payload_mass_inertia_and_grasp_geometry(mass, depth):
  cfg = agibot_x2_locomanipulation_flat_env_cfg(play=True)
  original_joint_defaults = dict(cfg.scene.entities["robot"].init_state.joint_pos)
  add_carry_box(cfg, PREDEFINED_POSES_X2[3], mass, depth=depth)
  cfg.scene.num_envs = 1
  model = Scene(cfg.scene, device="cpu").compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)

  assert model.body("carry_box/box").mass[0] == pytest.approx(mass)
  assert np.all(model.body("carry_box/box").inertia > 0)
  box_z_dof = model.joint("carry_box/box_joint").dofadr[0] + 2
  assert data.qfrc_bias[box_z_dof] == pytest.approx(mass * 9.81)
  assert model.neq == 2
  assert np.all(model.eq_type == mujoco.mjtEq.mjEQ_WELD)
  assert {model.body(int(body)).name for body in model.eq_obj1id} == {
    "robot/left_wrist_roll_link", "robot/right_wrist_roll_link",
  }
  box_geom = model.geom("carry_box/box_geom").id
  torso_geom = model.geom("robot/torso_collision").id
  distance = mujoco.mj_geomDistance(model, data, box_geom, torso_geom, 1.0, None)
  assert distance >= 0.029
  # Body contact is enabled, so clearance is physical rather than hidden by an
  # exclusion. Only the distal wrist assembly may bypass payload contact.
  for signature in model.exclude_signature:
    bodies = (int(signature) >> 16, int(signature) & 65535)
    names = [model.body(body).name for body in bodies]
    if "carry_box/box" in names:
      assert any("wrist" in name for name in names)
  np.testing.assert_allclose(data.efc_pos[:12], 0, atol=1e-12)
  assert np.isfinite(data.qacc).all()
  assert cfg.events["hand_force"].params["constant_force"]["z"] == 0
  assert cfg.actions["upper_body_motion"].fixed_upper_body_pose == PREDEFINED_POSES_X2[3]
  # Playback must not mutate the shared robot initial state used in training.
  fresh = agibot_x2_locomanipulation_flat_env_cfg()
  assert fresh.scene.entities["robot"].init_state.joint_pos == original_joint_defaults


@pytest.mark.parametrize("mass", [0, -1, float("nan"), float("inf")])
def test_invalid_payload_mass(mass):
  cfg = agibot_x2_locomanipulation_flat_env_cfg(play=True)
  with pytest.raises(ValueError, match="finite and positive"):
    add_carry_box(cfg, PREDEFINED_POSES_X2[3], mass)


def test_partial_reset_rotates_payload_with_robot_root():
  poses = torch.tensor([
    [0, 0, 0.68, 1, 0, 0, 0],
    [4, 5, 0.68, np.sqrt(0.5), 0, 0, np.sqrt(0.5)],
  ], dtype=torch.float32)
  written = {}
  def write(state, env_ids):
    written["state"] = state
    written["ids"] = env_ids
  box = SimpleNamespace(
    data=SimpleNamespace(default_root_state=torch.zeros(2, 13)),
    write_root_state_to_sim=write,
  )
  env = SimpleNamespace(
    sim=SimpleNamespace(forward=lambda: None), device="cpu",
    scene={"robot": SimpleNamespace(data=SimpleNamespace(root_link_pose_w=poses)),
           "carry_box": box},
  )
  _reset_box(env, torch.tensor([1]), (np.float64(0.2), 0.0, 0.3))
  torch.testing.assert_close(written["state"][0, :3], torch.tensor([4, 5.2, 0.98]))
  torch.testing.assert_close(written["state"][0, 3:7], poses[1, 3:7])
  assert written["ids"].tolist() == [1]


@pytest.mark.parametrize("name,target", [
  ("twist", [0.4, 0.0, 0.0]), ("base_height", [0.6]), ("waist_yaw", [0.1]),
])
def test_play_commands_compute_and_keyboard_override(name, target):
  cfg = agibot_x2_locomanipulation_flat_env_cfg(play=True)
  robot = SimpleNamespace(
    joint_names=["waist_yaw_joint"],
    data=SimpleNamespace(
      root_link_pos_w=torch.tensor([[0, 0, 0.68], [0, 0, 0.68]]),
      joint_pos=torch.zeros(2, 1), root_link_lin_vel_b=torch.zeros(2, 3),
      root_link_ang_vel_b=torch.zeros(2, 3),
      heading_w=torch.zeros(2),
    ),
  )
  env = SimpleNamespace(scene={"robot": robot}, num_envs=2, device="cpu", step_dt=0.02)
  term = cfg.commands[name].build(env)
  term.reset(torch.tensor([0, 1]))
  term.compute(0.02)
  override = SimpleNamespace(
    vel_active=True, target_vel_x=0.4, target_vel_y=0.0, target_ang_vel_z=0.0,
    height_active=True, target_height=0.6,
    waist_yaw_active=True, target_waist_yaw=0.1,
  )
  _patch_command_compute(term, override, name)
  other_timer = term.time_left[1].clone()
  other_command = term.command[1].clone()
  term.compute(0.02, torch.tensor([0]))
  torch.testing.assert_close(term.command[0], torch.tensor(target))
  torch.testing.assert_close(term.time_left[1], other_timer)
  torch.testing.assert_close(term.command[1], other_command)


def test_sampled_pose_configuration_keeps_sampling_and_nominal_defaults(tmp_path):
  cfg = agibot_x2_locomanipulation_flat_env_cfg(play=True)
  defaults = dict(cfg.scene.entities["robot"].init_state.joint_pos)
  from mjlab.entity import Entity
  entity = Entity(cfg.scene.entities["robot"])
  model = entity.spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  for name, value in CENTERED_TEST_POSE.items():
    data.qpos[model.joint(name).qposadr[0]] = value
  dof = np.array([data.qpos[model.joint(name).qposadr[0]] for name in entity.joint_names])
  path = tmp_path / "static_poses.pkl"
  joblib.dump({"pose": {"dof": np.repeat(dof[None], 2, axis=0), "fps": 30}}, path)
  add_sampled_carry_box(cfg, str(path), 10)
  action = cfg.actions["upper_body_motion"]
  assert action.motion_file == str(path)
  assert action.fixed_upper_body_pose is None
  assert action.pose_only and action.default_pose_ratio == 0
  assert action.locomotion_motion_file is None
  assert action.sample_clip_indices == (0,)
  assert cfg.scene.entities["robot"].init_state.joint_pos == defaults
  assert "eq_data" in cfg.events["reset_carry_box"].func.model_fields
  assert "body_inertia" in cfg.events["reset_carry_box"].func.model_fields
  joblib.dump({"pose": {"dof": np.array([dof, dof + 0.1]), "fps": 30}}, path)
  with pytest.raises(ValueError, match="static"):
    add_sampled_carry_box(agibot_x2_locomanipulation_flat_env_cfg(play=True), str(path), 10)


def test_sampled_box_geometry_stays_level_and_welds_match_dimensions():
  cfg = agibot_x2_locomanipulation_flat_env_cfg(play=True)
  add_carry_box(cfg, PREDEFINED_POSES_X2[3], 10)
  # Geometry fitting uses the robot's unprefixed standalone model.
  from mjlab.entity import Entity
  model = Entity(cfg.scene.entities["robot"]).spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  for name, value in CENTERED_TEST_POSE.items():
    data.qpos[model.joint(name).qposadr[0]] = value
  mujoco.mj_forward(model, data)
  for depth, height in ((0.2, 0.15), (0.4, 0.3)):
    center, quat, size, welds = _sampled_box_geometry(model, data, depth, height)
    assert size[0] == depth and size[2] == height and size[1] > 0
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, quat)
    rotation = rotation.reshape(3, 3)
    np.testing.assert_allclose(rotation[:, 2], [0, 0, 1], atol=1e-12)
    for side, weld in zip(("left", "right"), welds):
      hand = model.body(f"{side}_wrist_roll_link").id
      hand_rotation = data.xmat[hand].reshape(3, 3)
      grasp = center + rotation @ weld[:3]
      np.testing.assert_allclose(data.xpos[hand] + hand_rotation @ weld[3:6], grasp)
      palm = data.xpos[hand] + hand_rotation @ np.array([0, 0, -0.11])
      assert abs(np.linalg.norm(palm - grasp) - 0.044) < 0.001
      relative_rotation = np.zeros(9)
      mujoco.mju_quat2Mat(relative_rotation, weld[6:10])
      np.testing.assert_allclose(hand_rotation @ relative_rotation.reshape(3, 3), rotation, atol=1e-12)


def test_sampled_box_rejects_pose_instead_of_moving_box_away_from_hands():
  from mjlab.entity import Entity
  cfg = agibot_x2_locomanipulation_flat_env_cfg(play=True)
  model = Entity(cfg.scene.entities["robot"]).spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  for name, value in PREDEFINED_POSES_X2[3].items():
    data.qpos[model.joint(name).qposadr[0]] = value
  mujoco.mj_forward(model, data)
  with pytest.raises(ValueError, match="center"):
    _sampled_box_geometry(model, data, 0.4, 0.3)
