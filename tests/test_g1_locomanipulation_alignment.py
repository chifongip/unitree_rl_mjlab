"""Regression checks for G1 static arm targets, carry, and observer restore."""

from dataclasses import asdict
import joblib
import mujoco
import numpy as np
import pytest

from mjlab.tasks.registry import load_env_cfg
from scripts.generate_x2_carry_pose_dataset import has_self_collision, scalar_joint_info
from scripts.play import _load_params_overrides, _restore_torque_compensation
from scripts.train import _dump_env_overrides_yaml
from src.assets.robots.unitree_g1.g1_23dof_constants import G1_23DOF_XML
from src.assets.robots.unitree_g1.g1_constants import G1_XML
from src.tasks.locomanipulation.config.g1.reference import ARM_23, ARM_29, CARRY_23, CARRY_29
from src.tasks.locomanipulation.mdp.payload import SlidingBoxForceEvent


def test_eval_carry_presets_match_g1_references():
  from scripts.eval import POSE_PRESETS

  assert POSE_PRESETS["g1_carry"] == CARRY_29
  assert POSE_PRESETS["g1_23dof_carry"] == CARRY_23


def test_play_selects_carry_pose_for_g1_23dof():
  from scripts.play import PREDEFINED_POSES

  assert PREDEFINED_POSES[3] == CARRY_23


@pytest.mark.parametrize("robot,arm_names", [("G1", ARM_29), ("G1-23Dof", ARM_23)])
@pytest.mark.parametrize("suffix", ["Flat", "Rough", "Carry-Finetune-Flat"])
def test_g1_training_factory(robot, arm_names, suffix):
  cfg = load_env_cfg(f"Unitree-{robot}-Locomanipulation-{suffix}")
  action = cfg.actions["upper_body_motion"]
  assert action.pose_only and action.locomotion_motion_file is None
  assert action.observer.joint_names == arm_names
  assert action.observer.algorithm == "nonlinear" and action.observer.mode == "ACTIVE"
  height = cfg.commands["base_height"]
  assert (height.nominal_height, height.max_deviation_down, height.max_deviation_up) == (
    0.76, 0.26, 0.02,
  )
  assert ("waist_regulation" in cfg.rewards) == (robot == "G1")
  if suffix.startswith("Carry"):
    event = cfg.events["hand_force"]
    assert event.func is SlidingBoxForceEvent
    assert action.default_pose_ratio == 0.15
    assert event.params["no_force_ratio"] == 0.3
    assert event.params["torque_bounds"]["joint_names"] == arm_names
    assert cfg.curriculum["force_curriculum"].params["start_step"] == 0
    clips = joblib.load(action.motion_file)
    assert len(clips) > 1
    model = mujoco.MjModel.from_xml_path(str(G1_XML if robot == "G1" else G1_23DOF_XML))
    data = mujoco.MjData(model)
    names, addresses, ids = scalar_joint_info(model)
    assert tuple(names[i] for i in action.motion_dof_indices) == arm_names
    limits = model.jnt_range[[ids[name] for name in names]]
    hands = event.params["asset_cfg"].body_names
    up_local = np.array(event.params["hand_up_local"])
    for clip in clips.values():
      frame = clip["dof"][0]
      assert (clip["dof"] == frame).all()
      assert np.all((frame >= limits[:, 0]) & (frame <= limits[:, 1]))
      data.qpos[[addresses[name] for name in names]] = frame
      mujoco.mj_forward(model, data)
      assert not has_self_collision(model, data, 0.001)
      left, right = (data.body(hand) for hand in hands)
      span = left.xipos - right.xipos
      assert np.linalg.norm(span) >= event.params["payload"]["min_hand_separation_m"]
      up = np.mean(
        [body.xmat.reshape(3, 3) @ axis for body, axis in zip((left, right), up_local)],
        axis=0,
      )
      up -= span * (up @ span) / (span @ span)
      assert up[2] / np.linalg.norm(up) > 0.1
    reference = CARRY_29 if robot == "G1" else CARRY_23
    np.testing.assert_allclose(
      clips["carry_pose_0000"]["dof"][0, list(action.motion_dof_indices)],
      [reference[name] for name in arm_names],
    )


@pytest.mark.parametrize("robot", ["G1", "G1-23Dof"])
def test_g1_controller_restore(robot, tmp_path):
  task = f"Unitree-{robot}-Locomanipulation-Flat"
  source = load_env_cfg(task)
  source.actions["upper_body_motion"].observer.mode = "OFF"
  _dump_env_overrides_yaml(tmp_path / "env_overrides.yaml", asdict(source))
  saved = _load_params_overrides(tmp_path)
  cfg = load_env_cfg(task)
  _restore_torque_compensation(cfg, saved)
  assert cfg.actions["upper_body_motion"].observer.mode == "OFF"

  observer = saved["actions"]["upper_body_motion"]["observer"]
  observer.pop("inertia")
  with pytest.raises(ValueError, match="incomplete"):
    _restore_torque_compensation(cfg, saved)
  assert cfg.actions["upper_body_motion"].observer.mode == "OFF"

  _restore_torque_compensation(cfg, None)
  assert cfg.actions["upper_body_motion"].observer is None


@pytest.mark.parametrize("robot", ["G1", "G1-23Dof"])
def test_g1_observer_effort_and_partial_reset(robot, tmp_path, monkeypatch):
  import torch
  import warp
  from mjlab.envs import ManagerBasedRlEnv

  monkeypatch.setattr(warp.config, "kernel_cache_dir", str(tmp_path / "warp"))
  cfg = load_env_cfg(f"Unitree-{robot}-Locomanipulation-Flat", play=True)
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    obs, _ = env.reset()
    actor_dim, critic_dim, action_dim = (430, 108, 15) if robot == "G1" else (360, 94, 13)
    assert obs["actor"].shape == (2, actor_dim)
    assert obs["critic"].shape == (2, critic_dim)
    assert env.action_manager.total_action_dim == action_dim
    term = env.action_manager.get_term("upper_body_motion")
    observer = term._observer
    action = torch.zeros((2, env.action_manager.total_action_dim))
    for _ in range(8):
      env.step(action)
    assert torch.count_nonzero(observer.effort) > 0
    assert (observer.effort.abs() <= observer.limits).all()

    survivor = observer.effort[1].clone()
    env.reset(env_ids=torch.tensor([0]))
    assert not observer.initialized[0]
    assert torch.count_nonzero(observer.effort[0]) == 0
    torch.testing.assert_close(observer.effort[1], survivor)

    observer.cfg.mode = "OFF"
    env.step(action)
    assert torch.count_nonzero(observer.effort) == 0
  finally:
    env.close()


@pytest.mark.parametrize("robot", ["Agibot-X2", "Unitree-G1", "Unitree-G1-23Dof"])
@pytest.mark.parametrize("play", [False, True])
def test_locomanipulation_contact_capacity(robot, play):
  rough = load_env_cfg(f"{robot}-Locomanipulation-Rough", play=play)
  flat = load_env_cfg(f"{robot}-Locomanipulation-Flat", play=play)
  carry = load_env_cfg(f"{robot}-Locomanipulation-Carry-Finetune-Flat", play=play)
  assert rough.sim.nconmax == 128
  assert flat.sim.nconmax is None
  assert carry.sim.nconmax is None
