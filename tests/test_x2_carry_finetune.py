"""Tests for the X2 carry fine-tune load ramp and playback settings."""

from types import SimpleNamespace

import pytest

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from src.tasks.locomanipulation.config.x2.env_cfgs import (
  agibot_x2_locomanipulation_carry_finetune_flat_env_cfg,
)
from src.tasks.locomanipulation.mdp.curriculums import force_scale_staged
from src.tasks.locomanipulation.rl.runner import (
  X2_CarryFinetuneOnPolicyRunner,
  X2_LocomanipulationOnPolicyRunner,
)


def _fake_runner(step: int):
  event_cfg = SimpleNamespace(params={"force_scale": 0.0})
  stages = [
    {"step": 0, "scale": 0.0},
    {"step": 24, "scale": 0.5},
    {"step": 48, "scale": 1.0},
  ]
  force_cfg = CurriculumTermCfg(
    func=force_scale_staged,
    params={"event_name": "hand_force", "stages": stages, "start_step": 0},
  )
  env = SimpleNamespace(
    common_step_counter=step,
    event_manager=SimpleNamespace(get_term_cfg=lambda name: event_cfg),
    curriculum_manager=SimpleNamespace(get_term_cfg=lambda name: force_cfg),
  )
  runner = object.__new__(X2_CarryFinetuneOnPolicyRunner)
  runner.env = SimpleNamespace(unwrapped=env)
  return runner, env, force_cfg, event_cfg


def test_carry_force_ramp_starts_at_loaded_walking_step(monkeypatch):
  runner, env, force_cfg, event_cfg = _fake_runner(step=0)

  def load_checkpoint(self, path, load_cfg, strict, map_location):
    env.common_step_counter = 240_000
    return {"env_state": {"common_step_counter": 240_000}}

  monkeypatch.setattr(X2_LocomanipulationOnPolicyRunner, "load", load_checkpoint)
  runner.load("walking.pt")

  assert force_cfg.params["start_step"] == 240_000
  assert event_cfg.params["force_scale"] == 0.0
  env.common_step_counter += 25
  force_scale_staged(env, None, **force_cfg.params)
  assert event_cfg.params["force_scale"] == pytest.approx(0.5)


def test_carry_force_ramp_survives_carry_checkpoint(monkeypatch):
  runner, env, force_cfg, event_cfg = _fake_runner(step=0)

  def load_checkpoint(self, path, load_cfg, strict, map_location):
    env.common_step_counter = 240_050
    return {"carry_force_start_step": 240_000}

  saved_infos = {}

  def save_checkpoint(self, path, infos):
    saved_infos.update(infos)

  monkeypatch.setattr(X2_LocomanipulationOnPolicyRunner, "load", load_checkpoint)
  monkeypatch.setattr(X2_LocomanipulationOnPolicyRunner, "save", save_checkpoint)
  runner.load("carry.pt")
  runner.save("next.pt")

  assert force_cfg.params["start_step"] == 240_000
  assert event_cfg.params["force_scale"] == 1.0
  assert saved_infos["carry_force_start_step"] == 240_000


def test_carry_playback_inherits_parent_play_settings():
  cfg = agibot_x2_locomanipulation_carry_finetune_flat_env_cfg(play=True)

  assert cfg.commands["twist"].fixed_command == (0.0, 0.0, 0.0)
  assert cfg.events["hand_force"].params["constant_force"] == {
    "x": 0.0, "y": 0.0, "z": 0.0,
  }
  assert cfg.events["hand_force"].params["force_scale"] == 1.0
  assert cfg.actions["upper_body_motion"].fixed_upper_body_pose is not None
