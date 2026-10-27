"""Complete controller persistence through training, playback and evaluation."""

from copy import deepcopy
from dataclasses import asdict, replace

import pytest
import yaml

from mjlab.utils.os import dump_yaml
from scripts import eval as evaluation, play
from scripts.train import TrainConfig, _dump_env_overrides_yaml, _restore_training_torque_compensation
from src.assets.robots.agibot_x2.compensated_actuator import CompensatedPositionActuatorCfg
from src.tasks.locomanipulation.config.x2.env_cfgs import agibot_x2_locomanipulation_flat_env_cfg
from src.tasks.locomanipulation.config.x2.rl_cfg import agibot_x2_locomanipulation_ppo_runner_cfg
from src.tasks.locomanipulation.config.x2.torque_compensation import configure_x2_torque_compensation


@pytest.fixture(params=[play, evaluation], ids=["play", "eval"])
def checkpoint_io(request):
  return request.param


def _customized_training_config(algorithm="nonlinear", mode="ACTIVE"):
  cfg = agibot_x2_locomanipulation_flat_env_cfg()
  configure_x2_torque_compensation(cfg, algorithm=algorithm, mode=mode)
  action = cfg.actions["upper_body_motion"]
  observer = action.observer
  action.observer = replace(
    observer, inertia=tuple(v * 1.1 for v in observer.inertia),
    correction_limits_nm=(0.75,) * 14, friction_limit=(0.1,) * 14,
    friction_viscosity=tuple(v * 0.9 for v in observer.friction_viscosity),
    bandwidth_rad_s=5.5, slew_nm_s=1.25,
    release_slew_nm_s=6.5 if algorithm == "nonlinear" else 1.25,
    position_noise_std=0.001, velocity_noise_std=0.02,
  )
  robot = cfg.scene.entities["robot"]
  robot.articulation.actuators = tuple(
    replace(a, stiffness=55.0, damping=2.5, effort_limit=32.0)
    if "shoulder_pitch" in a.target_names_expr[0] else a
    for a in robot.articulation.actuators
  )
  return cfg


def _assert_controller_equal(training, playback):
  assert playback.actions["upper_body_motion"].observer == training.actions["upper_body_motion"].observer
  trained_arms = [a for a in training.scene.entities["robot"].articulation.actuators
                  if isinstance(a, CompensatedPositionActuatorCfg)]
  played_arms = [a for a in playback.scene.entities["robot"].articulation.actuators
                 if isinstance(a, CompensatedPositionActuatorCfg)]
  assert played_arms == trained_arms


@pytest.mark.parametrize("algorithm", ["linearized", "nonlinear"])
@pytest.mark.parametrize("mode", ["OFF", "SHADOW", "ACTIVE"])
def test_saved_yaml_restores_exact_controller_and_arm_settings(
  tmp_path, capsys, algorithm, mode, checkpoint_io,
):
  training = _customized_training_config(algorithm, mode)
  _dump_env_overrides_yaml(tmp_path / "env_overrides.yaml", asdict(training))
  overrides = checkpoint_io._load_params_overrides(tmp_path)
  playback = agibot_x2_locomanipulation_flat_env_cfg(play=True)
  observations = deepcopy(playback.observations)
  lower_action = deepcopy(playback.actions["joint_pos"])
  checkpoint_io._apply_env_overrides(playback, overrides)
  _assert_controller_equal(training, playback)
  assert playback.observations == observations
  assert playback.actions["joint_pos"] == lower_action
  assert "restored" in capsys.readouterr().out


def test_legacy_incomplete_overrides_are_supplemented_from_full_yaml(tmp_path, checkpoint_io):
  training = _customized_training_config()
  _dump_env_overrides_yaml(tmp_path / "env_overrides.yaml", asdict(training))
  dump_yaml(tmp_path / "env.yaml", asdict(training))
  data = yaml.safe_load((tmp_path / "env_overrides.yaml").read_text())
  observer = data["actions"]["upper_body_motion"]["observer"]
  for name in ("joint_names", "inertia", "correction_limits_nm", "friction_limit", "friction_viscosity"):
    observer.pop(name)
  data["scene"]["entities"]["robot"]["articulation"].pop("actuators")
  (tmp_path / "env_overrides.yaml").write_text(yaml.safe_dump(data))
  playback = agibot_x2_locomanipulation_flat_env_cfg(play=True)
  checkpoint_io._apply_env_overrides(playback, checkpoint_io._load_params_overrides(tmp_path))
  _assert_controller_equal(training, playback)


def test_full_yaml_fallback_restores_controller(tmp_path, checkpoint_io):
  training = _customized_training_config()
  dump_yaml(tmp_path / "env.yaml", asdict(training))
  playback = agibot_x2_locomanipulation_flat_env_cfg(play=True)
  checkpoint_io._apply_env_overrides(playback, checkpoint_io._load_params_overrides(tmp_path))
  _assert_controller_equal(training, playback)


def test_corrupt_legacy_full_yaml_cannot_discard_known_observer_metadata(tmp_path, checkpoint_io):
  training = _customized_training_config()
  _dump_env_overrides_yaml(tmp_path / "env_overrides.yaml", asdict(training))
  data = yaml.safe_load((tmp_path / "env_overrides.yaml").read_text())
  data["actions"]["upper_body_motion"]["observer"].pop("inertia")
  data["scene"]["entities"]["robot"]["articulation"].pop("actuators")
  (tmp_path / "env_overrides.yaml").write_text(yaml.safe_dump(data))
  (tmp_path / "env.yaml").write_text("broken: [")
  saved = checkpoint_io._load_params_overrides(tmp_path)
  assert saved is not None
  with pytest.raises(ValueError, match="complete arm actuator settings"):
    checkpoint_io._apply_env_overrides(agibot_x2_locomanipulation_flat_env_cfg(play=True), saved)


@pytest.mark.parametrize("broken", ["vectors", "actuators", "invalid_model"])
def test_incomplete_or_invalid_saved_controller_cannot_silently_use_pd(
  tmp_path, broken, checkpoint_io,
):
  training = _customized_training_config()
  _dump_env_overrides_yaml(tmp_path / "env_overrides.yaml", asdict(training))
  saved = checkpoint_io._load_params_overrides(tmp_path)
  if broken == "vectors":
    saved["actions"]["upper_body_motion"]["observer"].pop("inertia")
  elif broken == "actuators":
    saved["scene"]["entities"]["robot"]["articulation"].pop("actuators")
  else:
    saved["actions"]["upper_body_motion"]["observer"]["inertia"] = [-1.0] * 14
  playback = agibot_x2_locomanipulation_flat_env_cfg(play=True)
  original_actuators = deepcopy(playback.scene.entities["robot"].articulation.actuators)
  original_observer = deepcopy(playback.actions["upper_body_motion"].observer)
  with pytest.raises(ValueError):
    checkpoint_io._apply_env_overrides(playback, saved)
  assert playback.actions["upper_body_motion"].observer == original_observer
  assert playback.scene.entities["robot"].articulation.actuators == original_actuators


def test_pd_only_checkpoint_keeps_observer_disabled(tmp_path, checkpoint_io):
  training = agibot_x2_locomanipulation_flat_env_cfg()
  training.actions["upper_body_motion"].observer = None
  _dump_env_overrides_yaml(tmp_path / "env_overrides.yaml", asdict(training))
  playback = agibot_x2_locomanipulation_flat_env_cfg(play=True)
  configure_x2_torque_compensation(playback)
  checkpoint_io._apply_env_overrides(playback, checkpoint_io._load_params_overrides(tmp_path))
  assert playback.actions["upper_body_motion"].observer is None


@pytest.mark.parametrize("metadata", ["missing_field", "missing_params"])
def test_legacy_pd_checkpoint_disables_factory_observer(tmp_path, checkpoint_io, metadata):
  if metadata == "missing_field":
    legacy = asdict(agibot_x2_locomanipulation_flat_env_cfg())
    del legacy["actions"]["upper_body_motion"]["observer"]
    _dump_env_overrides_yaml(tmp_path / "env_overrides.yaml", legacy)
  playback = agibot_x2_locomanipulation_flat_env_cfg(play=True)
  checkpoint_io._apply_env_overrides(playback, checkpoint_io._load_params_overrides(tmp_path))
  assert playback.actions["upper_body_motion"].observer is None


def test_training_resume_restores_custom_controller_before_environment_creation(tmp_path):
  training = _customized_training_config()
  _dump_env_overrides_yaml(tmp_path / "params" / "env_overrides.yaml", asdict(training))
  env = agibot_x2_locomanipulation_flat_env_cfg()
  cfg = TrainConfig(env=env, agent=agibot_x2_locomanipulation_ppo_runner_cfg())
  _restore_training_torque_compensation(cfg, tmp_path / "model_1.pt")
  _assert_controller_equal(training, env)


def test_environment_limits_roundtrip_to_resume_play_and_eval(tmp_path, checkpoint_io):
  training = agibot_x2_locomanipulation_flat_env_cfg()
  configure_x2_torque_compensation(
    training, correction_limit_fraction=0.8, correction_limit_nm=0,
  )
  expected = (28.8, 28.8, 19.2, 19.2, 19.2, 1.76, 1.76) * 2
  assert training.actions["upper_body_motion"].observer.correction_limits_nm == pytest.approx(expected)
  _dump_env_overrides_yaml(tmp_path / "params" / "env_overrides.yaml", asdict(training))
  playback = agibot_x2_locomanipulation_flat_env_cfg(play=True)
  checkpoint_io._apply_env_overrides(playback, checkpoint_io._load_params_overrides(tmp_path / "params"))
  _assert_controller_equal(training, playback)
  cfg = TrainConfig(
    env=agibot_x2_locomanipulation_flat_env_cfg(),
    agent=agibot_x2_locomanipulation_ppo_runner_cfg(),
  )
  _restore_training_torque_compensation(cfg, tmp_path / "model_1.pt")
  _assert_controller_equal(training, cfg.env)


def test_environment_can_override_saved_controller_for_play_and_eval(tmp_path, checkpoint_io):
  training = _customized_training_config("linearized", "SHADOW")
  _dump_env_overrides_yaml(tmp_path / "env_overrides.yaml", asdict(training))
  playback = agibot_x2_locomanipulation_flat_env_cfg(play=True)
  configure_x2_torque_compensation(
    playback, correction_limit_fraction=0.8, correction_limit_nm=0,
  )
  playback.actions["upper_body_motion"].restore_observer_from_checkpoint = False
  expected = deepcopy(playback)
  checkpoint_io._apply_env_overrides(playback, checkpoint_io._load_params_overrides(tmp_path))
  _assert_controller_equal(expected, playback)


def test_environment_can_override_saved_controller_for_training_resume(tmp_path):
  training = _customized_training_config("linearized", "SHADOW")
  _dump_env_overrides_yaml(tmp_path / "params" / "env_overrides.yaml", asdict(training))
  env = agibot_x2_locomanipulation_flat_env_cfg()
  configure_x2_torque_compensation(env, correction_limit_fraction=0.8, correction_limit_nm=0)
  env.actions["upper_body_motion"].restore_observer_from_checkpoint = False
  expected = deepcopy(env)
  cfg = TrainConfig(env=env, agent=agibot_x2_locomanipulation_ppo_runner_cfg())
  _restore_training_torque_compensation(cfg, tmp_path / "model_1.pt")
  _assert_controller_equal(expected, env)


@pytest.mark.parametrize("metadata", ["disabled", "missing_field", "missing_params"])
def test_environment_can_enable_compensation_when_finetuning_pd_checkpoint(tmp_path, metadata):
  if metadata != "missing_params":
    saved = asdict(agibot_x2_locomanipulation_flat_env_cfg())
    if metadata == "disabled":
      saved["actions"]["upper_body_motion"]["observer"] = None
    else:
      del saved["actions"]["upper_body_motion"]["observer"]
    _dump_env_overrides_yaml(tmp_path / "params" / "env_overrides.yaml", saved)
  env = agibot_x2_locomanipulation_flat_env_cfg()
  env.actions["upper_body_motion"].restore_observer_from_checkpoint = False
  cfg = TrainConfig(env=env, agent=agibot_x2_locomanipulation_ppo_runner_cfg())
  _restore_training_torque_compensation(cfg, tmp_path / "model_1.pt")
  assert env.actions["upper_body_motion"].observer.mode == "ACTIVE"


@pytest.mark.parametrize("metadata", ["missing_field", "missing_params"])
def test_training_resume_legacy_pd_checkpoint_disables_factory_observer(tmp_path, metadata):
  if metadata == "missing_field":
    legacy = asdict(agibot_x2_locomanipulation_flat_env_cfg())
    del legacy["actions"]["upper_body_motion"]["observer"]
    _dump_env_overrides_yaml(tmp_path / "params" / "env_overrides.yaml", legacy)
  env = agibot_x2_locomanipulation_flat_env_cfg()
  cfg = TrainConfig(env=env, agent=agibot_x2_locomanipulation_ppo_runner_cfg())
  _restore_training_torque_compensation(cfg, tmp_path / "model_1.pt")
  assert env.actions["upper_body_motion"].observer is None


def test_fresh_training_keeps_factory_observer():
  env = agibot_x2_locomanipulation_flat_env_cfg()
  cfg = TrainConfig(env=env, agent=agibot_x2_locomanipulation_ppo_runner_cfg())
  _restore_training_torque_compensation(cfg, None)
  assert env.actions["upper_body_motion"].observer.mode == "ACTIVE"


def test_training_resume_respects_explicitly_disabled_saved_observer(tmp_path):
  training = agibot_x2_locomanipulation_flat_env_cfg()
  training.actions["upper_body_motion"].observer = None
  _dump_env_overrides_yaml(tmp_path / "params" / "env_overrides.yaml", asdict(training))
  env = agibot_x2_locomanipulation_flat_env_cfg()
  assert env.actions["upper_body_motion"].observer.mode == "ACTIVE"
  cfg = TrainConfig(env=env, agent=agibot_x2_locomanipulation_ppo_runner_cfg())
  _restore_training_torque_compensation(cfg, tmp_path / "model_1.pt")
  assert env.actions["upper_body_motion"].observer is None
