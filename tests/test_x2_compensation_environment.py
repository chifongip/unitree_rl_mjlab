"""Headless full-environment validation of restored compensation and native DR."""

from copy import deepcopy
from dataclasses import asdict
import gc

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.dr.actuator import pd_gains
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from scripts.play import _apply_env_overrides, _load_params_overrides
from scripts.train import _dump_env_overrides_yaml
from src.tasks.locomanipulation.config.x2.env_cfgs import agibot_x2_locomanipulation_flat_env_cfg
from src.tasks.locomanipulation.config.x2.torque_compensation import configure_x2_torque_compensation


@pytest.fixture(autouse=True)
def _release_environment_memory():
  yield
  # Environments and their action terms reference each other. Collect those
  # cycles between CPU/CUDA cases rather than retaining several full plants.
  gc.collect()
  if torch.cuda.is_available():
    torch.cuda.empty_cache()


@pytest.mark.parametrize("algorithm", ["linearized", "nonlinear"])
@pytest.mark.parametrize("mode", ["OFF", "SHADOW", "ACTIVE"])
@pytest.mark.parametrize("device", [
  "cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable",
  )),
])
def test_restored_controller_runs_with_native_randomization_and_partial_reset(tmp_path, algorithm, mode, device):
  training = agibot_x2_locomanipulation_flat_env_cfg()
  configure_x2_torque_compensation(training, algorithm=algorithm, mode=mode)
  _dump_env_overrides_yaml(tmp_path / "env_overrides.yaml", asdict(training))
  cfg = agibot_x2_locomanipulation_flat_env_cfg(play=True)
  observations = deepcopy(cfg.observations)
  _apply_env_overrides(cfg, _load_params_overrides(tmp_path))
  assert cfg.observations == observations
  cfg.scene.num_envs = 2
  cfg.seed = 20261027
  # Exercise EventManager field expansion and the actual startup event, with
  # mixed builtin leg actuators and compensated arm actuators.
  cfg.events["pd_gains"] = EventTermCfg(
    func=pd_gains, mode="startup",
    params={"kp_range": (0.8, 1.2), "kd_range": (0.8, 1.2),
            "asset_cfg": SceneEntityCfg("robot")},
  )
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    obs, _ = env.reset()
    assert obs["actor"].shape == (2, 430)
    assert obs["critic"].shape == (2, 108)
    assert env.action_manager.total_action_dim == 15
    term = env.action_manager.get_term("upper_body_motion")
    torch.testing.assert_close(term._observer.kp[0, :4], torch.full((4,), 50.0, device=device))
    nominal_cfg = deepcopy(term._observer.cfg)
    for _ in range(8):
      obs, reward, *_ = env.step(torch.zeros((2, 15), device=device))
      assert torch.isfinite(obs["actor"]).all()
      assert torch.isfinite(obs["critic"]).all()
      assert torch.isfinite(reward).all()
      assert torch.isfinite(term._observer.effort).all()
      assert (term._observer.effort.abs() <= term._observer.limits).all()
      if mode != "ACTIVE":
        assert torch.count_nonzero(term._observer.effort) == 0
      robot = env.scene["robot"]
      actual_limits = env.sim.model.actuator_forcerange[:, robot.indexing.ctrl_ids, 1]
      assert (robot.data.actuator_force.abs() <= actual_limits + 1e-5).all()
    if mode == "ACTIVE":
      assert torch.count_nonzero(term._observer.effort) > 0
    assert term._observer.cfg == nominal_cfg
    survivor = term._observer.estimate[1].clone()
    env.reset(env_ids=torch.tensor([0], device=device))
    assert not term._observer.initialized[0]
    assert torch.count_nonzero(term._observer.effort[0]) == 0
    torch.testing.assert_close(term._observer.estimate[1], survivor)
    assert torch.count_nonzero(robot.data.joint_effort_target[0]) == 0
    env.step(torch.zeros((2, 15), device=device))
    assert torch.count_nonzero(term._observer.effort[0]) == 0  # Reinitialize first.
  finally:
    env.close()
