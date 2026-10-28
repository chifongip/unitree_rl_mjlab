"""Full bilateral wrench bounds and independent MuJoCo reference checks."""

import mujoco
import numpy as np
import pytest
import torch

from src.tasks.locomanipulation.config.x2.env_cfgs import (
  agibot_x2_locomanipulation_carry_finetune_flat_env_cfg as carry_cfg,
)
from src.tasks.locomanipulation.mdp.events import maximum_wrench_scale
from src.tasks.locomanipulation.mdp.payload import (
  default_torque_bounds, enable_payload_playback, validate_torque_bounds,
)


def test_signed_scales_match_independent_direct_constraints():
  generator = torch.Generator().manual_seed(123)
  baseline = torch.rand(100, 5, generator=generator) * 30 - 15
  requested = torch.rand(100, 5, generator=generator) * 60 - 30
  requested[::3, 0] = 0
  lower, upper = torch.full_like(baseline, -9), torch.full_like(baseline, 9)
  scale, valid = maximum_wrench_scale(baseline, requested, lower, upper)
  grid = torch.linspace(0, 1, 10001)
  demand = baseline[:, None] - grid[None, :, None] * requested[:, None]
  direct = ((demand >= lower[:, None]) & (demand <= upper[:, None])).all(2)
  assert torch.equal(valid, direct.any(1))
  grid_max = torch.where(direct, grid, -1).amax(1)
  assert ((scale[valid] - grid_max[valid]).abs() <= 1.01e-4).all()
  effort = baseline - scale[:, None] * requested
  assert (effort[valid] >= lower[valid] - 2e-6).all()
  assert (effort[valid] <= upper[valid] + 2e-6).all()
  assert (scale[~valid] == 0).all()


def test_zero_request_helpful_load_and_conflicting_joint_intervals():
  baseline = torch.tensor([[1., 1.], [10., 0.], [10., 0.], [10., 0.]])
  requested = torch.tensor([[0., 0.], [2., 0.], [0., 1.], [2., 30.]])
  scale, valid = maximum_wrench_scale(
    baseline, requested, torch.full_like(baseline, -9), torch.full_like(baseline, 9),
  )
  assert valid.tolist() == [True, True, False, False]
  torch.testing.assert_close(scale, torch.tensor([1., 1., 0., 0.]))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("field", range(4))
def test_nonfinite_constraints_return_finite_zero_scale(bad, field):
  args = [torch.zeros(1, 1), torch.ones(1, 1), torch.full((1, 1), -9.), torch.full((1, 1), 9.)]
  args[field][:] = bad
  scale, valid = maximum_wrench_scale(*args)
  assert scale.item() == 0 and not valid.item()


@pytest.mark.parametrize("key,value", [
  ("enabled", 1), ("joint_names", ("arm", "arm")),
  ("effort_limit_scale", float("nan")), ("effort_limit_scale", 0),
  ("unknown_setting", 1),
])
def test_invalid_torque_settings_are_rejected(key, value):
  settings = default_torque_bounds(("arm",))
  settings[key] = value
  with pytest.raises(ValueError, match="torque_bounds"):
    validate_torque_bounds(settings)


@pytest.mark.parametrize("device", [
  "cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable",
  )),
])
def test_full_wrench_agrees_with_mujoco_without_mutating_state(device):
  from mjlab.envs import ManagerBasedRlEnv

  cfg = carry_cfg(play=True)
  enable_payload_playback(cfg)
  cfg.scene.num_envs = 2
  cfg.seed = 123
  cfg.events = {"hand_force": cfg.events["hand_force"]}
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    env.reset()
    event = env.event_manager.get_term_cfg("hand_force").func
    estimator = event._torque_estimator
    robot = env.scene["robot"]
    forces = torch.tensor([[[7., -11., -80.], [-3., 13., -70.]]], device=device).repeat(2, 1, 1)
    torques = torch.tensor([[[3., -8., 1.], [-4., 7., -2.]]], device=device).repeat(2, 1, 1)
    old = torch.cat((forces * .2, torques * .2), dim=-1)
    robot.write_external_wrench_to_sim(old[..., :3], old[..., 3:], body_ids=event._body_ids)
    before = {name: getattr(env.sim.data, name).clone() for name in ("qpos", "qvel", "qacc", "qfrc_actuator")}
    estimate = estimator.estimate_wrench_scale(env, forces, torques)
    for name, value in before.items():
      torch.testing.assert_close(getattr(env.sim.data, name).clone(), value)
    for n in range(2):
      data = mujoco.MjData(env.sim.mj_model)
      data.qpos[:] = env.sim.data.qpos[n].cpu().numpy()
      mujoco.mj_forward(env.sim.mj_model, data)
      generalized = np.zeros(env.sim.mj_model.nv)
      for h, body in enumerate(estimator._ee_global_body_ids):
        mujoco.mj_applyFT(
          env.sim.mj_model, data, forces[n, h].double().cpu().numpy(),
          torques[n, h].double().cpu().numpy(), data.xipos[body], body, generalized,
        )
      q = generalized[estimator._wrench_dof_adr.cpu().numpy()]
      current = robot.data.qfrc_actuator[n, estimator._wrench_joint_ids].cpu().numpy()
      np.testing.assert_allclose(estimate.requested_effort[n].cpu().numpy(), current - .8 * q, atol=4e-6)
      np.testing.assert_allclose(
        estimate.applied_effort[n].cpu().numpy(), current + (.2 - estimate.scale[n].item()) * q,
        atol=4e-6,
      )
    # Native limits can differ by environment, and reads must never mutate them.
    actuator, column = next((a, c) for a, c in estimator._wrench_actuators if getattr(a, "force_limit", None) is not None)
    initial = actuator.force_limit.clone()
    env.sim.expand_model_fields(("actuator_forcerange",))
    actuator.set_effort_limit(torch.tensor([0], device=device), initial[:1] * .5)
    native_before = env.sim.model.actuator_forcerange.clone()
    low, high = estimator._wrench_effort_bounds(env)
    torch.testing.assert_close(env.sim.model.actuator_forcerange.clone(), native_before)
    j = next(j for j, (a, c) in enumerate(estimator._wrench_actuators) if a is actuator and c == column)
    assert high[0, j].item() == pytest.approx(high[1, j].item() * .5)
    assert low[0, j] == -high[0, j]
  finally:
    env.close()
