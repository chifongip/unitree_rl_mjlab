"""Integration tests for the opt-in walk-to-stand experiment."""
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from src.tasks.locomanipulation.config.x2.env_cfgs import (
  agibot_x2_locomanipulation_transition_flat_env_cfg,
)
from src.tasks.locomanipulation.mdp.symmetry import _get_term_mirror_x2
from src.tasks.locomanipulation.config.x2.rl_cfg import agibot_x2_locomanipulation_ppo_runner_cfg
from src.tasks.locomanipulation.rl import X2_LocomanipulationOnPolicyRunner


@pytest.fixture(scope="module")
def env():
  cfg = agibot_x2_locomanipulation_transition_flat_env_cfg()
  cfg.scene.num_envs = 4
  cfg.seed = 42
  environment = ManagerBasedRlEnv(cfg, device="cpu")
  yield environment
  environment.close()


def test_dimensions_profiles_and_real_steps(env):
  obs, _ = env.reset()
  assert obs["actor"].shape == (4, 430)
  assert obs["critic"].shape == (4, 121)
  cmd = env.command_manager.get_term("twist")
  cmd.profile[:] = torch.tensor([0, 1, 2, 2])
  cmd.switch_at[:] = .02
  for _ in range(4):
    obs, reward, *_ = env.step(torch.zeros((4, 15)))
    assert torch.isfinite(reward).all()
    assert torch.isfinite(obs["critic"]).all()
  assert torch.count_nonzero(cmd.command[1:]) == 0
  assert torch.equal(env.command_manager.get_command("base_height"), torch.full((4, 1), .64))


def test_command_standing_and_bounded_transition(env, monkeypatch):
  env.reset()
  cmd = env.command_manager.get_term("twist")
  state = cmd.state()
  cmd.vel_command_b[:] = 0
  state.reset(torch.arange(4))
  env.common_step_counter += 1
  state.advance()
  # Standing from reset immediately uses the original rewards.
  assert torch.all(state.mode == 2)
  terms = [env.reward_manager.get_term_cfg(name) for name in
           ("stand_still", "leg_joint_vel_penalty", "base_drift_penalty")]
  for term in terms:
    params = {k: v for k, v in term.params.items()
              if k not in ("original_func", "transition_kind")}
    expected = term.func.func(env, **params)
    torch.testing.assert_close(term.func(env, **term.params), expected)
  assert (terms[0].func(env, **terms[0].params) > 0).all()

  cmd.vel_command_b[:, 0] = .5
  state.reconcile()
  assert torch.all(state.mode == 0)
  cmd.vel_command_b[:] = 0
  state.reconcile()
  assert torch.all(state.mode == 1)
  age = state.age.clone()
  state.critic()
  torch.testing.assert_close(age, state.age)

  # Deliberately bad posture, placement, speed and missing contacts cannot
  # extend the grace period or prevent the original standing rewards returning.
  data = SimpleNamespace(
    joint_pos=state.targets[0].expand(4, -1).clone() + .5,
    root_link_lin_vel_b=torch.ones((4, 3)),
  )
  monkeypatch.setattr(state, "robot", SimpleNamespace(data=data))
  monkeypatch.setattr(state, "joint_ids", list(range(len(state.names))))
  monkeypatch.setattr(state, "geometry", lambda: (
    torch.ones(4, 2, 3), torch.zeros(4, dtype=torch.long)))
  env.scene["feet_ground_contact"].data.found[:] = 0
  steps = round(cmd.cfg.transition_s / env.step_dt)
  for step in range(steps):
    env.common_step_counter += 1
    state.advance()
    old = state.age.clone()
    state.advance()
    torch.testing.assert_close(old, state.age)
    assert torch.all(state.mode == (1 if step < steps - 1 else 2))
    for term in terms:
      params = {k: v for k, v in term.params.items()
                if k not in ("original_func", "transition_kind")}
      value = term.func(env, **term.params)
      expected = torch.zeros_like(value) if step < steps - 1 else term.func.func(env, **params)
      torch.testing.assert_close(value, expected)
  for _ in range(10):
    env.common_step_counter += 1
    state.advance()
    assert torch.all(state.mode == 2)
  # The original combined command threshold still applies.
  cmd.vel_command_b[1] = torch.tensor([.06, 0., .05])
  state.reconcile()
  assert state.mode[1] == 0 and state.age[1] == 0
  cmd.vel_command_b[1] = 0
  state.reconcile()
  assert state.mode[1] == 1
  age = state.age.clone()
  state.reset(torch.tensor([0]))
  assert state.mode[0] == 2 and state.age[0] == 0
  assert state.mode[1] == 1
  torch.testing.assert_close(state.age[1:], age[1:])


def test_mirror_signed_placement_and_double_identity():
  rule = _get_term_mirror_x2("stop_transition")
  x = torch.tensor([[0., 1., 0., .5, .03, .02, .1, -.03, -.02, -.2]])
  original = x.clone()
  rule.apply(x, None)
  torch.testing.assert_close(x[0, 4:], torch.tensor([-.03, .02, .2, .03, -.02, -.1]))
  rule.apply(x, None)
  torch.testing.assert_close(x, original)


def test_stance_targets_follow_current_body_heading(env, monkeypatch):
  from mjlab.utils.lab_api.math import quat_apply, quat_mul
  env.reset()
  state = env.command_manager.get_term("twist").state()
  robot = env.scene["robot"].data
  data = SimpleNamespace(
    body_link_pos_w=robot.body_link_pos_w.clone(),
    body_link_quat_w=robot.body_link_quat_w.clone(),
    heading_w=robot.heading_w.clone(),
  )
  monkeypatch.setattr(state, "robot", SimpleNamespace(data=data))
  before = state.geometry()[0].clone()
  # Rotate the complete configuration in world space, including the current
  # body heading. Body-relative errors must be unchanged even after stopping.
  yaw = torch.tensor([.3, -.4, .7, -.8])
  rotation = torch.zeros((4, data.body_link_pos_w.shape[1], 4))
  rotation[..., 0] = torch.cos(yaw[:, None] / 2)
  rotation[..., 3] = torch.sin(yaw[:, None] / 2)
  data.body_link_pos_w = quat_apply(rotation, data.body_link_pos_w)
  data.body_link_quat_w = quat_mul(rotation, data.body_link_quat_w)
  data.heading_w += yaw
  torch.testing.assert_close(state.geometry()[0], before, atol=2e-6, rtol=0)
  # Change only the present heading: stance targets follow it, without a
  # reconcile()/command edge or historical heading update.
  data.heading_w += .3
  torch.testing.assert_close(state.geometry()[0][..., 2], before[..., 2] - .3,
                             atol=2e-6, rtol=0)


def test_training_from_scratch_updates_actor_and_critic(env, tmp_path):
  env.reset()
  from mjlab.rl import RslRlVecEnvWrapper
  cfg = asdict(agibot_x2_locomanipulation_ppo_runner_cfg())
  cfg["algorithm"]["num_learning_epochs"] = 1
  cfg["algorithm"]["num_mini_batches"] = 1
  cfg["num_steps_per_env"] = 4
  runner = X2_LocomanipulationOnPolicyRunner(RslRlVecEnvWrapper(env), cfg, device="cpu")
  actor_before = [p.detach().clone() for p in runner.alg._raw_actor.parameters()]
  critic_before = [p.detach().clone() for p in runner.alg._raw_critic.parameters()]
  runner.learn(2)
  assert any(not torch.equal(a, b) for a, b in zip(actor_before, runner.alg._raw_actor.parameters()))
  assert any(not torch.equal(a, b) for a, b in zip(critic_before, runner.alg._raw_critic.parameters()))
  assert all(torch.isfinite(p).all() for p in runner.alg._raw_actor.parameters())
  assert all(torch.isfinite(p).all() for p in runner.alg._raw_critic.parameters())
  runner.save(str(tmp_path / "model_1.pt"))
  assert (tmp_path / "policy.onnx").exists()


def test_potential_does_not_reward_speed_cycles():
  v = torch.tensor([0., .2, 0., .2, 0.])
  phi = -v
  reward = .99 * phi[1:] - phi[:-1]
  torch.testing.assert_close(sum(.99**i * x for i, x in enumerate(reward)), torch.tensor(0.),
                             atol=1e-7, rtol=0)


def test_evaluation_requires_physical_stopping():
  from scripts.eval_stopping import _is_stopped
  # Reward phase completion alone cannot turn drift, turning, or stepping into
  # an evaluation success; a stationary, grounded configuration does pass.
  velocity = torch.tensor([[0., 0.], [.2, 0.], [0., 0.], [0., 0.]])
  yaw_velocity = torch.tensor([0., 0., .2, 0.])
  contact = torch.tensor([[True, True], [True, True], [True, True], [True, False]])
  assert _is_stopped(velocity, yaw_velocity, contact).tolist() == [True, False, False, False]


def test_paired_evaluation_rejects_regressions():
  from copy import deepcopy
  from scripts.eval_stopping import compare
  row = dict(height=.64, command=[.5, 0., 0.], phase=0, censored=False, success=True, failed=False,
    post_touchdown_hip_yaw_rad=.2, touchdown_posture_rms=.1, contact_slip_m=.01,
    walking_velocity_mae=.1, walking_yaw_mae=.1, last_touchdown_s=.4,
    final_height_error_m=.01, final_placement_error_m=.01, final_heading_error_rad=.05)
  baseline = dict(seed=42, trials=[row], summary={"fall_rate": 0.})
  candidate = deepcopy(baseline)
  candidate["trials"][0].update(post_touchdown_hip_yaw_rad=.08, touchdown_posture_rms=.05)
  import json
  result = compare(baseline, candidate)
  assert result["passed"]
  json.dumps(result)  # CLI reports must remain JSON serializable.
  # Comparative style improvements cannot compensate for failure to stop.
  candidate["trials"][0]["success"] = False
  result = compare(baseline, candidate)
  assert not result["passed"] and not result["criteria"]["stopping_success"]
  candidate["trials"][0]["success"] = True
  for key, value, criterion in (
    ("final_height_error_m", .04, "height_accuracy"),
    ("final_placement_error_m", .04, "placement_accuracy"),
    ("final_heading_error_rad", .3, "heading_accuracy"),
  ):
    previous = candidate["trials"][0][key]
    candidate["trials"][0][key] = value
    result = compare(baseline, candidate)
    assert not result["passed"] and not result["criteria"][criterion]
    candidate["trials"][0][key] = previous
  candidate["trials"][0]["failed"] = True
  assert not compare(baseline, candidate)["passed"]
  candidate["seed"] = 43
  with pytest.raises(ValueError, match="match"):
    compare(baseline, candidate)


def test_held_out_evaluation_preserves_training_rng(monkeypatch):
  import numpy as np
  from scripts import eval_stopping
  before = torch.get_rng_state().clone()
  numpy_before = np.random.get_state()
  threads = torch.get_num_threads()
  def fake(*args):
    torch.manual_seed(123)
    np.random.seed(123)
    torch.set_num_threads(1)
    return {"ok": True}
  monkeypatch.setattr(eval_stopping, "_evaluate", fake)
  assert eval_stopping.evaluate("unused") == {"ok": True}
  assert torch.equal(torch.get_rng_state(), before)
  np.testing.assert_array_equal(np.random.get_state()[1], numpy_before[1])
  assert torch.get_num_threads() == threads


@torch.inference_mode()
def test_shaping_closes_at_transition_exits(env, monkeypatch):
  from src.tasks.locomanipulation.mdp.stopping import stopping_reward
  env.reset()
  cmd = env.command_manager.get_term("twist")
  state = cmd.state()
  for name in ("placement_weight", "swing_weight", "slip_weight"):
    monkeypatch.setattr(cmd.cfg, name, 0.)
  monkeypatch.setattr(cmd.cfg, "shaping_weight", 1.)
  speed = torch.full((env.num_envs,), .2)
  monkeypatch.setattr(state, "speed", lambda: speed)

  def rollout(brake_step, exit_kind="timer", initial_speed=.2):
    state.reset(torch.arange(env.num_envs))
    speed[:] = initial_speed
    cmd.vel_command_b[:, 0] = .5
    cmd.vel_command_b[:, 1:] = 0
    state.reconcile()
    cmd.vel_command_b[:] = 0
    state.reconcile()
    total = torch.zeros(env.num_envs)
    for step in range(1, round(cmd.cfg.transition_s / env.step_dt) + 4):
      env.common_step_counter += 1
      if step >= brake_step:
        speed[:] = 0
      if step == 10 and exit_kind == "command":
        cmd.vel_command_b[:, 0] = .5
      if step == 10 and exit_kind == "termination":
        env.reset_terminated[:] = True
      total += cmd.cfg.gamma ** (step - 1) * stopping_reward(env) * env.step_dt
    env.reset_terminated[:] = False
    return total

  # Both braking at expiry and exiting with residual speed must close the
  # potential. Extra standing steps must not pay the boundary increment again.
  for initial_speed in (.2, .4):
    for brake_step in (5, 59, 60, 100):
      torch.testing.assert_close(rollout(brake_step, initial_speed=initial_speed), torch.zeros(env.num_envs),
                                 atol=1e-6, rtol=0)
  for exit_kind in ("command", "termination"):
    torch.testing.assert_close(rollout(40, exit_kind), torch.zeros(env.num_envs),
                               atol=1e-6, rtol=0)


def test_settling_time_requires_sustained_stop():
  from scripts.eval_stopping import _settle_time
  stopped = torch.zeros(100, dtype=torch.bool)
  stopped[0] = True
  assert _settle_time(stopped, .02) is None
  stopped[75:] = True
  assert _settle_time(stopped, .02) == pytest.approx(1.5)
  assert _settle_time(torch.ones(24, dtype=torch.bool), .02) is None
  assert _settle_time(torch.ones(25, dtype=torch.bool), .02) == 0.


@torch.inference_mode()
def test_stand_to_walk_schedule_and_partial_reset(env, monkeypatch):
  cmd = env.command_manager.get_term("twist")
  monkeypatch.setattr(cmd.cfg, "standing_ratio", 0.)
  monkeypatch.setattr(cmd.cfg, "transition_ratio", 0.)
  monkeypatch.setattr(cmd.cfg, "start_walking_ratio", 1.)
  env.reset()
  state = cmd.state()
  assert (cmd.profile == 3).all()
  assert ((cmd.switch_at >= 2.) & (cmd.switch_at <= 4.)).all()
  sampled = cmd.walk_command.clone()
  assert (sampled.norm(dim=1) >= .1).all()
  assert torch.count_nonzero(cmd.command) == 0
  assert (state.mode == 2).all()

  # Give each environment a different scheduled start and check both sides of
  # the boundary without resetting its physical state or observation history.
  cmd.switch_at[:] = torch.tensor([2., 2.5, 3., 3.5])
  env.episode_length_buf[:] = 99
  cmd._update_command()
  assert torch.count_nonzero(cmd.command) == 0
  env.episode_length_buf[:] = 100
  root_before = env.scene["robot"].data.root_link_pos_w.clone()
  cmd._update_command()
  torch.testing.assert_close(cmd.command[0], sampled[0])
  assert torch.count_nonzero(cmd.command[1:]) == 0
  assert state.mode.tolist() == [0, 2, 2, 2]
  torch.testing.assert_close(env.scene["robot"].data.root_link_pos_w, root_before)
  assert (env.episode_length_buf == 100).all()

  env.episode_length_buf[:] = 180
  cmd._update_command()
  torch.testing.assert_close(cmd.command, sampled)
  assert (state.mode == 0).all()
  for _ in range(3):
    obs, reward, *_ = env.step(torch.zeros((4, 15)))
    assert torch.isfinite(reward).all()
    assert torch.isfinite(obs["actor"]).all()
    torch.testing.assert_close(cmd.command, sampled)

  # Resampling/resetting one episode must preserve the other walking commands.
  env.reset(env_ids=torch.tensor([0]))
  assert cmd.profile[0] == 3 and state.mode[0] == 2
  assert torch.count_nonzero(cmd.command[0]) == 0
  torch.testing.assert_close(cmd.walk_command[1:], sampled[1:])
  torch.testing.assert_close(cmd.command[1:], sampled[1:])
  assert (state.mode[1:] == 0).all()


@torch.inference_mode()
def test_stand_to_walk_disabled_in_play(env, monkeypatch):
  env.reset()
  cmd = env.command_manager.get_term("twist")
  monkeypatch.setattr(cmd.cfg, "sample_profiles", False)
  cmd.profile[:] = 3
  cmd.is_heading_env[:] = False
  cmd.is_standing_env[:] = False
  cmd.vel_command_b[:] = torch.tensor([.4, 0., 0.])
  cmd.walk_command[:] = .8
  cmd.switch_at[:] = 100.
  cmd._update_command()
  torch.testing.assert_close(cmd.command, torch.tensor([[.4, 0., 0.]]).expand(4, -1))


def test_transition_factory_preserves_loaded_transport_configuration():
  from src.tasks.locomanipulation.config.x2.env_cfgs import agibot_x2_locomanipulation_flat_env_cfg
  baseline = agibot_x2_locomanipulation_flat_env_cfg()
  transition = agibot_x2_locomanipulation_transition_flat_env_cfg()
  assert transition.events == baseline.events
  assert transition.actions == baseline.actions
  assert transition.curriculum == baseline.curriculum
  assert transition.terminations == baseline.terminations
  assert transition.observations["actor"] == baseline.observations["actor"]


@torch.inference_mode()
def test_transition_metrics_exclude_walking_postures(env, monkeypatch):
  env.reset()
  cmd = env.command_manager.get_term("twist")
  state = cmd.state()
  state.reset(torch.arange(4))
  cmd.vel_command_b[:] = 0
  cmd.vel_command_b[0, 0] = .5
  errors = torch.zeros(4, 2, 3)
  errors[0] = 10.  # Walking must not contaminate standing posture diagnostics.
  errors[1:, :, 0] = .02
  monkeypatch.setattr(state, "geometry", lambda: (errors, torch.zeros(4, dtype=torch.long)))
  env.common_step_counter += 1
  state.advance()
  torch.testing.assert_close(env.extras["log"]["Transition/placement_error"], torch.tensor(.02))
  cmd.vel_command_b[:, 0] = .5
  env.common_step_counter += 1
  state.advance()
  assert env.extras["log"]["Transition/placement_error"] == 0
  assert env.extras["log"]["Transition/hip_yaw_error"] == 0


@pytest.mark.parametrize('starting', [False, True])
@pytest.mark.parametrize('lead', [0, 1])
@torch.inference_mode()
def test_shared_foot_stages_and_early_completion(env, starting, lead):
  env.reset()
  cmd = env.command_manager.get_term('twist')
  state = cmd.state()
  state.reset(torch.arange(4))
  contact = env.scene['feet_ground_contact'].data.found
  contact[:] = 1
  cmd.vel_command_b[:] = 0
  if not starting:
    cmd.vel_command_b[:, 0] = .5
    state.reconcile()
    cmd.vel_command_b[:] = 0
  else:
    cmd.vel_command_b[:, 0] = .5
  state.reconcile()
  assert state.active.all() and (state.foot_stage == 0).all()
  mask = state.pose_mask()
  assert (mask[:, sum(state.leg_columns, [])] == 0).all()
  torso = [i for i in range(len(state.names)) if i not in sum(state.leg_columns, [])]
  assert (mask[:, torso] == 1).all()

  def tick(left, right):
    contact[:] = torch.tensor([left, right])
    env.common_step_counter += 1
    state.advance()

  airborne = [1, 1]
  airborne[lead] = 0
  tick(*airborne)
  assert (state.foot_stage == 0).all()  # one-sample contact noise
  tick(1, 1)
  tick(*airborne)
  tick(*airborne)
  assert (state.foot_stage[:, lead] == 1).all()
  assert (state.pose_mask()[:, state.leg_columns[lead]] == 1).all()
  assert (state.pose_mask()[:, state.leg_columns[1-lead]] == 0).all()
  tick(1, 1)
  assert (state.foot_stage[:, lead] == 1).all()
  tick(1, 1)
  assert state.touchdown[:, lead].all() and state.active.all()
  airborne = [1, 1]
  airborne[1-lead] = 0
  tick(*airborne)
  tick(*airborne)
  tick(1, 1)
  tick(1, 1)
  assert state.finished.all() and not state.deadline.any()
  assert not state.active.any()
  assert (state.mode == (0 if starting else 2)).all()
  assert (state.age < cmd.cfg.transition_s).all()


@torch.inference_mode()
def test_transition_reversal_airborne_initialization_and_partial_reset(env):
  env.reset()
  cmd = env.command_manager.get_term('twist')
  state = cmd.state()
  state.reset(torch.arange(4))
  env.scene['feet_ground_contact'].data.found[:] = torch.tensor([0, 1])
  state.air_ticks[:, 0] = 2  # confirmed airborne history before the command
  cmd.vel_command_b[:] = torch.tensor([.5, 0., 0.])
  state.reconcile()
  assert state.starting.all()
  assert (state.foot_stage == torch.tensor([1, 0])).all()
  cmd.vel_command_b[:] = 0
  state.reconcile()
  assert (state.mode == 1).all() and not state.starting.any()
  assert (state.age == 0).all()
  before = state.feet_critic().clone()
  state.reset(torch.tensor([0]))
  assert state.mode[0] == 2 and not state.active[0]
  torch.testing.assert_close(state.feet_critic()[1:], before[1:])


def test_transition_feet_mirror():
  mirror = _get_term_mirror_x2('transition_feet')
  value = torch.tensor([[1., 0., .5, 0., 1., .5]])
  original = value.clone()
  mirror.apply(value, None, history_length=2)
  torch.testing.assert_close(value, torch.tensor([[1., .5, 0., 0., .5, 1.]]))
  mirror.apply(value, None, history_length=2)
  torch.testing.assert_close(value, original)


@torch.inference_mode()
def test_pose_mask_removes_only_pending_leg_incentive(env, monkeypatch):
  env.reset()
  cmd = env.command_manager.get_term('twist')
  state = cmd.state()
  env.scene['feet_ground_contact'].data.found[:] = 1
  state.reset(torch.arange(4))
  cmd.vel_command_b[:] = torch.tensor([.5, 0., 0.])
  state.reconcile()
  term = env.reward_manager.get_term_cfg('pose')
  params = {k:v for k,v in term.params.items() if k not in ('original_func','transition_kind')}
  data = env.scene['robot'].data
  positions = data.joint_pos.clone()
  monkeypatch.setattr(type(data), 'joint_pos', property(lambda _: positions))
  joint = state.joint_ids[state.leg_columns[0][0]]
  original = data.joint_pos[:, joint].clone()
  try:
    masked = term.func.func(env, joint_mask=state.pose_mask(), **params)
    data.joint_pos[:, joint] += .5
    torch.testing.assert_close(term.func.func(env, joint_mask=state.pose_mask(), **params), masked)
    state.foot_stage[:, 0] = 1
    changed = term.func.func(env, joint_mask=state.pose_mask(), **params)
    data.joint_pos[:, joint] = original
    restored = term.func.func(env, joint_mask=state.pose_mask(), **params)
    assert not torch.allclose(changed, restored)
  finally:
    data.joint_pos[:, joint] = original


@pytest.mark.parametrize('starting', [False, True])
@torch.inference_mode()
def test_sole_guidance_is_landing_only(env, monkeypatch, starting):
  from src.tasks.locomanipulation.mdp.stopping import stopping_reward
  env.reset()
  cmd = env.command_manager.get_term('twist')
  state = cmd.state()
  monkeypatch.setattr(state, 'advance', lambda: None)
  monkeypatch.setattr(cmd.cfg, 'placement_weight', 0.)
  monkeypatch.setattr(cmd.cfg, 'shaping_weight', 0.)
  state.guidance_active[:] = True
  state.was_moving[:] = starting
  state.touchdown[:] = False
  state.landing_guidance[:] = False
  state.foot_stage[:] = 2
  state.foot_stage[:, 0] = 1
  contact = env.scene['feet_ground_contact'].data.found
  contact[:] = 1
  contact[:, 0] = 0
  errors = torch.zeros(4, 2, 3)
  def geometry(destination=False):
    assert destination
    return errors, torch.zeros(4, dtype=torch.long)
  monkeypatch.setattr(state, 'geometry', geometry)
  torch.testing.assert_close(stopping_reward(env), torch.zeros(4))
  state.landing_guidance[:, 0] = True
  aligned = stopping_reward(env)
  torch.testing.assert_close(aligned * env.step_dt, torch.full((4,), cmd.cfg.swing_weight))
  errors[:, 0, 2] = .5
  assert (stopping_reward(env) < aligned).all()
  contact[:] = 1
  state.landing_guidance[:] = False
  torch.testing.assert_close(stopping_reward(env), torch.zeros(4))
  state.touchdown[:, 0] = True
  state.landing_guidance[:, 0] = True
  assert (stopping_reward(env) > 0).all()
  state.guidance_active[:] = False
  torch.testing.assert_close(stopping_reward(env), torch.zeros(4))


@pytest.mark.parametrize('starting', [False, True])
@torch.inference_mode()
def test_completion_handles_a_foot_lifting_again(env, starting):
  env.reset()
  cmd = env.command_manager.get_term('twist')
  state = cmd.state()
  state.reset(torch.arange(4))
  contact = env.scene['feet_ground_contact'].data.found
  contact[:] = 1
  cmd.vel_command_b[:] = torch.tensor([.5, 0., 0.])
  state.reconcile()
  if not starting:
    cmd.vel_command_b[:] = 0
    state.reconcile()
  def tick(left, right):
    contact[:] = torch.tensor([left, right])
    env.common_step_counter += 1
    state.advance()
  for pair in [(0,1),(0,1),(1,1),(1,1),(1,0),(1,0),(0,1),(0,1)]:
    tick(*pair)
  assert (state.foot_stage == 2).all()  # both have landed once
  assert bool(state.finished.all()) == starting
  if not starting:
    assert state.active.all()
    tick(1,1)
    assert not state.finished.any()
    tick(1,1)
    assert state.finished.all() and not state.deadline.any()


@torch.inference_mode()
def test_contact_dropout_at_command_edge_is_not_a_swing(env):
  env.reset()
  cmd = env.command_manager.get_term('twist')
  state = cmd.state()
  state.reset(torch.arange(4))
  contact = env.scene['feet_ground_contact'].data.found
  contact[:] = 1
  cmd.vel_command_b[:] = torch.tensor([.5, 0., 0.])
  state.reconcile()
  # Single raw dropout, with no confirmed air history.
  contact[:] = 0
  cmd.vel_command_b[:] = 0
  state.reconcile()
  assert (state.foot_stage == 0).all()
  for _ in range(2):
    contact[:] = 1
    env.common_step_counter += 1
    state.advance()
  assert (state.foot_stage == 0).all()
  assert not state.finished.any() and not state.landing_paid.any()
  assert (state.pose_mask()[:, sum(state.leg_columns, [])] == 0).all()


@torch.inference_mode()
def test_landing_bonus_has_no_airtime_or_repeat_payment(env, monkeypatch):
  from src.tasks.locomanipulation.mdp.stopping import stopping_reward
  env.reset()
  cmd = env.command_manager.get_term('twist')
  state = cmd.state()
  contact = env.scene['feet_ground_contact'].data.found
  monkeypatch.setattr(cmd.cfg, 'shaping_weight', 0.)
  monkeypatch.setattr(state, 'geometry', lambda destination=False: (
    torch.zeros(4,2,3), torch.zeros(4,dtype=torch.long)))
  def rollout(air_frames, repeat=False):
    state.reset(torch.arange(4))
    contact[:] = 1
    cmd.vel_command_b[:] = torch.tensor([.5,0.,0.])
    state.reconcile()
    cmd.vel_command_b[:] = 0
    state.reconcile()
    total = torch.zeros(4)
    def tick(left,right):
      contact[:] = torch.tensor([left,right])
      env.common_step_counter += 1
      reward = stopping_reward(env) * env.step_dt
      return reward
    for _ in range(air_frames):
      reward = tick(0,1)
      torch.testing.assert_close(reward, torch.zeros(4))
      total += reward
    total += tick(1,1)
    total += tick(1,1)
    assert state.landing_paid[:,0].all()
    if repeat:
      for pair in [(0,1),(0,1),(1,1),(1,1)]:
        reward = tick(*pair)
        torch.testing.assert_close(reward, torch.zeros(4))
        total += reward
    for pair in [(1,0),(1,0),(1,1),(1,1),(1,1)]:
      total += tick(*pair)
    return total
  expected = torch.full((4,), 2 * (cmd.cfg.swing_weight + cmd.cfg.placement_weight))
  torch.testing.assert_close(rollout(3), expected)
  torch.testing.assert_close(rollout(12), expected)
  torch.testing.assert_close(rollout(3, repeat=True), expected)
