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
  assert obs["critic"].shape == (4, 123)
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
  state.reset(torch.arange(4))  # Begin from an established walking reference.
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
  positions = env.scene["robot"].data.joint_pos.clone()
  positions[:, state.joint_ids] = state.targets[0] + .5
  data = SimpleNamespace(
    joint_pos=positions,
    root_link_lin_vel_b=torch.ones((4, 3)),
  )
  monkeypatch.setattr(state, "robot", SimpleNamespace(data=data))
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
      if term.func.kind == "stand_still":
        columns = [state.joint_ids.index(j) for j in params["asset_cfg"].joint_ids]
        params["joint_reference"] = state.reference[:, columns]
      expected = (torch.zeros_like(value) if step < steps - 1 and term.func.kind == "stand_still"
                  else term.func.func(env, **params))
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



def test_transition_feet_mirror():
  mirror = _get_term_mirror_x2('transition_feet')
  value = torch.tensor([[1., 0., .5, 0., 1., .5]])
  original = value.clone()
  mirror.apply(value, None, history_length=2)
  torch.testing.assert_close(value, torch.tensor([[1., .5, 0., 0., .5, 1.]]))
  mirror.apply(value, None, history_length=2)
  torch.testing.assert_close(value, original)




def prepare_blend(env, starting):
  env.reset()
  cmd = env.command_manager.get_term('twist')
  state = cmd.state()
  state.reset(torch.arange(4))
  contact = env.scene['feet_ground_contact'].data.found
  contact[:] = 1
  cmd.vel_command_b[:] = torch.tensor([0. if starting else .5, 0., 0.])
  state.reconcile()
  source = state.reference.clone()
  cmd.vel_command_b[:, 0] = .5 if starting else 0.
  state.reconcile()
  def tick(left=1, right=1):
    contact[:] = torch.tensor([left, right])
    env.common_step_counter += 1
    state.advance()
  return cmd, state, source, tick


@pytest.mark.parametrize('starting', [False, True])
@pytest.mark.parametrize('lead', [0, 1])
@torch.inference_mode()
def test_partial_blend_freezes_on_early_landing_and_resumes(env, starting, lead):
  cmd, state, source, tick = prepare_blend(env, starting)
  torch.testing.assert_close(state.reference, source)
  for _ in range(3):
    tick()
  torch.testing.assert_close(state.reference, source)
  air = [1,1]
  air[lead] = 0
  tick(*air)  # unconfirmed contact dropout
  torch.testing.assert_close(state.reference, source)
  for _ in range(4):
    tick(*air)
  assert ((state.progress[:, lead] > 0) & (state.progress[:, lead] < 1)).all()
  intermediate = state.reference.clone()
  tick()
  tick()
  torch.testing.assert_close(state.reference, intermediate)
  credit = state.credited_progress.clone()
  for _ in range(round(cmd.cfg.transition_s / env.step_dt) + 2):
    tick()
  assert not state.active.any()  # deadline restores normal routing
  torch.testing.assert_close(state.reference, intermediate)
  assert not (state.credited_progress >= 1).all()
  for _ in range(16):
    tick(*air)
  tick()
  tick()
  assert (state.progress[:, lead] == 1).all()
  assert (state.credited_progress[:, lead] == 1).all()
  assert (state.credited_progress[:, 1-lead] == credit[:, 1-lead]).all()
  torch.testing.assert_close(state.reference[:, state.leg_columns[lead]],
    state.destination_reference[:, state.leg_columns[lead]])


@torch.inference_mode()
def test_blend_reversal_reset_and_reference_symmetry(env):
  cmd, state, _, tick = prepare_blend(env, True)
  for _ in range(5):
    tick(0,1)
  before = state.reference.clone()
  cmd.vel_command_b[:] = 0
  state.reconcile()
  torch.testing.assert_close(state.reference, before)
  torch.testing.assert_close(state.reference_at(state.source_blend), before)
  state.reset(torch.tensor([0]))
  cmd.vel_command_b[0] = 0
  state.reconcile()
  torch.testing.assert_close(state.reference[1:], before[1:])
  assert (state.progress[0] == 1).all()
  mirror = _get_term_mirror_x2('transition_reference')
  value = state.reference_critic().clone()
  original = value.clone()
  mirror.apply(value, None)
  torch.testing.assert_close(value, original[:,[1,0]])
  mirror.apply(value, None)
  torch.testing.assert_close(value, original)


@torch.inference_mode()
@pytest.mark.parametrize("starting", [False, True])
def test_landed_progress_credit_is_bounded_and_never_pays_for_waiting(env, monkeypatch, starting):
  from src.tasks.locomanipulation.mdp.stopping import stopping_reward
  cmd, state, _, _ = prepare_blend(env, starting)
  monkeypatch.setattr(cmd.cfg, 'shaping_weight', 0.)
  monkeypatch.setattr(state, 'geometry', lambda destination=False:(
    torch.zeros(4,2,3), torch.zeros(4,dtype=torch.long)))
  contact = env.scene['feet_ground_contact'].data.found
  total = torch.zeros(4)
  def tick(left,right):
    contact[:] = torch.tensor([left,right])
    env.common_step_counter += 1
    value = stopping_reward(env) * env.step_dt
    return value
  for _ in range(4):
    torch.testing.assert_close(tick(0,1), torch.zeros(4))
  total += tick(1,1)
  total += tick(1,1)
  assert ((state.credited_progress[:,0] > 0) & (state.credited_progress[:,0] < 1)).all()
  for _ in range(70):
    torch.testing.assert_close(tick(1,1), torch.zeros(4))
  for _ in range(16):
    total += tick(0,1)
  total += tick(1,1)
  total += tick(1,1)
  for _ in range(16):
    total += tick(1,0)
  total += tick(1,1)
  total += tick(1,1)
  torch.testing.assert_close(total, torch.full((4,), 2 * cmd.cfg.swing_weight if starting else 0.))
  for _ in range(4):
    torch.testing.assert_close(tick(0,0), torch.zeros(4))
  for _ in range(4):
    torch.testing.assert_close(tick(1,1), torch.zeros(4))


@torch.inference_mode()
def test_blended_reference_is_used_by_pose_and_stand_still(env, monkeypatch):
  cmd, state, _, tick = prepare_blend(env, False)
  for _ in range(4):
    tick(0,1)
  tick()
  tick()
  for _ in range(70):
    tick()
  # References remain intermediate after normal standing penalties return.
  assert not state.active.any() and (state.progress[:,0] < 1).all()
  data = env.scene['robot'].data
  positions = data.joint_pos.clone()
  positions[:, state.joint_ids] = state.reference
  monkeypatch.setattr(type(data),'joint_pos',property(lambda _:positions))
  for name in ('pose','stand_still'):
    term = env.reward_manager.get_term_cfg(name)
    value = term.func(env, **term.params)
    torch.testing.assert_close(value, torch.ones(4) if name == 'pose' else torch.zeros(4))


@torch.inference_mode()
def test_absolute_blend_describes_reference_after_multiple_reversals(env):
  cmd, state, _, tick = prepare_blend(env, True)
  for moving in (True, False, True, False):
    before = state.reference.clone()
    cmd.vel_command_b[:, 0] = .5 if moving else 0.
    state.reconcile()
    torch.testing.assert_close(state.reference, before)
    for _ in range(4):
      tick(0,1)
    tick()
    tick()
    coordinate = state.reference_critic()
    assert coordinate.shape == (4,2)
    assert ((coordinate >= 0) & (coordinate <= 1)).all()
    index = state.reference_height_index
    for foot, columns in enumerate(state.leg_columns):
      expected = ((1-coordinate[:, foot, None]) * state.targets[index][:, columns]
                  + coordinate[:, foot, None] * state.walk_targets[index][:, columns])
      torch.testing.assert_close(state.reference[:, columns], expected, atol=1e-6, rtol=0)


@torch.inference_mode()
def test_jump_cannot_advance_references_or_earn_landing_credit(env, monkeypatch):
  from src.tasks.locomanipulation.mdp.stopping import stopping_reward
  cmd, state, source, tick = prepare_blend(env, False)
  monkeypatch.setattr(cmd.cfg, 'shaping_weight', 0.)
  monkeypatch.setattr(state, 'geometry', lambda destination=False: (
    torch.zeros(4,2,3), torch.zeros(4,dtype=torch.long)))
  for _ in range(20):
    tick(0,0)
    torch.testing.assert_close(state.reference, source)
    assert (state.progress == 0).all()
    torch.testing.assert_close(stopping_reward(env), torch.zeros(4))
  tick()
  tick()
  assert (state.landing_credit == 0).all()
  torch.testing.assert_close(stopping_reward(env), torch.zeros(4))
  # Ordinary supported steps still progress after an unsupported hop.
  for _ in range(16):
    tick(0,1)
  tick()
  tick()
  assert (state.credited_progress[:,0] == 1).all()
  assert (state.credited_progress[:,1] == 0).all()


@torch.inference_mode()
def test_flight_freezes_existing_partial_blend(env):
  _, state, _, tick = prepare_blend(env, False)
  tick()
  tick()
  for _ in range(4):
    tick(0,1)
  before = state.reference.clone()
  progress = state.progress.clone()
  assert (progress[:,0] > 0).all()
  for _ in range(8):
    tick(0,0)
    torch.testing.assert_close(state.reference, before)
    torch.testing.assert_close(state.progress, progress)


@torch.inference_mode()
@pytest.mark.parametrize("name", ["track_linear_velocity", "track_angular_velocity",
                                 "leg_joint_vel_penalty", "base_drift_penalty"])
@pytest.mark.parametrize("mode", [1, 2])
def test_stopping_retains_original_stabilization(env, monkeypatch, name, mode):
  _, state, _, _ = prepare_blend(env, False)
  state.mode[:] = mode
  term = env.reward_manager.get_term_cfg(name)
  monkeypatch.setattr(state, 'advance', lambda: None)
  data = env.scene['robot'].data
  monkeypatch.setattr(type(data), 'root_link_lin_vel_b',
                      property(lambda _: torch.full((4, 3), .3)))
  monkeypatch.setattr(type(data), 'root_link_ang_vel_b',
                      property(lambda _: torch.full((4, 3), .2)))
  params = {k:v for k,v in term.params.items() if k not in ('original_func','transition_kind')}
  expected = term.func.func(env, **params)
  torch.testing.assert_close(term.func(env, **term.params), expected)


@torch.inference_mode()
@pytest.mark.parametrize("mode", [1, 2])
def test_stopping_landing_bonus_is_zero_even_after_deadline(env, monkeypatch, mode):
  from src.tasks.locomanipulation.mdp.stopping import stopping_reward
  _, state, _, _ = prepare_blend(env, False)
  state.mode[:] = mode
  state.landing_credit[:] = 1
  state.shaping[:] = .7
  monkeypatch.setattr(state, 'advance', lambda: None)
  monkeypatch.setattr(state, 'geometry', lambda destination=False: (
    torch.zeros(4, 2, 3), torch.zeros(4, dtype=torch.long)))
  torch.testing.assert_close(stopping_reward(env),
                            state.command.cfg.shaping_weight * state.shaping)


@torch.inference_mode()
def test_reversal_without_reference_motion_earns_no_credit(env):
  cmd, state, _, tick = prepare_blend(env, True)
  source = state.reference.clone()
  cmd.vel_command_b[:] = 0
  state.reconcile()
  torch.testing.assert_close(state.reference, source)
  assert (state.progress == 1).all() and (state.credited_progress == 1).all()
  for pair in [(0,1)]*16 + [(1,1)]*2 + [(1,0)]*16 + [(1,1)]*2:
    tick(*pair)
    assert not state.landing_credit.any()
    torch.testing.assert_close(state.reference, source)


@torch.inference_mode()
def test_reversal_only_restarts_legs_with_changed_references(env):
  cmd, state, _, tick = prepare_blend(env, True)
  for _ in range(4):
    tick(0,1)
  tick()
  tick()
  assert (state.blend[:,0] > 0).all() and (state.blend[:,1] == 0).all()
  cmd.vel_command_b[:] = 0
  state.reconcile()
  assert (state.progress[:,0] == 0).all()
  assert (state.progress[:,1] == 1).all()
  assert (state.credited_progress[:,1] == 1).all()


@torch.inference_mode()
def test_hop_before_landing_cannot_cash_out_previously_supported_progress(env):
  _, state, _, tick = prepare_blend(env, False)
  tick()
  tick()
  for _ in range(4):
    tick(0,1)
  assert (state.progress[:,0] > 0).all()
  tick(0,0)
  tick(0,0)
  tick()
  tick()
  assert not state.landing_credit.any()
  assert (state.credited_progress == 0).all()
  # Progress/reference are retained; a later supported landing can earn credit.
  for _ in range(16):
    tick(0,1)
  tick()
  tick()
  assert (state.landing_credit[:,0] > 0).all()
  assert (state.credited_progress[:,0] == 1).all()
