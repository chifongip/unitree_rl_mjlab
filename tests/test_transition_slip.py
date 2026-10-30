"""Regression coverage for planted-foot rotation and transition diagnostics."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from src.tasks.locomanipulation.mdp.stopping import contact_slip_cost, contact_slip_reward
from scripts.transition_metrics import foot_transition_metrics
from scripts.eval_stopping import compare


def test_contact_cost_separates_yaw_and_translation():
  contact = torch.tensor([[True, False]])
  velocity = torch.tensor([[[2., 0., 8.], [50., 50., 0.]]])
  angular = torch.tensor([[[7., 9., 3.], [50., 50., 50.]]])
  linear, yaw = contact_slip_cost(contact, velocity, angular)
  assert linear.item() == 4 and yaw.item() == 9
  # Pitch/roll angular motion and vertical translation are not independent costs.
  angular[..., 2] = 0
  assert contact_slip_cost(contact, velocity, angular)[1].item() == 0
  contact[:] = False
  assert all(v.item() == 0 for v in contact_slip_cost(contact, velocity, angular))


def test_slip_reward_is_command_independent_and_symmetric():
  quats = torch.zeros(3, 2, 4)
  quats[..., 0] = 1
  data = SimpleNamespace(body_link_ang_vel_w=torch.zeros(3, 2, 3),
    body_link_lin_vel_w=torch.zeros(3, 2, 3), body_link_quat_w=quats)
  data.body_link_ang_vel_w[..., 2] = 1
  # Nonzero centroid offset also exercises omega cross r.
  state = SimpleNamespace(robot=SimpleNamespace(data=data), body_ids=[0, 1],
    offsets=torch.tensor([[.1, 0., 0.], [.1, 0., 0.]]),
    command=SimpleNamespace(cfg=SimpleNamespace(slip_weight=.25, yaw_slip_weight=.25)))
  found = torch.ones(3, 2)
  env = SimpleNamespace(num_envs=3, extras={'log':{}},
    scene={'feet_ground_contact':SimpleNamespace(data=SimpleNamespace(found=found))},
    command_manager=SimpleNamespace(get_term=lambda _:SimpleNamespace(state=lambda:state)))
  # No mode/command is available: the physical cost must not depend on either.
  value = contact_slip_reward(env)
  torch.testing.assert_close(value, torch.full((3,), -.505))
  data.body_link_ang_vel_w[..., 2] *= -1
  torch.testing.assert_close(contact_slip_reward(env), value)
  found[:] = 0
  torch.testing.assert_close(contact_slip_reward(env), torch.zeros(3))


def trace_metrics(contact=None, ground_yaw=.1, air_yaw=10.):
  c = torch.ones(40, dtype=torch.bool) if contact is None else contact
  if contact is None:
    c[3:6] = False
  velocity = torch.zeros(40, 3)
  angular = torch.zeros(40, 3)
  angular[:, 2] = torch.where(c, ground_yaw, air_yaw)
  return foot_transition_metrics(c, torch.zeros(40, 3), torch.zeros(40), velocity,
                                 angular, torch.zeros(40), 0, 40, .02)


def test_contact_events_do_not_reward_missing_steps():
  m = trace_metrics()
  assert m['liftoff_s'] == pytest.approx(.06)
  assert m['touchdown_s'] == pytest.approx(.12)
  assert m['pre_liftoff_yaw_rad'] == pytest.approx(.006)
  assert m['contact_yaw_rad'] == pytest.approx(37 * .02 * .1)
  assert trace_metrics(ground_yaw=0.)['contact_yaw_rad'] == 0
  missing = trace_metrics(torch.ones(40, dtype=torch.bool))
  assert missing['missing_liftoff'] and missing['missing_touchdown']
  assert missing['pre_liftoff_yaw_rad'] is None
  assert missing['post_touchdown_yaw_rad'] is None
  c = torch.zeros(40, dtype=torch.bool)
  c[38:] = True
  late = trace_metrics(c)
  assert late['started_airborne'] and late['post_touchdown_censored']


def paired_reports():
  row = dict(height=.64,command=[.5,0.,0.],phase=0,seed=42,
    scenario='stand-walk-stop',load_n_per_hand=40.,carry_poses=True,
    carry_joint_names=['arm'],carry_pose_target=[.2],walk_start_s=3.,stop_command_s=8.,
    failed=False,censored=False,success=True,walking_velocity_mae=.1,walking_yaw_mae=.1,
    final_height_error_m=.01,final_placement_error_m=.01,final_heading_error_rad=.02,
    carry_pose_rms=.01,start_delay_s=.2,settle_s=.4,
    stand_to_walk=[trace_metrics(),trace_metrics()],walk_to_stand=[trace_metrics(),trace_metrics()])
  baseline = dict(seed=42,schema_version=3,trials=[row])
  candidate = deepcopy(baseline)
  for direction in ('stand_to_walk','walk_to_stand'):
    for foot in candidate['trials'][0][direction]:
      foot['contact_yaw_rad'] *= .4
  return baseline, candidate


def test_loaded_comparison_rejects_mismatch_and_missing_events():
  baseline, candidate = paired_reports()
  assert compare(baseline,candidate)['passed']
  for key, value in (('load_n_per_hand',20.),('carry_pose_target',[.3]),('seed',43),('walk_start_s',4.)):
    changed = deepcopy(candidate)
    changed['trials'][0][key] = value
    with pytest.raises(ValueError,match='match'):
      compare(baseline,changed)
  for report in (baseline, candidate):
    foot = report['trials'][0]['walk_to_stand'][0]
    foot['started_airborne'] = True
    foot['missing_liftoff'] = True
  assert compare(baseline,candidate)['passed']
  airborne = deepcopy(candidate)
  airborne['trials'][0]['stand_to_walk'][0]['started_airborne'] = True
  assert not compare(baseline,airborne)['passed']
  candidate['trials'][0]['stand_to_walk'][0]['missing_liftoff'] = True
  assert not compare(baseline,candidate)['passed']


def test_transition_task_has_only_one_slip_reward():
  from src.tasks.locomanipulation.config.x2.env_cfgs import agibot_x2_locomanipulation_transition_flat_env_cfg
  cfg = agibot_x2_locomanipulation_transition_flat_env_cfg()
  assert cfg.rewards['foot_slip'].func is contact_slip_reward
  assert cfg.rewards['foot_slip'].weight == 1.
  assert cfg.rewards['foot_slip'].params == {}


def test_start_metrics_use_first_landing_not_later_walking_steps():
  c = torch.ones(60, dtype=torch.bool)
  c[3:6] = False
  c[45:55] = False
  args = (c, torch.zeros(60, 3), torch.zeros(60), torch.zeros(60, 3),
          torch.zeros(60, 3), torch.zeros(60), 0, 60, .02)
  starting = foot_transition_metrics(*args, final_touchdown=False)
  stopping = foot_transition_metrics(*args, final_touchdown=True)
  assert starting['touchdown_s'] == pytest.approx(.12)
  assert not starting['post_touchdown_censored']
  assert stopping['touchdown_s'] == pytest.approx(1.1)
  assert stopping['post_touchdown_censored']


@pytest.mark.parametrize("key,value,criterion", [
  ("final_placement_error_m", .15, "final_placement_accuracy"),
  ("final_heading_error_rad", .6, "final_heading_accuracy"),
])
def test_loaded_comparison_requires_correct_stance(key, value, criterion):
  baseline, candidate = paired_reports()
  candidate['trials'][0][key] = value
  result = compare(baseline, candidate)
  assert not result['passed']
  assert not result['by_load']['40.0'][criterion]


@pytest.mark.parametrize("direction", ['stand_to_walk', 'walk_to_stand'])
@pytest.mark.parametrize("key", ['pre_liftoff_yaw_rad', 'pre_liftoff_slide_m',
                                'post_touchdown_yaw_rad', 'post_touchdown_slide_m',
                                'post_touchdown_hip_yaw_rad'])
def test_total_improvement_cannot_hide_event_regression(direction, key):
  baseline, candidate = paired_reports()
  candidate['trials'][0][direction][0][key] += .02
  result = compare(baseline, candidate)
  assert not result['passed']
  assert not result['by_load']['40.0'][direction + '_' + key + '_preserved']


def test_stable_support_foot_can_remain_planted_during_stop():
  baseline, candidate = paired_reports()
  stationary = trace_metrics(torch.ones(40, dtype=torch.bool), ground_yaw=0.)
  for report in (baseline, candidate):
    report['trials'][0]['walk_to_stand'][0] = deepcopy(stationary)
  assert stationary['missing_liftoff'] and stationary['missing_touchdown']
  assert stationary['pre_liftoff_yaw_rad'] is None
  assert compare(baseline, candidate)['passed']
  # Continuous support is not a free pass for sliding or pivoting.
  for key, value in [('contact_slide_m', .04), ('contact_yaw_rad', .2)]:
    bad = deepcopy(candidate)
    bad['trials'][0]['walk_to_stand'][0][key] = value
    result = compare(baseline, bad)
    assert not result['by_load']['40.0']['walk_to_stand_events_complete']
  interrupted = deepcopy(candidate)
  interrupted['trials'][0]['walk_to_stand'][0]['continuous_contact'] = False
  assert not compare(baseline, interrupted)['passed']


def test_old_loaded_reports_require_regeneration():
  baseline, candidate = paired_reports()
  baseline['schema_version'] = candidate['schema_version'] = 2
  with pytest.raises(ValueError, match='regenerate'):
    compare(baseline, candidate)
