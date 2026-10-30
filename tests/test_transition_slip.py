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
    failed=False,censored=False,success=True,reference_complete=True,walking_velocity_mae=.1,walking_yaw_mae=.1,
    final_height_error_m=.01,final_placement_error_m=.01,final_heading_error_rad=.02,
    carry_pose_rms=.01,start_delay_s=.2,settle_s=.4,
    flight={d:{'flight_s':0.,'peak_height_above_command_m':.02,'peak_vertical_speed_m_s':.1} for d in ('stand_to_walk','walk_to_stand')},
    stand_to_walk=[trace_metrics(),trace_metrics()],walk_to_stand=[trace_metrics(),trace_metrics()])
  baseline = dict(seed=42,schema_version=6,trials=[row])
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
  baseline['schema_version'] = candidate['schema_version'] = 5
  with pytest.raises(ValueError, match='regenerate'):
    compare(baseline, candidate)


def test_quiet_partial_reference_does_not_pass_loaded_comparison():
  baseline, candidate = paired_reports()
  candidate['trials'][0]['reference_complete'] = False
  result = compare(baseline, candidate)
  assert not result['passed']
  assert not result['by_load']['40.0']['reference_completion']


def test_flight_diagnostic_distinguishes_single_support():
  from scripts.transition_metrics import flight_metrics
  c = torch.tensor([[True,False],[False,True],[False,False],[True,True]])
  result = flight_metrics(c, torch.tensor([.64,.65,.68,.64]), torch.tensor([0.,.1,.4,-.2]), .64, .02)
  assert result['flight_s'] == pytest.approx(.02)
  assert result['peak_height_above_command_m'] == pytest.approx(.04)
  assert result['peak_vertical_speed_m_s'] == pytest.approx(.4)
  baseline, candidate = paired_reports()
  candidate['trials'][0]['flight']['walk_to_stand']['flight_s'] = .02
  result = compare(baseline, candidate)
  assert not result['passed']
  assert not result['by_load']['40.0']['walk_to_stand_flight_preserved']


@pytest.mark.parametrize('key,value', [('peak_height_above_command_m',.08), ('peak_vertical_speed_m_s',.4)])
def test_height_and_vertical_motion_regression_cannot_hide_behind_contact(key, value):
  baseline, candidate = paired_reports()
  candidate['trials'][0]['flight']['walk_to_stand'][key] = value
  result = compare(baseline,candidate)
  assert not result['passed']
  assert not result['by_load']['40.0']['walk_to_stand_' + key + '_preserved']


def test_low_baseline_success_does_not_allow_low_candidate_success():
  baseline, candidate = paired_reports()
  for report in (baseline,candidate):
    report['trials'] = [deepcopy(report['trials'][0]) for _ in range(5)]
    for i,row in enumerate(report['trials']):
      row['phase'] = i
      row['success'] = i == 0
  result = compare(baseline,candidate)
  assert not result['passed']
  assert not result['by_load']['40.0']['complete_transition_success']


def test_full_interval_flight_metric_includes_a_late_hop():
  from scripts.transition_metrics import flight_metrics
  contact = torch.ones(200,2,dtype=torch.bool)
  contact[100:105] = False  # Flight at 2 seconds, past the old 1.5-second window.
  z = torch.full((200,),.64)
  vz = torch.zeros(200)
  assert flight_metrics(contact,z,vz,.64,.02)['flight_s'] == pytest.approx(.1)


def test_stopping_step_metrics_count_both_feet_and_preimpact_speed():
  from scripts.transition_metrics import stopping_step_metrics
  contact = torch.ones(8, 2, dtype=torch.bool)
  contact[2:4] = False
  contact[5:6, 0] = False
  sole = torch.zeros(8, 2, 3)
  sole[3, :, 2] = -.7
  sole[5, 0, 2] = -.2
  velocity = torch.zeros(8, 2)
  velocity[:5, 0] = .5
  position = torch.zeros(8, 3)
  quaternion = torch.zeros(8, 4)
  quaternion[:, 0] = 1
  force = torch.zeros(8, 2)
  force[4] = 123.
  result = stopping_step_metrics(contact, sole, velocity, position, quaternion, force, 2, .02)
  assert result['touchdown_count'] == 3
  assert result['post_braking_touchdowns'] == 1
  assert result['peak_downward_touchdown_speed_m_s'] == pytest.approx(.7)
  assert result['peak_contact_force_n'] == 123.
  assert result['roll_pitch_range_rad'] == [0., 0.]
  velocity[:] = .5
  result = stopping_step_metrics(contact, sole, velocity, position, quaternion, force, 2, .02)
  assert result['post_braking_touchdowns'] is None


def stepping_reports():
  a, b = paired_reports()
  for report in (a, b):
    row = report['trials'][0]
    row['hip_motion'] = {'stand_to_walk': {k:[.1,.1] for k in
      ('peak_velocity_rad_s','peak_acceleration_rad_s2','peak_jerk_rad_s3')}}
    row['stopping_steps'] = dict(version=1, post_braking_touchdowns=3,
      peak_downward_touchdown_speed_m_s=.5, peak_contact_force_n=100.,
      roll_pitch_range_rad=[.1,.1], horizontal_sway_range_m=[.1,.1])
  b['trials'][0]['stopping_steps']['post_braking_touchdowns'] = 1
  b['trials'][0]['reference_complete'] = False
  return a, b


def test_physical_stopping_comparison_does_not_require_reference_completion():
  a, b = stepping_reports()
  assert compare(a,b,stopping_steps=True)['passed']
  b['trials'][0]['stopping_steps'].pop('version')
  with pytest.raises(ValueError, match='Regenerate'):
    compare(a,b,stopping_steps=True)


@pytest.mark.parametrize('key,value', [
  ('post_braking_touchdowns',3), ('post_braking_touchdowns',None),
  ('peak_contact_force_n',101.), ('peak_downward_touchdown_speed_m_s',.6),
  ('roll_pitch_range_rad',[.2,.1]), ('horizontal_sway_range_m',[.2,.1])])
def test_stopping_comparison_rejects_regression(key, value):
  a, b = stepping_reports()
  b['trials'][0]['stopping_steps'][key] = value
  assert not compare(a,b,stopping_steps=True)['passed']
