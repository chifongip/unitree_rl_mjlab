"""Conservation, state isolation and compatibility of reduced carrying loads."""

from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import matrix_from_quat, quat_from_euler_xyz, quat_mul
from scripts.play import _apply_env_overrides, _load_params_overrides
from scripts.train import TrainConfig, _dump_env_overrides_yaml, _prepare_payload
from src.tasks.locomanipulation.config.x2.env_cfgs import (
  agibot_x2_locomanipulation_carry_finetune_flat_env_cfg as carry_cfg,
  agibot_x2_locomanipulation_flat_env_cfg as flat_cfg,
  agibot_x2_locomanipulation_rough_env_cfg as rough_cfg,
)
from src.tasks.locomanipulation.config.x2.rl_cfg import agibot_x2_locomanipulation_ppo_runner_cfg
from src.tasks.locomanipulation.mdp.events import TriangleWaveForceEvent
from src.tasks.locomanipulation.mdp.observations import wrist_external_force
from src.tasks.locomanipulation.mdp.payload import (
  SlidingBoxForceEvent,
  default_payload_params,
  distribute_hand_wrench,
  enable_payload_playback,
  restore_payload_params,
  validate_payload_params,
  advance_frictional_contents,
  box_rotation,
  rotational_acceleration,
)


def make_event(n=3, **parameters):
  data = SimpleNamespace(
    body_com_pos_w=torch.tensor([[[0.0, 0.2, 1.0], [0.0, -0.2, 1.0]]]).repeat(n, 1, 1),
    body_com_lin_vel_w=torch.zeros(n, 2, 3),
    # X2 side grasps use wrist local +X as box up.
    body_link_quat_w=torch.tensor([np.sqrt(0.5), 0, -np.sqrt(0.5), 0], dtype=torch.float32).repeat(
      n, 2, 1
    ),
    body_external_wrench=torch.zeros(n, 2, 6),
    root_link_quat_w=torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1),
  )

  def write(forces, torques, body_ids, env_ids=None):
    ids = torch.arange(n)[slice(None) if env_ids is None else env_ids]
    bodies = torch.tensor(body_ids)
    data.body_external_wrench[ids[:, None], bodies, :3] = forces
    data.body_external_wrench[ids[:, None], bodies, 3:] = torques

  asset = SimpleNamespace(data=data, write_external_wrench_to_sim=write)
  env = SimpleNamespace(
    num_envs=n,
    device="cpu",
    step_dt=0.02,
    scene={"robot": asset},
    sim=SimpleNamespace(
      mj_model=SimpleNamespace(opt=SimpleNamespace(gravity=np.array([0, 0, -9.81])))
    ),
  )
  payload = default_payload_params()
  payload.update(mass_range_kg=(4.0, 4.0), initial_offset_m=0.0)
  payload.update(parameters)
  params = dict(
    asset_cfg=SceneEntityCfg("robot", body_names=("left", "right")),
    payload=payload,
    force_scale=1.0,
    no_force_ratio=0.0,
  )
  params["asset_cfg"].body_ids = [0, 1]
  event = SlidingBoxForceEvent(SimpleNamespace(params=params), env)
  return event, env, params, data


def call(event, env, params):
  event(env, None, **params)


def tilt_hands(data, *, roll=0.0, pitch=0.0, yaw=0.0):
  n = data.body_com_pos_w.shape[0]
  quat = quat_from_euler_xyz(
    torch.full((n,), float(roll)), torch.full((n,), float(pitch)), torch.full((n,), float(yaw))
  )
  rotation = matrix_from_quat(quat)
  offsets = torch.tensor([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]]).repeat(n, 1, 1)
  data.body_com_pos_w[:] = torch.einsum("bij,bkj->bki", rotation, offsets) + torch.tensor(
    [0, 0, 1.0]
  )
  grip = torch.tensor([np.sqrt(0.5), 0, -np.sqrt(0.5), 0], dtype=torch.float32).repeat(n, 2, 1)
  data.body_link_quat_w[:] = quat_mul(quat[:, None].expand(-1, 2, -1), grip)
  return rotation


def test_box_frame_tracks_pitch_roll_and_yaw_from_both_grasps():
  _, _, _, data = make_event(n=1)
  expected = tilt_hands(data, roll=0.15, pitch=0.3, yaw=0.4)
  rotation, separation, valid = box_rotation(data.body_com_pos_w, data.body_link_quat_w)
  assert valid.all()
  torch.testing.assert_close(separation, torch.tensor([0.4]))
  torch.testing.assert_close(rotation, expected)


@pytest.mark.parametrize("pitch,should_slide", [(0.08, False), (0.35, True)])
def test_gravity_drives_sliding_only_above_static_friction_angle(pitch, should_slide):
  event, env, params, data = make_event(n=1, damping_per_s=0.0)
  rotation = tilt_hands(data, pitch=pitch)
  call(event, env, params)
  if should_slide:
    expected = 9.81 * (np.sin(pitch) - 0.1 * np.cos(pitch))
    assert event._velocity[0, 0].item() == pytest.approx(expected * env.step_dt, rel=1e-5)
    # Hand forces include the true contents tangential acceleration, not just weight.
    expected_acceleration_w = rotation[0, :, 0] * expected
    expected_force = torch.tensor([0.0, 0, -39.24]) - 3.0 * expected_acceleration_w
    torch.testing.assert_close(data.body_external_wrench[0, :, :3].sum(0), expected_force)
  else:
    torch.testing.assert_close(event._velocity, torch.zeros_like(event._velocity))
    torch.testing.assert_close(event._position, torch.zeros_like(event._position))
    torch.testing.assert_close(
      data.body_external_wrench[0, :, :3].sum(0), torch.tensor([0.0, 0, -39.24])
    )


def test_normal_loading_sets_friction_and_sliding_friction_is_lower_than_stiction():
  # Same tangential drive: low normal load slips, high normal load sticks.
  position = torch.zeros(2, 2)
  velocity = torch.zeros_like(position)
  drive = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
  _, next_velocity, _ = advance_frictional_contents(
    position,
    velocity,
    drive,
    torch.tensor([4.0, 10.0]),
    torch.ones(2),
    0.15,
    0.10,
    0.0,
    0.1,
    0.02,
  )
  torch.testing.assert_close(next_velocity[:, 0], torch.tensor([0.012, 0.0]))
  # A moving mass continues under drive below mu_s*N but above mu_k*N.
  _, next_velocity, _ = advance_frictional_contents(
    position[:1],
    torch.tensor([[0.1, 0.0]]),
    torch.tensor([[1.2, 0.0]]),
    torch.tensor([10.0]),
    torch.ones(2),
    0.15,
    0.10,
    0.0,
    0.1,
    0.02,
  )
  torch.testing.assert_close(next_velocity, torch.tensor([[0.104, 0.0]]))


def test_wall_bounce_transfers_impulse_and_loses_energy():
  velocity = torch.tensor([[1.0, 0.0]])
  position, next_velocity, acceleration = advance_frictional_contents(
    torch.tensor([[0.099, 0.0]]),
    velocity,
    torch.zeros(1, 2),
    torch.tensor([9.81]),
    torch.tensor([0.1, 0.1]),
    0,
    0,
    0,
    0.1,
    0.02,
  )
  torch.testing.assert_close(position, torch.tensor([[0.1, 0.0]]))
  torch.testing.assert_close(next_velocity, torch.tensor([[-0.1, 0.0]]))
  torch.testing.assert_close(-3 * acceleration * 0.02, torch.tensor([[3.3, 0.0]]))
  assert next_velocity.square().sum() < velocity.square().sum()


def test_euler_centripetal_and_coriolis_terms_have_physical_signs():
  omega = torch.tensor([[0.0, 0, 2.0]])
  alpha = torch.tensor([[0.0, 0, 1.0]])
  offset = torch.tensor([[0.1, 0, 0]])
  torch.testing.assert_close(
    rotational_acceleration(omega, alpha, offset), torch.tensor([[-0.4, 0.1, 0.0]])
  )
  event, env, params, data = make_event(
    n=1,
    static_friction_coefficient=0,
    kinetic_friction_coefficient=0,
    damping_per_s=0,
    acceleration_filter_s=0,
  )
  call(event, env, params)
  theta = 2 * env.step_dt
  previous_quat = quat_from_euler_xyz(torch.zeros(1), torch.zeros(1), torch.tensor([-theta]))
  event._rotation[:] = matrix_from_quat(previous_quat)
  angular_speed = np.sin(theta) / env.step_dt
  event._omega[:, 2] = angular_speed
  event._velocity[:, 0] = 1.0
  call(event, env, params)
  assert event._velocity[0, 1].item() == pytest.approx(-2 * angular_speed * env.step_dt, rel=1e-4)


def test_shell_rotational_inertia_is_transmitted_as_grip_torque():
  event, env, params, data = make_event(
    n=1, contents_fraction=0, com_height_m=0, acceleration_filter_s=0
  )
  call(event, env, params)
  theta = 0.02
  tilt_hands(data, pitch=theta)
  call(event, env, params)
  alpha_y = np.sin(theta) / env.step_dt**2
  inertia_y = 4.0 / 12 * (0.25**2 + 0.12**2)
  assert data.body_external_wrench[0, :, 4].sum().item() == pytest.approx(
    -inertia_y * alpha_y, rel=1e-4
  )
  torch.testing.assert_close(
    data.body_external_wrench[0, :, :3].sum(0), torch.tensor([0.0, 0, -39.24])
  )


def test_restore_discards_obsolete_tag_without_mutating_saved_settings():
  cfg = carry_cfg()
  saved_payload = default_payload_params()
  saved_payload["model_version"] = 2
  saved = {"events": {"hand_force": {"params": {"payload": saved_payload}}}}
  restore_payload_params(cfg, saved)
  assert cfg.events["hand_force"].params["payload"] == default_payload_params()
  assert saved_payload["model_version"] == 2


def test_invalid_restore_does_not_partially_change_payload_parameters():
  cfg = carry_cfg()
  original = deepcopy(cfg.events["hand_force"].params)
  saved = deepcopy(original)
  saved["payload"]["mass_range_kg"] = (1.0, 3.0)
  saved["no_force_ratio"] = 1.5
  with pytest.raises(ValueError, match="no_force_ratio"):
    restore_payload_params(cfg, {"events": {"hand_force": {"params": saved}}}, training=True)
  assert cfg.events["hand_force"].params == original


def test_narrow_grasp_uses_the_same_box_geometry_for_inertia_and_display():
  event, env, params, data = make_event(
    n=1, contents_fraction=0, com_height_m=0, acceleration_filter_s=0
  )
  data.body_com_pos_w[0, :, 1] = torch.tensor([0.06, -0.06])
  call(event, env, params)
  theta = 0.02
  quat = quat_from_euler_xyz(torch.tensor([theta]), torch.zeros(1), torch.zeros(1))
  rotation = matrix_from_quat(quat)
  offsets = data.body_com_pos_w - torch.tensor([0.0, 0, 1.0])
  data.body_com_pos_w[:] = torch.einsum("bij,bkj->bki", rotation, offsets) + torch.tensor(
    [0.0, 0, 1.0]
  )
  data.body_link_quat_w[:] = quat_mul(quat[:, None].expand(-1, 2, -1), data.body_link_quat_w)
  call(event, env, params)
  torch.testing.assert_close(event._half_size, torch.tensor([[0.125, 0.105, 0.06]]))
  torque_x = data.body_external_wrench[0, :, 3:].sum(0)[0]
  # Include the force-pair moment in the total shell rotational reaction.
  offsets = data.body_com_pos_w[0] - data.body_com_pos_w[0].mean(0)
  torque_x += torch.cross(offsets, data.body_external_wrench[0, :, :3], dim=-1).sum(0)[0]
  inertia_x = 4 / 12 * (0.21**2 + 0.12**2)
  assert torque_x.item() == pytest.approx(-inertia_x * np.sin(theta) / env.step_dt**2, rel=1e-4)


def test_static_weight_and_existing_critic_encoding():
  event, env, params, data = make_event()
  call(event, env, params)
  expected = torch.zeros(3, 2, 6)
  expected[:, :, 2] = -19.62
  torch.testing.assert_close(data.body_external_wrench, expected)
  observed = wrist_external_force(env, params["asset_cfg"])
  expected_force = expected[:, :, :3].flatten(1)
  torch.testing.assert_close(observed, expected_force.sign() * expected_force.abs().log1p())


def test_viewer_overlay_matches_contents_and_com_without_mutating_dynamics():
  event, env, params, data = make_event(n=1)
  event._position[:] = torch.tensor([[0.04, -0.02]])
  call(event, env, params)
  shapes = {"boxes": [], "spheres": [], "arrows": [], "cylinders": []}
  visualizer = SimpleNamespace(
    get_env_indices=lambda count: [0],
    add_box=lambda **values: shapes["boxes"].append(values),
    add_sphere=lambda **values: shapes["spheres"].append(values),
    add_arrow=lambda **values: shapes["arrows"].append(values),
    add_cylinder=lambda **values: shapes["cylinders"].append(values),
  )
  wrench = data.body_external_wrench.clone()
  position = event._position.clone()
  event.debug_vis(visualizer)
  assert len(shapes["boxes"]) == 1
  assert len(shapes["arrows"]) == 2
  assert len(shapes["cylinders"]) == 1
  box = shapes["boxes"][0]
  contents, com = shapes["spheres"]
  np.testing.assert_allclose(box["center"], [0, 0, 0.95])
  np.testing.assert_allclose(contents["center"], [0.04, -0.02, 0.915])
  np.testing.assert_allclose(com["center"], [0.03, -0.015, 0.92375])
  torch.testing.assert_close(data.body_external_wrench, wrench)
  torch.testing.assert_close(event._position, position)
  params["constant_force"] = {"z": 0.0}
  call(event, env, params)
  for values in shapes.values():
    values.clear()
  event.debug_vis(visualizer)
  assert not any(shapes.values())


def test_displaced_contents_changes_load_sharing_and_forward_moment():
  event, env, params, data = make_event(n=2)
  event._position[:] = torch.tensor([[0, 0.04], [0.04, 0]])
  call(event, env, params)
  torch.testing.assert_close(data.body_external_wrench[0, :, 2], torch.tensor([-22.563, -16.677]))
  torch.testing.assert_close(data.body_external_wrench[1, :, 4], torch.full((2,), 0.5886))
  torch.testing.assert_close(event._position, torch.tensor([[0, 0.04], [0.04, 0]]))


def test_random_wrench_distribution_conserves_force_and_moment():
  generator = torch.Generator().manual_seed(7)
  hands = torch.randn(64, 2, 3, generator=generator)
  force, moment = torch.randn(2, 64, 3, generator=generator)
  forces, torques = distribute_hand_wrench(hands, force, moment)
  offsets = hands - hands.mean(1, keepdim=True)
  torch.testing.assert_close(forces.sum(1), force)
  torch.testing.assert_close((torch.cross(offsets, forces, dim=-1) + torques).sum(1), moment)


def test_frictionless_contents_do_not_add_horizontal_inertia_before_wall():
  event, env, params, data = make_event(
    static_friction_coefficient=0,
    kinetic_friction_coefficient=0,
    damping_per_s=0,
    acceleration_filter_s=0,
  )
  call(event, env, params)
  data.body_com_lin_vel_w[:, :, 0] = 0.2  # 10 m/s^2 over one event interval.
  call(event, env, params)
  torch.testing.assert_close(event._velocity[:, 0], torch.full((3,), -0.2))
  torch.testing.assert_close(event._position[:, 0], torch.full((3,), -0.004))
  # Only the 1 kg shell accelerates; contents inertia cancels through slip.
  torch.testing.assert_close(data.body_external_wrench[:, :, 0].sum(1), torch.full((3,), -10.0))


def test_friction_stops_without_reversing_and_walls_transfer_momentum():
  position = torch.tensor([[0.0, 0.0], [0.099, 0.0]])
  velocity = torch.tensor([[0.01, 0.0], [1.0, 0.0]])
  next_position, next_velocity, acceleration = advance_frictional_contents(
    position,
    velocity,
    torch.zeros_like(position),
    torch.ones(2),
    torch.tensor([0.1, 0.1]),
    1.0,
    1.0,
    0,
    0,
    0.02,
  )
  torch.testing.assert_close(next_velocity, torch.zeros_like(velocity))
  torch.testing.assert_close(next_position, torch.tensor([[0.0, 0.0], [0.1, 0.0]]))
  torch.testing.assert_close(acceleration * 0.02, -velocity)


def test_zero_scale_no_force_runtime_changes_and_invalid_grasp():
  event, env, params, data = make_event()
  for scale, ratio in [(0.0, 0.0), (1.0, 1.0)]:
    params.update(force_scale=scale, no_force_ratio=ratio)
    call(event, env, params)
    assert torch.count_nonzero(data.body_external_wrench) == 0
  params.update(force_scale=1.0, no_force_ratio=0.0)
  data.body_com_pos_w[0, 1] = data.body_com_pos_w[0, 0]
  data.body_com_lin_vel_w[:] = 10  # First active sample must initialize history.
  call(event, env, params)
  assert event._no_force_mask.tolist() == [True, False, False]
  assert torch.count_nonzero(data.body_external_wrench[0]) == 0
  assert torch.count_nonzero(event._acceleration) == 0
  assert torch.isfinite(data.body_external_wrench).all()


def test_constant_override_and_reentry_do_not_create_acceleration_impulse():
  event, env, params, data = make_event()
  call(event, env, params)
  params["constant_force"] = {"x": 1.0, "z": -5.0}
  call(event, env, params)
  torch.testing.assert_close(data.body_external_wrench[:, :, 0], torch.ones(3, 2))
  params["constant_force"] = None
  data.body_com_lin_vel_w[:] = 20
  call(event, env, params)
  assert torch.count_nonzero(event._acceleration) == 0
  torch.testing.assert_close(data.body_external_wrench[:, :, 2], torch.full((3, 2), -19.62))


def test_partial_and_slice_reset_clear_only_selected_wrenches_and_state():
  event, env, params, data = make_event()
  call(event, env, params)
  event._position[:] = 0.03
  event._velocity[:] = 0.1
  before = {
    name: getattr(event, name)[1].clone()
    for name in (
      "_position",
      "_velocity",
      "_mass",
      "_episode_draw",
      "_initialized",
      "_omega",
      "_angular_acceleration",
      "_rotation",
    )
  }
  wrench = data.body_external_wrench[1].clone()
  for ids in (torch.tensor([0]), slice(0, 1)):
    event.reset(ids)
    assert torch.count_nonzero(data.body_external_wrench[0]) == 0
    assert not event._initialized[0]
    for name, value in before.items():
      torch.testing.assert_close(getattr(event, name)[1], value)
    torch.testing.assert_close(data.body_external_wrench[1], wrench)


def test_curriculum_scales_mass_and_mirrored_contents_swap_hand_loading():
  event, env, params, data = make_event(n=2)
  params["force_scale"] = 0.5
  event._position[:] = torch.tensor([[0.02, 0.04], [0.02, -0.04]])
  call(event, env, params)
  wrench = data.body_external_wrench
  torch.testing.assert_close(wrench[:, :, 2].sum(1), torch.full((2,), -19.62))
  reflection = torch.tensor([1, -1, 1, -1, 1, -1])  # Force polar, torque axial.
  torch.testing.assert_close(wrench[0].flip(0) * reflection, wrench[1])


def test_wall_impulse_and_final_state_agree_at_two_timesteps():
  results = []
  for dt in (0.02, 0.01):
    position = torch.zeros(1, 2)
    velocity = torch.tensor([[0.5, 0.0]])
    impulse = torch.zeros_like(velocity)
    for _ in range(round(0.5 / dt)):
      position, velocity, acceleration = advance_frictional_contents(
        position,
        velocity,
        torch.zeros_like(position),
        torch.ones(1),
        torch.tensor([0.1, 0.1]),
        0,
        0,
        0,
        0,
        dt,
      )
      impulse -= 3.0 * acceleration * dt
    results.append((position, velocity, impulse))
  for coarse, fine in zip(*results):
    torch.testing.assert_close(coarse, fine)
  torch.testing.assert_close(results[0][2], torch.tensor([[1.5, 0.0]]))


@pytest.mark.parametrize(
  "name,value",
  [
    ("mass_range_kg", (4, 1)),
    ("travel_xy_m", (-1, 1)),
    ("contents_fraction", 1.1),
    ("damping_per_s", -1),
    ("initial_offset_m", 0.5),
    ("com_height_m", float("nan")),
    ("kinetic_friction_coefficient", 0.3),
    ("wall_restitution", 1.1),
  ],
)
def test_invalid_payload_configuration(name, value):
  params = default_payload_params()
  params[name] = value
  with pytest.raises(ValueError):
    validate_payload_params(params)


def test_carry_configuration_isolated_from_basic_tasks_and_preserves_rewards():
  base = flat_cfg()
  observations, rewards = deepcopy(base.observations), deepcopy(base.rewards)
  carry = carry_cfg()
  assert carry.events["hand_force"].func is SlidingBoxForceEvent
  assert carry.rewards == rewards
  assert carry.observations["actor"] == observations["actor"]
  for name, term in carry.observations["critic"].terms.items():
    if name != "wrist_force":
      assert term == observations["critic"].terms[name]
  for basic in (base, flat_cfg(), rough_cfg()):
    assert basic.events["hand_force"].func is TriangleWaveForceEvent
    assert basic.observations["critic"].terms["wrist_force"].params["asset_cfg"].body_names == (
      "left_wrist_yaw_link",
      "right_wrist_yaw_link",
    )
    assert (
      basic.actions["upper_body_motion"].motion_file
      != carry.actions["upper_body_motion"].motion_file
    )


def test_payload_roundtrip_train_play_and_eval_preserves_play_overrides(tmp_path):
  from scripts.eval import _apply_env_overrides as eval_overrides

  training = carry_cfg()
  training.events["hand_force"].params["payload"]["mass_range_kg"] = (1.5, 3.5)
  training.events["hand_force"].params["no_force_ratio"] = 0.45
  params_dir = tmp_path / "params"
  _dump_env_overrides_yaml(params_dir / "env_overrides.yaml", asdict(training))
  saved = _load_params_overrides(params_dir)
  for apply in (_apply_env_overrides, eval_overrides):
    playback = carry_cfg(play=True)
    apply(playback, saved)
    assert (
      playback.events["hand_force"].params["payload"]
      == training.events["hand_force"].params["payload"]
    )
    assert playback.events["hand_force"].params["constant_force"] == {"x": 0, "y": 0, "z": 0}
  resumed = carry_cfg()
  _prepare_payload(
    TrainConfig(env=resumed, agent=agibot_x2_locomanipulation_ppo_runner_cfg()),
    tmp_path / "model_1.pt",
  )
  assert (
    resumed.events["hand_force"].params["payload"]
    == training.events["hand_force"].params["payload"]
  )
  assert resumed.events["hand_force"].params["no_force_ratio"] == 0.45
  with pytest.raises(ValueError, match="carry task"):
    restore_payload_params(flat_cfg(), saved)
  del saved["events"]["hand_force"]["params"]["payload"]["contents_fraction"]
  with pytest.raises(ValueError, match="required fields"):
    restore_payload_params(carry_cfg(), saved)


def test_explicit_payload_playback_uses_carry_dataset_and_physical_box_disables_load():
  from scripts.play import PREDEFINED_POSES_X2
  from src.tasks.locomanipulation.config.x2.carry_box import add_carry_box

  cfg = carry_cfg(play=True)
  enable_payload_playback(cfg)
  assert cfg.actions["upper_body_motion"].fixed_upper_body_pose is None
  assert cfg.events["hand_force"].params["constant_force"] is None
  add_carry_box(cfg, PREDEFINED_POSES_X2[3], 1.0)
  assert cfg.events["hand_force"].params["constant_force"] == {"x": 0, "y": 0, "z": 0}
  with pytest.raises(ValueError, match="carry task"):
    enable_payload_playback(flat_cfg(play=True))


def test_eval_can_switch_between_payload_constant_and_unloaded_conditions():
  from scripts.eval import POSE_PRESETS, _configure_env_base, _configure_force_and_pose
  from scripts.play import PREDEFINED_POSES_X2

  event, env, params, data = make_event()
  assert POSE_PRESETS["x2_carry"] == PREDEFINED_POSES_X2[3]
  configured = carry_cfg(play=True)
  _configure_env_base(configured, 2, 0.64, 100)
  assert "force_range_max" not in configured.events["hand_force"].params
  cfg = SimpleNamespace(func=event, params=params)
  env.event_manager = SimpleNamespace(get_term_cfg=lambda name: cfg)
  env.action_manager = SimpleNamespace(
    get_term=lambda name: SimpleNamespace(set_fixed_pose=lambda pose: None)
  )
  wrapped = SimpleNamespace(unwrapped=env)
  for condition, expected_total_z in (("payload", -39.24), ("medium", -30.0), ("none", 0)):
    _configure_force_and_pose(wrapped, condition, "x2_carry")
    call(event, env, params)
    torch.testing.assert_close(
      data.body_external_wrench[:, :, 2].sum(1), torch.full((3,), float(expected_total_z))
    )


def test_payload_full_environment_observations_rewards_and_partial_reset():
  from mjlab.envs import ManagerBasedRlEnv

  cfg = carry_cfg(play=True)
  enable_payload_playback(cfg)
  cfg.scene.num_envs = 2
  cfg.seed = 123
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    obs, _ = env.reset()
    event_cfg = env.event_manager.get_term_cfg("hand_force")
    event = event_cfg.func
    assert env.sim.mj_model.nq == 36  # Robot only: 7 root + 29 robot joints.
    assert env.sim.mj_model.neq == 0
    for _ in range(12):
      obs, reward, *_ = env.step(torch.zeros(2, 15))
      assert obs["actor"].shape == (2, 430)
      assert obs["critic"].shape == (2, 108)
      assert torch.isfinite(obs["actor"]).all()
      assert torch.isfinite(obs["critic"]).all()
      assert torch.isfinite(reward).all()
      assert torch.isfinite(event._position).all()
      assert (event._position.abs() <= event._travel).all()
    robot = env.scene["robot"]
    forces = robot.data.body_external_wrench[:, event._body_ids, :3]
    assert torch.count_nonzero(forces) > 0
    observation_cfg = cfg.observations["critic"].terms["wrist_force"]
    observed = observation_cfg.func(env, **observation_cfg.params)
    assert torch.count_nonzero(observed) > 0
    # The same existing reward functions must see a valid load mask.
    for name in ("stand_still", "leg_joint_vel_penalty"):
      term = env.reward_manager.get_term_cfg(name)
      assert torch.isfinite(term.func(env, **term.params)).all()
    survivor = event._position[1].clone()
    survivor_wrench = robot.data.body_external_wrench[1].clone()
    env.reset(env_ids=torch.tensor([0]))
    assert not event._initialized[0]
    torch.testing.assert_close(event._position[1], survivor)
    torch.testing.assert_close(robot.data.body_external_wrench[1], survivor_wrench)
    assert torch.count_nonzero(robot.data.body_external_wrench[0]) == 0
  finally:
    env.close()
