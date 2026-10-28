"""Actuator routing, policy-interface preservation and observer cadence."""

from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from mjlab.actuator import ActuatorCmd, BuiltinPositionActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.actuator.builtin_group import BuiltinActuatorGroup
from src.assets.robots.agibot_x2.compensated_actuator import CompensatedPositionActuatorCfg
from src.instinct_mj.controllers import BatchedJointObserver, JointObserverCfg
from src.tasks.locomanipulation.config.x2.env_cfgs import (
  agibot_x2_locomanipulation_carry_finetune_flat_env_cfg,
  agibot_x2_locomanipulation_flat_env_cfg,
  agibot_x2_locomanipulation_rough_env_cfg,
)
from src.tasks.locomanipulation.config.x2 import env_cfgs
from src.tasks.locomanipulation.config.x2.torque_compensation import (
  X2_OBSERVER_JOINT_NAMES,
  configure_x2_torque_compensation,
)
from src.tasks.locomanipulation.mdp.upper_body_action import UpperBodyMotionAction


@pytest.mark.parametrize("factory", [
  agibot_x2_locomanipulation_rough_env_cfg,
  agibot_x2_locomanipulation_flat_env_cfg,
  agibot_x2_locomanipulation_carry_finetune_flat_env_cfg,
])
@pytest.mark.parametrize("play", [False, True])
def test_default_compensation_preserves_all_policy_interfaces(factory, play, monkeypatch):
  cfg = factory(play=play)
  observer = cfg.actions["upper_body_motion"].observer
  assert observer.algorithm == "nonlinear"
  assert observer.mode == "ACTIVE"
  assert observer.joint_names == X2_OBSERVER_JOINT_NAMES
  expected_limits = (28.8, 28.8, 19.2, 19.2, 19.2, 1.76, 1.76) * 2
  assert observer.correction_limits_nm == pytest.approx(expected_limits)
  # Compare with the exact factory behavior before default compensation.
  with monkeypatch.context() as patch:
    patch.setattr(env_cfgs, "configure_x2_torque_compensation", lambda *args, **kwargs: None)
    baseline = factory(play=play)
  assert cfg.observations == baseline.observations
  assert cfg.rewards == baseline.rewards
  assert cfg.actions["joint_pos"] == baseline.actions["joint_pos"]
  current = asdict(cfg.actions["upper_body_motion"])
  original = asdict(baseline.actions["upper_body_motion"])
  current.pop("observer")
  original.pop("observer")
  assert current == original
  for old, new in zip(baseline.scene.entities["robot"].articulation.actuators,
                      cfg.scene.entities["robot"].articulation.actuators):
    assert asdict(old) == asdict(new)  # Gains, limits, joint patterns unchanged.
    if isinstance(new, CompensatedPositionActuatorCfg):
      assert any("shoulder" in p or "elbow" in p or "wrist" in p for p in new.target_names_expr)
  assert factory(play=play).actions["upper_body_motion"].observer is not observer


@pytest.mark.parametrize("limit", [None, 0, -1, float("nan"), float("inf")])
def test_preset_rejects_invalid_limits_without_mutating_configuration(limit):
  cfg = agibot_x2_locomanipulation_flat_env_cfg()
  robot = cfg.scene.entities["robot"]
  original_observer = deepcopy(cfg.actions["upper_body_motion"].observer)
  actuators = tuple(
    BuiltinPositionActuatorCfg(**{**asdict(a), "effort_limit": limit})
    if "elbow" in a.target_names_expr[0] else a
    for a in robot.articulation.actuators
  )
  robot.articulation.actuators = actuators
  with pytest.raises(ValueError, match="finite positive effort limit"):
    configure_x2_torque_compensation(cfg)
  assert robot.articulation.actuators is actuators
  assert cfg.actions["upper_body_motion"].observer == original_observer


@pytest.mark.parametrize("fraction,cap,expected", [
  (0.8, 8.0, (8.0, 8.0, 8.0, 8.0, 8.0, 1.76, 1.76)),
  (0.8, 0.0, (28.8, 28.8, 19.2, 19.2, 19.2, 1.76, 1.76)),
  (0.5, 10.0, (10.0, 10.0, 10.0, 10.0, 10.0, 1.1, 1.1)),
  (1.0, 0.0, (36.0, 36.0, 24.0, 24.0, 24.0, 2.2, 2.2)),
])
def test_configurable_correction_fraction_and_cap(fraction, cap, expected):
  cfg = agibot_x2_locomanipulation_flat_env_cfg()
  configure_x2_torque_compensation(
    cfg, correction_limit_fraction=fraction, correction_limit_nm=cap,
  )
  assert cfg.actions["upper_body_motion"].observer.correction_limits_nm == pytest.approx(expected * 2)


@pytest.mark.parametrize("parameters", [
  {"correction_limit_fraction": value} for value in (0, -0.1, 1.1, float("nan"), float("inf"))
] + [
  {"correction_limit_nm": value} for value in (-1, float("nan"), float("inf"))
])
def test_invalid_correction_settings_do_not_mutate_configuration(parameters):
  cfg = agibot_x2_locomanipulation_flat_env_cfg()
  observer = cfg.actions["upper_body_motion"].observer
  actuators = cfg.scene.entities["robot"].articulation.actuators
  with pytest.raises(ValueError, match="Correction limit"):
    configure_x2_torque_compensation(cfg, **parameters)
  assert cfg.actions["upper_body_motion"].observer is observer
  assert cfg.scene.entities["robot"].articulation.actuators is actuators


def test_preset_matches_joint_transmissions_and_leaves_tendons_unchanged():
  cfg = agibot_x2_locomanipulation_flat_env_cfg()
  robot = cfg.scene.entities["robot"]
  tendon = BuiltinPositionActuatorCfg(
    target_names_expr=X2_OBSERVER_JOINT_NAMES, transmission_type=TransmissionType.TENDON,
    stiffness=50, damping=3, effort_limit=2,
  )
  robot.articulation.actuators = (*robot.articulation.actuators, tendon)
  configure_x2_torque_compensation(cfg)
  assert robot.articulation.actuators[-1] is tendon


def test_preset_requires_playback_on_robot_entity():
  cfg = agibot_x2_locomanipulation_flat_env_cfg()
  cfg.actions["upper_body_motion"].entity_name = "other_robot"
  with pytest.raises(ValueError, match="upper-body motion playback"):
    configure_x2_torque_compensation(cfg)


@pytest.mark.parametrize("change", [
  {"stiffness": 0}, {"stiffness": float("inf")},
  {"damping": -1}, {"damping": float("nan")},
  {"effort_limit": 0}, {"effort_limit": float("inf")},
])
def test_compensated_actuator_rejects_invalid_parameters(change):
  cfg = CompensatedPositionActuatorCfg(
    target_names_expr=("arm",), stiffness=50, damping=3, effort_limit=2,
  )
  with pytest.raises(ValueError):
    replace(cfg, **change)


@pytest.mark.parametrize("gain_scale", [0.5, 1.0, 1.5])
@pytest.mark.parametrize("target,correction", [(0.02, 0.0), (0.02, 1.0), (1.0, 4.0), (-1.0, -4.0)])
def test_native_actuator_combines_pd_and_effort_before_clipping(target, correction, gain_scale):
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body()
  body.add_joint(name="arm", type=mujoco.mjtJoint.mjJNT_HINGE)
  body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.1, 0, 0], mass=1)
  cfg = CompensatedPositionActuatorCfg(
    target_names_expr=("arm",), stiffness=50.0, damping=3.0, effort_limit=2.0,
  )
  entity = SimpleNamespace()
  actuator = cfg.build(entity, [0], ["arm"])
  actuator.edit_spec(spec, ["arm"])
  model = spec.compile()
  model.actuator_gainprm[0, 0] *= gain_scale
  model.actuator_biasprm[0, 1] *= gain_scale
  entity.data = SimpleNamespace(model=SimpleNamespace(
    actuator_gainprm=torch.from_numpy(model.actuator_gainprm.copy()).unsqueeze(0),
  ))
  actuator._global_ctrl_ids = torch.tensor([0])
  data = mujoco.MjData(model)
  data.qpos[0], data.qvel[0] = 0.01, 0.02
  def tensor(value):
    return torch.tensor([[value]], dtype=torch.float64)
  cmd = ActuatorCmd(
    position_target=tensor(target), velocity_target=tensor(0),
    effort_target=tensor(correction), pos=tensor(0.01), vel=tensor(0.02),
  )
  stored_target = cmd.position_target.clone()
  data.ctrl[:] = actuator.compute(cmd).numpy().ravel()
  mujoco.mj_forward(model, data)
  expected = np.clip(50 * gain_scale * (target - 0.01) - 3 * 0.02 + correction, -2, 2)
  assert data.actuator_force[0] == pytest.approx(expected)
  torch.testing.assert_close(cmd.position_target, stored_target)
  # The actual mjlab dispatch must call compute rather than identity batching.
  _, custom = BuiltinActuatorGroup.process([actuator])
  assert custom == (actuator,)


@pytest.mark.parametrize("algorithm", ["linearized", "nonlinear"])
def test_mujoco_closed_loop_compensates_constant_load(algorithm):
  spec = mujoco.MjSpec()
  spec.option.timestep = 0.005
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  body = spec.worldbody.add_body()
  body.add_joint(name="arm", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1])
  body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.1, 0, 0], mass=1)
  cfg = CompensatedPositionActuatorCfg(
    target_names_expr=("arm",), stiffness=50, damping=3, effort_limit=2, armature=0.03,
  )
  entity = SimpleNamespace()
  actuator = cfg.build(entity, [0], ["arm"])
  actuator.edit_spec(spec, ["arm"])
  model = spec.compile()
  actuator._global_ctrl_ids = torch.tensor([0])
  entity.data = SimpleNamespace(model=SimpleNamespace(
    actuator_gainprm=torch.from_numpy(model.actuator_gainprm.copy()).unsqueeze(0),
  ))
  observer = BatchedJointObserver(
    JointObserverCfg(
      joint_names=("arm",), algorithm=algorithm,
      inertia=0.034, correction_limits_nm=1.0,
      release_slew_nm_s=16 if algorithm == "nonlinear" else 2,
    ), (50,), (3,), (2,), num_envs=1, dtype=torch.float64,
  )
  compensated, pd_only = mujoco.MjData(model), mujoco.MjData(model)
  zero = torch.zeros((1, 1), dtype=torch.float64)
  for index in range(400):
    q = torch.from_numpy(compensated.qpos.copy()).reshape(1, 1)
    qd = torch.from_numpy(compensated.qvel.copy()).reshape(1, 1)
    if index % 4 == 0:
      observer.step(zero, q, qd, 0.02)
    command = ActuatorCmd(zero, zero, observer.effort, q, qd)
    compensated.ctrl[:] = actuator.compute(command).numpy().ravel()
    for state in (compensated, pd_only):
      state.qfrc_applied[:] = -0.5
      mujoco.mj_step(model, state)
      assert abs(state.actuator_force[0]) <= 2
  assert abs(pd_only.qpos[0]) > 0.009
  assert abs(compensated.qpos[0]) < 0.001
  assert observer.effort.item() == pytest.approx(0.5, abs=0.02)


def _action():
  action = object.__new__(UpperBodyMotionAction)
  action._env = SimpleNamespace(num_envs=2, device="cpu")
  action._step_dt = 0.02
  action._joint_ids = [1, 2]
  action._observer_joint_ids = torch.tensor([2, 1])
  action._observer_target_cols = [1, 0]
  action._observer_update_pending = True
  action._num_upper_dofs = 2
  action._waist_zero_cols = []
  action._fixed_pose = torch.zeros(2)
  action._pose_target = torch.zeros((2, 2))
  action._default_joint_pos = torch.zeros((2, 2))
  action._gather_targets = lambda: action._pose_target
  cfg = JointObserverCfg(
    joint_names=("right_elbow_joint", "left_elbow_joint"),
    inertia=0.1, friction_viscosity=40, correction_limits_nm=1.0,
    release_slew_nm_s=16,
  )
  action._observer = BatchedJointObserver(cfg, (50, 50), (3, 3), (24, 24), num_envs=2)
  data = SimpleNamespace(
    joint_pos=torch.zeros((2, 3)), joint_pos_biased=torch.zeros((2, 3)),
    joint_vel=torch.full((2, 3), -0.1),
    joint_effort_target=torch.zeros((2, 3)), joint_pos_target=torch.zeros((2, 3)),
  )
  def setter(field, values, joint_ids, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    if isinstance(ids, torch.Tensor):
      ids = ids[:, None]
    getattr(data, field)[ids, joint_ids] = values
  action._entity = SimpleNamespace(
    data=data, _data=data,
    joint_names=["waist_yaw_joint", "left_elbow_joint", "right_elbow_joint"],
    set_joint_position_target=lambda values, **kw: setter("joint_pos_target", values, **kw),
    set_joint_effort_target=lambda values, **kw: setter("joint_effort_target", values, **kw),
    write_joint_state_to_sim=lambda *args, **kw: None,
  )
  return action


def test_observer_initialization_resolves_explicit_joint_order():
  action = _action()
  actuators = []
  for name, joint_id, kp in (
    ("left_elbow_joint", 1, 50), ("right_elbow_joint", 2, 20),
  ):
    cfg = CompensatedPositionActuatorCfg(
      target_names_expr=(name,), stiffness=kp, damping=3, effort_limit=24,
    )
    actuators.append(cfg.build(action._entity, [joint_id], [name]))
  # A tendon may share a joint's name, but belongs to a separate target space.
  tendon_cfg = BuiltinPositionActuatorCfg(
    target_names_expr=("left_elbow_joint",), transmission_type=TransmissionType.TENDON,
    stiffness=10, damping=1,
  )
  actuators.append(tendon_cfg.build(action._entity, [0], ["left_elbow_joint"]))
  action._entity.actuators = actuators
  action._initialize_observer(action._observer.cfg)
  assert action._observer_joint_ids.tolist() == [2, 1]
  assert action._observer_target_cols == [1, 0]
  torch.testing.assert_close(action._observer.kp, torch.tensor([[20.0, 50.0]]))


def test_action_updates_once_per_control_period_and_routes_arms_only(monkeypatch):
  action = _action()
  calls = []
  original_step = action._observer.step
  def step(*args):
    calls.append(args[-1])
    return original_step(*args)
  monkeypatch.setattr(action._observer, "step", step)
  for _ in range(3):
    action.process_actions(torch.empty((2, 0)))
    for _ in range(4):
      action.apply_actions()
  assert calls == [0.02] * 3
  assert action.action_dim == 0
  assert torch.count_nonzero(action._entity.data.joint_effort_target[:, 0]) == 0
  torch.testing.assert_close(
    action._entity.data.joint_effort_target[:, action._observer_joint_ids], action._observer.effort
  )


def test_action_partial_reset_clears_queued_effort_before_fixed_pose_return():
  action = _action()
  for _ in range(3):
    action.process_actions(torch.empty((2, 0)))
    action.apply_actions()
  survivor = action._observer.effort[1].clone()
  assert torch.count_nonzero(survivor) > 0
  action.reset(torch.tensor([0]))
  assert torch.count_nonzero(action._entity.data.joint_effort_target[0]) == 0
  torch.testing.assert_close(action._observer.effort[1], survivor)
  assert not action._observer.initialized[0]


def test_pose_teleport_clears_observer_history_and_effort():
  action = _action()
  action._observer.effort[:] = 0.5
  action._entity.data.joint_effort_target[:, 1:] = 0.5
  action.set_fixed_pose({"left_elbow_joint": -0.5})
  assert torch.count_nonzero(action._observer.effort) == 0
  assert torch.count_nonzero(action._entity.data.joint_effort_target) == 0
  assert not action._observer.initialized.any()
