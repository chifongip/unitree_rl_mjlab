"""Exercise the standard mjlab randomization events and native actuator output."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from mjlab.actuator import ActuatorCmd
from mjlab.actuator.fused_group import FusedActuatorGroup
from mjlab.envs.mdp.dr.actuator import effort_limits, pd_gains
from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.assets.robots.agibot_x2.compensated_actuator import CompensatedPositionActuatorCfg


def _environment():
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body()
  body.add_joint(name="arm", type=mujoco.mjtJoint.mjJNT_HINGE)
  body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.1, 0, 0], mass=1)
  entity = SimpleNamespace(indexing=SimpleNamespace(ctrl_ids=torch.tensor([0])))
  actuator = CompensatedPositionActuatorCfg(
    target_names_expr=("arm",), stiffness=50, damping=3, effort_limit=2,
  ).build(entity, [0], ["arm"])
  actuator.edit_spec(spec, ["arm"])
  native = spec.compile()
  model = SimpleNamespace(**{
    field: torch.from_numpy(getattr(native, field).copy()).unsqueeze(0).repeat(2, 1, 1)
    for field in ("actuator_gainprm", "actuator_biasprm", "actuator_forcerange")
  })
  entity.data = SimpleNamespace(model=model)
  actuator.initialize(native, model, SimpleNamespace(nworld=2), device="cpu")
  entity.actuators = [actuator]
  env = SimpleNamespace(num_envs=2, device="cpu", scene={"robot": entity},
                        sim=SimpleNamespace(model=model))
  return env, actuator, native


@pytest.mark.parametrize("operation", ["scale", "abs"])
def test_standard_gain_event_updates_native_pd_and_preserves_other_worlds(operation):
  env, actuator, native = _environment()
  kp, kd = ((1.4, 4 / 3) if operation == "scale" else (70.0, 4.0))
  for _ in range(2):  # Scaling remains relative to defaults, not the last sample.
    pd_gains(env, torch.tensor([1]), (kp, kp), (kd, kd), operation=operation,
             asset_cfg=SceneEntityCfg("robot"))
  model = env.sim.model
  torch.testing.assert_close(model.actuator_gainprm[:, 0, 0], torch.tensor([50.0, 70.0], dtype=torch.float64))
  torch.testing.assert_close(model.actuator_biasprm[:, 0, 1], torch.tensor([-50.0, -70.0], dtype=torch.float64))
  torch.testing.assert_close(model.actuator_biasprm[:, 0, 2], torch.tensor([-3.0, -4.0], dtype=torch.float64))
  # The event changes the physical law, and correction remains additive.
  native.actuator_gainprm[:] = model.actuator_gainprm[1].numpy()
  native.actuator_biasprm[:] = model.actuator_biasprm[1].numpy()
  values = [torch.full((2, 1), x) for x in (0.02, 0.0, 0.5, 0.01, 0.02)]
  command = ActuatorCmd(*values)
  state = mujoco.MjData(native)
  state.qpos[0], state.qvel[0] = 0.01, 0.02
  state.ctrl[:] = actuator.compute(command)[1].numpy()
  mujoco.mj_forward(native, state)
  assert state.actuator_force[0] == pytest.approx(70 * 0.01 - 4 * 0.02 + 0.5)
  assert actuator.cfg.stiffness == 50  # Observer's nominal gain remains fixed.
  _, custom = FusedActuatorGroup.process([actuator])
  assert custom == (actuator,)


def test_standard_gain_event_supports_all_worlds():
  env, actuator, _ = _environment()
  pd_gains(env, None, (0.8, 0.8), (0.5, 0.5), asset_cfg=SceneEntityCfg("robot"))
  torch.testing.assert_close(actuator.stiffness, torch.full((2, 1), 40.0))
  assert np.allclose(env.sim.model.actuator_gainprm[:, 0, 0], 40.0)


@pytest.mark.parametrize("operation,value", [("scale", 0.5), ("abs", 1.0)])
def test_standard_effort_limit_event_updates_native_clamp(operation, value):
  env, actuator, native = _environment()
  effort_limits(env, torch.tensor([1]), (value, value), operation=operation,
                asset_cfg=SceneEntityCfg("robot"))
  torch.testing.assert_close(env.sim.model.actuator_forcerange[:, 0],
                            torch.tensor([[-2.0, 2.0], [-1.0, 1.0]], dtype=torch.float64))
  native.actuator_forcerange[:] = env.sim.model.actuator_forcerange[1].numpy()
  state = mujoco.MjData(native)
  state.ctrl[:] = 1.0
  mujoco.mj_forward(native, state)
  assert state.actuator_force[0] == pytest.approx(1.0)


def test_invalid_native_gain_does_not_update_the_plant():
  env, actuator, _ = _environment()
  original = env.sim.model.actuator_gainprm.clone()
  with pytest.raises(ValueError):
    pd_gains(env, None, (0.0, 0.0), (1.0, 1.0), asset_cfg=SceneEntityCfg("robot"))
  torch.testing.assert_close(env.sim.model.actuator_gainprm, original)
  torch.testing.assert_close(actuator.stiffness, torch.full((2, 1), 50.0))
