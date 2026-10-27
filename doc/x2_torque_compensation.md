# X2 training torque compensation

The controller ports RoboJuDo-Plus's synchronous linearized and nonlinear
joint disturbance observers to batched Torch tensors. It controls the 14 arms;
the policy still controls the same 15 leg/waist joints. Actor and critic
observations, their ordering, history, corruption, and dimensions are unchanged.
No observer output is added to either observation group.

The X2 rough, flat and carry fine-tuning factories enable the nonlinear observer
in ACTIVE mode by default, for both training and playback. No CLI flag is needed:

```bash
python scripts/train.py Agibot-X2-Locomanipulation-Carry-Finetune-Flat \
  --env.scene.num-envs 4096
```

Controller configuration belongs to the X2 environment factory in
`src/tasks/locomanipulation/config/x2/env_cfgs.py`. Flat and carry tasks inherit
these settings from the rough factory:

```python
configure_x2_torque_compensation(
  cfg,
  algorithm="nonlinear",
  mode="ACTIVE",
  correction_limit_fraction=0.8,
  correction_limit_nm=0.0,
)
```

The default `correction_limit_nm=0` allows 80% of every arm actuator's torque
limit without an absolute Nm cap. Set a positive value, such as `8.0`, to apply
an additional cap. The fraction must lie in `(0, 1]`; the cap
must be finite and nonnegative. Select `algorithm="linearized"` for the comparison
controller. The training script has no torque-compensation configuration flags.

The upper-body action's environment setting `restore_observer_from_checkpoint`
defaults to `True`. Resume, playback and evaluation restore the saved observer and
arm-actuator settings, including bounds, before constructing the environment.
To fine-tune with the current environment's controller instead, set
`restore_observer_from_checkpoint=False` in the environment factory. This also
allows compensation when fine-tuning a legacy PD-only checkpoint. The selected
controller settings are saved with the new run for subsequent resume/play/eval.
PD headroom, total torque clipping and the existing slew rates still apply.

For programmatic use, the factories also enable compensation automatically:

```python
from mjlab.envs import ManagerBasedRlEnv
from src.tasks.locomanipulation.config.x2.env_cfgs import (
  agibot_x2_locomanipulation_carry_finetune_flat_env_cfg,
)
cfg = agibot_x2_locomanipulation_carry_finetune_flat_env_cfg()
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
```

Existing task names remain unchanged. Training's
`env_overrides.yaml` retains all observer vectors and arm-actuator settings.
Standard checkpoint playback and evaluation restore those settings before
environment construction, including custom gains, effort limits, noise and slew
parameters.
Older incomplete overrides are supplemented from the run's full `env.yaml`;
missing model/actuator settings raise an error instead of selecting PD-only
control or guessing current preset coefficients. PD-only checkpoints retain
their disabled observer during training resume, playback and evaluation. Legacy
checkpoints with no observer field or no saved environment parameters also use
PD-only control. Fresh training keeps the ACTIVE factory default.

`configure_x2_torque_compensation` can override the factory preset before
environment construction. It accepts `linearized`
and modes `OFF` or `SHADOW`. SHADOW estimates but issues zero correction and
predicts using that issued zero. Configuration is captured at construction; build
a new environment to change algorithms or modes.

The X2 preset copies the frozen simulation reference from
`RoboJuDo-Plus/robojudo/config/x2/observer_parameters.py`: 8 rad/s bandwidth,
0.3 Nm nominal friction, and the 14 inertia/viscosity coefficients in explicit
left-seven/right-seven name order. Correction bounds are
`0.8 * actuator_limit` by default; growth slew is 2 Nm/s. The nonlinear observer
releases toward zero at 16 Nm/s and waits until a later update to grow opposite
torque. The linearized observer uses symmetric 2 Nm/s slew.

At each control interval, the upper-body action samples its existing position
target, encoder position (including configured encoder bias), and joint velocity.
The controller predicts the previous interval using its previous target and
accepted correction, estimates disturbance, applies anti-windup and PD headroom
limits, then acknowledges the issued correction. At the default 20 ms control
period it holds that correction over four 5 ms physics steps. Native PD and
combined torque clipping continue at every physics step. Safety headroom removal
can be faster than normal correction slew, matching deployment.

The custom position actuator converts additive torque to an internal control
offset `tau_comp / Kp`. The original stored position target stays unchanged.
This retains MuJoCo's native PD integration and clips PD plus correction within
the original actuator force limits. The observer model does not read simulated
mass, gravity torque, acceleration, payload force, or dynamics matrices.
Payload physics and force events continue unchanged.

The actuator implements mjlab's standard PD parameter-setting interface while
retaining native MuJoCo position elements. Existing `dr.pd_gains` and
`dr.effort_limits` events work for compensated arms, including per-world resets
and scaling relative to defaults. Gain changes update the native gain/bias
parameters; limit changes update native force ranges. The observer retains its
fixed nominal model, so randomized plant gains do not leak into its estimator.

The observer has optional `position_noise_std` (rad) and `velocity_noise_std`
(rad/s) settings, independently configured from policy observation corruption.
Both default to zero (ideal joint sensing); these are not measured hardware
noise specifications. Set them before constructing the environment if needed.
The port currently assumes fresh synchronous feedback and zero command transport
delay; it does not emulate deployment timestamp replay, feedback dropouts or
delayed actuator commands. The helper rejects arm actuator command delays.

Partial episode resets clear only those worlds' histories and queued effort.
Pose teleports also clear history and effort. First feedback initializes history
without issuing compensation. Nonfinite input in one world clears its correction
and reinitializes that world's history without affecting other worlds.

The coefficients and torque bounds are simulation reference settings. Match the
settings actually used in deployment before interpreting transfer results; the
port does not qualify physical torque limits or hardware performance.

Validation:

```bash
PYTHONPATH="" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests -v -p no:launch_testing
```

`tests/fixtures/x2_observer_reference.npz` was generated independently from the
RoboJuDo NumPy runtime in ACTIVE and SHADOW modes for both algorithms. Its JSON
manifest identifies source hashes, timing and synthetic feedback provenance.
Tests compare the single- and double-precision Torch port with those reference outputs,
check bounds, release, headroom, resets, native actuator routing, control cadence,
and unchanged observation/action/reward definitions for all X2 task factories.
Additional regression tests exercise YAML round-trips, legacy full-YAML recovery,
standard gain/limit randomization and all six algorithm/mode combinations in
headless two-world X2 environments with partial resets on CPU and CUDA where
available. Short PPO training/resume checks establish training-path compatibility;
they do not establish convergence, full-scale throughput or hardware performance.
