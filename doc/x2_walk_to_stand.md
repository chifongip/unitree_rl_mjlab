# X2 locomanipulation transition training

Use `Agibot-X2-Locomanipulation-Transition-Flat` to train a new policy:

```bash
PYTHONPATH="" .venv/bin/python scripts/train.py Agibot-X2-Locomanipulation-Transition-Flat \
  --env.scene.num-envs=4096
```

The task uses the existing X2 PPO configuration, learning rate, height curriculum,
upper-body/force curricula, randomization, and checkpoint runner. There is no
checkpoint migration, critic-only warm-up, special optimizer handling, or automatic
evaluation during checkpoint saving. Existing Flat, Rough, and Carry tasks are
unchanged.

The primary objective is carrying and transporting heavy objects. Transition
practice supports pickup, loaded transport, and placement. The existing force
and upper-body curricula remain responsible for exposure to loads and poses.
The episode mix is fixed throughout training; for full 20-second episodes it
allocates about 58.5% of command time to standing and 41.5% to walking. This
mix has not been established as optimal for loaded transport.

## Changes specific to the transition

Episode profiles provide 30% sustained standing, 30% walking for 2–4 seconds
followed by standing, 20% standing for 2–4 seconds followed by walking, and
20% continued walking. The new fraction is configurable as `start_walking_ratio`.
Stand-to-walk uses the existing walking rewards immediately when the sampled
velocity command is released, without resetting the robot. Profiles persist through each
episode. Height and waist-yaw commands are sampled using their existing curricula
and held for the episode, keeping stopping trials separate from simultaneous
command changes.

A shared state tracks both command transitions. Each foot progresses from
awaiting lift, to swing, to landed. Contact changes require two control samples;
a foot with confirmed airborne history when the command changes starts in swing.
Raw contact dropouts at the command boundary cannot bypass confirmation. Startup
finishes after both feet have landed once and may continue stepping. Stopping
also requires current confirmed support from both feet, so a foot that lifts
again cannot cause premature completion. Both directions retain the unconditional
`transition_s=1.2` second deadline (two gait cycles).
The deadline is unconditional: there are no posture-success gates. Standing
from reset activates the original standing rewards immediately. Reversing the
command starts a new transition toward the latest destination; resets clear only
the selected environments.

While a foot awaits lift, its six leg joints are masked from the destination
pose reward, retaining the original denominator and torso contributions. During
swing and after landing, its destination posture is active. Walking retains its
original joint tolerances; stopping retains the soft posture tolerance. The active pose reference guides the swinging leg. A one-time bonus evaluates
sole heading at its first confirmed landing in both directions; stopping also
evaluates relative standing foot placement. `swing_weight=0.5` and
`placement_weight=1.0` are per-landing bonus weights. The reward returns a rate
(dividing by control dt) so the reward manager's dt scaling yields one bonus,
independent of swing duration. Waiting in the air earns no guidance bonus, and
repeated lift/landing cycles do not pay again. These replace the old bilateral
placement and hip-only swing rewards.
No posture interpolation or fixed stepping order is introduced.

Walking velocity and gait rewards remain active during startup. During stopping,
strict stillness, base drift, velocity tracking and periodic contact terms are
deferred until completion or the deadline. Contact slip penalties remain active
throughout. Potential shaping accounts for entry, exit, reversal and termination,
so there is no initial-speed bonus or constant elapsed-time penalty.

The actor retains its 430-input layout and phase encoding. The existing ten
critic transition features are supplemented by three values: startup-active,
left foot stage and right foot stage (0, 0.5, 1). This gives 121 critic inputs.
Symmetry swaps the foot stages and reflects the signed placement errors.
Existing checkpoints support actor-only evaluation; train the new critic from
scratch. Height tables retain nearest-height lookup with no interpolation.

`Transition/standing_fraction` reports the standing reward phase rather than
measured pose success. Placement and hip-yaw errors are averaged only over
standing commands, including stopping, and report zero when none are present.

Reward weights, sampling fractions, and the grace period (`transition_s`)
are fields of the task's twist command configuration. Foot geometry is derived
from the existing standing table using the generator's sole-centroid convention.
Placement and toe-out targets use the current body-heading frame, so turning
stops do not require the actor to remember a heading captured at the stop request.

## Optional evaluation

Evaluation is a standalone stopping diagnostic and does not run inside training.
By default it inherits the play configuration, including fixed arms and play-mode hand
forces; that default invocation does not reproduce the training load/pose
distribution or evaluate stand-to-walk transitions. Its acceptance result does not establish heavy-object
transport capability. That requires separate evaluation of loaded walking,
turning, starts, stops, and object stability.

Run the stopping diagnostic with:

```bash
PYTHONPATH="" .venv/bin/python scripts/eval_stopping.py \
  --checkpoint logs/rsl_rl/x2_locomanipulation/RUN/model_5000.pt \
  --output /tmp/stopping.json
```

Keep the checkpoint's saved `params/` directory alongside it so evaluation
restores the arm controller. Use `--smoke --phases 4 --heights .64` for a short
check. Use `--baseline /tmp/baseline.json` to compare paired grids.

The tool measures posture error at final touchdown, hip-yaw adjustment over the
following 0.5 seconds, contact slip/rotation, stopping time, height tracking,
walking/turning error, and falls. Stop success requires horizontal speed below
0.1 m/s, yaw speed below 0.1 rad/s, and bilateral contact throughout the final
0.5 seconds, independently of the training reward phase. Settling time is the
start of the first continuous 0.5-second stop, or null when none occurs.
It preserves the training process's RNG state and CPU thread count if invoked programmatically.

Paired acceptance additionally requires at least 90% successful stops with no
success-rate regression, final height and placement errors at most 3 cm, and
final heading errors at most 10 degrees. Comparative style improvements cannot
override these absolute stopping requirements.

These are training and evaluation changes, not a converged policy. Results from
the earlier checkpoint-adaptation pilot do not establish performance when
training from scratch. Validate the trained model before deployment.

The design is informed by
[van Marum et al. (2024)](https://arxiv.org/html/2404.19173v2)
and [Peng et al. (2025)](https://arxiv.org/html/2505.20619v3).

## Consistent planted-foot slip costs

The transition task uses one `foot_slip` reward for all command phases. It sums
contacting sole-centroid horizontal velocity squared and world-vertical foot
angular velocity squared. The twist configuration exposes `slip_weight=0.25`
and `yaw_slip_weight=0.25`; the reward term itself has weight 1. Airborne feet
are unpenalized. Independent roll/pitch angular costs are omitted to permit
foot rollover. The former walking-only slip term is replaced, and the stopping
reward no longer adds another slip penalty.

This isolates the missing walking-phase yaw cost. The per-foot transition guidance above shares this physical penalty. Episode
ratios and the load curricula are unchanged.
The two cost components are logged independently under `Transition/`.

## Loaded start/stop diagnostic

Use explicit carry poses and per-hand downward forces to evaluate both
transitions. These options override the fixed-arm/default force settings of the
legacy stopping diagnostic:

```bash
PYTHONPATH="" .venv/bin/python scripts/eval_stopping.py \
  --checkpoint logs/rsl_rl/x2_locomanipulation/2026-10-30_01-25-12/model_10000.pt \
  --scenario stand-walk-stop --carry-poses --loads-n 0 20 40 \
  --heights .64 .5 .3 --phases 4 --seeds 42 43 \
  --output /tmp/x2-transition-baseline.json
```

Each trial stands for 3 seconds plus a phase offset, walks for 5 seconds, then
stands for at least 4 seconds. The command grid includes forward/backward,
both lateral directions, and both turning directions. Initial pose sampling is
seeded for every height; actual joint names and targets are saved in each row
for replay and exact comparison. Every comparison checks scenario, load, pose,
seed, height, command, and command-switch timing. Loaded comparisons require version 3 reports; regenerate both baseline and
candidate reports with the current evaluator.

Per-foot metrics include grounded sliding/yaw, motion before first liftoff,
liftoff and touchdown positions, touchdown world yaw, and motion in the next
0.5 seconds. Start metrics end after the first touchdown and its stabilization
window; stop metrics use the final touchdown. Missing events produce null event
metrics and explicit flags. A stopping foot may remain planted if contact is
continuous and its accumulated sliding and yaw remain within 3 cm and 10 degrees.
That support foot needs no artificial lift/landing event; its event metrics stay
null and its full contact-motion costs are still compared. Physical stopping
success remains separate from transition-event availability.

Repeat the command with the candidate checkpoint and
`--baseline /tmp/x2-transition-baseline.json`. Comparisons require at least 50%
lower median grounded yaw and no increase in its 95th percentile or sliding,
separately for each direction and load. They also require no regression in
comparable pre-liftoff and post-touchdown motion (including hip yaw). Final
foot-placement error must be at most 3 cm and heading error at most 10 degrees.
They reject incomplete/censored events except valid continuous stopping support,
and reject falls in comparison traces. Tracking/height/carry-pose errors must remain
within 10% or 0.01 absolute of the baseline per trial, and start/settling delays
within 0.2 seconds. These are engineering acceptance targets, not guarantees
from the literature. Forces approximate a carried load; no physical box or
object-retention test is included.

Use `--smoke --phases 1 --heights .64 --seeds 42 --loads-n 0 40` for a short
execution check. Train the candidate from scratch with seed 42 and the existing
10,001-iteration PPO budget before comparing learned behavior. A checkpoint
trained with the old reward cannot demonstrate the new reward's effectiveness.


The loaded diagnostic also records transition duration, completion reason
(`landed` or `deadline`), and per-foot stage-change times for both directions.
Comparisons require matching maximum duration and guidance protocol. Regenerate
baseline reports with the same evaluator before comparing a newly trained model.
