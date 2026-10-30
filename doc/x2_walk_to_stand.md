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

## Contact-aware reference blending

Standing and walking references are blended independently for each six-joint leg.
A planted leg tracks its frozen reference, not a newly switched destination.
After confirmed liftoff with the opposite foot in confirmed support, smoothstep
advances toward the destination using the
existing nominal swing duration: gait period times one minus stance fraction,
currently 0.264 seconds. Early landing freezes the intermediate reference. On
another swing, a new smoothstep segment covers the remaining progress using
nominal swing duration times remaining progress. No new timing parameter is
introduced. Double-foot flight freezes both references and cannot generate new
landing-progress credit. A swing that includes double-foot flight cannot pay
landing credit, including progress earned before that flight. The reference is
retained, and a later supported placement can earn the unpaid credit. A supported
step may resume progress after landing. Reference position is continuous at reversals and contact events;
unexpected touchdown can interrupt reference velocity.

A command reversal snapshots the current blended reference and changes its
destination. Reset episodes initialize directly to their current command's
reference. Contact changes still require two samples. The common waist reference
is preserved. Height selection remains discrete: a changed nearest-height row
selects that row directly, so continuity across height-command changes is not
claimed. Training holds height fixed within an episode.

During stopping, full linear-velocity and angular-velocity tracking remain
active at their existing weights, along with leg-joint-velocity and base-drift
penalties. This restores the objective of braking and stabilizing throughout
the grace period. Pose relaxation and gait routing retain their existing behavior.

The 1.2-second routing deadline restores normal standing rewards unconditionally.
It never forces a planted leg's reference to its endpoint. Reference updates and
diagnostic landing progress can continue on later swings after that deadline;
reward credit continues only for startup. Both
pose and stand-still rewards track the current reference, including partial
references after stopping. This replaces binary pose masking. Existing walking
joint tolerances and stopping soft-pose tolerances are retained.

Command reversals that leave a leg reference already at its destination mark
that leg complete without new credit, avoiding rewards for unnecessary steps.
Each confirmed landing still records reference progress for diagnostics.
Only stand-to-walk earns a heading-weighted landing-progress bonus; a standing
command earns no landing bonus, even after the routing deadline. The
`placement_weight` configuration field is retained for compatibility but no
longer contributes to this reward. Startup credit retains RewardManager dt
compensation and its per-command bound. Braking potential shaping and contact
slip penalties remain active.

The actor remains at 430 inputs. The critic adds only two absolute leg-blend
coordinates to its existing 121 inputs, giving 123: zero means standing, one
means walking. These describe the current reference even after a reversal; local
transition progress alone would not. Joint references are reconstructed from
these coordinates and the commanded-height tables. Symmetry swaps the two
coordinates without sign changes. Existing checkpoints support actor-only
evaluation; train the new critic from scratch. The actor must infer reference
state from commands and proprioceptive history. This compact critic input has
not yet been compared against the full reference input in trained-policy tests.

Reference completion is separate from physical stopping and reward routing.
Diagnostics log mean progress and completion fraction. Legacy exact-stance comparisons require
at least 95% reference completion among successful trials, and at least 95%
complete transitions (physical stop plus reference completion) across all trials. A quiet partial stance
does not pass those legacy comparisons. Use `--stopping-comparison` for the
physical-stop experiment below. No additional hip-motion penalties are introduced.

Dense validation sampled 29,106 mixed-leg configurations with neutral and two
carry poses across all table heights. There were no joint-limit violations or
new penetrating contacts in the trained 0.30–0.64 m range. Maximum foot-path
excursion beyond the endpoint box was 4.3 mm, with no heading overshoot. At
0.22–0.26 m, some carry-pose blends introduce elbow/thigh collisions. These
heights are outside the current training range and are not validated for loaded
reference blending. This finite scan is not a guarantee for every arm pose.

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
seed, height, command, and command-switch timing. Loaded comparisons require version 6 reports; regenerate both baseline and
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


Add `--save-traces` to include actual/reference joint positions, leg progress,
and joint velocities. Reports include first-1.5-second hip speed, acceleration,
and jerk peaks in both directions, plus reference completion times and counts.
Physical stop success is reported separately from complete-transition success.
Regenerate both reports with guidance protocol `supported-reference-blend-v1` and schema 6.


Both transition directions now report unsupported flight time, peak base rise
above commanded height, and peak world-vertical speed over the full command
interval. Hip smoothness metrics retain the first-1.5-second window. Late hops
therefore cannot evade flight checks after the routing deadline.
Paired loaded comparisons reject increased flight time and more than 10%/0.01
regression in base rise or vertical speed. This distinguishes
ordinary single-support steps from hopping without imposing double support.

The test-3 checkpoint reproduced 20–40 ms of double-foot flight when stopping
in two loaded forward trials, versus none for test-2. The blend/reward fix does
not alter an already trained actor: validate it in a new training experiment.

## Stopping stabilization experiment

The first correction for excessive stopping steps preserves both posture tables,
supported reference blending, the 1.2-second deadline, and startup guidance.
It removes stopping landing bonuses and restores linear/angular tracking,
leg-velocity and drift penalties during stopping. Actor/critic input dimensions
and the other tasks are unchanged. Existing actors must be retrained to test the
new objective; replaying test-4 only establishes the baseline.

Train from scratch with the same test-4 seed, environment count, and budget:

```bash
PYTHONPATH="" .venv/bin/python scripts/train.py Agibot-X2-Locomanipulation-Transition-Flat \
  --env.scene.num-envs=4096 --agent.seed=42 --agent.max-iterations=10001 \
  --agent.run-name=stopping-stabilization
```

Generate the baseline and candidate reports using the same command (replace
CHECKPOINT and REPORT for each run):

```bash
PYTHONPATH="" .venv/bin/python scripts/eval_stopping.py \
  --checkpoint CHECKPOINT --output REPORT --scenario stand-walk-stop \
  --carry-poses --loads-n 0 --smoke --phases 8 --heights .64 --seeds 42 43 \
  --save-traces
```

`--smoke` selects the requested 0.5 m/s forward command; here eight phases and
two seeds give 16 trials. For the candidate, add
`--baseline BASELINE_REPORT --stopping-comparison` to select physical-stop
acceptance instead of the legacy exact-stance criteria. Both reports must have
`stopping_steps.version=1`; regenerate older baselines. These are additive
fields in report schema 6, so legacy comparisons remain available.

The new comparison requires fewer mean post-braking touchdowns, successful
uncensored stops, no falls or stopping flight, and no increase in touchdown
speed, peak contact force, roll/pitch range, or horizontal excursion. Settling
and startup delay may increase by at most 0.2 seconds. Startup hip-motion peaks
and walking errors use 10%/0.01 tolerances; startup flight must not increase.
Reference completion remains a separate diagnostic, not a physical-stop gate.

Touchdowns and pre-impact downward sole velocity are sampled at policy rate.
Post-braking counts begin strictly after the first stopping sample with
horizontal speed below 0.1 m/s; missing braking is reported as null and fails
comparison. Contact force is the net force per foot: the evaluator enables
sensor history across every physics substep and takes its peak norm, rather
than averaging substep peaks away. Body sway is reported as world-horizontal
position range and roll/pitch range over the full stop interval. These metrics
include normal braking excursion; they are paired comparisons, not universal
stability thresholds. Raw policy-rate contact, sole velocity, root pose, and
per-step force peaks are included with `--save-traces`.
