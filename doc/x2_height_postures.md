# X2 recorded height postures and knees-out corrections

The knees-out table starts from `scripts/postures_x2_recorded.py`, which is
resampled from the real-robot joint measurements in `scripts/height_postures.csv`.
It preserves the recorded waist angles exactly and the grounded pelvis
orientation at each height. It does not use the nominal home pose as its
reference.

The generator solves `q_new(height) = q_recorded(height) + delta(height)` for
the 12 leg joints. Joint-pitch deviation and changes to the preceding correction
are penalized. Both feet must stay flat; each knee must move outward and retract
relative to the recorded knee origin. Foot placement and toe-out are preferences,
so the solver can find a stance without forcing fixed foot poses at every height.
The grounded pelvis orientation stays fixed during IK, retaining the recording's
pelvis and torso pitch instead of forcing an upright base.

Splay fades in smoothly from zero at 0.64 m to full strength at 0.40 m. At full
strength, the default minimum outward displacement is 0.10 m per knee and the
minimum retraction is 0.03 m. The preferred toe-out is 30 degrees per foot and the
preferred outward foot displacement is 0.05 m per foot. The hard leg-pitch change
limit is 0.35 rad; the default generated table requires at most 0.054 rad.

The original recorded table remains the default for existing tasks. The corrected
table is a separate file for use when needed; no additional environment profile
is registered. To select it, change the posture filename in
`src/tasks/locomanipulation/config/x2/env_cfgs.py` from
`postures_x2_recorded.py` to `postures_x2_knees_out.py`. That existing loader
supplies both `pose` and `stand_still` rewards with the same named leg/waist joint
selection. Waist yaw retains its independent command and is excluded from
posture rewards.

## Generate and inspect

Use the repository's Python 3.11 environment, for example `.venv/bin/python`.
To resample a baseline from the recorded CSV:

```bash
python scripts/resample_height_postures.py scripts/height_postures.csv \
  --output /tmp/postures_x2_recorded.py
```

The checked-in reference includes zero waist-pitch targets over 0.60-0.64 m.
Resampling the raw CSV produces small negative waist-pitch values there. The
correction generator uses the checked-in reference by default, preserving those
existing upright targets as well as the recorded crouch targets.

The resampler retains its original behavior: leg pitch joints are symmetrically
averaged, leg roll/yaw joints are zeroed, and waist yaw/roll are zeroed. It is used
to prepare the recorded baseline; knees-out corrections happen afterwards in the
separate generator.

Generate a correction of the checked-in recorded table:

```bash
python scripts/generate_knees_out_postures.py \
  --reference scripts/postures_x2_recorded.py \
  --output /tmp/postures_x2_knees_out.py \
  --report /tmp/postures_x2_knees_out.json \
  --plot /tmp/x2_knees_out_postures.png
```

The script rejects infeasible heights and invalid rounded postures before writing
the table. The JSON report includes achieved height, pelvis orientation change,
knee displacements, foot headings/width, pitch changes, self-penetration and the
unloaded center-of-mass support margin for every height. Existing output files
require `--overwrite`. To regenerate the separate checked-in correction table,
set `--output scripts/postures_x2_knees_out.py --overwrite`.

```bash
python scripts/playback_height_postures.py scripts/postures_x2_knees_out.py --validate-only
python scripts/playback_height_postures.py scripts/postures_x2_knees_out.py --once
```

The reference CSV and original posture table remain editable. After adjusting
the original recorded table, rerun the generator with that table as `--reference`
to refresh the separate knees-out output. Changes are not propagated
automatically. If modifying leg angles directly, verify or resample their grounded
height first: the generator rejects references whose height labels no longer
match their grounded height. The generator reads its reference without changing
it and rejects an output path that points to the reference file.

After selecting corrected targets, use the normal `--agent.resume`,
`--agent.load-run` and `--agent.load-checkpoint` training options to fine-tune an
existing walking policy. Loading the new reward table does not change the
behavior of an already trained policy by itself.

## Measured kinematic changes

The shipped table contains all 22 original heights from 0.22 m through 0.64 m.
Numbers below describe each knee; the two sides are mirrored.

| Pelvis height | Outward displacement | Retraction | Maximum leg-pitch change |
| --- | --- | --- | --- |
| 0.64 m | 0 cm | 0 cm | 0 rad |
| 0.50 m | 7.81 cm | 1.87 cm | 0.0532 rad |
| 0.40 m | 13.13 cm | 3.00 cm | 0.0330 rad |
| 0.22 m | 14.33 cm | 3.42 cm | 0.0185 rad |

Maximum height error after rounding and regrounding is 0.033 mm. All rounded
joint values remain within the MJCF hard limits, with no detected self-penetration
greater than 1 mm and the unloaded COM projection inside the foot support hull.
The additional joint correction changes by at most 0.065 rad between adjacent
height samples. The original recording's sharp pitch change between its lowest
two samples is retained.

![Recorded and corrected X2 leg geometry](x2_knees_out_postures.png)

The plotted polylines connect hip-roll joint origins, knee origins and foot
support centroids; they illustrate joint geometry rather than robot collision
meshes. As in the existing recording/playback pipeline, height and flatness use
foot capsule **centerline endpoints** as the sole-point datum. Capsule radii are
not included in that datum, so these heights are for comparison with the existing
training table, rather than a new calibration of physical sole height.

These checks establish kinematic feasibility and unloaded static support only.
Foot positions and headings vary across heights: the default stance widens from
0.274 m at 0.64 m pelvis height to 0.374 m at 0.40 m. Playback directly sets joint
poses and does not simulate the contact forces needed for that transition. These
references therefore do not establish a squat with planted feet; that requires
fixed foot positions/headings across heights and subsequent policy validation.
They do not validate load carrying, ground friction, actuator torque, walking
transitions or balance under disturbances. Knee-origin retraction is a clearance
proxy; an actual carried object still requires checking its collision geometry
and the full leg shapes during policy evaluation.
