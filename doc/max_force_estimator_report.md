# T-Pose Maximum Force Report

## Scope and Method

This report evaluates `MaxForceEstimator` at the third predefined upper-body
pose in `scripts/play.py` (selected with F10). X2 uses shoulder roll angles of
±1.57 rad with straight elbows. G1 and G1-23DOF use shoulder roll angles of
±1.57 rad and elbow angles of +1.57 rad. Other arm joints remain at the zero or
model-default values specified by the pose definitions.

Each flat environment was loaded with `play=True`, one CPU simulation, and
startup domain randomization disabled. The robot was placed exactly at the
T-pose with zero velocity and matching position targets before forward
dynamics. The estimator retained its 90% actuator-limit envelope. Instantaneous
arm actuator forces were 0 Nm at this exact target, making the reported signed
bounds symmetric.

## Nominal Per-Hand Capacity

Forces were evaluated at each end-effector center (`offset = 0`). Values are
one-axis capacities; the three maxima cannot be applied simultaneously because
the force event uses Dirichlet axis scaling.

| Robot | End effector | ±X (N) | ±Y (N) | ±Z (N) |
|---|---|---:|---:|---:|
| Agibot X2 | Left wrist | 142.81 | 400.98 | 93.62 |
| Agibot X2 | Right wrist | 141.40 | 400.71 | 93.18 |
| Unitree G1 | Left wrist | 38.52 | 2596.92 | 51.30 |
| Unitree G1 | Right wrist | 38.52 | 2596.92 | 51.30 |
| Unitree G1-23DOF | Left hand | 108.20 | 2374.93 | 57.45 |
| Unitree G1-23DOF | Right hand | 108.20 | 2374.93 | 57.45 |

Taking the weaker hand, the nominal robot-level capacities are **X2: (141.40,
400.71, 93.18) N**, **G1: (38.52, 2596.92, 51.30) N**, and **G1-23DOF:
(108.20, 2374.93, 57.45) N**. The very large Y estimates for both G1 variants
come from near-zero Y-force Jacobian coefficients in this pose. They should not
be treated as physical payload ratings.

## Application-Point Sensitivity

Training samples a body-frame application offset inside a ±5 cm cube. The table
shows the lowest one-axis capacity found among its eight corners, taking the
weaker hand. This corner sweep illustrates sensitivity but is not a proof over
every interior point.

| Robot | X (N) | Y (N) | Z (N) |
|---|---:|---:|---:|
| Agibot X2 | 101.75 | 242.00 | 81.37 |
| Unitree G1 | 26.31 | 80.65 | 35.99 |
| Unitree G1-23DOF | 86.78 | 364.93 | 50.79 |

## Effective Training Force

The play configurations request only world-frame downward force, capped at 40 N
per hand: `(Fx, Fy, Fz) = (0, 0, [-40, 0]) N`. At the end-effector center, all
robots therefore accept the full 40 N cap. Across the sampled offset corners:

- X2 remains capped at 40 N, below its 81.37 N minimum corner estimate.
- G1 can be estimator-limited to 35.99 N, below the configured 40 N cap.
- G1-23DOF remains capped at 40 N, below its 50.79 N minimum corner estimate.

These are instantaneous model-based limits. During playback, posture error and
controller effort make the estimator asymmetric and can reduce capacity further.
Contacts, modeling error, and force points not represented by the corner sweep
also affect the result.
