"""Correct recorded X2 height postures toward a symmetric knees-out crouch.

The recorded waist and grounded pelvis orientation are retained. Bounded IK
changes the legs as little as possible while keeping both feet flat, moving
the knees outward, and reducing their forward position relative to the pelvis.
Height uses the same sole-point datum as playback_height_postures.py.

Example:
  python scripts/generate_knees_out_postures.py \
    --output scripts/postures_x2_knees_out.py --plot /tmp/knees_out.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

from compute_height_postures import (
  LOWER_BODY_JOINTS, ROBOT_CONFIGS, get_foot_geom_info, get_joint_info,
  get_pelvis_body_id, load_model,
)
from playback_height_postures import (
  PostureFrame, apply_grounded_frame, load_python_frames,
)
from resample_height_postures import EXPORTED_JOINTS


DEFAULT_REFERENCE = Path(__file__).resolve().with_name("postures_x2_recorded.py")
MIRROR = np.array([1, -1, -1, 1, 1, -1])


@dataclass(frozen=True)
class CorrectionCfg:
  """Knee displacement targets at and below full_splay_height, in metres."""

  start_height: float = 0.64
  full_splay_height: float = 0.40
  knee_outward: float = 0.10
  knee_retraction: float = 0.03
  toe_out_deg: float = 30.0
  foot_outward: float = 0.05
  max_foot_width: float = 0.65
  max_pitch_change: float = 0.35
  smoothness: float = 0.2

  def __post_init__(self):
    values = np.array(list(vars(self).values()))
    if not np.isfinite(values).all():
      raise ValueError("correction settings must be finite")
    if self.start_height <= self.full_splay_height:
      raise ValueError("start_height must exceed full_splay_height")
    if min(self.knee_outward, self.knee_retraction, self.toe_out_deg,
           self.foot_outward, self.smoothness) < 0:
      raise ValueError("displacements, toe-out and smoothness must be nonnegative")
    if self.max_foot_width <= 0 or self.max_pitch_change <= 0:
      raise ValueError("maximum foot width and pitch change must be positive")

  def blend(self, height: float) -> float:
    """Smoothly increase splay as the commanded height falls."""
    t = np.clip((self.start_height - height)
                / (self.start_height - self.full_splay_height), 0.0, 1.0)
    return float(t * t * (3 - 2 * t))


class RecordedPostureCorrector:
  """Solve leg corrections with the recorded waist and pelvis pose fixed."""

  def __init__(self, cfg: CorrectionCfg):
    self.cfg = cfg
    robot = ROBOT_CONFIGS["x2"]
    self.model, self.data = load_model(robot["xml"], robot["assets_dir"])
    _, self.qids, self.lower, self.upper = get_joint_info(self.model)
    self.feet, left, right = get_foot_geom_info(self.model, 7, True)
    self.ankles = (left, right)
    self.pelvis = get_pelvis_body_id(self.model)
    self.knees = [self.model.body(f"{side}_knee_link").id
                  for side in ("left", "right")]
    self.hips = [self.model.body(f"{side}_hip_roll_link").id
                 for side in ("left", "right")]
    self.previous_delta = np.zeros(12)
    self.previous_height: float | None = None

  def geometry(self):
    """Return knee origins, foot support points and foot headings."""
    points = np.array([
      self.feet[side] @ self.data.xmat[body].reshape(3, 3).T
      + self.data.xpos[body]
      for side, body in zip(("left", "right"), self.ankles)
    ])
    rotations = self.data.xmat[list(self.ankles)].reshape(2, 3, 3)
    headings = np.arctan2(rotations[:, 1, 0], rotations[:, 0, 0])
    return self.data.xpos[self.knees].copy(), points, headings

  def reference(self, frame: PostureFrame):
    """Recover the recording's floating-base pose before modifying any legs."""
    addresses = {name: int(self.model.jnt_qposadr[self.model.joint(name).id])
                 for name in frame.positions}
    required = set(EXPORTED_JOINTS) - frame.positions.keys()
    if required:
      raise ValueError(f"reference is missing joints: {sorted(required)}")
    for name, value in frame.positions.items():
      joint = self.model.joint(name)
      if not np.isfinite(value) or (joint.limited[0]
          and not joint.range[0] <= value <= joint.range[1]):
        raise ValueError(f"reference joint {name} is nonfinite or outside limits")
    height, spread = apply_grounded_frame(
      self.model, self.data, frame, addresses, self.feet, *self.ankles, self.pelvis,
    )
    if abs(height - frame.height_label) > 0.0002 or spread > 0.0002:
      raise ValueError(f"reference at {frame.height_label:.3f} m is not grounded "
                       f"at its labelled height (height={height:.6f}, spread={spread:.6f})")
    return self.data.qpos.copy()

  def solve(self, frame: PostureFrame) -> tuple[dict, dict]:
    """Return a validated, rounded posture and geometry diagnostics."""
    base = self.reference(frame)
    ref_q = base[self.qids].copy()
    ref_knees, ref_points, ref_heading = self.geometry()
    ref_hips = self.data.xpos[self.hips].copy()
    ref_centers = ref_points.mean(axis=1)
    ref_com = self.data.subtree_com[self.pelvis, :2].copy()
    blend = self.cfg.blend(frame.height_label)
    sign = np.array([1.0, -1.0])
    target_heading = ref_heading + sign * np.deg2rad(self.cfg.toe_out_deg) * blend
    target_centers = ref_centers[:, :2].copy()
    target_centers[:, 1] += sign * self.cfg.foot_outward * blend
    lower, upper = self.lower.copy(), self.upper.copy()
    pitch_ids = [0, 3, 4, 6, 9, 10]
    lower[pitch_ids] = np.maximum(lower[pitch_ids], ref_q[pitch_ids] - self.cfg.max_pitch_change)
    upper[pitch_ids] = np.minimum(upper[pitch_ids], ref_q[pitch_ids] + self.cfg.max_pitch_change)

    def apply(q):
      self.data.qpos[:] = base
      self.data.qpos[self.qids] = q
      mujoco.mj_forward(self.model, self.data)
      return self.geometry()

    def ground(q):
      _, points, _ = apply(q)
      rotations = self.data.xmat[list(self.ankles)].reshape(2, 3, 3)
      return np.concatenate([points[:, :, 2].mean(axis=1),
                             0.1 * rotations[:, 2, :2].ravel()])

    def clearance(q):
      knees, points, _ = apply(q)
      centers = points.mean(axis=1)
      rotations = self.data.xmat[list(self.ankles)].reshape(2, 3, 3)
      return np.concatenate([
        sign * (knees[:, 1] - ref_knees[:, 1]) - self.cfg.knee_outward * blend,
        ref_knees[:, 0] - knees[:, 0] - self.cfg.knee_retraction * blend,
        sign * centers[:, 1] - 0.075,
        [self.cfg.max_foot_width - (centers[0, 1] - centers[1, 1])],
        rotations[:, 2, 2] - 0.99,
      ])

    def objective(q):
      _, points, headings = apply(q)
      delta = q - ref_q
      # Pitch changes are expensive; roll/yaw are free to create the new stance.
      scales = np.tile([0.20, 0.8, 0.8, 0.20, 0.15, 0.35], 2)
      cost = np.sum((delta / scales) ** 2)
      cost += np.sum(((points.mean(axis=1)[:, :2] - target_centers) / 0.04) ** 2)
      angle_error = np.arctan2(np.sin(headings - target_heading),
                               np.cos(headings - target_heading))
      cost += np.sum((angle_error / 0.20) ** 2)
      cost += np.sum(((delta[:6] - MIRROR * delta[6:]) / 0.01) ** 2)
      cost += np.sum(((self.data.subtree_com[self.pelvis, :2] - ref_com) / 0.03) ** 2)
      if self.previous_height is not None:
        step_scale = max(abs(self.previous_height - frame.height_label) / 0.02, 1.0)
        cost += self.cfg.smoothness * np.sum(
          ((delta - self.previous_delta) / (0.10 * step_scale)) ** 2)
      # Discourage self-contact during optimization; reject remaining penetration below.
      cost += sum((min(float(c.dist) + 0.001, 0.0) / 0.001) ** 2
                  for c in self.data.contact[:self.data.ncon])
      return float(cost)

    if blend == 0:
      q = ref_q
      message = "unchanged recorded posture"
    else:
      starts = [np.clip(ref_q + self.previous_delta, lower, upper), ref_q]
      candidates = []
      for start in starts:
        result = minimize(objective, start, method="SLSQP", bounds=list(zip(lower, upper)),
                          constraints=[{"type": "eq", "fun": ground},
                                       {"type": "ineq", "fun": clearance}],
                          options={"maxiter": 400, "ftol": 1e-10})
        if (result.success and np.max(np.abs(ground(result.x))) < 1e-6
            and np.min(clearance(result.x)) >= -1e-6):
          candidates.append(result)
      if not candidates:
        raise RuntimeError(f"No feasible recorded-posture correction at "
                           f"{frame.height_label:.3f} m: {result.message}; "
                           f"ground error={np.max(np.abs(ground(result.x))):.6g}, "
                           f"clearance residual={np.min(clearance(result.x)):.6g}")
      result = min(candidates, key=lambda item: item.fun)
      q, message = result.x, str(result.message)

    posture = {name: frame.positions[name] for name in EXPORTED_JOINTS}
    posture.update({name: round(float(value), 4) for name, value in zip(LOWER_BODY_JOINTS, q)})
    rounded = np.array([posture[name] for name in LOWER_BODY_JOINTS])
    knees, points, headings = apply(rounded)
    hips = self.data.xpos[self.hips].copy()
    hull = ConvexHull(points[:, :, :2].reshape(-1, 2))
    com = self.data.subtree_com[self.pelvis, :2]
    com_margin = -np.max(hull.equations[:, :2] @ com + hull.equations[:, 2])
    penetration = max([0.0, *[-float(c.dist) for c in self.data.contact[:self.data.ncon]]])
    if (np.max(np.abs(points[:, :, 2])) > 0.0002
        or np.min(clearance(rounded)) < -0.0002
        or np.any(rounded < lower - 1e-10) or np.any(rounded > upper + 1e-10)
        or penetration > 0.001 or com_margin < 0):
      raise RuntimeError(f"Rounded posture at {frame.height_label:.3f} m failed validation: "
                         f"foot_z={np.max(np.abs(points[:, :, 2])):.6f}, "
                         f"penetration={penetration:.6f}, COM margin={com_margin:.6f}")
    corrected = PostureFrame(frame.height_label, frame.odometry_z, posture)
    addresses = {name: int(self.model.jnt_qposadr[self.model.joint(name).id]) for name in posture}
    achieved, spread = apply_grounded_frame(
      self.model, self.data, corrected, addresses, self.feet, *self.ankles, self.pelvis)
    orientation_change = 2 * np.arccos(np.clip(abs(np.dot(base[3:7], self.data.qpos[3:7])), 0, 1))
    if abs(achieved - frame.height_label) > 0.0002 or np.rad2deg(orientation_change) > 0.05:
      raise RuntimeError("rounded posture changed recorded height or pelvis orientation")
    self.previous_delta = rounded - ref_q
    self.previous_height = frame.height_label
    metrics = {
      "height": frame.height_label, "blend": blend, "achieved_height": achieved,
      "foot_height_spread": spread, "pelvis_rotation_change_deg": float(np.rad2deg(orientation_change)),
      "knee_outward_m": (sign * (knees[:, 1] - ref_knees[:, 1])).tolist(),
      "knee_retraction_m": (ref_knees[:, 0] - knees[:, 0]).tolist(),
      "toe_out_deg": (sign * np.rad2deg(headings)).tolist(),
      "foot_width_m": float(points[0, :, 1].mean() - points[1, :, 1].mean()),
      "max_leg_pitch_change_rad": float(np.max(np.abs(self.previous_delta[pitch_ids]))),
      "max_self_penetration_m": penetration, "com_support_margin_m": float(com_margin),
      "reference_knees": ref_knees.tolist(), "corrected_knees": knees.tolist(),
      "reference_hips": ref_hips.tolist(), "corrected_hips": hips.tolist(),
      "reference_feet": ref_points.tolist(), "corrected_feet": points.tolist(),
      "solver_message": message,
    }
    return posture, metrics


def generate(reference: Path, cfg: CorrectionCfg) -> tuple[dict, list[dict]]:
  """Correct a table from standing downwards, using adjacent corrections as seeds."""
  solver = RecordedPostureCorrector(cfg)
  frames, _ = load_python_frames(reference, solver.model)
  table, diagnostics = {}, []
  for frame in reversed(frames):
    posture, metrics = solver.solve(frame)
    table[frame.height_label] = posture
    diagnostics.append(metrics)
    print(f"Height {frame.height_label:.2f}: outward={min(metrics['knee_outward_m']):.3f} m, "
          f"retraction={min(metrics['knee_retraction_m']):.3f} m, "
          f"pitch change={metrics['max_leg_pitch_change_rad']:.3f} rad", flush=True)
  return dict(sorted(table.items())), list(reversed(diagnostics))


def format_corrected_postures(table: dict, reference: Path) -> str:
  """Export the validated values, retaining recorded waist precision exactly."""
  lines = [f'"""Knees-out corrections of recorded X2 postures from {reference.name}."""',
           "", "", "HEIGHT_POSTURES = {"]
  for height, posture in table.items():
    lines.append(f"    {height!r}: {{")
    for name in EXPORTED_JOINTS:
      lines.append(f'        "{name}": {posture[name]!r},')
    lines.append("    },")
  lines.extend(["}", ""])
  return "\n".join(lines)


def plot_comparison(diagnostics: list[dict], path: Path) -> None:
  """Plot joint origins and support outlines at representative recorded heights."""
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  selected = [diagnostics[i] for i in np.unique(np.linspace(0, len(diagnostics) - 1, 3).astype(int))]
  fig, axes = plt.subplots(2, len(selected), figsize=(4 * len(selected), 7), squeeze=False)
  for col, row in enumerate(selected):
    for ax, coordinates, view in zip(axes[:, col], [(1, 0), (0, 2)], ["Top", "Side"]):
      for prefix, color, label in [("reference", "#888888", "Recorded"),
                                    ("corrected", "#1479bd", "Knees out")]:
        knees = np.array(row[f"{prefix}_knees"])
        hips = np.array(row[f"{prefix}_hips"])
        feet = np.array(row[f"{prefix}_feet"])
        for side in range(2):
          chain = np.array([hips[side], knees[side], feet[side].mean(axis=0)])
          ax.plot(chain[:, coordinates[0]], chain[:, coordinates[1]], "o-", color=color,
                  label=label if side == 0 else None)
          if view == "Top":
            outline = feet[side, ConvexHull(feet[side, :, :2]).vertices]
            outline = np.concatenate([outline, outline[:1]])
            ax.plot(outline[:, 1], outline[:, 0], color=color)
      ax.set_title(f"{view}: pelvis {row['height']:.2f} m")
      ax.set_xlabel("Left (+Y), m" if view == "Top" else "Forward (+X), m")
      ax.set_ylabel("Forward (+X), m" if view == "Top" else "Height (+Z), m")
      if view == "Side":
        ax.set_xlim(-0.35, 0.25)
        ax.set_ylim(-0.02, 0.68)
      ax.set_aspect("equal")
      ax.grid(alpha=0.2)
      ax.legend()
  fig.suptitle("Recorded X2 posture and constrained knees-out correction (kinematics)")
  fig.tight_layout()
  path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(path, dpi=160)
  plt.close(fig)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--report", type=Path, help="JSON diagnostics (default: output.with_suffix('.json'))")
  parser.add_argument("--plot", type=Path)
  parser.add_argument("--overwrite", action="store_true")
  defaults = CorrectionCfg()
  help_text = {
    "start_height": "Height above which the recorded posture is unchanged (m)",
    "full_splay_height": "Height at which the full correction is requested (m)",
    "knee_outward": "Minimum outward displacement of each knee at full splay (m)",
    "knee_retraction": "Minimum backward displacement of each knee at full splay (m)",
    "toe_out_deg": "Preferred additional toe-out per foot at full splay (degrees)",
    "foot_outward": "Preferred outward displacement per foot at full splay (m)",
    "max_foot_width": "Maximum lateral separation of foot support centroids (m)",
    "max_pitch_change": "Maximum change from each recorded leg-pitch joint (rad)",
    "smoothness": "Weight on changes to the preceding height's correction",
  }
  for name, value in vars(defaults).items():
    parser.add_argument("--" + name.replace("_", "-"), type=float, default=value,
                        help=help_text[name])
  args = parser.parse_args()
  cfg = CorrectionCfg(**{name: getattr(args, name) for name in vars(defaults)})
  report = args.report or args.output.with_suffix(".json")
  paths = [args.output, report, *([args.plot] if args.plot else [])]
  if len({p.resolve() for p in paths}) != len(paths) or args.reference.resolve() in {p.resolve() for p in paths}:
    parser.error("reference, output, report and plot must be different paths")
  for path in paths:
    if path.exists() and not args.overwrite:
      parser.error(f"{path} already exists; use --overwrite")
  table, diagnostics = generate(args.reference, cfg)
  text = format_corrected_postures(table, args.reference)
  for path in paths:
    path.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(text)
  report.write_text(json.dumps({"reference": str(args.reference.resolve()),
                                "settings": vars(cfg), "postures": diagnostics}, indent=2) + "\n")
  if args.plot:
    plot_comparison(diagnostics, args.plot)
  print(f"Saved {len(table)} validated postures to {args.output}; diagnostics: {report}")


if __name__ == "__main__":
  main()
