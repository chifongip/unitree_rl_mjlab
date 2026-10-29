"""Generate symmetric X2 squat references with a constant ground-fixed stance.

Standing uses the corrected table; walking retains the recorded reference.
Height retains playback's capsule-centerline sole datum.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from compute_height_postures import (
  LOWER_BODY_JOINTS, ROBOT_CONFIGS, get_foot_geom_info, get_joint_info,
  get_pelvis_body_id, load_model,
)
from playback_height_postures import (
  PostureFrame, apply_grounded_frame, load_python_frames, model_joint_qpos_addresses,
)
from resample_height_postures import EXPORTED_JOINTS

DEFAULT_REFERENCE = Path(__file__).resolve().with_name("postures_x2_recorded.py")
MIRROR = np.array([1, -1, -1, 1, 1, -1])


@dataclass(frozen=True)
class CorrectionCfg:
  """Fixed stance and soft knee preferences; distances are metres."""

  start_height: float = 0.64
  full_splay_height: float = 0.40
  knee_outward: float = 0.10
  knee_retraction: float = 0.03
  stance_width: float = 0.374
  toe_out_deg: float = 30.0
  max_pelvis_shift: float = 0.10
  smoothness: float = 0.2

  def __post_init__(self):
    if not np.isfinite(list(vars(self).values())).all():
      raise ValueError("correction settings must be finite")
    if self.start_height <= self.full_splay_height:
      raise ValueError("start_height must exceed full_splay_height")
    if min(self.knee_outward, self.knee_retraction, self.toe_out_deg,
           self.smoothness, self.max_pelvis_shift) < 0 or self.stance_width <= 0:
      raise ValueError("preferences must be nonnegative and stance_width positive")

  def blend(self, height: float) -> float:
    t = np.clip((self.start_height - height)
                / (self.start_height - self.full_splay_height), 0.0, 1.0)
    return float(t * t * (3 - 2 * t))


@dataclass
class Solution:
  frame: PostureFrame
  q: np.ndarray
  dx: float
  metrics: dict


class RecordedPostureCorrector:
  """Bounded symmetric IK with independent bilateral export validation."""

  def __init__(self, cfg: CorrectionCfg):
    self.cfg = cfg
    robot = ROBOT_CONFIGS["x2"]
    self.model, self.data = load_model(robot["xml"], robot["assets_dir"])
    _, self.qids, self.lower, self.upper = get_joint_info(self.model)
    self.feet, left, right = get_foot_geom_info(self.model, 7, True)
    self.ankles = (left, right)
    self.pelvis = get_pelvis_body_id(self.model)
    self.knees = [self.model.body(f"{side}_knee_link").id for side in ("left", "right")]
    self.hips = [self.model.body(f"{side}_hip_roll_link").id for side in ("left", "right")]
    self.addresses = model_joint_qpos_addresses(self.model)
    self.lo = np.maximum(self.lower[:6], np.where(MIRROR > 0, self.lower[6:], -self.upper[6:]))
    self.hi = np.minimum(self.upper[:6], np.where(MIRROR > 0, self.upper[6:], -self.lower[6:]))

  def geometry(self):
    points = np.array([
      self.feet[side] @ self.data.xmat[body].reshape(3, 3).T + self.data.xpos[body]
      for side, body in zip(("left", "right"), self.ankles)
    ])
    rotations = self.data.xmat[list(self.ankles)].reshape(2, 3, 3)
    headings = np.arctan2(rotations[:, 1, 0], rotations[:, 0, 0])
    return self.data.xpos[self.knees].copy(), points, headings

  def reference(self, frame: PostureFrame):
    missing = set(EXPORTED_JOINTS) - frame.positions.keys()
    if missing:
      raise ValueError(f"reference is missing joints: {sorted(missing)}")
    for name, value in frame.positions.items():
      joint = self.model.joint(name)
      if not np.isfinite(value) or (joint.limited[0] and not joint.range[0] <= value <= joint.range[1]):
        raise ValueError(f"reference joint {name} is nonfinite or outside limits")
    height, spread = apply_grounded_frame(
      self.model, self.data, frame, self.addresses, self.feet, *self.ankles, self.pelvis)
    if abs(height - frame.height_label) > 0.0002 or spread > 0.0002:
      raise ValueError(f"reference at {frame.height_label:.3f} m is not grounded at its labelled height")
    q = self.data.qpos[self.qids]
    angles = Rotation.from_quat(self.data.qpos[3:7][[1, 2, 3, 0]]).as_euler("xyz")
    if (not np.allclose(q[:6], MIRROR * q[6:], atol=1e-8, rtol=0)
        or abs(angles[0]) > 1e-8 or abs(angles[2]) > 1e-8
        or abs(frame.positions["waist_yaw_joint"]) > 1e-8
        or abs(frame.positions["waist_roll_joint"]) > 1e-8):
      raise ValueError("reference must be sagittally symmetric")
    return self.data.qpos.copy()

  def prepare(self, frames: list[PostureFrame]):
    self.frames = sorted(frames, key=lambda f: f.height_label)
    self.heights = np.array([f.height_label for f in self.frames])
    self.bases = [self.reference(f) for f in self.frames]
    self.reference(self.frames[-1])
    self.centers = self.geometry()[1].mean(axis=1)
    self.centers[:, 1] = [self.cfg.stance_width / 2, -self.cfg.stance_width / 2]
    self.centers[:, 2] = 0
    angles = np.deg2rad([self.cfg.toe_out_deg, -self.cfg.toe_out_deg])
    self.rotations = np.array([
      [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
      for a in angles
    ])
    self.targets = np.array([
      (self.feet[side] - self.feet[side].mean(axis=0)) @ r.T + c
      for side, r, c in zip(("left", "right"), self.rotations, self.centers)
    ])

  def reference_at(self, height: float):
    if height in self.heights:
      i = int(np.flatnonzero(self.heights == height)[0])
      return self.bases[i].copy(), dict(self.frames[i].positions)
    raise ValueError(f"height {height} is not in the recorded reference table")

  def apply(self, q, base, dx):
    self.data.qpos[:] = base
    self.data.qpos[self.qids] = np.r_[q, MIRROR * q]
    self.data.qpos[0] += dx
    mujoco.mj_forward(self.model, self.data)

  def measure(self, frame: PostureFrame) -> dict:
    base, _ = self.reference_at(frame.height_label)
    self.data.qpos[:] = base
    mujoco.mj_forward(self.model, self.data)
    ref_knees, ref_points, _ = self.geometry()
    ref_hips = self.data.xpos[self.hips].copy()
    achieved, spread = apply_grounded_frame(
      self.model, self.data, frame, self.addresses, self.feet, *self.ankles, self.pelvis)
    knees, points, headings = self.geometry()
    rotations = self.data.xmat[list(self.ankles)].reshape(2, 3, 3)
    angle_error = Rotation.from_matrix(
      self.rotations.transpose(0, 2, 1) @ rotations).magnitude()
    hull = ConvexHull(points[:, :, :2].reshape(-1, 2))
    margin = -np.max(hull.equations[:, :2] @ self.data.subtree_com[self.pelvis, :2]
                     + hull.equations[:, 2])
    q = self.data.qpos[self.qids]
    shift = self.data.qpos[0] - base[0]
    return {
      "height": frame.height_label, "achieved_height": achieved,
      "foot_height_spread": spread,
      "foot_position_error_m": float(np.max(np.linalg.norm(points - self.targets, axis=2))),
      "foot_rotation_error_deg": float(np.rad2deg(np.max(angle_error))),
      "pelvis_rotation_change_deg": float(np.rad2deg(2 * np.arccos(np.clip(
        abs(np.dot(base[3:7], self.data.qpos[3:7])), 0, 1)))),
      "pelvis_fore_aft_shift_m": float(shift),
      "joint_limit_violation_rad": max(0., float(np.max(self.lower - q)), float(np.max(q - self.upper))),
      "max_self_penetration_m": max([0., *[-float(c.dist) for c in self.data.contact[:self.data.ncon]]]),
      "com_support_margin_m": float(margin),
      "knee_outward_m": (np.array([1, -1]) * (knees[:, 1] - ref_knees[:, 1])).tolist(),
      "knee_retraction_m": (ref_knees[:, 0] - (knees[:, 0] - shift)).tolist(),
      "toe_out_deg": (np.array([1, -1]) * np.rad2deg(headings)).tolist(),
      "foot_width_m": float(points[0, :, 1].mean() - points[1, :, 1].mean()),
      "reference_knees": ref_knees.tolist(), "corrected_knees": knees.tolist(),
      "reference_hips": ref_hips.tolist(), "corrected_hips": self.data.xpos[self.hips].tolist(),
      "reference_feet": ref_points.tolist(), "corrected_feet": points.tolist(),
      "target_feet": self.targets.tolist(),
    }

  @staticmethod
  def acceptable(m: dict, internal: bool = True) -> bool:
    distance, angle = (0.00016, 0.04) if internal else (0.0002, 0.05)
    return (
      m["foot_position_error_m"] <= distance and m["foot_rotation_error_deg"] <= angle
      and abs(m["achieved_height"] - m["height"]) <= distance
      and m["pelvis_rotation_change_deg"] <= angle
      and m["joint_limit_violation_rad"] <= 1e-10
      and m["max_self_penetration_m"] <= 0.001 and m["com_support_margin_m"] >= 0
    )

  def pelvis_offsets(self, preferred: float):
    """Try coarse offsets and exact bounds, then refine gaps to at most 0.1 mm."""
    limit = self.cfg.max_pelvis_shift
    tried = set()
    coarse = [preferred, 0.]
    coarse += [sign * round(float(x), 12)
               for x in np.arange(0.0025, limit + 1e-9, 0.0025) for sign in (1, -1)]
    coarse += [limit, -limit]
    for dx in coarse:
      if abs(dx) <= limit and dx not in tried:
        tried.add(dx)
        yield dx
    while True:
      points = sorted(tried)
      midpoints = [(a + b) / 2 for a, b in zip(points[:-1], points[1:])
                   if b - a > 0.0001 + 1e-12]
      if not midpoints:
        return
      for dx in sorted(midpoints, key=lambda x: (abs(x), -x)):
        tried.add(dx)
        yield dx

  def solve(self, height: float, seeds: list[Solution]) -> Solution:
    base, positions = self.reference_at(height)
    preferred = float(np.mean([s.dx for s in seeds])) if seeds else 0.
    starts = [s.q for s in seeds] + [base[self.qids][:6]]
    last_metrics = {}
    for dx in self.pelvis_offsets(preferred):
      candidates = []
      for start in starts:
        def residual(q):
          self.apply(q, base, dx)
          _, points, _ = self.geometry()
          rotation = self.data.xmat[self.ankles[0]].reshape(3, 3)
          return np.r_[points[0].mean(axis=0) - self.centers[0],
                       0.1 * Rotation.from_matrix(self.rotations[0].T @ rotation).as_rotvec()]
        fit = least_squares(
          residual, np.clip(start, self.lo + 1e-10, self.hi - 1e-10),
          bounds=(self.lo, self.hi), max_nfev=120,
          ftol=1e-12, xtol=1e-12, gtol=1e-12)
        posture = dict(positions)
        posture.update({n: round(float(v), 6)
                        for n, v in zip(LOWER_BODY_JOINTS, np.r_[fit.x, MIRROR * fit.x])})
        frame = PostureFrame(height, height, posture)
        m = self.measure(frame)
        m["solver_message"] = str(fit.message)
        m["solver_success"] = bool(fit.success)
        last_metrics = m
        if self.acceptable(m):
          candidates.append(Solution(frame, fit.x.copy(), dx, m))
      if candidates:
        def preference(candidate):
          q = candidate.q
          cost = np.sum(((q - base[self.qids][:6]) / np.array([.2, .8, .8, .2, .15, .35])) ** 2)
          blend = self.cfg.blend(height)
          for key, target in (("knee_outward_m", self.cfg.knee_outward),
                              ("knee_retraction_m", self.cfg.knee_retraction)):
            cost += np.sum((np.minimum(np.array(candidate.metrics[key]) - target * blend, 0) / .04) ** 2)
          cost += self.cfg.smoothness * sum(np.sum(((q - s.q) / .1) ** 2) for s in seeds)
          return float(cost)
        return min(candidates, key=preference)
    raise RuntimeError(f"No feasible fixed-foot posture found at {height:.6f} m "
                       f"after offset search with gaps <= 0.1 mm; last candidate: {last_metrics}")


def generate(reference: Path, cfg: CorrectionCfg) -> tuple[dict, list[dict]]:
  """Solve only recorded heights and validate serialized and float32 poses."""
  solver = RecordedPostureCorrector(cfg)
  frames, _ = load_python_frames(reference, solver.model)
  solver.prepare(frames)
  solutions = {}
  previous = []
  for frame in reversed(frames):
    solution = solver.solve(frame.height_label, previous)
    solutions[frame.height_label] = solution
    previous = [solution]
    print(f"Height {frame.height_label:.3f}: fixed feet, pelvis shift={solution.dx:.4f} m", flush=True)
  keys = sorted(solutions)
  # Validate the exact Python representation that downstream loaders execute.
  namespace = {}
  exec(format_corrected_postures(
    {h: solutions[h].frame.positions for h in keys}, reference), namespace)
  for h in keys:
    solutions[h].frame = PostureFrame(h, h, namespace["HEIGHT_POSTURES"][h])
  worst = {"checks": len(keys), "foot_position_error_m": 0., "foot_rotation_error_deg": 0.,
           "height_error_m": 0., "minimum_com_support_margin_m": float("inf")}
  diagnostics = []
  for h in keys:
    solution = solutions[h]
    # Rewards store the nearest table row in float32; no intermediate poses.
    positions = {n: float(np.float32(v)) for n, v in solution.frame.positions.items()}
    m = solver.measure(PostureFrame(h, h, positions))
    if not solver.acceptable(m, internal=False):
      raise RuntimeError(f"Float32 pose validation failed: {m}")
    for name in ("foot_position_error_m", "foot_rotation_error_deg"):
      worst[name] = max(worst[name], m[name])
    worst["height_error_m"] = max(worst["height_error_m"], abs(m["achieved_height"] - h))
    worst["minimum_com_support_margin_m"] = min(worst["minimum_com_support_margin_m"], m["com_support_margin_m"])
    # Check the executed export as well as its float32 representation.
    metrics = solver.measure(solution.frame)
    if not solver.acceptable(metrics):
      raise RuntimeError(f"Serialized pose validation failed: {metrics}")
    metrics.update(solver_message=solution.metrics["solver_message"],
                   solver_success=solution.metrics["solver_success"], float32_validation=worst)
    diagnostics.append(metrics)
  return {h: solutions[h].frame.positions for h in keys}, diagnostics


def format_corrected_postures(table: dict, reference: Path) -> str:
  """Export the validated values, retaining recorded waist precision exactly."""
  lines = [f'"""Fixed-foot squat corrections of recorded X2 postures from {reference.name}."""',
           "", "", "HEIGHT_POSTURES = {"]
  for height, posture in table.items():
    lines.append(f"    {height!r}: {{")
    for name in [*EXPORTED_JOINTS, *sorted(set(posture) - set(EXPORTED_JOINTS))]:
      lines.append(f'        "{name}": {float(posture[name])!r},')
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
                                    ("corrected", "#1479bd", "Fixed-foot squat")]:
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
  fig.suptitle("Recorded X2 posture and fixed-foot squat (kinematics)")
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
    "start_height": "Height above which soft knee preferences fade to zero (m)",
    "full_splay_height": "Height at which the full correction is requested (m)",
    "knee_outward": "Preferred outward displacement of each knee at full splay (m)",
    "knee_retraction": "Preferred backward displacement of each knee at full splay (m)",
    "toe_out_deg": "Constant outward heading per foot (degrees)",
    "stance_width": "Fixed separation of sole centroids (m)",
    "max_pelvis_shift": "Maximum fore/aft change from recorded pelvis position (m)",
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
