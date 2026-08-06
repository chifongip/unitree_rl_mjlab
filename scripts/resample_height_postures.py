"""Resample recorded X2 postures onto a uniform MuJoCo-height grid.

The source CSV comes from ``record_height_postures.py``.  Before resampling,
left/right leg pitch joints are averaged and all leg roll/yaw joints are set
to zero.  Waist roll is also set to zero to preserve sagittal symmetry.  Joint
positions are clamped to the X2 MJCF hard limits before heights are computed
with the free-base, flat-feet solve in
``playback_height_postures.py``.  The exported ``HEIGHT_POSTURES`` table
contains the 12 leg joints and 3 waist joints.

Usage:
    python scripts/resample_height_postures.py height_postures.csv \
        --output scripts/postures_x2_recorded.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco
from scipy.optimize import brentq

from compute_height_postures import (
    LOWER_BODY_JOINTS,
    ROBOT_CONFIGS,
    get_foot_geom_info,
    get_pelvis_body_id,
    load_model,
)
from playback_height_postures import (
    PostureFrame,
    apply_grounded_frame,
    load_csv_frames,
)


WAIST_JOINTS = [
    "waist_yaw_joint",
    "waist_pitch_joint",
    "waist_roll_joint",
]
EXPORTED_JOINTS = [*LOWER_BODY_JOINTS, *WAIST_JOINTS]
SYMMETRIC_LEG_PAIRS = [
    ("left_hip_pitch_joint", "right_hip_pitch_joint"),
    ("left_knee_joint", "right_knee_joint"),
    ("left_ankle_pitch_joint", "right_ankle_pitch_joint"),
]
ZERO_LEG_JOINTS = [
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in ("hip_roll", "hip_yaw", "ankle_roll")
]


def grounded_height(
    model,
    data,
    frame,
    qpos_addresses,
    foot_geoms,
    left_ankle_body,
    right_ankle_body,
    pelvis_body_id,
):
    """Return the pelvis height after solving the floating-base pose."""
    height, _ = apply_grounded_frame(
        model,
        data,
        frame,
        qpos_addresses,
        foot_geoms,
        left_ankle_body,
        right_ankle_body,
        pelvis_body_id,
    )
    return height


def interpolate_frame(low: PostureFrame, high: PostureFrame, alpha: float):
    """Linearly interpolate all recorded joint positions."""
    positions = {
        name: low.positions[name]
        + alpha * (high.positions[name] - low.positions[name])
        for name in low.positions
    }
    return PostureFrame(
        height_label=low.height_label
        + alpha * (high.height_label - low.height_label),
        odometry_z=low.odometry_z
        + alpha * (high.odometry_z - low.odometry_z),
        positions=positions,
    )


def symmetrize_leg_posture(frame: PostureFrame) -> PostureFrame:
    """Make the posture invariant under sagittal left-right reflection."""
    positions = dict(frame.positions)
    for left_name, right_name in SYMMETRIC_LEG_PAIRS:
        mean_position = 0.5 * (
            positions[left_name] + positions[right_name]
        )
        positions[left_name] = mean_position
        positions[right_name] = mean_position
    for name in ZERO_LEG_JOINTS:
        positions[name] = 0.0
    positions["waist_roll_joint"] = 0.0
    return PostureFrame(
        height_label=frame.height_label,
        odometry_z=frame.odometry_z,
        positions=positions,
    )


def clamp_joint_limits(
    frame: PostureFrame,
    model: mujoco.MjModel,
) -> PostureFrame:
    """Clamp all recorded scalar joints to their MuJoCo hard limits."""
    positions = dict(frame.positions)
    for name, position in positions.items():
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if joint_id < 0:
            raise ValueError(f"MuJoCo model has no joint {name!r}")
        if model.jnt_limited[joint_id]:
            lower, upper = model.jnt_range[joint_id]
            positions[name] = max(float(lower), min(float(upper), position))
    return PostureFrame(
        height_label=frame.height_label,
        odometry_z=frame.odometry_z,
        positions=positions,
    )


def target_heights(min_height: float, max_height: float, step: float):
    """Build an inclusive grid using integer indices to avoid float drift."""
    start_index = math.ceil((min_height - 1e-12) / step)
    end_index = math.floor((max_height + 1e-12) / step)
    return [round(index * step, 10) for index in range(start_index, end_index + 1)]


def find_source_pair(
    frames_with_heights: list[tuple[float, PostureFrame]], target: float
):
    """Find two neighboring source frames, allowing endpoint extrapolation."""
    if target <= frames_with_heights[0][0]:
        return frames_with_heights[0], frames_with_heights[1]
    if target >= frames_with_heights[-1][0]:
        return frames_with_heights[-2], frames_with_heights[-1]
    for low, high in zip(frames_with_heights, frames_with_heights[1:]):
        if low[0] <= target <= high[0]:
            return low, high
    raise RuntimeError(f"could not bracket target height {target}")


def solve_target_posture(
    target: float,
    low_item: tuple[float, PostureFrame],
    high_item: tuple[float, PostureFrame],
    height_fn,
    frame_transform,
):
    """Solve interpolation alpha so MuJoCo height matches the target."""
    low_height, low_frame = low_item
    high_height, high_frame = high_item
    if high_height <= low_height:
        raise ValueError("source heights must be strictly increasing")

    linear_alpha = (target - low_height) / (high_height - low_height)

    def candidate(alpha):
        return frame_transform(interpolate_frame(low_frame, high_frame, alpha))

    def residual(alpha):
        return height_fn(candidate(alpha)) - target

    if 0.0 <= linear_alpha <= 1.0:
        alpha = brentq(residual, 0.0, 1.0, xtol=1e-10)
    elif linear_alpha > 1.0:
        upper = max(1.25, linear_alpha * 1.5)
        while residual(upper) < 0.0 and upper < 4.0:
            upper *= 1.5
        alpha = brentq(residual, 1.0, upper, xtol=1e-10)
    else:
        lower = min(-0.25, linear_alpha * 1.5)
        while residual(lower) > 0.0 and lower > -3.0:
            lower *= 1.5
        alpha = brentq(residual, lower, 0.0, xtol=1e-10)

    return candidate(alpha), alpha


def resample(
    model,
    data,
    frames,
    qpos_addresses,
    foot_geoms,
    left_ankle_body,
    right_ankle_body,
    pelvis_body_id,
    targets,
):
    """Resample recorded frames at the requested MuJoCo pelvis heights."""
    def height_fn(frame):
        return grounded_height(
            model,
            data,
            frame,
            qpos_addresses,
            foot_geoms,
            left_ankle_body,
            right_ankle_body,
            pelvis_body_id,
        )

    def prepare_frame(frame):
        return clamp_joint_limits(symmetrize_leg_posture(frame), model)

    prepared_frames = [prepare_frame(frame) for frame in frames]
    source = sorted((height_fn(frame), frame) for frame in prepared_frames)
    for (low_height, _), (high_height, _) in zip(source, source[1:]):
        if high_height <= low_height:
            raise ValueError("recorded MuJoCo heights are not strictly monotonic")

    results = {}
    solve_metadata = {}
    for target in targets:
        low_item, high_item = find_source_pair(source, target)
        frame, alpha = solve_target_posture(
            target, low_item, high_item, height_fn, prepare_frame
        )
        results[target] = {
            name: frame.positions[name] for name in EXPORTED_JOINTS
        }
        solve_metadata[target] = {
            "achieved_height": height_fn(frame),
            "alpha": alpha,
            "source_range": (low_item[0], high_item[0]),
        }
    return results, solve_metadata, (source[0][0], source[-1][0])


def format_postures(results: dict[float, dict[str, float]]) -> str:
    """Format a table compatible with scripts/postures_x2.py."""
    lines = [
        '"""Real-robot X2 postures resampled by MuJoCo pelvis height."""',
        "",
        "",
        "HEIGHT_POSTURES = {",
    ]
    for height, posture in results.items():
        lines.append(f"    {height}: {{")
        for name in EXPORTED_JOINTS:
            lines.append(f'        "{name}": {posture[name]:.4f},')
        lines.append("    },")
    lines.extend(["}", ""])
    return "\n".join(lines)


def joint_limit_violations(model, results):
    """Return exported target heights whose positions exceed MJCF limits."""
    violations = {}
    for name in EXPORTED_JOINTS:
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if not model.jnt_limited[joint_id]:
            continue
        lower, upper = model.jnt_range[joint_id]
        invalid_heights = [
            height
            for height, posture in results.items()
            if not lower <= posture[name] <= upper
        ]
        if invalid_heights:
            violations[name] = (float(lower), float(upper), invalid_heights)
    return violations


def validate_rounded_results(
    source_text,
    model,
    data,
    frames,
    qpos_addresses,
    foot_geoms,
    left_ankle_body,
    right_ankle_body,
    pelvis_body_id,
):
    """Measure height error after the exported four-decimal rounding."""
    namespace = {}
    exec(source_text, namespace)
    template = frames[0]
    errors = []
    for target, posture in namespace["HEIGHT_POSTURES"].items():
        positions = dict(template.positions)
        positions.update(posture)
        frame = PostureFrame(target, target, positions)
        achieved = grounded_height(
            model,
            data,
            frame,
            qpos_addresses,
            foot_geoms,
            left_ankle_body,
            right_ankle_body,
            pelvis_body_id,
        )
        errors.append(abs(achieved - target))
    return max(errors)


def finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be finite")
    return parsed


def positive_float(value: str) -> float:
    parsed = finite_float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", type=Path, help="Recorded posture CSV")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination Python posture table",
    )
    parser.add_argument("--min-height", type=finite_float, default=0.22)
    parser.add_argument("--max-height", type=finite_float, default=0.64)
    parser.add_argument("--height-step", type=positive_float, default=0.02)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it exists",
    )
    args = parser.parse_args()
    if args.max_height < args.min_height:
        parser.error("--max-height must be greater than or equal to --min-height")
    return args


def main():
    args = parse_args()
    cfg = ROBOT_CONFIGS["x2"]
    model, data = load_model(cfg["xml"], cfg["assets_dir"])
    frames, qpos_addresses = load_csv_frames(args.csv_file, model)
    missing_joints = [name for name in EXPORTED_JOINTS if name not in qpos_addresses]
    if missing_joints:
        raise ValueError(f"MuJoCo model is missing exported joints: {missing_joints}")

    foot_geoms, left_ankle_body, right_ankle_body = get_foot_geom_info(
        model, cfg["foot_geom_count"], cfg["is_capsule"]
    )
    pelvis_body_id = get_pelvis_body_id(model)
    targets = target_heights(
        args.min_height, args.max_height, args.height_step
    )
    results, metadata, source_range = resample(
        model,
        data,
        frames,
        qpos_addresses,
        foot_geoms,
        left_ankle_body,
        right_ankle_body,
        pelvis_body_id,
        targets,
    )
    output_text = format_postures(results)
    max_rounded_error = validate_rounded_results(
        output_text,
        model,
        data,
        frames,
        qpos_addresses,
        foot_geoms,
        left_ankle_body,
        right_ankle_body,
        pelvis_body_id,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else "x"
    with args.output.open(mode, encoding="utf-8") as output_file:
        output_file.write(output_text)

    extrapolated = [
        target
        for target, values in metadata.items()
        if not 0.0 <= values["alpha"] <= 1.0
    ]
    print(
        f"Saved {len(results)} postures with {len(EXPORTED_JOINTS)} joints "
        f"to {args.output}"
    )
    print(
        f"Source MuJoCo height range: {source_range[0]:.6f}-"
        f"{source_range[1]:.6f} m"
    )
    print(f"Extrapolated targets: {extrapolated or 'none'}")
    print(f"Maximum height error after rounding: {max_rounded_error:.6f} m")
    violations = joint_limit_violations(model, results)
    for name, (lower, upper, heights) in violations.items():
        print(
            f"WARNING: {name} exceeds MJCF range [{lower:.4f}, {upper:.4f}] "
            f"at target heights {heights}; recorded values were preserved"
        )


if __name__ == "__main__":
    main()
