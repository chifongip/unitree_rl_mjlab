"""Check self-collision statistics for motion data.

Loads motion data (pkl) and runs MuJoCo collision detection to count how many
frames produce self-collisions. Unnamed geoms fall back to parent body names.

Can also clean the data: drop individual collision frames (``--clean``) or
split clips at collision boundaries into contiguous clean segments (``--split``).

Robots: g1_23dof (default), g1, x2.

Usage:
    python scripts/check_motion_collisions.py --robot <robot> --motion-file <path> [options]

Options:
    --robot            Robot variant (default: g1_23dof)
    --motion-file      Path to motion pkl (default: G1 ACCAD data)
    --show             Open MuJoCo viewer for visual playback
    --clip <str>       Filter to clips containing this substring
    --collision-only   Only show collision frames (with --show)
    --clean            Remove collision frames and save cleaned data
    --split            Split clips at collision boundaries into clean segments
    --min-segment-len  Minimum segment length in frames for --split (default: 60)

Examples:
    # Headless scan
    python scripts/check_motion_collisions.py --robot x2 --motion-file src/assets/data/x2/amass/amass_all.pkl

    # Visual playback
    python scripts/check_motion_collisions.py --robot g1 --motion-file src/assets/data/g1/bones_seed/bones_seed_locomotion.pkl --show

    # Drop collision frames (creates temporal gaps)
    python scripts/check_motion_collisions.py --robot g1_23dof --motion-file src/assets/data/g1/bones_seed/bones_seed_locomotion.pkl --clean

    # Split into clean segments (preserves continuity, 0 collisions)
    python scripts/check_motion_collisions.py --robot g1_23dof --motion-file src/assets/data/g1/bones_seed/bones_seed_locomotion.pkl --split --min-segment-len 60
"""

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import mujoco
import mujoco.viewer
import numpy as np


@dataclass
class RobotConfig:
    """Configuration for a robot variant."""
    name: str
    joint_names: list[str]
    # Indices into the motion data DOF array to extract this robot's DOFs.
    # None means use all columns directly.
    motion_dof_indices: tuple[int, ...] | None = None
    spec_fn: object = None  # Callable -> MjSpec
    default_height: float = 0.785  # Pelvis height for standing pose.


def _make_g1_23dof_config() -> RobotConfig:
    from src.assets.robots.unitree_g1.g1_23dof_constants import get_spec
    return RobotConfig(
        name="g1_23dof",
        joint_names=[
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "waist_yaw_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
        ],
        motion_dof_indices=(12, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26),
        spec_fn=get_spec,
    )


def _make_g1_config() -> RobotConfig:
    from src.assets.robots.unitree_g1.g1_constants import get_spec
    return RobotConfig(
        name="g1",
        joint_names=[
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint", "left_elbow_joint",
            "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint", "right_elbow_joint",
            "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        ],
        motion_dof_indices=None,  # use all 29 columns directly
        spec_fn=get_spec,
    )


def _make_x2_config() -> RobotConfig:
    from src.assets.robots.agibot_x2.x2_constants import get_spec
    # Motion data (29 DOF) is already in X2 joint order:
    #   0-11: leg, 12-14: waist (yaw/pitch/roll), 15-21: L arm, 22-28: R arm.
    return RobotConfig(
        name="x2",
        joint_names=[
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint", "left_elbow_joint",
            "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint", "right_elbow_joint",
            "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
        ],
        motion_dof_indices=None,
        spec_fn=get_spec,
        default_height=0.68,
    )


ROBOT_CONFIGS = {
    "g1_23dof": _make_g1_23dof_config,
    "g1": _make_g1_config,
    "x2": _make_x2_config,
}

# Default lower-body pose from HOME_KEYFRAME (same for both robots).
DEFAULT_LOWER_BODY = {
    "left_hip_pitch_joint": -0.1, "right_hip_pitch_joint": -0.1,
    "left_hip_roll_joint": 0.0, "right_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0, "right_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.3, "right_knee_joint": 0.3,
    "left_ankle_pitch_joint": -0.2, "right_ankle_pitch_joint": -0.2,
    "left_ankle_roll_joint": 0.0, "right_ankle_roll_joint": 0.0,
}

MOTION_FILE = Path(__file__).resolve().parent.parent / "src" / "assets" / "data" / "g1" / "accad" / "accad_all.pkl"


def build_qpos_index(model, joint_names: list[str]):
    """Build a mapping from joint name to qpos address."""
    idx = {}
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        idx[name] = model.jnt_qposadr[jid]
    return idx


def set_frame(data, qpos_idx, joint_names: list[str], lower_body_dof, upper_body_dof,
              default_height: float = 0.785):
    """Set the robot's qpos for one frame of motion data."""
    # Floating base: position + identity quaternion.
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[2] = default_height
    data.qpos[3] = 1.0  # quat w
    data.qpos[4] = 0.0
    data.qpos[5] = 0.0
    data.qpos[6] = 0.0

    # Lower body (indices 0-11, same ordering in both 29-DOF and 23-DOF).
    for i, name in enumerate(joint_names[:12]):
        adr = qpos_idx[name]
        if name in DEFAULT_LOWER_BODY:
            data.qpos[adr] = DEFAULT_LOWER_BODY[name]
        else:
            data.qpos[adr] = lower_body_dof[i]

    # Upper body.
    for i, name in enumerate(joint_names[12:]):
        adr = qpos_idx[name]
        data.qpos[adr] = upper_body_dof[i]


def extract_dof(dof: np.ndarray, motion_dof_indices: tuple[int, ...] | None):
    """Extract lower and upper body DOFs from motion data."""
    assert dof.shape[1] >= 12, f"Motion data has {dof.shape[1]} columns, expected ≥12"
    lower = dof[:, :12]
    if motion_dof_indices is not None:
        upper = dof[:, list(motion_dof_indices)]
    else:
        upper = dof[:, 12:]
    return lower, upper


def _geom_or_body_name(model, geom_id):
    """Return geom name, or parent body name if geom is unnamed."""
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    if name is None:
        body_id = model.geom_bodyid[geom_id]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    return name


def check_collision(model, data):
    """Run FK + collision detection, return list of (name1, name2) pairs."""
    mujoco.mj_forward(model, data)
    mujoco.mj_collision(model, data)
    pairs = []
    for i in range(data.ncon):
        c = data.contact[i]
        n1 = _geom_or_body_name(model, c.geom1)
        n2 = _geom_or_body_name(model, c.geom2)
        pairs.append((n1, n2))
    return pairs


def split_clean_segments(
    motion_data: dict,
    collision_masks: dict[str, np.ndarray],
    min_len: int = 60,
) -> dict:
    """Split clips at collision boundaries, keeping clean segments as new clips.

    Each contiguous collision-free segment ≥ min_len frames becomes its own
    clip named ``{original_clip}__seg{N}``. This preserves temporal continuity
    within each segment — no interpolation or frame dropping occurs.

    Args:
        motion_data: Original motion data dict.
        collision_masks: Per-clip boolean array (True = collision frame).
        min_len: Minimum segment length in frames.

    Returns:
        Dict of clean segments with same value structure.
    """
    segments: dict = {}
    total_orig = 0
    total_clean = 0

    for clip_name, clip_data in motion_data.items():
        mask = collision_masks[clip_name]
        dof = np.array(clip_data["dof"], dtype=np.float32)
        n_frames = len(mask)
        total_orig += n_frames

        clean = ~mask
        padded = np.concatenate([[False], clean, [False]])
        changes = np.diff(padded.astype(int))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]

        for i, (s, e) in enumerate(zip(starts, ends)):
            seg_len = e - s
            if seg_len >= min_len:
                seg_name = f"{clip_name}__seg{i}"
                segments[seg_name] = {
                    "dof": dof[s:e].astype(np.float32),
                    "fps": clip_data["fps"],
                }
                total_clean += seg_len

    removed = total_orig - total_clean
    pct_removed = removed / total_orig * 100 if total_orig > 0 else 0
    print(f"\nSplit cleaning (min segment={min_len}f): "
          f"{len(motion_data):,} clips → {len(segments):,} segments "
          f"({total_orig:,} → {total_clean:,} frames, {removed:,} removed, {pct_removed:.1f}%)")

    return segments


def run_headless(
    robot_cfg: RobotConfig,
    motion_file: Path,
    clean: bool = False,
    split: bool = False,
    min_segment_len: int = 60,
):
    """Headless scan: check all frames and print statistics."""
    spec = robot_cfg.spec_fn()
    model = spec.compile()
    data = mujoco.MjData(model)
    qpos_idx = build_qpos_index(model, robot_cfg.joint_names)

    motion_data = joblib.load(motion_file)

    total_frames = 0
    total_collision_frames = 0
    global_pair_clips: dict[str, set[str]] = {}
    collision_masks: dict[str, np.ndarray] = {}

    for clip_name, clip_data in motion_data.items():
        dof = np.array(clip_data["dof"])
        n_frames = dof.shape[0]
        lower_body, upper_body = extract_dof(dof, robot_cfg.motion_dof_indices)

        clip_mask = np.zeros(n_frames, dtype=bool)

        for f in range(n_frames):
            set_frame(data, qpos_idx, robot_cfg.joint_names, lower_body[f], upper_body[f],
                      default_height=robot_cfg.default_height)
            pairs = check_collision(model, data)
            if pairs:
                clip_mask[f] = True
                for pair in pairs:
                    key = f"{pair[0]} <-> {pair[1]}"
                    global_pair_clips.setdefault(key, set()).add(clip_name)

        total_frames += n_frames
        total_collision_frames += clip_mask.sum()
        collision_masks[clip_name] = clip_mask

    pct = total_collision_frames / total_frames * 100 if total_frames > 0 else 0
    print(f"Overall: {total_frames} frames, {total_collision_frames} collisions ({pct:.2f}%)")

    if global_pair_clips:
        _print_summary_table(global_pair_clips)

    if clean or split:
        if split:
            print()
            print(f"Splitting at collision boundaries (min segment={min_segment_len}f)...")
            cleaned = split_clean_segments(motion_data, collision_masks, min_segment_len)
            suffix = f"{robot_cfg.name}_split"
        else:
            print()
            print("Cleaning motion data (dropping collision frames)...")
            cleaned = {}
            total_clean = 0
            dropped_empty = 0
            for clip_name, clip_data in motion_data.items():
                keep = ~collision_masks[clip_name]
                n_clean = keep.sum()
                if n_clean == 0:
                    dropped_empty += 1
                    continue
                total_clean += n_clean
                cleaned[clip_name] = {
                    "dof": np.array(clip_data["dof"])[keep],
                    "fps": clip_data["fps"],
                }
            total_removed = total_frames - total_clean
            pct_removed = total_removed / total_frames * 100 if total_frames > 0 else 0
            print(f"  Overall: {total_frames:,} -> {total_clean:,} frames "
                  f"({total_removed:,} removed, {pct_removed:.1f}%)")
            if dropped_empty > 0:
                print(f"  ({dropped_empty} fully-colliding clips dropped)")
            suffix = f"{robot_cfg.name}_clean"

        out_path = motion_file.parent / f"{motion_file.stem}_{suffix}.pkl"
        joblib.dump(cleaned, out_path)
        print(f"  Saved to: {out_path}")


def _body_part(geom_name: str) -> str:
    """Simplify a collision geom name to its body part."""
    g = geom_name.replace("_collision", "")
    # Collapse footN suffixes (foot1..foot12) to "foot".
    g = re.sub(r"foot\d+$", "foot", g)
    for prefix in ("left_", "right_"):
      if g.startswith(prefix):
        g = g[len(prefix):]
        break
    return g


def _categorize_pair(pair: str) -> str | None:
    """Return a category label for a collision pair, or None to skip."""
    parts = pair.split(" <-> ")
    if len(parts) != 2:
        return None
    p1 = _body_part(parts[0])
    p2 = _body_part(parts[1])
    if p1 > p2:
        p1, p2 = p2, p1
    return f"{p1} <-> {p2}"


def _print_summary_table(global_pair_clips: dict[str, set[str]]):
    """Print a summary table of collision categories."""
    categories: dict[str, set[str]] = {}
    for pair, clips in global_pair_clips.items():
        cat = _categorize_pair(pair)
        if cat is not None:
            categories.setdefault(cat, set()).update(clips)

    if not categories:
        return

    rows = sorted(categories.items(), key=lambda x: len(x[1]), reverse=True)

    cat_width = max(len(r[0]) for r in rows)
    cat_width = max(cat_width, len("Collision pair"))
    clips_width = max(len(str(len(r[1]))) for r in rows)
    clips_width = max(clips_width, len("Clips affected"))

    header = f"| {'Collision pair':<{cat_width}} | {'Clips affected':>{clips_width}} |"
    sep = f"+{'-' * (cat_width + 2)}+{'-' * (clips_width + 2)}+"

    print()
    print("Summary:")
    print(sep)
    print(header)
    print(sep)
    for cat, clips in rows:
        print(f"| {cat:<{cat_width}} | {len(clips):>{clips_width}} |")
    print(sep)


def run_show(args, robot_cfg: RobotConfig):
    """Visual playback with MuJoCo viewer."""
    spec = robot_cfg.spec_fn()
    model = spec.compile()
    data = mujoco.MjData(model)
    qpos_idx = build_qpos_index(model, robot_cfg.joint_names)

    motion_data = joblib.load(args.motion_file)

    if args.clip:
        motion_data = {
            k: v for k, v in motion_data.items()
            if args.clip.lower() in k.lower()
        }
        if not motion_data:
            print(f"No clips matching '{args.clip}'")
            return

    print(f"Playing {len(motion_data)} clips ({robot_cfg.name}). Enable contacts: viewer menu → Rendering → Contacts")
    print("Press ESC to exit.\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for clip_name, clip_data in motion_data.items():
            dof = np.array(clip_data["dof"])
            n_frames = dof.shape[0]
            fps = clip_data["fps"]
            lower_body, upper_body = extract_dof(dof, robot_cfg.motion_dof_indices)
            dt = 1.0 / fps

            clip_collisions = 0
            print(f"Clip: {clip_name} ({n_frames} frames, {fps} fps)")

            for f in range(n_frames):
                if not viewer.is_running():
                    return

                set_frame(data, qpos_idx, robot_cfg.joint_names, lower_body[f], upper_body[f],
                          default_height=robot_cfg.default_height)
                pairs = check_collision(model, data)
                is_collision = len(pairs) > 0

                if is_collision:
                    clip_collisions += 1

                if args.collision_only and not is_collision:
                    continue

                viewer.sync()
                if args.collision_only and is_collision:
                    print(f"  Frame {f}: COLLISION - {pairs}")
                    time.sleep(0.5)
                else:
                    time.sleep(dt)

            print(f"  Collisions: {clip_collisions}/{n_frames}")
            print()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", type=str, default="g1_23dof", choices=list(ROBOT_CONFIGS.keys()),
                        help="Robot variant (default: g1_23dof)")
    parser.add_argument("--motion-file", type=Path, default=MOTION_FILE, help="Path to motion pkl file")
    parser.add_argument("--show", action="store_true", help="Open MuJoCo viewer for visual inspection")
    parser.add_argument("--clip", type=str, default=None, help="Filter to clips containing this substring")
    parser.add_argument("--collision-only", action="store_true", help="Only show collision frames (with --show)")
    clean_split = parser.add_mutually_exclusive_group()
    clean_split.add_argument("--clean", action="store_true",
                             help="Remove collision frames and save cleaned data")
    clean_split.add_argument("--split", action="store_true",
                             help="Split clips at collision boundaries, keeping clean segments as separate clips")
    parser.add_argument("--min-segment-len", type=int, default=60,
                        help="Minimum clean segment length in frames for --split (default: 60)")
    args = parser.parse_args()

    if args.min_segment_len <= 0:
        parser.error("--min-segment-len must be > 0")
    if args.min_segment_len != 60 and not args.split:
        print("Warning: --min-segment-len is only used with --split, ignoring")


    robot_cfg = ROBOT_CONFIGS[args.robot]()

    if args.show:
        run_show(args, robot_cfg)
    else:
        run_headless(robot_cfg, args.motion_file, clean=args.clean,
                     split=args.split, min_segment_len=args.min_segment_len)


if __name__ == "__main__":
    main()
