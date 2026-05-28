"""Check self-collision statistics for motion data on the G1 robot.

Loads the ACCAD motion dataset, optionally remaps to 23-DOF, and runs MuJoCo
collision detection to count how many frames produce self-collisions.

Usage:
    # 23-DOF (default) — headless scan
    python scripts/check_motion_collisions.py

    # 29-DOF — headless scan
    python scripts/check_motion_collisions.py --robot g1

    # Visual playback
    python scripts/check_motion_collisions.py --show
    python scripts/check_motion_collisions.py --robot g1 --show

    # Specific clip
    python scripts/check_motion_collisions.py --show --clip "walk turn right"

    # Only collision frames
    python scripts/check_motion_collisions.py --show --collision-only

    # Remove collision frames and save cleaned data
    python scripts/check_motion_collisions.py --clean
    python scripts/check_motion_collisions.py --robot g1 --clean

    # Verify cleaned data has no collisions
    python scripts/check_motion_collisions.py --motion-file src/assets/data/g1/accad_all_23dof_clean.pkl
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import mujoco
import mujoco.viewer
import numpy as np


@dataclass
class RobotConfig:
    """Configuration for a G1 robot variant."""
    name: str
    joint_names: list[str]
    # Indices into the 29-DOF motion data to extract this robot's DOFs.
    # None means use all columns directly (29-DOF).
    motion_dof_indices: tuple[int, ...] | None = None
    spec_fn: object = None  # Callable -> MjSpec


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


ROBOT_CONFIGS = {
    "g1_23dof": _make_g1_23dof_config,
    "g1": _make_g1_config,
}

# Default lower-body pose from HOME_KEYFRAME (same for both robots).
DEFAULT_LOWER_BODY = {
    "left_hip_pitch_joint": -0.1, "right_hip_pitch_joint": -0.1,
    "left_knee_joint": 0.3, "right_knee_joint": 0.3,
    "left_ankle_pitch_joint": -0.2, "right_ankle_pitch_joint": -0.2,
}

MOTION_FILE = Path(__file__).resolve().parent.parent / "src" / "assets" / "data" / "g1" / "accad_all.pkl"


def build_qpos_index(model, joint_names: list[str]):
    """Build a mapping from joint name to qpos address."""
    idx = {}
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        idx[name] = model.jnt_qposadr[jid]
    return idx


def set_frame(data, qpos_idx, joint_names: list[str], lower_body_dof, upper_body_dof):
    """Set the robot's qpos for one frame of motion data."""
    # Floating base: position + identity quaternion.
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[2] = 0.785
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
    lower = dof[:, :12]
    if motion_dof_indices is not None:
        upper = dof[:, list(motion_dof_indices)]
    else:
        upper = dof[:, 12:]
    return lower, upper


def check_collision(model, data):
    """Run FK + collision detection, return list of (geom1_name, geom2_name) pairs."""
    mujoco.mj_forward(model, data)
    mujoco.mj_collision(model, data)
    pairs = []
    for i in range(data.ncon):
        c = data.contact[i]
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
        pairs.append((g1, g2))
    return pairs


def run_headless(robot_cfg: RobotConfig, motion_file: Path, clean: bool = False):
    """Headless scan: check all frames and print statistics."""
    spec = robot_cfg.spec_fn()
    model = spec.compile()
    data = mujoco.MjData(model)
    qpos_idx = build_qpos_index(model, robot_cfg.joint_names)

    motion_data = joblib.load(motion_file)

    total_frames = 0
    total_collision_frames = 0
    global_pair_clips: dict[str, set[str]] = {}
    clean_frames: dict[str, list[int]] = {}

    for clip_name, clip_data in motion_data.items():
        dof = np.array(clip_data["dof"])
        n_frames = dof.shape[0]
        lower_body, upper_body = extract_dof(dof, robot_cfg.motion_dof_indices)

        clip_collision_frames = 0
        collision_details: dict[str, list[int]] = {}
        clip_clean = []

        for f in range(n_frames):
            set_frame(data, qpos_idx, robot_cfg.joint_names, lower_body[f], upper_body[f])
            pairs = check_collision(model, data)
            if pairs:
                clip_collision_frames += 1
                for pair in pairs:
                    key = f"{pair[0]} <-> {pair[1]}"
                    collision_details.setdefault(key, []).append(f)
                    global_pair_clips.setdefault(key, set()).add(clip_name)
            else:
                clip_clean.append(f)

        total_frames += n_frames
        total_collision_frames += clip_collision_frames
        clean_frames[clip_name] = clip_clean

        pct = clip_collision_frames / n_frames * 100 if n_frames > 0 else 0
        print(f"Clip: {clip_name}")
        print(f"  Frames: {n_frames}, Collisions: {clip_collision_frames} ({pct:.1f}%)")
        if collision_details:
            for pair, frames in collision_details.items():
                print(f"  {pair}: {len(frames)} frames")
        print()

    pct = total_collision_frames / total_frames * 100 if total_frames > 0 else 0
    print(f"Overall: {total_frames} frames, {total_collision_frames} collisions ({pct:.2f}%)")

    if global_pair_clips:
        _print_summary_table(global_pair_clips)

    if clean:
        print()
        print("Cleaning motion data...")
        cleaned = {}
        total_clean = 0
        for clip_name, clip_data in motion_data.items():
            idx = clean_frames[clip_name]
            n_orig = np.array(clip_data["dof"]).shape[0]
            n_clean = len(idx)
            removed = n_orig - n_clean
            total_clean += n_clean
            print(f"  {clip_name}: {n_orig} -> {n_clean} frames ({removed} removed)")
            cleaned[clip_name] = {
                "dof": np.array(clip_data["dof"])[idx],
                "fps": clip_data["fps"],
            }

        total_removed = total_frames - total_clean
        pct_removed = total_removed / total_frames * 100 if total_frames > 0 else 0
        print(f"\n  Overall: {total_frames} -> {total_clean} frames ({total_removed} removed, {pct_removed:.2f}%)")

        out_path = motion_file.parent / f"{motion_file.stem}_{robot_cfg.name}_clean.pkl"
        joblib.dump(cleaned, out_path)
        print(f"  Saved to: {out_path}")


def _body_part(geom_name: str) -> str:
    """Simplify a collision geom name to its body part."""
    g = geom_name.replace("_collision", "")
    for part in ("foot1", "foot2", "foot3", "foot4", "foot5", "foot6", "foot7"):
        if g.endswith(part):
            g = g.rsplit(part, 1)[0] + "foot"
            break
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

                set_frame(data, qpos_idx, robot_cfg.joint_names, lower_body[f], upper_body[f])
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
    parser.add_argument("--clean", action="store_true", help="Remove collision frames and save cleaned data")
    args = parser.parse_args()

    robot_cfg = ROBOT_CONFIGS[args.robot]()

    if args.show:
        run_show(args, robot_cfg)
    else:
        run_headless(robot_cfg, args.motion_file, clean=args.clean)


if __name__ == "__main__":
    main()
