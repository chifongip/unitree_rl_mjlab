"""Convert BONES-SEED locomotion data to pkl format for locomanipulation.

Reads the BONES-SEED dataset metadata CSV and G1 joint-angle CSVs, filters by
locomotion categories and movement types (walking/jogging/running), converts
degrees to radians, and outputs a pkl file matching the format expected by
UpperBodyMotionAction.

Usage:
    python scripts/convert_bones_seed_locomotion.py [options]

Options:
    --bones-seed-dir  Path to BONES-SEED dataset (default: /home/ubuntu/BONES-SEED)
    --output          Output pkl path (default: src/assets/data/g1/bones_seed/bones_seed_locomotion.pkl)
    --source-fps      Source FPS of CSV data (default: 120)
    --target-fps      Target FPS after downsampling (default: 30)
    --target-clips    Target number of clips after sampling (default: 1256)
    --dedup           Keep one random clip per motion description
    --seed            Random seed for --dedup and sampling (default: 42)
    --no-sample       Disable sampling; keep all matching clips
"""

import argparse
import csv
import random
from collections import defaultdict, Counter
from pathlib import Path

import joblib
import numpy as np


DOF_COL_START = 7  # Skip Frame + root_translateXYZ + root_rotateXYZ
DOF_COL_END = 36  # 29 DOF columns (7..35 inclusive)
NUM_DOFS = 29
DEG2RAD = np.pi / 180.0

# Locomotion categories in BONES-SEED.
LOCOMOTION_CATEGORIES = {
    "Basic Locomotion Neutral",
    "Basic Locomotion Styles",
    "Advanced Locomotion",
}

# Movement types to include (content_type_of_movement field).
LOCOMOTION_MOVEMENT_TYPES = {
    "walking",
    "walking, turning",
    "jogging",
    "jogging, turning",
    "running",
    "running, turning",
}

# Exclude clips whose content_short_description contains any of these substrings.
EXCLUDE_DESCRIPTION_SUBSTRINGS = [
    "injured",
    "Injured",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bones-seed-dir",
        type=Path,
        default=Path("/home/ubuntu/BONES-SEED"),
        help="Path to BONES-SEED dataset root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/assets/data/g1/bones_seed/bones_seed_locomotion.pkl"),
        help="Output pkl path",
    )
    parser.add_argument(
        "--source-fps",
        type=int,
        default=120,
        help="Source FPS of CSV data",
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=30,
        help="Target FPS after downsampling",
    )
    parser.add_argument(
        "--target-clips",
        type=int,
        default=1256,
        help="Target number of clips after sampling",
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        help="Keep one random clip per motion description",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for --dedup and sampling",
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Disable sampling; keep all matching clips",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    bones_seed_dir = args.bones_seed_dir
    metadata_path = bones_seed_dir / "seed_metadata_v004.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")

    # Read metadata and filter clips.
    print(f"Reading metadata from {metadata_path}...")
    all_rows = []
    with open(metadata_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Filter: non-mirror, locomotion categories, walking/jogging/running only.
            if row["is_mirror"] == "True":
                continue
            if row["category"] not in LOCOMOTION_CATEGORIES:
                continue
            mov_type = row.get("content_type_of_movement", "")
            if mov_type not in LOCOMOTION_MOVEMENT_TYPES:
                continue
            # Exclude injured/unnatural gait clips.
            desc = row.get("content_short_description", "")
            if any(s in desc for s in EXCLUDE_DESCRIPTION_SUBSTRINGS):
                continue
            all_rows.append(row)

    print(f"Filtered to {len(all_rows)} clips (locomotion, walking/jogging/running, non-injured)")

    # Deduplicate: keep one random clip per motion description.
    rng = random.Random(args.seed)
    if args.dedup:
        groups = defaultdict(list)
        for row in all_rows:
            groups[row["content_short_description"]].append(row)
        all_rows = [rng.choice(rows) for rows in groups.values()]
        print(f"Deduplicated to {len(all_rows)} clips (one per motion type, seed={args.seed})")

    # Sample to target clip count using stratified sampling by movement type.
    if not args.no_sample and len(all_rows) > args.target_clips:
        # Group by movement type for stratified sampling.
        type_groups = defaultdict(list)
        for row in all_rows:
            type_groups[row.get("content_type_of_movement", "unknown")].append(row)

        # Compute per-type allocation proportional to available count.
        total = len(all_rows)
        sampled = []
        allocated = 0
        type_items = sorted(type_groups.items(), key=lambda x: -len(x[1]))
        for i, (mov_type, rows) in enumerate(type_items):
            if i == len(type_items) - 1:
                # Last group takes the remainder to hit target exactly.
                take = args.target_clips - allocated
            else:
                take = max(1, int(args.target_clips * len(rows) / total))
            take = min(take, len(rows))
            sampled.extend(rng.sample(rows, take))
            allocated += take

        all_rows = sampled
        print(f"Sampled {len(all_rows)} clips (target={args.target_clips})")

    # Show movement type distribution after sampling.
    mov_counts = Counter(r.get("content_type_of_movement", "") for r in all_rows)
    print("Movement type distribution:")
    for mov, count in mov_counts.most_common():
        print(f"  {mov}: {count}")

    # Downsampling setup.
    source_fps = args.source_fps
    target_fps = args.target_fps
    if target_fps > source_fps:
        raise ValueError(f"target_fps ({target_fps}) cannot exceed source_fps ({source_fps})")
    stride = source_fps // target_fps
    if source_fps % target_fps != 0:
        print(f"Warning: {source_fps} / {target_fps} is not integer, stride={stride}")
    print(f"Downsampling: {source_fps} -> {target_fps} FPS (stride={stride})")

    # Convert CSV files to pkl format.
    motion_data = {}
    skipped = 0
    for i, row in enumerate(all_rows):
        csv_rel_path = row["move_g1_path"]
        csv_path = bones_seed_dir / csv_rel_path

        if not csv_path.exists():
            skipped += 1
            continue

        # Read CSV: columns are Frame, root_translateXYZ, root_rotateXYZ, 29 DOFs.
        try:
            data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        except Exception:
            skipped += 1
            continue

        if data.ndim != 2 or data.shape[1] < DOF_COL_END:
            skipped += 1
            continue

        # Extract 29 DOF columns and convert degrees to radians.
        dof = data[:, DOF_COL_START:DOF_COL_END].astype(np.float32) * DEG2RAD

        # Downsample to target FPS.
        if stride > 1:
            dof = dof[::stride]

        clip_name = row["move_name"]
        # Deduplicate clip names (multiple actors for same move).
        if clip_name in motion_data:
            clip_name = f"{clip_name}__{row['actor_uid']}"

        motion_data[clip_name] = {"dof": dof, "fps": target_fps}

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(all_rows)} clips...")

    print(f"Converted {len(motion_data)} clips ({skipped} skipped)")

    if not motion_data:
        print("No clips converted. Check filter criteria.")
        return

    # Save pkl.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(motion_data, args.output)
    print(f"Saved to {args.output}")

    # Quick sanity check.
    sample_key = next(iter(motion_data))
    sample = motion_data[sample_key]
    print(f"\nSample clip: {sample_key}")
    print(f"  Shape: {sample['dof'].shape}")
    print(f"  FPS: {sample['fps']}")
    print(f"  DOF range: [{sample['dof'].min():.4f}, {sample['dof'].max():.4f}] rad")

    # Summary stats.
    total_frames = sum(v["dof"].shape[0] for v in motion_data.values())
    print(f"\nSummary:")
    print(f"  Clips: {len(motion_data)}")
    print(f"  Total frames: {total_frames:,}")
    print(f"  Duration: {total_frames / target_fps / 60:.1f} min")


if __name__ == "__main__":
    main()
