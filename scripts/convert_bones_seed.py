"""Convert BONES-SEED CSV motion data to pkl format for locomanipulation.

Reads the BONES-SEED dataset metadata CSV and G1 joint-angle CSVs, filters by
category and mirror settings, converts degrees to radians, and outputs a pkl
file matching the format expected by UpperBodyMotionAction.

Output format: dict[str, dict] where each value has:
    "dof": np.ndarray of shape (num_frames, 29), dtype float32, in radians
    "fps": int (target FPS, default 30)

Usage:
    python scripts/convert_bones_seed.py [options]

Options:
    --bones-seed-dir  Path to BONES-SEED dataset (default: /home/ubuntu/BONES-SEED)
    --categories      Comma-separated categories (default: Gestures,Communication,Baseline)
    --output          Output pkl path (default: src/assets/data/g1/bones_seed/bones_seed.pkl)
    --source-fps      Source FPS of CSV data (default: 120)
    --target-fps      Target FPS after downsampling (default: 30)
    --dedup           Keep one random clip per motion description
    --seed            Random seed for --dedup (default: 42)
"""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np


DOF_COL_START = 7  # Skip Frame + root_translateXYZ + root_rotateXYZ
DOF_COL_END = 36  # 29 DOF columns (7..35 inclusive)
NUM_DOFS = 29
DEG2RAD = np.pi / 180.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bones-seed-dir",
        type=Path,
        default=Path("/home/ubuntu/BONES-SEED"),
        help="Path to BONES-SEED dataset root",
    )
    parser.add_argument(
        "--categories",
        type=lambda s: [c.strip() for c in s.split(",")],
        default=["Gestures", "Communication", "Baseline"],
        help="Comma-separated categories to include",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/assets/data/g1/bones_seed/bones_seed.pkl"),
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
        "--dedup",
        action="store_true",
        help="Keep one random clip per motion description",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for --dedup",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    bones_seed_dir = args.bones_seed_dir
    categories = set(args.categories)
    metadata_path = bones_seed_dir / "seed_metadata_v004.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")

    # Read metadata and filter clips.
    print(f"Reading metadata from {metadata_path}...")
    selected = []
    with open(metadata_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["is_mirror"] != "False":
                continue
            if row["category"] not in categories:
                continue
            selected.append(row)

    print(f"Selected {len(selected)} clips from categories: {sorted(categories)}")

    # Deduplicate: keep one random clip per motion description.
    if args.dedup:
        rng = random.Random(args.seed)
        groups = defaultdict(list)
        for row in selected:
            groups[row["content_short_description"]].append(row)
        selected = [rng.choice(rows) for rows in groups.values()]
        print(f"Deduplicated to {len(selected)} clips (one per motion type, seed={args.seed})")

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
    for i, row in enumerate(selected):
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
            print(f"  Processed {i + 1}/{len(selected)} clips...")

    print(f"Converted {len(motion_data)} clips ({skipped} skipped)")

    if not motion_data:
        print("No clips converted. Check --categories values.")
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


if __name__ == "__main__":
    main()
