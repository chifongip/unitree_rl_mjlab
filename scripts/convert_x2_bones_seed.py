#!/usr/bin/env python3
"""Convert retargeted X2 (Agibot) BONES-SEED qpos CSVs into G1-format pickles.

The retarget pipeline (robot_retargeter/bash/retarget_bones_seed.sh) wrote one qpos
CSV per G1 pickle key, named ``<key>_from_g1_agibot_x2.csv``, into a single directory
that contains the union of the G1 manipulation and locomotion clip sets. This script
reuses the two G1 pickles only to enumerate the key sets and the manip/loco split, then
converts each matching X2 CSV into the same on-disk format the G1 pickles use:

    motion_data : dict[str, {"dof": (num_frames, 29) float32 radians, "fps": int}]

The X2 qpos CSV is headerless MuJoCo qpos in radians. Columns 0-6 are the floating base
(3 root translation + 4 root quaternion); columns 7-35 are the 29 X2 body joints shared
with the training model (x2_ultra_no_head.xml); columns 36-37 are 2 head joints that the
training model drops. The CSV is at 120 FPS, so we stride-downsample to 30 FPS to match
the G1 pickles.

Usage:
    python scripts/convert_x2_bones_seed.py
    python scripts/convert_x2_bones_seed.py --output-dir /tmp/x2_bones_seed   # dry run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_X2_CSV_DIR = Path("/home/ubuntu/robot_retargeter/output_data/robot_motion/bones_agibot_x2")
DEFAULT_MANIP_PKL = REPO_ROOT / "src/assets/data/g1/bones_seed/bones_seed.pkl"
DEFAULT_LOCO_PKL = REPO_ROOT / "src/assets/data/g1/bones_seed/bones_seed_locomotion.pkl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "src/assets/data/x2/bones_seed"
DEFAULT_RETARGETER_XML = Path("/home/ubuntu/robot_retargeter/asset/robot/agibot_x2/x2_ultra.xml")
DEFAULT_TRAINING_XML = REPO_ROOT / "src/assets/robots/agibot_x2/xmls/x2_ultra_no_head.xml"


def _hinge_joint_columns(model: mujoco.MjModel) -> dict[str, int]:
    """Map each 1-DOF (hinge/slide) joint name to its qpos start column."""
    cols: dict[str, int] = {}
    for i in range(model.njnt):
        if model.jnt_type[i] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            cols[name] = int(model.jnt_qposadr[i])
    return cols


def _hinge_joint_names(model: mujoco.MjModel) -> list[str]:
    names = []
    for i in range(model.njnt):
        if model.jnt_type[i] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i))
    return names


def build_dof_columns(retargeter_xml: Path, training_xml: Path) -> tuple[list[int], int, int]:
    """Return (cols, retargeter_nq, num_dofs).

    ``cols`` maps each training 29-DOF joint (in training order) to its qpos column in
    the retargeter CSV. Raises if a training joint is absent from the retargeter model.
    """
    retargeter = mujoco.MjModel.from_xml_path(str(retargeter_xml))
    training = mujoco.MjModel.from_xml_path(str(training_xml))

    retargeter_cols = _hinge_joint_columns(retargeter)
    training_names = _hinge_joint_names(training)

    cols: list[int] = []
    for name in training_names:
        if name not in retargeter_cols:
            raise KeyError(f"Training joint '{name}' not found in retargeter model {retargeter_xml}")
        cols.append(retargeter_cols[name])

    return cols, retargeter.nq, len(training_names)


def convert_set(
    keys: list[str],
    x2_csv_dir: Path,
    dof_cols: list[int],
    retargeter_nq: int,
    stride: int,
    target_fps: int,
) -> tuple[dict, list[str]]:
    """Convert one clip set. Returns (motion_data, missing_keys)."""
    out: dict = {}
    missing: list[str] = []
    for key in keys:
        csv_path = x2_csv_dir / f"{key}_from_g1_agibot_x2.csv"
        if not csv_path.exists():
            missing.append(key)
            continue
        qpos = np.loadtxt(str(csv_path), delimiter=",")
        if qpos.ndim == 1:
            qpos = qpos[None, :]
        if qpos.ndim != 2 or qpos.shape[1] != retargeter_nq:
            raise ValueError(
                f"{csv_path.name}: expected {retargeter_nq} qpos columns, got {qpos.shape[1]}"
            )
        qpos = qpos[::stride].astype(np.float32)
        dof = qpos[:, dof_cols]
        out[key] = {"dof": dof, "fps": int(target_fps)}
    return out, missing


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert X2 BONES-SEED qpos CSVs to G1-format pickles.")
    p.add_argument("--x2-csv-dir", type=Path, default=DEFAULT_X2_CSV_DIR)
    p.add_argument("--manip-pkl", type=Path, default=DEFAULT_MANIP_PKL)
    p.add_argument("--loco-pkl", type=Path, default=DEFAULT_LOCO_PKL)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--retargeter-xml", type=Path, default=DEFAULT_RETARGETER_XML)
    p.add_argument("--training-xml", type=Path, default=DEFAULT_TRAINING_XML)
    p.add_argument("--source-fps", type=int, default=120)
    p.add_argument("--target-fps", type=int, default=30)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stride = args.source_fps // args.target_fps
    if stride < 1:
        raise ValueError(f"source_fps ({args.source_fps}) must be >= target_fps ({args.target_fps})")

    dof_cols, retargeter_nq, num_dofs = build_dof_columns(args.retargeter_xml, args.training_xml)
    print(f"[setup] retargeter nq={retargeter_nq}, training dof cols={num_dofs}, stride={stride}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    manip_keys = list(joblib.load(args.manip_pkl).keys())
    loco_keys = list(joblib.load(args.loco_pkl).keys())
    print(f"[setup] manip keys={len(manip_keys)}, loco keys={len(loco_keys)}")

    manip_data, manip_missing = convert_set(
        manip_keys, args.x2_csv_dir, dof_cols, retargeter_nq, stride, args.target_fps
    )
    loco_data, loco_missing = convert_set(
        loco_keys, args.x2_csv_dir, dof_cols, retargeter_nq, stride, args.target_fps
    )

    joblib.dump(manip_data, args.output_dir / "bones_seed.pkl")
    joblib.dump(loco_data, args.output_dir / "bones_seed_locomotion.pkl")

    manip_frames = sum(v["dof"].shape[0] for v in manip_data.values())
    loco_frames = sum(v["dof"].shape[0] for v in loco_data.values())
    print(f"[done] wrote {args.output_dir}/bones_seed.pkl ({len(manip_data)} clips, {manip_frames} frames)")
    print(f"[done] wrote {args.output_dir}/bones_seed_locomotion.pkl ({len(loco_data)} clips, {loco_frames} frames)")
    if manip_missing:
        print(f"[warn] {len(manip_missing)} manip keys had no CSV: {manip_missing[:5]}")
    if loco_missing:
        print(f"[warn] {len(loco_missing)} loco keys had no CSV: {loco_missing[:5]}")


if __name__ == "__main__":
    main()
