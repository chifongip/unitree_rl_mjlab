"""Play recorded or resampled height postures in the X2 MuJoCo model.

Inputs can be a CSV produced by ``record_height_postures.py`` or a Python file
containing a ``HEIGHT_POSTURES`` table.  Joint angles are applied by name, and
the floating base is solved so both soles lie on z=0.  This avoids using the
provisional leg-odometry height to place the robot in the viewer.

Usage:
    python scripts/playback_height_postures.py height_postures.csv
    python scripts/playback_height_postures.py scripts/postures_x2_recorded.py
    python scripts/playback_height_postures.py height_postures.csv --once
    python scripts/playback_height_postures.py height_postures.csv --validate-only
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from scipy.optimize import least_squares

from compute_height_postures import (
    ROBOT_CONFIGS,
    compute_foot_ground_z,
    compute_foot_world_xy,
    get_foot_geom_info,
    get_pelvis_body_id,
    load_model,
)


POSITION_SUFFIX = "_position"


@dataclass(frozen=True)
class PostureFrame:
    """One named robot posture and its provisional height metadata."""

    height_label: float
    odometry_z: float
    positions: dict[str, float]


def model_joint_qpos_addresses(model: mujoco.MjModel) -> dict[str, int]:
    """Return scalar joint qpos addresses, excluding the floating base."""
    addresses = {}
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name is not None:
            addresses[name] = int(model.jnt_qposadr[joint_id])
    return addresses


def joint_name_from_column(column: str) -> str | None:
    """Extract the joint name from a recorder CSV position column."""
    if "__" not in column or not column.endswith(POSITION_SUFFIX):
        return None
    _, encoded_name = column.split("__", maxsplit=1)
    return encoded_name[: -len(POSITION_SUFFIX)]


def load_csv_frames(
    csv_path: Path, model: mujoco.MjModel
) -> tuple[list[PostureFrame], dict[str, int]]:
    """Load and validate recorder CSV frames against the MuJoCo model."""
    qpos_addresses = model_joint_qpos_addresses(model)
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} has no CSV header")

        joint_columns: dict[str, str] = {}
        for column in reader.fieldnames:
            joint_name = joint_name_from_column(column)
            if joint_name is None:
                continue
            if joint_name in joint_columns:
                raise ValueError(f"duplicate CSV position for joint {joint_name!r}")
            joint_columns[joint_name] = column

        expected = set(qpos_addresses)
        recorded = set(joint_columns)
        missing = sorted(expected - recorded)
        additional = sorted(recorded - expected)
        if missing or additional:
            raise ValueError(
                "CSV/model joint mismatch: "
                f"missing={missing}, additional={additional}"
            )

        required_metadata = {"height_label_m", "odometry_z_m"}
        missing_metadata = required_metadata - set(reader.fieldnames)
        if missing_metadata:
            raise ValueError(
                f"CSV is missing metadata columns {sorted(missing_metadata)}"
            )

        frames = []
        for line_number, row in enumerate(reader, start=2):
            try:
                height_label = float(row["height_label_m"])
                odometry_z = float(row["odometry_z_m"])
                positions = {
                    name: float(row[column])
                    for name, column in joint_columns.items()
                }
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid numeric value on CSV line {line_number}: {exc}"
                ) from exc

            values = [height_label, odometry_z, *positions.values()]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(
                    f"non-finite numeric value on CSV line {line_number}"
                )
            frames.append(PostureFrame(height_label, odometry_z, positions))

    if not frames:
        raise ValueError(f"{csv_path} contains no posture rows")
    return frames, qpos_addresses


def load_python_frames(
    posture_path: Path, model: mujoco.MjModel
) -> tuple[list[PostureFrame], dict[str, int]]:
    """Load a Python HEIGHT_POSTURES table and validate its named joints."""
    namespace = {}
    exec(posture_path.read_text(encoding="utf-8"), namespace)
    table = namespace.get("HEIGHT_POSTURES")
    if not isinstance(table, dict) or not table:
        raise ValueError(f"{posture_path} has no non-empty HEIGHT_POSTURES dict")

    qpos_addresses = model_joint_qpos_addresses(model)
    frames = []
    expected_names = None
    for height_value, posture in table.items():
        if not isinstance(posture, dict) or not posture:
            raise ValueError(f"posture at height {height_value!r} is not a dict")
        try:
            height = float(height_value)
            positions = {name: float(value) for name, value in posture.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid posture at height {height_value!r}: {exc}"
            ) from exc
        if not math.isfinite(height) or not all(
            math.isfinite(value) for value in positions.values()
        ):
            raise ValueError(f"non-finite posture at height {height_value!r}")

        additional = sorted(set(positions) - set(qpos_addresses))
        if additional:
            raise ValueError(f"posture has joints absent from model: {additional}")
        names = set(positions)
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise ValueError(
                f"joint-name set changes at height {height_value!r}"
            )
        frames.append(PostureFrame(height, height, positions))

    frames.sort(key=lambda frame: frame.height_label)
    return frames, qpos_addresses


def load_frames(
    posture_path: Path, model: mujoco.MjModel
) -> tuple[list[PostureFrame], dict[str, int]]:
    """Load recorder CSV or exported Python posture data by file suffix."""
    if posture_path.suffix.lower() == ".csv":
        return load_csv_frames(posture_path, model)
    if posture_path.suffix.lower() == ".py":
        return load_python_frames(posture_path, model)
    raise ValueError(
        f"unsupported posture file {posture_path}; expected .csv or .py"
    )


def apply_grounded_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    frame: PostureFrame,
    qpos_addresses: dict[str, int],
    foot_geoms: dict[str, np.ndarray],
    left_ankle_body: int,
    right_ankle_body: int,
    pelvis_body_id: int,
) -> tuple[float, float]:
    """Apply a posture and solve the free base so both soles lie on z=0."""
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0

    # Joint positions remain exactly as recorded.  Only the floating base is
    # adjusted to align the best-fit plane through both soles with the ground.
    data.qpos[0:3] = 0.0
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    for name, position in frame.positions.items():
        data.qpos[qpos_addresses[name]] = position

    def set_base(z: float, roll: float, pitch: float) -> None:
        cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
        cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
        data.qpos[2] = z
        # Intrinsic XYZ Euler rotation with yaw fixed at zero.  Ground contact
        # does not constrain yaw, so retaining zero avoids arbitrary rotation.
        data.qpos[3:7] = (cr * cp, sr * cp, cr * sp, -sr * sp)

    def sole_heights(base_pose: np.ndarray | tuple[float, float, float]):
        set_base(*base_pose)
        mujoco.mj_forward(model, data)
        return np.concatenate(
            [
                compute_foot_ground_z(
                    data, left_ankle_body, foot_geoms["left"]
                ),
                compute_foot_ground_z(
                    data, right_ankle_body, foot_geoms["right"]
                ),
            ]
        )

    initial_foot_z = sole_heights((0.0, 0.0, 0.0))
    solution = least_squares(
        sole_heights,
        x0=(-float(initial_foot_z.mean()), 0.0, 0.0),
        bounds=((-2.0, -math.pi / 3.0, -math.pi / 3.0),
                (2.0, math.pi / 3.0, math.pi / 3.0)),
        max_nfev=100,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    if not solution.success:
        raise RuntimeError(f"free-base grounding failed: {solution.message}")

    foot_z = sole_heights(solution.x)
    # The least-squares plane is centered on z=0.  Lift it by the remaining
    # sub-millimetre residual so no sole point penetrates the visual floor.
    solution.x[0] -= float(foot_z.min())
    set_base(*solution.x)
    mujoco.mj_forward(model, data)

    # Horizontal translation is unconstrained by a flat floor; center the foot
    # support points around the viewer origin while leaving joint angles intact.
    foot_xy = np.concatenate(
        [
            compute_foot_world_xy(
                data, left_ankle_body, foot_geoms["left"]
            ),
            compute_foot_world_xy(
                data, right_ankle_body, foot_geoms["right"]
            ),
        ]
    )
    data.qpos[0:2] -= foot_xy.mean(axis=0)
    mujoco.mj_forward(model, data)

    grounded_foot_z = sole_heights(
        (float(data.qpos[2]), float(solution.x[1]), float(solution.x[2]))
    )
    pelvis_height = float(data.xpos[pelvis_body_id, 2])
    foot_height_spread = float(np.ptp(grounded_foot_z))
    return pelvis_height, foot_height_spread


def initialize_ground_plane(viewer) -> None:
    """Add a visual ground plane at z=0 to the passive viewer."""
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[0],
        mujoco.mjtGeom.mjGEOM_PLANE,
        np.array([2.0, 2.0, 0.1]),
        np.array([0.0, 0.0, 0.0]),
        np.eye(3).flatten(),
        np.array([0.25, 0.3, 0.35, 0.45], dtype=np.float32),
    )
    viewer.user_scn.ngeom = 1


def validate_frames(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    frames: list[PostureFrame],
    qpos_addresses: dict[str, int],
    foot_geoms: dict[str, np.ndarray],
    left_ankle_body: int,
    right_ankle_body: int,
    pelvis_body_id: int,
) -> list[tuple[float, float]]:
    """Apply every frame headlessly and return grounded height diagnostics."""
    diagnostics = []
    for frame in frames:
        diagnostics.append(
            apply_grounded_frame(
                model,
                data,
                frame,
                qpos_addresses,
                foot_geoms,
                left_ankle_body,
                right_ankle_body,
                pelvis_body_id,
            )
        )
    return diagnostics


def playback(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    frames: list[PostureFrame],
    qpos_addresses: dict[str, int],
    foot_geoms: dict[str, np.ndarray],
    left_ankle_body: int,
    right_ankle_body: int,
    pelvis_body_id: int,
    frame_duration: float,
    once: bool,
) -> None:
    """Play the recorded frames in capture order in a passive viewer."""
    print(f"Opening viewer with {len(frames)} frames. Press Ctrl+C to stop.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        initialize_ground_plane(viewer)
        try:
            while viewer.is_running():
                for index, frame in enumerate(frames, start=1):
                    if not viewer.is_running():
                        return
                    pelvis_height, foot_spread = apply_grounded_frame(
                        model,
                        data,
                        frame,
                        qpos_addresses,
                        foot_geoms,
                        left_ankle_body,
                        right_ankle_body,
                        pelvis_body_id,
                    )
                    viewer.sync()
                    print(
                        f"Frame {index:02d}/{len(frames)}: "
                        f"odom={frame.odometry_z:.3f} m, "
                        f"grounded pelvis={pelvis_height:.3f} m, "
                        f"foot spread={foot_spread:.3f} m"
                    )
                    deadline = time.monotonic() + frame_duration
                    while viewer.is_running() and time.monotonic() < deadline:
                        time.sleep(min(0.02, deadline - time.monotonic()))
                if once:
                    return
        except KeyboardInterrupt:
            print("\nViewer closed.")


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError(
            "value must be a finite number greater than zero"
        )
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "posture_file",
        type=Path,
        help="Recorder CSV or exported HEIGHT_POSTURES Python file",
    )
    parser.add_argument(
        "--frame-duration",
        type=positive_float,
        default=0.25,
        help="Seconds to display each posture (default: 0.25)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Play the CSV once instead of looping",
    )
    parser.add_argument(
        "--sort-height",
        action="store_true",
        help="Play from lowest to highest odometry label instead of capture order",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and apply all frames without opening a viewer",
    )
    parser.add_argument(
        "--sole-tolerance",
        type=positive_float,
        default=0.003,
        help="Maximum sole-point height spread in metres (default: 0.003)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ROBOT_CONFIGS["x2"]
    model, data = load_model(cfg["xml"], cfg["assets_dir"])
    frames, qpos_addresses = load_frames(args.posture_file, model)
    if args.sort_height:
        frames.sort(key=lambda frame: frame.height_label)

    foot_geoms, left_ankle_body, right_ankle_body = get_foot_geom_info(
        model, cfg["foot_geom_count"], cfg["is_capsule"]
    )
    pelvis_body_id = get_pelvis_body_id(model)
    diagnostics = validate_frames(
        model,
        data,
        frames,
        qpos_addresses,
        foot_geoms,
        left_ankle_body,
        right_ankle_body,
        pelvis_body_id,
    )
    pelvis_heights = [height for height, _ in diagnostics]
    foot_spreads = [spread for _, spread in diagnostics]
    max_foot_spread = max(foot_spreads)
    print(
        f"Validated {len(frames)} frames and {len(frames[0].positions)} "
        f"posture joints; "
        f"grounded pelvis range={min(pelvis_heights):.3f}-"
        f"{max(pelvis_heights):.3f} m; "
        f"maximum foot-height spread={max_foot_spread:.6f} m"
    )
    if max_foot_spread > args.sole_tolerance:
        raise ValueError(
            f"sole contact check failed: {max_foot_spread:.6f} m exceeds "
            f"--sole-tolerance={args.sole_tolerance:.6f} m"
        )
    if not args.validate_only:
        playback(
            model,
            data,
            frames,
            qpos_addresses,
            foot_geoms,
            left_ankle_body,
            right_ankle_body,
            pelvis_body_id,
            args.frame_duration,
            args.once,
        )


if __name__ == "__main__":
    main()
