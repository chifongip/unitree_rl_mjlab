"""Compute default lower-body joint postures for each commanded base height.

Uses MuJoCo forward kinematics + scipy optimization to find joint angles
that place the pelvis at a target height with both feet on the ground
(z=0) and the torso upright.

Usage:
    python scripts/compute_height_postures.py --robot g1                  # G1, compute + print
    python scripts/compute_height_postures.py --robot x2                  # X2, compute + print
    python scripts/compute_height_postures.py --robot x2 --save postures.py   # compute + save
    python scripts/compute_height_postures.py --robot x2 --show           # also open MuJoCo viewer
    python scripts/compute_height_postures.py --postures-file postures.py # load + replay in viewer

Output: a Python dict that can be saved to a file and imported by env_cfgs.py
as the height_postures parameter for the variable_posture reward.
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from scipy.optimize import minimize

# ── Robot configurations ────────────────────────────────────────────────────

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"

# G1
G1_XML = _SRC_DIR / "assets" / "robots" / "unitree_g1" / "xmls" / "g1.xml"
G1_ASSETS_DIR = G1_XML.parent / "assets"
G1_NOMINAL_HEIGHT = 0.76
G1_TARGET_HEIGHTS = sorted(set(
    [round(h, 2) for h in np.arange(0.50, 0.79, 0.02)] + [G1_NOMINAL_HEIGHT]
))
G1_FOOT_GEOM_COUNT = 7   # capsules
G1_IS_CAPSULE = True
G1_ALIGN_OFFSET = np.array([-0.05, 0.0])

# X2
X2_XML = _SRC_DIR / "assets" / "robots" / "agibot_x2" / "xmls" / "x2_ultra_no_head.xml"
X2_ASSETS_DIR = X2_XML.parent / "assets"
X2_NOMINAL_HEIGHT = 0.66
X2_TARGET_HEIGHTS = sorted(set(
    [round(h, 2) for h in np.arange(0.40, 0.67, 0.02)] + [X2_NOMINAL_HEIGHT]
))
X2_FOOT_GEOM_COUNT = 12  # spheres
X2_IS_CAPSULE = False
X2_ALIGN_OFFSET = np.array([0.0, 0.0])

ROBOT_CONFIGS = {
    "g1": {
        "xml": G1_XML,
        "assets_dir": G1_ASSETS_DIR,
        "nominal_height": G1_NOMINAL_HEIGHT,
        "target_heights": G1_TARGET_HEIGHTS,
        "foot_geom_count": G1_FOOT_GEOM_COUNT,
        "is_capsule": G1_IS_CAPSULE,
        "align_offset": G1_ALIGN_OFFSET,
    },
    "x2": {
        "xml": X2_XML,
        "assets_dir": X2_ASSETS_DIR,
        "nominal_height": X2_NOMINAL_HEIGHT,
        "target_heights": X2_TARGET_HEIGHTS,
        "foot_geom_count": X2_FOOT_GEOM_COUNT,
        "is_capsule": X2_IS_CAPSULE,
        "align_offset": X2_ALIGN_OFFSET,
    },
}

# Lower-body joints to optimize (12 DOF) — same names for G1 and X2.
LOWER_BODY_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]


def load_model(xml_path, assets_dir):
    """Load an MJCF model with mesh assets."""
    spec = mujoco.MjSpec.from_file(str(xml_path))
    assets = {}
    for f in assets_dir.iterdir():
        if f.is_file():
            assets[f.name] = f.read_bytes()
    spec.assets = assets
    model = spec.compile()
    data = mujoco.MjData(model)
    return model, data


def get_joint_info(model):
    """Get qpos indices and joint limits for the lower-body joints."""
    joint_ids = []
    qpos_ids = []
    lower = []
    upper = []
    for name in LOWER_BODY_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        joint_ids.append(jid)
        qpos_adr = model.jnt_qposadr[jid]
        qpos_ids.append(qpos_adr)
        lower.append(model.jnt_range[jid, 0])
        upper.append(model.jnt_range[jid, 1])
    return joint_ids, qpos_ids, np.array(lower), np.array(upper)


def get_foot_geom_info(model, foot_geom_count, is_capsule=True):
    """Get foot geom local points in the ankle_roll_link frame.

    For capsules (G1): each geom produces 2 endpoints (14 points per foot).
    For spheres (X2): each geom produces 1 point (12 points per foot).
    """
    foot_geoms = {"left": [], "right": []}
    for side in ("left", "right"):
        for i in range(1, foot_geom_count + 1):
            name = f"{side}_foot{i}_collision"
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            pos = model.geom_pos[gid].copy()
            if is_capsule:
                quat = model.geom_quat[gid].copy()      # [4] (wxyz)
                half_len = model.geom_size[gid, 1]       # scalar
                axis = np.array([0.0, 0.0, 1.0])
                rot = _quat_rotate(quat, axis)
                foot_geoms[side].append(pos - rot * half_len)
                foot_geoms[side].append(pos + rot * half_len)
            else:
                foot_geoms[side].append(pos)
        foot_geoms[side] = np.array(foot_geoms[side])

    left_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    right_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    return foot_geoms, left_body, right_body


def _quat_rotate(quat, vec):
    """Rotate vec by quaternion (wxyz convention)."""
    w, x, y, z = quat
    t = 2.0 * np.cross([x, y, z], vec)
    return vec + w * t + np.cross([x, y, z], t)


def compute_foot_ground_z(data, body_id, local_points):
    """Compute world-frame z for local_points on the given body."""
    pos = data.xpos[body_id]                 # [3]
    mat = data.xmat[body_id].reshape(3, 3)   # [3, 3]
    world_pts = local_points @ mat.T + pos   # [N, 3]
    return world_pts[:, 2]


def compute_foot_world_xy(data, body_id, local_points):
    """Compute world-frame xy for local_points on the given body."""
    pos = data.xpos[body_id]
    mat = data.xmat[body_id].reshape(3, 3)
    world_pts = local_points @ mat.T + pos
    return world_pts[:, :2]


def compute_foot_centroid(data, left_body, right_body, left_local, right_local):
    """Compute world-frame XY centroid of all foot geom points."""
    left_xy = compute_foot_world_xy(data, left_body, left_local)
    right_xy = compute_foot_world_xy(data, right_body, right_local)
    all_xy = np.concatenate([left_xy, right_xy], axis=0)
    return np.mean(all_xy, axis=0)  # [2]


def get_pelvis_body_id(model):
    """Get body ID for pelvis."""
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")


def compute_posture_for_height(
    model,
    data,
    target_z,
    qpos_ids,
    joint_lower,
    joint_upper,
    foot_geoms,
    left_ankle_body,
    right_ankle_body,
    pelvis_body_id,
    q_init,
    align_offset,
):
    """Find lower-body joint angles that place pelvis at target_z with feet on ground."""
    left_local = foot_geoms["left"]
    right_local = foot_geoms["right"]

    def objective(q):
        # Set lower-body joint angles.
        for i, qp in enumerate(qpos_ids):
            data.qpos[qp] = q[i]
        # Set pelvis height (free joint z is qpos[2]).
        data.qpos[2] = target_z
        # Set free joint orientation to identity (upright).
        data.qpos[3] = 1.0  # qw
        data.qpos[4] = 0.0  # qx
        data.qpos[5] = 0.0  # qy
        data.qpos[6] = 0.0  # qz

        mujoco.mj_forward(model, data)

        # Foot z-position error: all foot geom points should be at z=0.
        left_z = compute_foot_ground_z(data, left_ankle_body, left_local)
        right_z = compute_foot_ground_z(data, right_ankle_body, right_local)
        foot_error = np.sum(left_z**2) + np.sum(right_z**2)

        # Torso orientation error: keep upright.
        pelvis_mat = data.xmat[pelvis_body_id].reshape(3, 3)
        tilt_error = pelvis_mat[2, 0] ** 2 + pelvis_mat[2, 1] ** 2

        # Pelvis-foot alignment: pelvis xy should be at the foot polygon centroid.
        left_xy = compute_foot_world_xy(data, left_ankle_body, left_local)
        right_xy = compute_foot_world_xy(data, right_ankle_body, right_local)
        all_xy = np.concatenate([left_xy, right_xy], axis=0)
        foot_centroid = np.mean(all_xy, axis=0)
        pelvis_xy = data.xpos[pelvis_body_id, :2]
        align_error = np.sum((pelvis_xy - (foot_centroid + align_offset)) ** 2)

        # Regularization: stay close to initial guess.
        reg = 0.01 * np.sum((q - q_init) ** 2)

        # Symmetry penalty: left and right should be mirror images.
        sym_error = 0.0
        sym_error += (q[0] - q[6]) ** 2   # hip_pitch
        sym_error += (q[1] + q[7]) ** 2   # hip_roll (negate)
        sym_error += (q[2] + q[8]) ** 2   # hip_yaw (negate)
        sym_error += (q[3] - q[9]) ** 2   # knee
        sym_error += (q[4] - q[10]) ** 2  # ankle_pitch
        sym_error += (q[5] + q[11]) ** 2  # ankle_roll (negate)

        # Sagittal plane constraint: strongly penalize hip_roll and hip_yaw
        # to keep legs in the sagittal plane (no sideways splay).
        sagittal_penalty = 0.0
        sagittal_penalty += q[1] ** 2 + q[7] ** 2   # hip_roll L/R
        sagittal_penalty += q[2] ** 2 + q[8] ** 2   # hip_yaw L/R

        # Prefer slight knee bend over fully straight legs.
        knee_slack = 0.0
        knee_slack += max(0, 0.05 - q[3]) ** 2   # left knee
        knee_slack += max(0, 0.05 - q[9]) ** 2   # right knee

        return (1000.0 * foot_error + 100.0 * tilt_error + 100.0 * align_error
                + 0.01 * reg + 1.0 * sym_error + 200.0 * sagittal_penalty
                + 50.0 * knee_slack)

    bounds = list(zip(joint_lower, joint_upper))

    result = minimize(
        objective,
        x0=q_init,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    return result.x, result.fun


def compute_postures(model, data, qpos_ids, joint_lower, joint_upper,
                     foot_geoms, left_ankle_body, right_ankle_body,
                     pelvis_body_id, target_heights, align_offset):
    """Compute postures for all target heights."""
    # Initial guess: HOME_KEYFRAME values for lower body.
    # hip_pitch=-0.1, knee=0.3, ankle_pitch=-0.2 (same for G1 and X2).
    q_home = np.array([
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,   # left leg
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,   # right leg (mirror)
    ])
    q_home = np.clip(q_home, joint_lower, joint_upper)

    pts_per_foot = len(foot_geoms["left"])
    print(f"Model loaded: {model.nq} qpos, {model.nv} qvel")
    print(f"Lower-body joints: {len(LOWER_BODY_JOINTS)}")
    print(f"Foot geoms: {pts_per_foot} points per foot")
    print()

    results = {}
    q_prev = q_home.copy()

    for target_z in target_heights:
        # Try warm-start from previous, HOME_KEYFRAME, and random restarts.
        candidates = []
        for q_init_try in [q_prev, q_home]:
            q_try, cost_try = compute_posture_for_height(
                model, data, target_z,
                qpos_ids, joint_lower, joint_upper,
                foot_geoms, left_ankle_body, right_ankle_body,
                pelvis_body_id,
                q_init_try,
                align_offset,
            )
            candidates.append((q_try, cost_try))
        # Random restarts to escape local minima.
        rng = np.random.RandomState(int(target_z * 1000))
        for _ in range(5):
            q_rand = rng.uniform(joint_lower, joint_upper)
            q_try, cost_try = compute_posture_for_height(
                model, data, target_z,
                qpos_ids, joint_lower, joint_upper,
                foot_geoms, left_ankle_body, right_ankle_body,
                pelvis_body_id,
                q_rand,
                align_offset,
            )
            candidates.append((q_try, cost_try))
        # Pick the best.
        q_opt, cost = min(candidates, key=lambda x: x[1])

        # Verify: compute actual foot z and pelvis z.
        for i, qp in enumerate(qpos_ids):
            data.qpos[qp] = q_opt[i]
        data.qpos[2] = target_z
        data.qpos[3] = 1.0
        data.qpos[4] = 0.0
        data.qpos[5] = 0.0
        data.qpos[6] = 0.0
        mujoco.mj_forward(model, data)

        left_z = compute_foot_ground_z(data, left_ankle_body, foot_geoms["left"])
        right_z = compute_foot_ground_z(data, right_ankle_body, foot_geoms["right"])
        pelvis_actual_z = data.xpos[pelvis_body_id, 2]

        # Build posture dict.
        posture = {}
        for i, name in enumerate(LOWER_BODY_JOINTS):
            posture[name] = round(float(q_opt[i]), 4)

        results[round(target_z, 3)] = posture

        print(f"Height {target_z:.3f}m:")
        print(f"  pelvis_actual_z = {pelvis_actual_z:.4f}m")
        print(f"  foot_z range = [{left_z.min():.4f}, {left_z.max():.4f}] / [{right_z.min():.4f}, {right_z.max():.4f}]")
        print(f"  cost = {cost:.6f}")
        print(f"  joints = {posture}")
        print()

        q_prev = q_opt.copy()

    return results


def format_postures(results):
    """Format HEIGHT_POSTURES dict as a Python source string."""
    lines = [
        '"""Computed height postures from compute_height_postures.py."""',
        "",
        "",
        "HEIGHT_POSTURES = {",
    ]
    for h, posture in results.items():
        lines.append(f"    {h}: {{")
        for name, val in posture.items():
            lines.append(f'        "{name}": {val},')
        lines.append("    },")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def load_postures(path):
    """Load a HEIGHT_POSTURES dict from a Python file."""
    text = Path(path).read_text()
    ns = {}
    exec(text, ns)
    if "HEIGHT_POSTURES" in ns:
        return ns["HEIGHT_POSTURES"]
    # Fallback: grab the first dict value.
    for v in ns.values():
        if isinstance(v, dict):
            return v
    raise ValueError(f"No dict found in {path}")


def show_in_viewer(model, data, results, qpos_ids, foot_geoms, left_ankle_body, right_ankle_body, align_offset):
    """Open MuJoCo viewer and cycle through each height posture."""
    heights = sorted(results.keys())
    print(f"\nViewer: cycling through {len(heights)} heights. Press Ctrl+C to quit.")
    print("  Each height is shown for 3 seconds.\n")

    mat = np.eye(3).flatten()
    red = np.array([1.0, 0.0, 0.0, 0.8], dtype=np.float32)
    blue = np.array([0.0, 0.0, 1.0, 0.8], dtype=np.float32)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        try:
            while True:
                for h in heights:
                    posture = results[h]
                    # Set lower-body joints.
                    for i, qp in enumerate(qpos_ids):
                        data.qpos[qp] = posture[LOWER_BODY_JOINTS[i]]
                    # Set pelvis height and upright orientation.
                    data.qpos[2] = h
                    data.qpos[3] = 1.0
                    data.qpos[4] = 0.0
                    data.qpos[5] = 0.0
                    data.qpos[6] = 0.0
                    mujoco.mj_forward(model, data)

                    # Compute foot centroid.
                    centroid_xy = compute_foot_centroid(
                        data, left_ankle_body, right_ankle_body,
                        foot_geoms["left"], foot_geoms["right"],
                    )
                    # Red sphere: foot centroid on the ground plane.
                    centroid_pos = np.array([centroid_xy[0], centroid_xy[1], 0.0])
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[0],
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([0.02, 0.0, 0.0]),
                        centroid_pos, mat, red,
                    )
                    # Blue sphere: alignment target (centroid + offset).
                    target_xy = centroid_xy + align_offset
                    target_pos = np.array([target_xy[0], target_xy[1], 0.0])
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[1],
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([0.02, 0.0, 0.0]),
                        target_pos, mat, blue,
                    )
                    viewer.user_scn.ngeom = 2

                    viewer.sync()
                    print(f"  Height = {h:.3f}m  centroid = [{centroid_xy[0]:.3f}, {centroid_xy[1]:.3f}]  target = [{target_xy[0]:.3f}, {target_xy[1]:.3f}]")
                    time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nViewer closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=["g1", "x2"], default="g1",
                        help="Robot model to use (default: g1)")
    parser.add_argument("--show", action="store_true", help="Open MuJoCo viewer to inspect postures")
    parser.add_argument("--postures-file", type=str, help="Load existing HEIGHT_POSTURES dict from a Python file instead of computing")
    parser.add_argument("--save", type=str, help="Save HEIGHT_POSTURES dict to a Python file")
    args = parser.parse_args()

    cfg = ROBOT_CONFIGS[args.robot]

    if args.postures_file:
        results = load_postures(args.postures_file)
        model, data = load_model(cfg["xml"], cfg["assets_dir"])
        _, qpos_ids, _, _ = get_joint_info(model)
        foot_geoms, left_ankle_body, right_ankle_body = get_foot_geom_info(
            model, cfg["foot_geom_count"], cfg["is_capsule"])
        print(f"Loaded {len(results)} height postures from {args.postures_file}")
        show_in_viewer(model, data, results, qpos_ids, foot_geoms, left_ankle_body, right_ankle_body, cfg["align_offset"])
    else:
        model, data = load_model(cfg["xml"], cfg["assets_dir"])
        _, qpos_ids, joint_lower, joint_upper = get_joint_info(model)
        foot_geoms, left_ankle_body, right_ankle_body = get_foot_geom_info(
            model, cfg["foot_geom_count"], cfg["is_capsule"])
        pelvis_body_id = get_pelvis_body_id(model)

        results = compute_postures(
            model, data, qpos_ids, joint_lower, joint_upper,
            foot_geoms, left_ankle_body, right_ankle_body,
            pelvis_body_id, cfg["target_heights"], cfg["align_offset"],
        )

        text = format_postures(results)
        if args.save:
            Path(args.save).write_text(text)
            print(f"Saved to {args.save}")
        else:
            print(text)
        if args.show:
            show_in_viewer(model, data, results, qpos_ids, foot_geoms, left_ankle_body, right_ankle_body, cfg["align_offset"])
