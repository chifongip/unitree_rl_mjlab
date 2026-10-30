"""Paired walk-to-stand evaluation; writes metrics, never trains the policy."""
import argparse
import contextlib
from dataclasses import asdict
import io
import json
import random
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.lab_api.math import quat_apply
from src.tasks.locomanipulation.config.x2.env_cfgs import agibot_x2_locomanipulation_transition_flat_env_cfg
from src.tasks.locomanipulation.config.x2.rl_cfg import agibot_x2_locomanipulation_ppo_runner_cfg
from src.tasks.locomanipulation.rl import X2_LocomanipulationOnPolicyRunner
if __package__:
  from .transition_metrics import foot_transition_metrics, compare_loaded_transitions, flight_metrics, stopping_step_metrics, compare_stopping_steps
  from .play import _load_params_overrides, _restore_torque_compensation
else:
  from transition_metrics import foot_transition_metrics, compare_loaded_transitions, flight_metrics, stopping_step_metrics, compare_stopping_steps
  from play import _load_params_overrides, _restore_torque_compensation


def _is_stopped(velocity, yaw_velocity, contact):
  """Measure physical stopping independently of the training reward phase."""
  return (velocity.norm(dim=-1) < .1) & (yaw_velocity.abs() < .1) & contact.all(dim=-1)


def _settle_time(stopped, dt, hold_s=.5):
  """Return the start of the first sustained stop, or None."""
  frames = max(1, round(hold_s / dt))
  if stopped.numel() < frames:
    return None
  starts = torch.where(stopped.unfold(0, frames, 1).all(dim=-1))[0]
  return float(starts[0] * dt) if len(starts) else None


def evaluate(checkpoint, heights=(.64, .5, .3), phases=8, cases=None, seed=42,
             scenario="walk-stop", load_n=None, carry_poses=False, save_traces=False):
  """Run held-out evaluation without changing the training process RNG/threads."""
  threads = torch.get_num_threads()
  cpu_rng, numpy_rng, python_rng = torch.get_rng_state(), np.random.get_state(), random.getstate()
  cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
  try:
    return _evaluate(checkpoint, heights, phases, cases, seed, scenario, load_n, carry_poses, save_traces)
  finally:
    torch.set_num_threads(threads)
    torch.set_rng_state(cpu_rng)
    np.random.set_state(numpy_rng)
    random.setstate(python_rng)
    if cuda_rng is not None:
      torch.cuda.set_rng_state_all(cuda_rng)
    import gc
    gc.collect()


def _evaluate(checkpoint, heights, phases, cases, seed, scenario="walk-stop", load_n=None, carry_poses=False, save_traces=False):
  """Evaluate a fixed grid with the actor interface used in deployment."""
  if scenario not in ("walk-stop", "stand-walk-stop") or phases < 1 or not heights:
    raise ValueError("Invalid scenario, phases, or empty height grid")
  if load_n is not None and (not np.isfinite(load_n) or load_n < 0):
    raise ValueError("load_n must be a finite nonnegative downward force")
  torch.set_num_threads(1)
  if cases is None:
    cases = [(v, 0., 0.) for v in (.2, .5, 1., -.2, -.35, -.5)]
    cases += [(0., v, 0.) for v in (.2, .35, .5, -.2, -.35, -.5)]
    cases += [(0., 0., v) for v in (-.5, .5)]
  cfg = agibot_x2_locomanipulation_transition_flat_env_cfg(play=True)
  _restore_torque_compensation(cfg, _load_params_overrides(Path(checkpoint).parent / "params"))
  if not cases:
    raise ValueError("Command grid must not be empty")
  if carry_poses:
    upper = cfg.actions["upper_body_motion"]
    upper.fixed_upper_body_pose = None
    upper.motion_file = str(Path(__file__).resolve().parents[1] / "src/assets/data/x2/bones_seed/carry_poses_x2.pkl")
    upper.locomotion_motion_file = None
    upper.pose_only = True
    upper.default_pose_ratio = 0.
  if load_n is not None:
    cfg.events["hand_force"].params["constant_force"] = {"x": 0., "y": 0., "z": -load_n}
    cfg.events["hand_force"].params["no_force_ratio"] = 0.
  cfg.scene.num_envs = phases * len(cases)
  cfg.seed = seed
  for sensor in cfg.scene.sensors:
    if sensor.name == "feet_ground_contact":
      sensor.history_length = cfg.decimation
  cfg.events.pop("randomize_terrain", None)
  cfg.commands["twist"].heading_command = False
  cfg.commands["twist"].ranges.heading = None
  with contextlib.redirect_stdout(io.StringIO()):
    raw = ManagerBasedRlEnv(cfg, device="cpu")
    env = RslRlVecEnvWrapper(raw)
    runner_cfg = asdict(agibot_x2_locomanipulation_ppo_runner_cfg())
    runner_cfg["algorithm"]["symmetry_cfg"] = False
    runner = X2_LocomanipulationOnPolicyRunner(env, runner_cfg, device="cpu")
    runner.load(str(checkpoint), load_cfg={"actor": True}, map_location="cpu")
    policy = runner.get_inference_policy(device="cpu")
  term = raw.command_manager.get_term("twist")
  dt = raw.step_dt
  stops = torch.tensor([round((2 + i * .6 / phases) / dt) for _ in cases for i in range(phases)])
  starts = torch.zeros_like(stops)
  if scenario == "stand-walk-stop":
    starts = stops + round(1. / dt)
    stops = starts + round(5. / dt)
  duration_steps = round(6.6 / dt) if scenario == "walk-stop" else int(stops.max()) + round(4. / dt)
  commands = torch.tensor(cases).repeat_interleave(phases, dim=0)
  def scheduled(env_ids=None):
    active = (raw.episode_length_buf >= starts) & (raw.episode_length_buf < stops)
    term.vel_command_b[:] = commands * active[:, None]
    term.is_heading_env[:] = False
    if term.stop_state is not None:
      term.stop_state.reconcile()
  term._update_command = scheduled
  rows = []
  try:
    for height in heights:
      raw.command_manager.get_term("base_height").cfg.fixed_height = height
      torch.manual_seed(seed)  # Pair initial poses/randomization across heights and checkpoints.
      obs, _ = env.reset()
      state = term.state()
      q_ids = state.joint_ids
      yaw_ids = state.hip_yaw_ids
      upper = raw.action_manager.get_term("upper_body_motion")
      carry_ids = upper._joint_ids
      carry_names = [raw.scene["robot"].joint_names[j] for j in carry_ids]
      pose_targets = upper._pose_target.clone()
      trace = []
      failed = torch.zeros(raw.num_envs, dtype=torch.bool)
      with torch.no_grad():
        for step in range(duration_steps + 1):
          if step:
            obs, _, done, _ = env.step(policy(obs))
            failed |= done.bool()
          r = raw.scene["robot"].data
          error, index = state.geometry()
          q_error = r.joint_pos[:, q_ids] - state.targets[index]
          contact = raw.scene["feet_ground_contact"].data.found > 0
          angular = r.body_link_ang_vel_w[:, state.body_ids]
          offsets = quat_apply(r.body_link_quat_w[:, state.body_ids], state.offsets[None].expand(raw.num_envs, -1, -1))
          sole_velocity = r.body_link_lin_vel_w[:, state.body_ids] + torch.linalg.cross(angular, offsets)
          slip = (sole_velocity[..., :2].norm(dim=-1) * contact).sum(1)
          rotation = (angular[..., 2].abs() * contact).sum(1)
          centers = r.body_link_pos_w[:, state.body_ids] + offsets
          axis = torch.zeros_like(offsets)
          axis[..., 0] = 1
          axis = quat_apply(r.body_link_quat_w[:, state.body_ids], axis)
          world_yaw = torch.atan2(axis[..., 1], axis[..., 0])
          carry_error = (r.joint_pos[:, carry_ids] - pose_targets).square().mean(1).sqrt()
          trace.append((r.joint_pos[:, q_ids].clone(), q_error.clone(), contact.clone(),
            r.root_link_pos_w[:, 2].clone(),
            _is_stopped(r.root_link_lin_vel_b[:, :2], r.root_link_ang_vel_b[:, 2], contact),
            r.root_link_lin_vel_b[:, :2].clone(), error.clone(), slip.clone(), rotation.clone(), r.root_link_ang_vel_b[:, 2].clone(),
            centers.clone(), world_yaw.clone(), sole_velocity.clone(), angular.clone(), carry_error.clone(),
            state.foot_stage.clone(), state.finished.clone(), state.deadline.clone(), state.age.clone(),
            state.reference.clone(), state.progress.clone(), state.credited_progress.clone(),
            r.joint_vel[:, q_ids].clone(), r.root_link_lin_vel_w[:, 2].clone(),
            r.root_link_pos_w.clone(), r.root_link_quat_w.clone(),
            raw.scene["feet_ground_contact"].data.force_history.norm(dim=-1).amax(dim=-1).clone()))
      q, qe, contact, z, stopped, velocity, error, slip, rotation, yaw_velocity = [torch.stack([t[i] for t in trace]) for i in range(10)]
      centers, world_yaw, sole_v, angular, carry_error = [torch.stack([t[i] for t in trace]) for i in range(10, 15)]
      stages, finished, deadline, age = [torch.stack([t[i] for t in trace]) for i in range(15, 19)]
      references, progress, credited, qvel = [torch.stack([t[i] for t in trace]) for i in range(19, 23)]
      vertical_velocity = torch.stack([t[23] for t in trace])
      position, quaternion, force_peak = [torch.stack([t[j] for t in trace]) for j in range(24, 27)]
      for i in range(raw.num_envs):
        start = int(stops[i])
        walk_start = int(starts[i])
        walk_measure_start = walk_start + round(.5 / dt)
        touchdowns = [t for t in range(start + 1, len(trace)) if (contact[t, i] & ~contact[t - 1, i]).any()]
        last = touchdowns[-1] if touchdowns else start
        end = last + round(.5 / dt) + 1
        motion = (q[last:min(end, len(trace)), i, yaw_ids] - q[last, i, yaw_ids]).abs().max().item()
        row = dict(height=height, command=commands[i].tolist(), phase=i % phases,
          transition_s=term.cfg.transition_s, transition_guidance="supported-reference-blend-v1", swing_duration=state.swing_duration,
          reference_progress=progress[-1, i].tolist(),
          reference_complete=bool((credited[-1, i] >= 1).all()),
          seed=seed, scenario=scenario, load_n_per_hand=load_n, carry_poses=carry_poses,
          carry_joint_names=carry_names, carry_pose_target=pose_targets[i].tolist(),
          walk_start_s=walk_start*dt, stop_command_s=start*dt,
          carry_pose_rms=carry_error[walk_measure_start:start, i].mean().item(),
          failed=bool(failed[i]), censored=end > len(trace),
          success=bool(stopped[-round(.5/dt):, i].all()) and not bool(failed[i]) and end <= len(trace),
          settle_s=_settle_time(stopped[start:, i], dt),
          post_touchdown_hip_yaw_rad=motion, touchdown_posture_rms=qe[last, i].square().mean().sqrt().item(),
          last_touchdown_s=(last-start)*dt,
          contact_slip_m=slip[start:, i].sum().item()*dt,
          contact_rotation_rad=rotation[start:, i].sum().item()*dt,
          final_height_error_m=abs(z[-1, i].item() - height),
          walking_velocity_mae=(velocity[walk_measure_start:start, i] - commands[i, :2]).norm(dim=1).mean().item(),
          walking_yaw_mae=(yaw_velocity[walk_measure_start:start, i] - commands[i, 2]).abs().mean().item(),
          final_placement_error_m=error[-1, i, :, :2].norm(dim=-1).max().item(),
          final_heading_error_rad=error[-1, i, :, 2].abs().max().item())
        row["stopping_steps"] = stopping_step_metrics(
          contact[:, i], sole_v[:, i], velocity[:, i], position[:, i], quaternion[:, i],
          force_peak[:, i], start, dt)
        if save_traces:
          row["stopping_trace"] = dict(dt=dt, contact=contact[:, i].tolist(),
            sole_velocity=sole_v[:, i].tolist(), position=position[:, i].tolist(),
            quaternion=quaternion[:, i].tolist(), force_peak_n=force_peak[:, i].tolist())
          row["reference_trace"] = dict(dt=dt, joint_names=state.names,
            position=q[:, i].tolist(), reference=references[:, i].tolist(),
            progress=progress[:, i].tolist(), velocity=qvel[:, i].tolist())
        row["transition_diagnostics"] = {}
        for direction, begin, end_frame in (
          ("stand_to_walk", walk_start, start), ("walk_to_stand", start, len(trace))):
          measurement_end = min(end_frame, begin + round(1.5 / dt))
          row.setdefault("flight", {})[direction] = flight_metrics(
            contact[begin:end_frame, i], z[begin:end_frame, i],
            vertical_velocity[begin:end_frame, i], height, dt)
          hip_ids = state.hip_yaw_ids + [state.names.index(f"{side}_hip_roll_joint")
                                       for side in ("left", "right")]
          velocity_window = qvel[begin:measurement_end, i, :][:, hip_ids]
          acceleration = velocity_window.diff(dim=0) / dt
          jerk = acceleration.diff(dim=0) / dt
          row.setdefault("hip_motion", {})[direction] = dict(
            joint_names=[state.names[j] for j in hip_ids],
            peak_velocity_rad_s=velocity_window.abs().amax(0).tolist(),
            peak_acceleration_rad_s2=acceleration.abs().amax(0).tolist(),
            peak_jerk_rad_s3=jerk.abs().amax(0).tolist())
          reference_done = torch.where((credited[begin:end_frame, i] >= 1).all(1))[0]
          completions = torch.where(finished[begin:end_frame, i])[0]
          completion = begin + int(completions[0]) if len(completions) else None
          row["transition_diagnostics"][direction] = dict(
            reference_completion_s=None if not len(reference_done) else float(reference_done[0] * dt),
            duration_s=None if completion is None else float(age[completion, i]),
            reason=None if completion is None else ("deadline" if deadline[completion, i] else "landed"),
            stage_changes=[[
              dict(time_s=(t-begin)*dt, stage=int(stages[t, i, foot]))
              for t in range(begin, end_frame)
              if t == begin or stages[t, i, foot] != stages[t-1, i, foot]] for foot in range(2)])
          row[direction] = [foot_transition_metrics(
            contact[:, i, foot], centers[:, i, foot], world_yaw[:, i, foot],
            sole_v[:, i, foot], angular[:, i, foot], q[:, i, yaw_ids[foot]],
            begin, end_frame, dt, final_touchdown=direction == "walk_to_stand") for foot in range(2)]
        delays = [f["liftoff_s"] for f in row["stand_to_walk"] if f["liftoff_s"] is not None]
        row["start_delay_s"] = min(delays) if delays else None
        rows.append(row)
      print(f"Evaluated {len(cases) * phases} trials at {height:.2f} m", flush=True)
  finally:
    env.close()
  return dict(schema_version=6, checkpoint=str(checkpoint), seed=seed, trials=rows,
    summary=dict(reference_completion_rate=float(np.mean([r["reference_complete"] for r in rows])),
      complete_transition_rate=float(np.mean([r["success"] and r["reference_complete"] for r in rows])),
      success_rate=float(np.mean([r["success"] for r in rows])),
      fall_rate=float(np.mean([r["failed"] for r in rows])),
      median_post_touchdown_hip_yaw_rad=float(np.median([r["post_touchdown_hip_yaw_rad"] for r in rows])),
      max_height_error_m=max(r["final_height_error_m"] for r in rows)))


def compare(baseline, candidate, stopping_steps=False):
  """Evaluate fixed acceptance thresholds on paired, uncensored trials."""
  a, b = baseline["trials"], candidate["trials"]
  if not a or baseline["seed"] != candidate["seed"] or len(a) != len(b):
    raise ValueError("Evaluation grids/seeds must match")
  if baseline.get("schema_version") != candidate.get("schema_version"):
    raise ValueError("Evaluation schema versions must match")
  for x, y in zip(a, b):
    keys = ("seed", "scenario", "load_n_per_hand", "carry_poses", "carry_joint_names",
            "carry_pose_target", "walk_start_s", "stop_command_s", "transition_s", "transition_guidance", "swing_duration")
    if any(x.get(k) != y.get(k) for k in keys):
      raise ValueError("Evaluation scenarios, loads, poses, and timing must match")
    if (x["height"], x["command"], x["phase"]) != (y["height"], y["command"], y["phase"]):
      raise ValueError("Evaluation grids must match")
  if stopping_steps:
    return compare_stopping_steps(a, b)
  if a[0].get("scenario") == "stand-walk-stop":
    if baseline.get("schema_version") != 6:
      raise ValueError("Loaded transition reports must use schema version 6; regenerate both reports")
    if any(r.get("load_n_per_hand") is None for r in a):
      raise ValueError("Loaded transition comparisons require explicit loads")
    return compare_loaded_transitions(a, b)
  def median(rows, key):
    return float(np.median([r[key] for r in rows]))
  def success_rate(rows):
    return float(np.mean([r["success"] and not r["failed"] and not r["censored"] for r in rows]))
  candidate_success = success_rate(b)
  result = {
    "no_censored_trials": not any(r["censored"] for r in a + b),
    "stopping_success": candidate_success >= .9,
    "stopping_success_preserved": candidate_success >= success_rate(a),
    "height_accuracy": max(r["final_height_error_m"] for r in b) <= .03,
    "placement_accuracy": max(r["final_placement_error_m"] for r in b) <= .03,
    "heading_accuracy": max(r["final_heading_error_rad"] for r in b) <= np.deg2rad(10),
    "hip_yaw_improved": median(b, "post_touchdown_hip_yaw_rad") <= .5 * median(a, "post_touchdown_hip_yaw_rad"),
    "posture_improved": median(b, "touchdown_posture_rms") < median(a, "touchdown_posture_rms"),
    "slip_not_increased": median(b, "contact_slip_m") <= median(a, "contact_slip_m"),
    "fall_rate_preserved": bool(np.mean([r["failed"] for r in b]) <= np.mean([r["failed"] for r in a]) + .01),
    "walking_preserved": median(b, "walking_velocity_mae") <= 1.1 * median(a, "walking_velocity_mae"),
    "turning_preserved": median(b, "walking_yaw_mae") <= 1.1 * median(a, "walking_yaw_mae"),
    "no_delayed_final_step": median(b, "last_touchdown_s") <= median(a, "last_touchdown_s") + .3,
  }
  # NumPy comparisons produce np.bool_, which JSON cannot serialize.
  result = {name: bool(value) for name, value in result.items()}
  return {"passed": all(result.values()), "criteria": result}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--heights", nargs="+", type=float, default=[.64, .5, .3])
  parser.add_argument("--phases", type=int, default=8)
  parser.add_argument("--baseline", type=Path, help="Compare against a paired evaluation JSON")
  parser.add_argument("--smoke", action="store_true", help="Only forward 0.5 m/s")
  parser.add_argument("--scenario", choices=["walk-stop", "stand-walk-stop"], default="walk-stop")
  parser.add_argument("--loads-n", nargs="+", type=float, help="Downward force per hand, e.g. 0 20 40")
  parser.add_argument("--carry-poses", action="store_true")
  parser.add_argument("--seeds", nargs="+", type=int, default=[42])
  parser.add_argument("--save-traces", action="store_true", help="Include joint/reference/progress trajectories")
  parser.add_argument("--stopping-comparison", action="store_true",
                      help="Compare physical stopping and impacts instead of exact stance completion")
  args = parser.parse_args()
  if args.stopping_comparison and not args.baseline:
    parser.error("--stopping-comparison requires --baseline")
  if args.phases < 1:
    parser.error("phases must be positive")
  cases = [(.5, 0., 0.)] if args.smoke else None
  if cases is None and args.scenario == "stand-walk-stop":
    cases = [(.5, 0., 0.), (-.3, 0., 0.), (0., .2, 0.), (0., -.2, 0.), (0., 0., .5), (0., 0., -.5)]
  reports = [evaluate(args.checkpoint, args.heights, args.phases, cases, seed,
                      args.scenario, load, args.carry_poses, args.save_traces)
             for seed in args.seeds for load in (args.loads_n or [None])]
  report = reports[0]
  if len(reports) > 1:
    rows = [row for result in reports for row in result["trials"]]
    report = dict(schema_version=6, checkpoint=str(args.checkpoint), seed=args.seeds,
                  trials=rows, summary=dict(
                    reference_completion_rate=float(np.mean([r["reference_complete"] for r in rows])),
                    complete_transition_rate=float(np.mean([r["success"] and r["reference_complete"] for r in rows])),
                    success_rate=float(np.mean([r["success"] for r in rows])),
                    fall_rate=float(np.mean([r["failed"] for r in rows])),
                    max_height_error_m=max(r["final_height_error_m"] for r in rows)))
  if args.baseline:
    report["comparison"] = compare(json.loads(args.baseline.read_text()), report, stopping_steps=args.stopping_comparison)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
  main()
