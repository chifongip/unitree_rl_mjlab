"""Per-foot contact events and paired loaded-transition acceptance metrics."""
import numpy as np


def foot_transition_metrics(contact, position, yaw, velocity, angular, hip_yaw, start, end, dt, final_touchdown=True):
  """Measure a single foot over [start, end), preserving missing events as null."""
  contact = contact.bool()
  lifts = [i for i in range(max(1, start), end) if contact[i - 1] and not contact[i]]
  touches = [i for i in range(max(1, start), end) if not contact[i - 1] and contact[i]]
  first = lifts[0] if lifts else None
  last = (touches[-1] if final_touchdown else touches[0]) if touches else None
  def integrals(a, b):
    c = contact[a:b]
    return (float((velocity[a:b, :2].norm(dim=-1) * c).sum() * dt),
            float((angular[a:b, 2].abs() * c).sum() * dt))
  # Starting ends after the first landing plus its stabilization window;
  # stopping includes the final landing and the remainder of the stop.
  metric_end = end if final_touchdown or last is None else min(end, last + round(.5 / dt) + 1)
  slide, rotate = integrals(start, metric_end)
  pre = integrals(start, first) if first is not None and contact[start] else (None, None)
  post_end = None if last is None else last + round(.5 / dt) + 1
  post = integrals(last, post_end) if post_end is not None and post_end <= end else (None, None)
  return dict(
    continuous_contact=bool(contact[start:end].all()),
    started_airborne=not bool(contact[start]), missing_liftoff=first is None,
    missing_touchdown=last is None, post_touchdown_censored=last is not None and post_end > end,
    liftoff_s=None if first is None else (first - start) * dt,
    touchdown_s=None if last is None else (last - start) * dt,
    contact_slide_m=slide, contact_yaw_rad=rotate,
    pre_liftoff_slide_m=pre[0], pre_liftoff_yaw_rad=pre[1],
    post_touchdown_slide_m=post[0], post_touchdown_yaw_rad=post[1],
    post_touchdown_hip_yaw_rad=None if post[0] is None else float((hip_yaw[last:post_end] - hip_yaw[last]).abs().max()),
    liftoff_position=None if first is None else position[first].tolist(),
    touchdown_position=None if last is None else position[last].tolist(),
    touchdown_world_yaw=None if last is None else float(yaw[last]),
  )


def flight_metrics(contact, height, vertical_velocity, commanded_height, dt):
  """Measure unsupported flight separately from ordinary single-support steps."""
  return dict(flight_s=float((~contact.any(dim=-1)).sum() * dt),
              peak_height_above_command_m=float((height - commanded_height).clamp_min(0).max()),
              peak_vertical_speed_m_s=float(vertical_velocity.abs().max()))


def compare_loaded_transitions(a, b):
  """Compare each direction/load separately; missing events never count as zero."""
  groups = {}
  for load in sorted({r['load_n_per_hand'] for r in a}):
    aa = [r for r in a if r['load_n_per_hand'] == load]
    bb = [r for r in b if r['load_n_per_hand'] == load]
    criteria = {'uncorrupted_trials': not any(r['failed'] or r['censored'] for r in aa + bb)}
    for direction in ('stand_to_walk', 'walk_to_stand'):
      fa = [f for r in aa for f in r[direction]]
      fb = [f for r in bb for f in r[direction]]
      def stationary_support(f):
        return (direction == 'walk_to_stand' and f['continuous_contact']
                and f['contact_slide_m'] <= .03 and f['contact_yaw_rad'] <= np.deg2rad(10))

      def valid(f):
        if stationary_support(f):
          return True
        # A foot already in swing at the stop command need only land.
        needs_liftoff = direction == 'stand_to_walk' or not f['started_airborne']
        return not ((needs_liftoff and f['missing_liftoff']) or f['missing_touchdown']
                    or f['post_touchdown_censored']
                    or (direction == 'stand_to_walk' and f['started_airborne']))
      criteria[direction + '_flight_preserved'] = all(
        y['flight'][direction]['flight_s'] <= x['flight'][direction]['flight_s'] + 1e-8
        for x, y in zip(aa, bb))
      for key in ('peak_height_above_command_m', 'peak_vertical_speed_m_s'):
        criteria[direction + '_' + key + '_preserved'] = all(
          y['flight'][direction][key] <= x['flight'][direction][key] + max(.1 * x['flight'][direction][key], .01)
          for x, y in zip(aa, bb))
      criteria[direction + '_events_complete'] = all(valid(f) for f in fa + fb)
      for key, factor in (('contact_yaw_rad', .5), ('contact_slide_m', 1.)):
        va, vb = [f[key] for f in fa], [f[key] for f in fb]
        criteria[direction + '_' + key + '_median'] = np.median(vb) <= factor * np.median(va)
        criteria[direction + '_' + key + '_p95'] = np.percentile(vb, 95) <= np.percentile(va, 95)
      for key in ('pre_liftoff_slide_m', 'pre_liftoff_yaw_rad',
                  'post_touchdown_slide_m', 'post_touchdown_yaw_rad',
                  'post_touchdown_hip_yaw_rad'):
        # Missing events stay unavailable, rather than becoming zero. Validity
        # above admits only complete events or a stable continuous support foot.
        pairs = [(x[key], y[key]) for x, y in zip(fa, fb)
                 if x[key] is not None and y[key] is not None]
        criteria[direction + '_' + key + '_preserved'] = all(y <= x + 1e-8 for x, y in pairs)
    successful = [r for r in bb if r['success']]
    criteria['complete_transition_success'] = sum(r['success'] and r['reference_complete'] for r in bb) / len(bb) >= .95
    criteria['reference_completion'] = bool(successful) and sum(r['reference_complete'] for r in successful) / len(successful) >= .95
    criteria['final_placement_accuracy'] = all(r['final_placement_error_m'] <= .03 for r in bb)
    criteria['final_heading_accuracy'] = all(r['final_heading_error_rad'] <= np.deg2rad(10) for r in bb)
    criteria['no_additional_falls'] = sum(r['failed'] for r in bb) <= sum(r['failed'] for r in aa)
    criteria['stop_success_preserved'] = sum(r['success'] for r in bb) >= sum(r['success'] for r in aa)
    for key in ('walking_velocity_mae', 'walking_yaw_mae', 'final_height_error_m', 'carry_pose_rms'):
      # Protect each command/height stratum, not just the pooled mean.
      criteria[key] = all(y[key] <= x[key] + max(.1 * x[key], .01) for x, y in zip(aa, bb))
    for key in ('start_delay_s', 'settle_s'):
      criteria[key] = all(x[key] is not None and y[key] is not None and y[key] <= x[key] + .2 for x, y in zip(aa, bb))
    groups[str(load)] = {k: bool(v) for k, v in criteria.items()}
  return dict(passed=all(all(g.values()) for g in groups.values()), by_load=groups)


def stopping_step_metrics(contact, sole_velocity, velocity, position, quaternion, force_peak, start, dt):
  """Stopping impacts and sway; touchdown velocity uses the preceding policy frame."""
  import torch
  touches = contact[1:] & ~contact[:-1]
  times, feet = torch.where(touches[max(start - 1, 0):])
  times = times + max(start, 1)
  slow = torch.where(velocity[start:].norm(dim=-1) < .1)[0]
  brake = None if not len(slow) else start + int(slow[0])
  down = (-sole_velocity[times - 1, feet, 2]).clamp_min(0)
  w, x, y, z = quaternion[start:].unbind(-1)
  roll = torch.atan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
  pitch = torch.asin((2 * (w*y - z*x)).clamp(-1, 1))
  xy = position[start:, :2]
  # Range is translation invariant and captures fore/aft and lateral excursion.
  return dict(version=1, touchdown_count=len(times),
    braking_s=None if brake is None else (brake-start)*dt,
    post_braking_touchdowns=None if brake is None else int((times > brake).sum()),
    touchdown_times_s=((times-start)*dt).tolist(),
    touchdown_feet=feet.tolist(), downward_touchdown_speed_m_s=down.tolist(),
    peak_downward_touchdown_speed_m_s=float(down.max()) if len(down) else 0.,
    peak_contact_force_n=float(force_peak[start+1:].max()),
    roll_pitch_range_rad=[float(roll.max()-roll.min()), float(pitch.max()-pitch.min())],
    horizontal_sway_range_m=(xy.max(0).values-xy.min(0).values).tolist())


def compare_stopping_steps(baseline, candidate):
  """Compare matched grids without requiring a prescribed stance completion."""
  a, b = baseline, candidate
  if any(r.get('stopping_steps', {}).get('version') != 1 for r in a+b):
    raise ValueError('Regenerate both reports with stopping-step diagnostics version 1')
  criteria = dict(
    successful_stops=all(r['success'] and not r['failed'] and not r['censored'] for r in b),
    no_stopping_flight=all(r['flight']['walk_to_stand']['flight_s'] == 0 for r in b),
    fewer_post_braking_steps=(all(r['stopping_steps']['post_braking_touchdowns'] is not None for r in a+b)
      and np.mean([r['stopping_steps']['post_braking_touchdowns'] for r in b]) <
          np.mean([r['stopping_steps']['post_braking_touchdowns'] for r in a])),
    settling_preserved=all(x['settle_s'] is not None and y['settle_s'] is not None
      and y['settle_s'] <= x['settle_s'] + .2 for x,y in zip(a,b)),
    startup_preserved=all(x['start_delay_s'] is not None and y['start_delay_s'] is not None
      and y['start_delay_s'] <= x['start_delay_s'] + .2
      and y['flight']['stand_to_walk']['flight_s'] <= x['flight']['stand_to_walk']['flight_s']
      and all(np.max(y['hip_motion']['stand_to_walk'][key]) <=
              max(1.1 * np.max(x['hip_motion']['stand_to_walk'][key]), np.max(x['hip_motion']['stand_to_walk'][key]) + .01)
              for key in ('peak_velocity_rad_s', 'peak_acceleration_rad_s2', 'peak_jerk_rad_s3'))
      for x,y in zip(a,b)),
    walking_preserved=all(y[key] <= max(x[key]*1.1, x[key]+.01)
      for x,y in zip(a,b) for key in ('walking_velocity_mae','walking_yaw_mae')))
  for key in ('peak_downward_touchdown_speed_m_s', 'peak_contact_force_n',
              'roll_pitch_range_rad', 'horizontal_sway_range_m'):
    criteria[key + '_preserved'] = all(np.all(np.asarray(y['stopping_steps'][key]) <=
      np.asarray(x['stopping_steps'][key]) + 1e-6) for x,y in zip(a,b))
  criteria = {k:bool(v) for k,v in criteria.items()}
  return dict(passed=all(criteria.values()), criteria=criteria)
