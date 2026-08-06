#!/usr/bin/python3
"""Record one real-robot posture for each observed height interval.

The odometry height is only a provisional label.  The recorded named joint
positions can later be loaded into the X2 MuJoCo model to calculate a more
accurate height.

Before running, source both ROS 2 and AIMDK::

    source /opt/ros/humble/setup.bash
    source /home/ubuntu/aimdk/install/setup.bash
    /usr/bin/python3 scripts/record_height_postures.py

ROS Humble on this machine uses Python 3.10, so do not run this script through
the repository's Python 3.11 Conda environment.

Stop recording with Ctrl+C.  Each CSV row is flushed immediately so that
completed samples survive an interrupted recording session.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Any

import rclpy
from aimdk_msgs.msg import JointStateArray
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


JOINT_TOPICS = {
  "leg": "/aima/hal/joint/leg/state",
  "waist": "/aima/hal/joint/waist/state",
  "arm": "/aima/hal/joint/arm/state",
}
ODOMETRY_TOPIC = "/aima/mc/leg_odometry"


def timestamp_ns(stamp: Any) -> int:
  """Convert a ROS builtin_interfaces/Time-like value to nanoseconds."""
  return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def height_bin_index(height: float, step: float) -> int:
  """Return the nearest height-bin index, avoiding Python's ties-to-even round."""
  scaled = height / step
  return int(math.copysign(math.floor(abs(scaled) + 0.5), scaled))


def parse_joint_positions(msg: JointStateArray) -> dict[str, float]:
  """Extract a validated name-to-position mapping from a joint message."""
  positions: dict[str, float] = {}
  for joint in msg.joints:
    name = joint.name.strip()
    if not name:
      raise ValueError("joint message contains an empty joint name")
    if name in positions:
      raise ValueError(f"joint message contains duplicate name {name!r}")
    position = float(joint.position)
    if not math.isfinite(position):
      raise ValueError(f"joint {name!r} has non-finite position {position}")
    positions[name] = position
  if not positions:
    raise ValueError("joint message contains no joints")
  return positions


class HeightPostureRecorder(Node):
  """Collect the first fresh full-body pose observed in each height bin."""

  def __init__(
    self,
    output_path: Path,
    height_step: float,
    max_age: float,
    overwrite: bool,
  ) -> None:
    super().__init__("height_posture_recorder")
    self._height_step = height_step
    self._max_age_ns = int(max_age * 1_000_000_000)
    self._latest_joints: dict[str, tuple[JointStateArray, int]] = {}
    self._joint_names: dict[str, tuple[str, ...]] | None = None
    self._recorded_bins: set[int] = set()
    self._recorded_count = 0
    self._duplicate_bin_count = 0
    self._invalid_count = 0
    self._last_wait_reason: str | None = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    self._output_file = output_path.open(mode, newline="", encoding="utf-8")
    self._writer: csv.DictWriter[str] | None = None
    self.output_path = output_path.resolve()

    self._joint_subscriptions = []
    for group, topic in JOINT_TOPICS.items():
      subscription = self.create_subscription(
        JointStateArray,
        topic,
        lambda msg, group=group: self._joint_callback(group, msg),
        qos_profile_sensor_data,
      )
      self._joint_subscriptions.append(subscription)
    self._odometry_subscription = self.create_subscription(
      Odometry,
      ODOMETRY_TOPIC,
      self._odometry_callback,
      qos_profile_sensor_data,
    )

    self.get_logger().info(f"Recording height postures to {self.output_path}")
    self.get_logger().info(
      f"Saving one posture per {self._height_step:g} m height bin"
    )

  def _joint_callback(self, group: str, msg: JointStateArray) -> None:
    self._latest_joints[group] = (msg, time.monotonic_ns())

  def _fresh_joint_messages(
    self, now_ns: int
  ) -> tuple[dict[str, JointStateArray] | None, str | None]:
    missing = [group for group in JOINT_TOPICS if group not in self._latest_joints]
    if missing:
      return None, f"waiting for topics: {', '.join(missing)}"

    stale = [
      group
      for group, (_, received_ns) in self._latest_joints.items()
      if now_ns - received_ns > self._max_age_ns
    ]
    if stale:
      return None, f"waiting for fresh topics: {', '.join(stale)}"

    return {
      group: self._latest_joints[group][0]
      for group in JOINT_TOPICS
    }, None

  def _extract_positions(
    self, messages: dict[str, JointStateArray]
  ) -> tuple[dict[str, dict[str, float]] | None, str | None]:
    try:
      positions = {
        group: parse_joint_positions(messages[group])
        for group in JOINT_TOPICS
      }
    except ValueError as exc:
      return None, str(exc)

    names = {
      group: tuple(sorted(group_positions))
      for group, group_positions in positions.items()
    }
    if self._joint_names is None:
      self._joint_names = names
    elif names != self._joint_names:
      details = []
      for group in JOINT_TOPICS:
        expected = set(self._joint_names[group])
        actual = set(names[group])
        if expected != actual:
          details.append(
            f"{group}: missing={sorted(expected - actual)}, "
            f"additional={sorted(actual - expected)}"
          )
      return None, "joint-name set changed; " + "; ".join(details)
    return positions, None

  def _initialize_writer(self) -> None:
    assert self._joint_names is not None
    fieldnames = [
      "record_time_ns",
      "height_bin_index",
      "height_label_m",
      "odometry_z_m",
      "odometry_stamp_ns",
    ]
    for group in JOINT_TOPICS:
      fieldnames.extend(
        [
          f"{group}_stamp_ns",
          f"{group}_meas_stamp_ns",
          f"{group}_sequence",
        ]
      )
    for group in JOINT_TOPICS:
      fieldnames.extend(
        f"{group}__{name}_position" for name in self._joint_names[group]
      )
    self._writer = csv.DictWriter(self._output_file, fieldnames=fieldnames)
    self._writer.writeheader()
    self._output_file.flush()

  def _odometry_callback(self, msg: Odometry) -> None:
    height = float(msg.pose.pose.position.z)
    if not math.isfinite(height):
      self._warn_once(f"ignoring non-finite odometry height {height}")
      self._invalid_count += 1
      return

    bin_index = height_bin_index(height, self._height_step)
    if bin_index in self._recorded_bins:
      self._duplicate_bin_count += 1
      return

    messages, reason = self._fresh_joint_messages(time.monotonic_ns())
    if messages is None:
      self._warn_once(reason or "joint data unavailable")
      return

    positions, reason = self._extract_positions(messages)
    if positions is None:
      self._warn_once(reason or "invalid joint data")
      self._invalid_count += 1
      return

    if self._writer is None:
      self._initialize_writer()

    row: dict[str, int | float] = {
      "record_time_ns": time.time_ns(),
      "height_bin_index": bin_index,
      "height_label_m": bin_index * self._height_step,
      "odometry_z_m": height,
      "odometry_stamp_ns": timestamp_ns(msg.header.stamp),
    }
    for group, joint_msg in messages.items():
      row[f"{group}_stamp_ns"] = timestamp_ns(joint_msg.header.stamp)
      row[f"{group}_meas_stamp_ns"] = timestamp_ns(
        joint_msg.header.meas_stamp
      )
      row[f"{group}_sequence"] = int(joint_msg.header.sequence)
      assert self._joint_names is not None
      for name in self._joint_names[group]:
        row[f"{group}__{name}_position"] = positions[group][name]

    assert self._writer is not None
    self._writer.writerow(row)
    self._output_file.flush()
    self._recorded_bins.add(bin_index)
    self._recorded_count += 1
    self._last_wait_reason = None
    self.get_logger().info(
      f"Recorded {bin_index * self._height_step:.3f} m bin "
      f"(raw z={height:.4f} m, total={self._recorded_count})"
    )

  def _warn_once(self, reason: str) -> None:
    if reason != self._last_wait_reason:
      self.get_logger().warning(reason)
      self._last_wait_reason = reason

  def close(self) -> None:
    if not self._output_file.closed:
      self._output_file.flush()
      self._output_file.close()

  def summary(self) -> str:
    return (
      f"Recorded {self._recorded_count} height bins to {self.output_path}; "
      f"ignored {self._duplicate_bin_count} already-recorded bin samples and "
      f"{self._invalid_count} invalid samples"
    )


def positive_float(value: str) -> float:
  parsed = float(value)
  if not math.isfinite(parsed) or parsed <= 0.0:
    raise argparse.ArgumentTypeError("value must be a finite number greater than zero")
  return parsed


def default_output_path() -> Path:
  timestamp = time.strftime("%Y%m%d_%H%M%S")
  return Path(f"height_postures_{timestamp}.csv")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="CSV output path (default: height_postures_<timestamp>.csv)",
  )
  parser.add_argument(
    "--height-step",
    type=positive_float,
    default=0.01,
    help="Height interval in metres (default: 0.01)",
  )
  parser.add_argument(
    "--max-age",
    type=positive_float,
    default=0.2,
    help="Maximum age in seconds of cached joint messages (default: 0.2)",
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite --output if it already exists",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  output_path = args.output or default_output_path()

  rclpy.init()
  recorder: HeightPostureRecorder | None = None
  try:
    recorder = HeightPostureRecorder(
      output_path=output_path,
      height_step=args.height_step,
      max_age=args.max_age,
      overwrite=args.overwrite,
    )
    rclpy.spin(recorder)
  except KeyboardInterrupt:
    pass
  finally:
    if recorder is not None:
      recorder.close()
      print(recorder.summary())
      recorder.destroy_node()
    if rclpy.ok():
      rclpy.shutdown()


if __name__ == "__main__":
  main()
