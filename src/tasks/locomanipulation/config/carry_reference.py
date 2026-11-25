"""Robot-specific inputs shared by carry-pose generation and training."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CarryPoseReference:
  xml: Path
  joint_count: int
  arm_names: tuple[str, ...]
  hands: tuple[str, str]
  pose: dict[str, float]
  dataset: Path
  motion_dof_indices: tuple[int, ...]
  supports_ground_parallel: bool = False
  hand_up_local: tuple[tuple[float, float, float], ...] | None = None
