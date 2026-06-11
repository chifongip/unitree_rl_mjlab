"""Waist yaw angle command term for locomanipulation task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class WaistYawCommand(CommandTerm):
    cfg: WaistYawCommandCfg

    def __init__(self, cfg: WaistYawCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]
        self._waist_yaw_command = torch.zeros(self.num_envs, 1, device=self.device)
        self.metrics["error_waist_yaw"] = torch.zeros(self.num_envs, device=self.device)
        # Resolve waist_yaw_joint index for metrics.
        joint_names = self.robot.joint_names
        self._waist_yaw_idx = joint_names.index("waist_yaw_joint")

    @property
    def command(self) -> torch.Tensor:
        return self._waist_yaw_command

    def _update_metrics(self) -> None:
        max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt
        actual_yaw = self.robot.data.joint_pos[:, self._waist_yaw_idx]
        cmd_yaw = self._waist_yaw_command[:, 0]
        self.metrics["error_waist_yaw"] += torch.abs(cmd_yaw - actual_yaw) / max_command_step

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if self.cfg.fixed_waist_yaw is not None:
            self._waist_yaw_command[env_ids, 0] = self.cfg.fixed_waist_yaw
            return
        lo = self.cfg.ranges[0] * self.cfg.yaw_scale
        hi = self.cfg.ranges[1] * self.cfg.yaw_scale
        r = torch.empty(len(env_ids), device=self.device)
        self._waist_yaw_command[env_ids, 0] = r.uniform_(lo, hi)
        nominal_mask = r.uniform_(0.0, 1.0) < self.cfg.nominal_yaw_ratio
        self._waist_yaw_command[env_ids[nominal_mask], 0] = 0.0

    def _update_command(self) -> None:
        pass


@dataclass(kw_only=True)
class WaistYawCommandCfg(CommandTermCfg):
    entity_name: str
    ranges: tuple[float, float] = (-1.5708, 1.5708)
    """Min/max commanded waist yaw angle in radians (default ±90°)."""
    fixed_waist_yaw: float | None = None
    """If set, always command this yaw instead of random sampling."""
    yaw_scale: float = 0.0
    """Curriculum scale in [0, 1]. 0 = zero yaw only, 1 = full range."""
    nominal_yaw_ratio: float = 0.0
    """Probability of pinning a resampled env to yaw=0 (nominal forward-facing)."""

    def build(self, env: ManagerBasedRlEnv) -> WaistYawCommand:
        return WaistYawCommand(self, env)
