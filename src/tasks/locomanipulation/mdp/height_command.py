"""Base height command term for locomanipulation task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class BaseHeightCommand(CommandTerm):
    cfg: BaseHeightCommandCfg

    def __init__(self, cfg: BaseHeightCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]
        self._height_command = torch.zeros(self.num_envs, 1, device=self.device)
        self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._height_command

    def _update_metrics(self) -> None:
        max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt
        actual_z = self.robot.data.root_link_pos_w[:, 2]
        cmd_z = self._height_command[:, 0]
        self.metrics["error_height"] += torch.abs(cmd_z - actual_z) / max_command_step

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if self.cfg.fixed_height is not None:
            self._height_command[env_ids, 0] = self.cfg.fixed_height
            return
        lo = self.cfg.nominal_height - self.cfg.max_deviation * self.cfg.height_scale
        hi = self.cfg.nominal_height
        r = torch.empty(len(env_ids), device=self.device)
        self._height_command[env_ids, 0] = r.uniform_(lo, hi)
        # Override: keep a fraction at nominal height.
        nominal_mask = r.uniform_(0.0, 1.0) < self.cfg.nominal_height_ratio
        self._height_command[env_ids[nominal_mask], 0] = self.cfg.nominal_height

    def _update_command(self) -> None:
        pass


@dataclass(kw_only=True)
class BaseHeightCommandCfg(CommandTermCfg):
    entity_name: str
    ranges: tuple[float, float] = (0.5, 0.785)
    """Min/max absolute commanded height in world frame (meters)."""
    fixed_height: float | None = None
    """If set, always command this height instead of random sampling."""
    nominal_height: float = 0.785
    """Nominal standing height (meters). The upper bound of the sampling range."""
    max_deviation: float = 0.285
    """Maximum downward deviation from nominal (meters). nominal - min(ranges)."""
    height_scale: float = 0.0
    """Curriculum scale in [0, 1]. 0 = nominal only, 1 = full range."""
    nominal_height_ratio: float = 0.0
    """Fraction of envs that always command nominal_height."""

    def build(self, env: ManagerBasedRlEnv) -> BaseHeightCommand:
        return BaseHeightCommand(self, env)
