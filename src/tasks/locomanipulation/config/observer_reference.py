"""Frozen nominal observer coefficients in explicit arm joint order."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ObserverReference:
  joint_names: tuple[str, ...]
  inertia: tuple[float, ...]
  friction_limit: float
  friction_viscosity: tuple[float, ...]
  bandwidth_rad_s: float
