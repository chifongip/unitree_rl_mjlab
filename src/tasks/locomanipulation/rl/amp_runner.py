"""AMP on-policy runner for locomanipulation with ONNX export.

Extends the rsl_rl v6 ``AMPOnPolicyRunner`` with:
- Config adaptation from mjlab RslRlAmpRunnerCfg format to v6 format.
- Symmetry augmentation support.
- Automatic ONNX export on every save.
"""

from __future__ import annotations

import os

import torch
import wandb
from tensordict import TensorDict

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from rsl_rl.runners import AMPOnPolicyRunner as _BaseAMPOnPolicyRunner


# ---------------------------------------------------------------------------
# Config adapter
# ---------------------------------------------------------------------------

def _adapt_amp_config(train_cfg: dict) -> dict:
  """Move root-level AMP params into ``cfg["algorithm"]`` for v6 AMPPPO.

  The mjlab ``RslRlAmpRunnerCfg`` dataclass stores AMP fields at the top
  level.  v6 ``AMPPPO.construct_algorithm()`` reads them from
  ``cfg["algorithm"]``.
  """
  alg = train_cfg.setdefault("algorithm", {})

  _amp_keys = [
    "amp_reward_coef",
    "amp_motion_files",
    "amp_num_preload_transitions",
    "amp_task_reward_lerp",
    "amp_discr_hidden_dims",
    "min_normalized_std",
    "amp_body_names",
    "amp_anchor_name",
  ]
  for key in _amp_keys:
    if key in train_cfg:
      alg.setdefault(key, train_cfg.pop(key))

  # Fix obs_groups tuple-to-list.
  if "obs_groups" in train_cfg:
    train_cfg["obs_groups"] = {
      k: list(v) for k, v in train_cfg["obs_groups"].items()
    }

  # Ensure algorithm sub-dict has keys the v6 expects.
  # Note: Do NOT set symmetry_cfg here — the runner handles it with its own default.
  alg.setdefault("rnd_cfg", None)

  # Strip keys from actor/critic that MLPModel doesn't accept.
  _model_strip_keys = {"cnn_cfg", "rnn_type", "rnn_hidden_dim", "rnn_num_layers", "class_name"}
  for section in ("actor", "critic"):
    if section in train_cfg:
      for key in _model_strip_keys:
        train_cfg[section].pop(key, None)

  return train_cfg


# ---------------------------------------------------------------------------
# ONNX export helpers
# ---------------------------------------------------------------------------

class _OnnxPolicyWrapper(torch.nn.Module):
  """Wrap v6 MLPModel actor + obs normalizer for ONNX export."""

  def __init__(self, actor, obs_normalizer=None, obs_key: str = "actor"):
    super().__init__()
    self.actor = actor
    self.obs_normalizer = obs_normalizer
    self.obs_key = obs_key

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    if self.obs_normalizer is not None:
      obs = self.obs_normalizer(obs)
    td = TensorDict(
      {self.obs_key: obs}, batch_size=obs.shape[0], device=obs.device
    )
    return self.actor(td, stochastic_output=False)


# ---------------------------------------------------------------------------
# AMP runner for locomanipulation
# ---------------------------------------------------------------------------

class LocomanipulationAMPOnPolicyRunner(_BaseAMPOnPolicyRunner):
  """AMP runner with symmetry support and ONNX export for locomanipulation."""

  env: RslRlVecEnvWrapper

  _DEFAULT_SYMMETRY_CFG = {
    "use_data_augmentation": True,
    "use_mirror_loss": True,
    "mirror_loss_coeff": 1.0,
    "data_augmentation_func": "src.tasks.locomanipulation.mdp.symmetry.g1_locomanipulation_symmetry",
  }

  def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
    # Adapt config format for v6.
    train_cfg = _adapt_amp_config(train_cfg)

    # Convert boolean symmetry_cfg to dict.
    alg_cfg = train_cfg.get("algorithm", {})
    enable_symmetry = alg_cfg.pop("symmetry_cfg", True)
    if enable_symmetry:
      alg_cfg["symmetry_cfg"] = self._DEFAULT_SYMMETRY_CFG.copy()
    else:
      alg_cfg["symmetry_cfg"] = None

    super().__init__(env, train_cfg, log_dir, device)

    # Remove env reference from symmetry_cfg for serialization.
    if "symmetry_cfg" in self.cfg.get("algorithm", {}):
      self.cfg["algorithm"]["symmetry_cfg"] = None

  # ------------------------------------------------------------------
  # Load (unify interface with OnPolicyRunner)
  # ------------------------------------------------------------------

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
    **kwargs,
  ):
    # AMPOnPolicyRunner.load() uses load_optimizer instead of load_cfg/strict.
    return super().load(path, load_optimizer=True, map_location=map_location)

  # ------------------------------------------------------------------
  # ONNX export
  # ------------------------------------------------------------------

  def _export_policy_to_onnx(self, path: str, filename: str = "policy.onnx"):
    actor = self.alg._raw_actor
    obs_normalizer = getattr(actor, "obs_normalizer", None)

    if obs_normalizer is not None and hasattr(obs_normalizer, "mean"):
      obs_dim = obs_normalizer.mean.numel()
    else:
      obs_dim = self.alg._raw_actor.obs_dim

    wrapper = _OnnxPolicyWrapper(actor, obs_normalizer, obs_key="actor")
    wrapper.to("cpu")
    wrapper.eval()

    dummy_input = torch.zeros(1, obs_dim)
    os.makedirs(path, exist_ok=True)
    torch.onnx.export(
      wrapper,
      dummy_input,
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      input_names=["obs"],
      output_names=["actions"],
      dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
    )

    # Restore to training device.
    actor.to(self.device)
    if obs_normalizer is not None and hasattr(obs_normalizer, "_mean"):
      obs_normalizer.to(self.device)

  # ------------------------------------------------------------------
  # Save (with ONNX export)
  # ------------------------------------------------------------------

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    self._export_policy_to_onnx(policy_path, filename)

    run_name: str = (
      wandb.run.name
      if self.logger.logger_type in ("wandb", "WandbLogWriter") and wandb.run
      else "local"
    )
    onnx_path = os.path.join(policy_path, filename)
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    attach_metadata_to_onnx(onnx_path, metadata)

    if self.logger.logger_type in ("wandb", "WandbLogWriter"):
      wandb.save(
        policy_path + filename, base_path=os.path.dirname(policy_path)
      )
