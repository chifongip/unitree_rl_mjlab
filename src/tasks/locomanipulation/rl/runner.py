import os

import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner


class LocomanipulationOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  _DEFAULT_SYMMETRY_CFG = {
    "use_data_augmentation": True,
    "use_mirror_loss": True,
    "mirror_loss_coeff": 1.0,
    "data_augmentation_func": "src.tasks.locomanipulation.mdp.symmetry.g1_locomanipulation_symmetry",
  }

  def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
    # Pop symmetry_cfg before super().__init__() to avoid passing it as **kwargs
    # to PPO.__init__(), which would conflict with resolve_symmetry_config.
    # Also avoids polluting train_cfg (which train.py later dumps to YAML).
    enable_symmetry = train_cfg.get("algorithm", {}).pop("symmetry_cfg", True)

    super().__init__(env, train_cfg, log_dir, device)

    # Inject symmetry into the constructed algorithm without mutating self.cfg,
    # since train.py holds the same dict reference and will dump it to YAML.
    if enable_symmetry:
      from rsl_rl.utils import resolve_callable
      symmetry_cfg = self._DEFAULT_SYMMETRY_CFG.copy()
      symmetry_cfg["data_augmentation_func"] = resolve_callable(
        symmetry_cfg["data_augmentation_func"]
      )
      symmetry_cfg["_env"] = self.env
      self.alg.symmetry = symmetry_cfg

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    self.export_policy_to_onnx(policy_path, filename)
    run_name: str = (
      wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    onnx_path = os.path.join(policy_path, filename)
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    attach_metadata_to_onnx(onnx_path, metadata)
    if self.logger.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class G1_23DOF_LocomanipulationOnPolicyRunner(LocomanipulationOnPolicyRunner):
  """Runner for G1-23DOF locomanipulation with 23-DOF symmetry function."""

  _DEFAULT_SYMMETRY_CFG = {
    **LocomanipulationOnPolicyRunner._DEFAULT_SYMMETRY_CFG,
    "data_augmentation_func": "src.tasks.locomanipulation.mdp.symmetry.g1_23dof_locomanipulation_symmetry",
  }
