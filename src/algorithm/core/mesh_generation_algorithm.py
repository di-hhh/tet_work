from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm

from src.helpers.qol import add_to_dictionary, aggregate_metrics, prefix_keys, safe_mean
from src.helpers.torch_util import count_parameters, detach

if TYPE_CHECKING:
    from src.algorithm.dataloader.mesh_generation_data import MeshGenerationData
    from src.algorithm.dataloader.mesh_generation_dataset import MeshGenerationDataset
    from src.algorithm.normalizer import RunningNormalizer


class MeshGenerationAlgorithm(LightningModule, ABC):
    """
    Abstract class for the full mesh generation algorithm. This class provides a structured approach
    to training and evaluating a deep learning model for mesh generation.
    """

    def __init__(self, algorithm_config: DictConfig, train_dataset: "MeshGenerationDataset"):
        super().__init__()

        self.config = algorithm_config
        self.train_dataset = train_dataset

        from src.algorithm.util.parse_input_types import get_mesh_node_type
        from src.algorithm.prediction_transform import get_transform

        self.mesh_node_type = get_mesh_node_type(self.config.sizing_field_interpolation_type)
        self.force_mesh_generation = self.config.force_mesh_generation
        self.gmsh_kwargs: Dict[str, float] = self._get_gmsh_kwargs(gmsh_config=self.config.gmsh)
        self.max_mesh_elements: float = self._get_max_mesh_elements(self.config.max_mesh_elements)

        self.evaluation_frequency = self.config.evaluation_frequency
        self.plotting_sample_idxs: List[int] = self.config.plotting.sample_idxs
        self.plot_frequency: int = self.config.plotting.frequency
        self.plot_initial_epoch: bool = self.config.plotting.initial_epoch

        self.model = self._get_model()
        self.prediction_transform = get_transform(transform_config=algorithm_config.prediction_transform)
        self.criterion = self._get_optimization_criterion()
        if hasattr(self.criterion, "weighted_imitation_config") and not getattr(self.criterion, "weighted_imitation_config"):
            # [CodeX] 沿用现有配置树初始化 weighted imitation，避免 loss 与 Hydra 配置脱节。
            self.criterion.weighted_imitation_config = algorithm_config.get("weighted_imitation") or {}
        if hasattr(self.criterion, "weighted_imitation_config"):
            if isinstance(self.criterion.weighted_imitation_config, DictConfig):
                # [CodeX] 将结构化 Hydra 配置转成可写普通字典，避免新增 physics loss 系数时被 struct 限制拦截。
                self.criterion.weighted_imitation_config = OmegaConf.to_container(
                    self.criterion.weighted_imitation_config,
                    resolve=False,
                )
            # [CodeX] 将新增的模型侧修正损失系数注入 loss 配置，保持用户可从算法根配置直接控制。
            self.criterion.weighted_imitation_config["lambda_expert_aux"] = float(
                algorithm_config.get("lambda_expert_aux", self.criterion.weighted_imitation_config.get("lambda_expert_aux", 0.25))
            )
            self.criterion.weighted_imitation_config["lambda_corr_reg"] = float(
                algorithm_config.get("lambda_corr_reg", self.criterion.weighted_imitation_config.get("lambda_corr_reg", 1.0e-3))
            )
            self.criterion.weighted_imitation_config["lambda_corr_aux"] = float(
                algorithm_config.get("lambda_corr_aux", self.criterion.weighted_imitation_config.get("lambda_corr_aux", 0.5))
            )
        self.normalizer = self._get_normalizer()

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.grad_norms = []

        self.save_hyperparameters("algorithm_config")
        self._weighted_baseline_init_report = None

    ###################
    # Algorithm setup #
    ###################

    def _get_model(self):
        raise NotImplementedError

    def _get_optimization_criterion(self):
        raise NotImplementedError

    def _get_normalizer(self) -> "RunningNormalizer":
        from src.algorithm.normalizer import get_normalizer

        normalizer = get_normalizer(
            normalizer_config=self.config.normalizer,
            example_input=self.train_dataset.first.observation,
            prediction_transform=self.prediction_transform,
        )
        [normalizer.update_normalizers(x.observation) for x in self.train_dataset.data]
        return normalizer

    def _get_architecture_config_with_physics_codex(self) -> DictConfig:
        architecture_config = OmegaConf.create(OmegaConf.to_container(self.config.architecture, resolve=False))
        # [CodeX] 将算法级 physics correction 配置注入 architecture config，避免重构现有 get_gnn 接口。
        architecture_config.enable_physics_correction_branch = bool(
            self.config.get("enable_physics_correction_branch", False)
        )
        architecture_config.gate_activation = self.config.get("gate_activation", "sigmoid")
        architecture_config.gate_max = float(self.config.get("gate_max", 1.0))
        architecture_config.gate_init_bias = float(self.config.get("gate_init_bias", -2.5))
        architecture_config.physics_readout_init_std = float(self.config.get("physics_readout_init_std", 1.0e-3))
        architecture_config.inference_missing_physics_fallback = self.config.get(
            "inference_missing_physics_fallback",
            "gate_zero",
        )
        return architecture_config

    def initialize_from_weighted_baseline_checkpoint_codex(self) -> Dict[str, object]:
        checkpoint_path = self.config.get("init_from_weighted_baseline_checkpoint")
        if checkpoint_path in [None, False, ""]:
            self._weighted_baseline_init_report = {
                "applied": False,
                "checkpoint_path": None,
                "loaded_keys": 0,
                "adapted_keys": [],
                "skipped_keys": [],
            }
            return self._weighted_baseline_init_report

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        source_state_dict = checkpoint.get("state_dict", checkpoint)
        current_state_dict = self.state_dict()
        loadable_state_dict = {}
        adapted_keys = []
        skipped_keys = []

        for key, checkpoint_value in source_state_dict.items():
            if key not in current_state_dict:
                continue
            current_value = current_state_dict[key]
            checkpoint_value = checkpoint_value.to(dtype=current_value.dtype)

            if current_value.shape == checkpoint_value.shape:
                loadable_state_dict[key] = checkpoint_value
                continue

            adapted_value = self._adapt_checkpoint_tensor_codex(
                current_value=current_value,
                checkpoint_value=checkpoint_value,
            )
            if adapted_value is not None:
                loadable_state_dict[key] = adapted_value
                adapted_keys.append(key)
            else:
                skipped_keys.append(key)

        load_result = self.load_state_dict(loadable_state_dict, strict=False)
        self._weighted_baseline_init_report = {
            "applied": True,
            "checkpoint_path": checkpoint_path,
            "loaded_keys": len(loadable_state_dict),
            "adapted_keys": adapted_keys,
            "skipped_keys": skipped_keys,
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
        }
        return self._weighted_baseline_init_report

    @staticmethod
    def _adapt_checkpoint_tensor_codex(
        *,
        current_value: torch.Tensor,
        checkpoint_value: torch.Tensor,
    ) -> torch.Tensor | None:
        # [CodeX] 兼容“旧输入特征维度 +1”的 checkpoint 迁移：旧列直接复制，新 physics 列保持零影响。
        if current_value.ndim == 2 and checkpoint_value.ndim == 2:
            if current_value.shape[0] == checkpoint_value.shape[0] and current_value.shape[1] == checkpoint_value.shape[1] + 1:
                adapted = torch.zeros_like(current_value)
                adapted[:, : checkpoint_value.shape[1]] = checkpoint_value
                return adapted
        if current_value.ndim == 1 and checkpoint_value.ndim == 1:
            if current_value.shape[0] == checkpoint_value.shape[0] + 1:
                adapted = current_value.clone()
                adapted[: checkpoint_value.shape[0]] = checkpoint_value
                return adapted
        return None

    def _get_gmsh_kwargs(self, gmsh_config: DictConfig) -> Dict:
        from src.mesh_util.sizing_field_util import get_sizing_field

        min_sizing_field = gmsh_config.get("min_sizing_field")
        if min_sizing_field.startswith("x"):
            factor = 1 / float(min_sizing_field[1:])
            min_sizing_field = factor * np.min([np.min(get_sizing_field(mesh)) for mesh in self.train_dataset.expert_meshes])
        max_sizing_field = gmsh_config.get("max_sizing_field")
        if max_sizing_field.startswith("x"):
            factor = float(max_sizing_field[1:])
            max_sizing_field = factor * np.max([np.max(get_sizing_field(mesh)) for mesh in self.train_dataset.expert_meshes])

        return {"min_sizing_field": min_sizing_field, "max_sizing_field": max_sizing_field}

    def _get_max_mesh_elements(self, max_elements: str | float) -> float:
        if isinstance(max_elements, str):
            max_data_elements = max([mesh.nelements for mesh in self.train_dataset.expert_meshes])
            if max_elements == "auto":
                max_elements = int(max_data_elements * 1.5)
            elif max_elements.startswith("x"):
                factor = float(max_elements[1:])
                max_elements = int(max_data_elements * factor)
        return max_elements

    def configure_optimizers(self) -> Optimizer | Dict[str, Optimizer | Dict[str, LRScheduler | str]]:
        from src.algorithm.optimizer import get_optimizer_and_scheduler

        return get_optimizer_and_scheduler(
            optimizer_dict=self.config.optimizer,
            model=self.model,
            num_epochs=self.trainer.max_epochs,
        )

    ##################
    # Start training #
    ##################
    def on_train_start(self):
        validation_loader = self.trainer.val_dataloaders
        self._log_constant_metrics(validation_loader)
        self._log_constant_plots(dataloader=validation_loader, prefix="val")

    def _log_constant_metrics(self, validation_loader) -> None:
        train_expert_meshes = self.train_dataset.expert_meshes
        initial_meshes = [x.mesh for x in self.train_dataset.data]
        constant_metrics = {
            "min_sizing_field": self.gmsh_kwargs.get("min_sizing_field"),
            "max_sizing_field": self.gmsh_kwargs.get("max_sizing_field"),
            "max_mesh_elements": self.max_mesh_elements,
            "mean_expert_elements": np.mean([mesh.nelements for mesh in train_expert_meshes]),
            "max_expert_elements": np.max([mesh.nelements for mesh in train_expert_meshes]),
            "min_expert_elements": np.min([mesh.nelements for mesh in train_expert_meshes]),
            "mean_expert_vertices": np.mean([mesh.nvertices for mesh in train_expert_meshes]),
            "max_expert_vertices": np.max([mesh.nvertices for mesh in train_expert_meshes]),
            "min_expert_vertices": np.min([mesh.nvertices for mesh in train_expert_meshes]),
            "mean_initial_elements": np.mean([mesh.nelements for mesh in initial_meshes]),
            "max_initial_elements": np.max([mesh.nelements for mesh in initial_meshes]),
            "min_initial_elements": np.min([mesh.nelements for mesh in initial_meshes]),
            "mean_initial_vertices": np.mean([mesh.nvertices for mesh in initial_meshes]),
            "max_initial_vertices": np.max([mesh.nvertices for mesh in initial_meshes]),
            "min_initial_vertices": np.min([mesh.nvertices for mesh in initial_meshes]),
            "num_network_parameters": count_parameters(self.model),
        }
        metric_dict_list = {}
        for data in tqdm(validation_loader, desc="Initial Metrics".title()):
            data: "MeshGenerationData"
            sample_dict = self._evaluate_initial_sample(data=data)
            add_to_dictionary(metric_dict_list, new_scalars=sample_dict)
        metric_dict_list = {key: safe_mean(value) for key, value in metric_dict_list.items()}
        metric_dict_list = prefix_keys(metric_dict_list, prefix="constant", separator=".")
        constant_metrics = prefix_keys(constant_metrics, prefix="constant", separator="/")
        constant_metrics = constant_metrics | metric_dict_list

        self.log_dict(constant_metrics, on_step=False, prog_bar=True)

    def _evaluate_initial_sample(self, data: "MeshGenerationData") -> Dict[str, float]:
        from src.algorithm.util.amber_util import get_reconstructed_mesh
        from src.mesh_util.mesh_metrics import MeshMetrics

        initial_mesh = data.mesh
        expert_mesh = data.expert_mesh
        initial2expert_similarity_metrics = MeshMetrics(
            metric_config=self.config.mesh_metrics,
            reference_mesh=expert_mesh,
            evaluated_mesh=initial_mesh,
            fem_problem=data.fem_problem,
        )()
        initial2expert_similarity_metrics = prefix_keys(initial2expert_similarity_metrics, prefix="initial")
        reconstructed_mesh = get_reconstructed_mesh(expert_mesh, gmsh_kwargs=self.gmsh_kwargs)
        reconstruction2expert_similarity_metrics = MeshMetrics(
            metric_config=self.config.mesh_metrics,
            reference_mesh=expert_mesh,
            evaluated_mesh=reconstructed_mesh,
            fem_problem=data.fem_problem,
        )()
        reconstruction2expert_similarity_metrics = prefix_keys(reconstruction2expert_similarity_metrics, prefix="rec")
        return reconstruction2expert_similarity_metrics | initial2expert_similarity_metrics

    def _log_constant_plots(self, dataloader: DataLoader, prefix: str) -> None:
        from src.algorithm.visualization.amber_visualization import get_reference_plot

        for plotting_sample_idx in self.plotting_sample_idxs:
            if len(dataloader.dataset) <= plotting_sample_idx:
                continue
            data = dataloader.dataset[plotting_sample_idx]
            expert_plot = get_reference_plot(
                reference_mesh=data.expert_mesh,
                fem_problem=copy.deepcopy(data.fem_problem),
                reference_name="Expert",
            )
            self._log_plots({f"expert/{prefix}{plotting_sample_idx}": expert_plot})

    #################
    # Training loop #
    #################
    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        if hasattr(self.criterion, "set_stage_context"):
            self.criterion.set_stage_context(
                current_epoch=self.current_epoch,
                max_epochs=self.trainer.max_epochs,
                resumed_from_checkpoint=bool(getattr(self.trainer, "ckpt_path", None)),
            )  # [CodeX] 在不重构训练循环的前提下，让 loss 能根据 epoch / checkpoint 状态切换阶段设置。
        loss, scalars = self._training_step(batch, batch_idx)
        if hasattr(self.criterion, "last_loss_metrics"):
            loss_metrics = self.criterion.last_loss_metrics
            scalars = scalars | {
                "loss_unweighted": loss_metrics.get("unweighted_loss", scalars.get("loss", float(loss.detach().item()))),
                "loss_weighted": loss_metrics.get("weighted_loss", scalars.get("loss", float(loss.detach().item()))),
                "imitation_weight_mean": loss_metrics.get("weight_mean", 1.0),
                "imitation_weight_std": loss_metrics.get("weight_std", 0.0),
                "imitation_weight_min": loss_metrics.get("weight_min", 1.0),
                "imitation_weight_max": loss_metrics.get("weight_max", 1.0),
                "imitation_weight_q50": loss_metrics.get("weight_q50", 1.0),
                "imitation_weight_q80": loss_metrics.get("weight_q80", 1.0),
                "imitation_weight_q95": loss_metrics.get("weight_q95", 1.0),
                "imitation_weight_top20_ratio": loss_metrics.get("weight_top20_ratio", 0.2),
                "imitation_weight_effective_sample_ratio": loss_metrics.get("weight_effective_sample_ratio", 1.0),
                "imitation_weight_loaded": loss_metrics.get("weights_loaded", 0.0),
                "imitation_weight_fallback": loss_metrics.get("weights_fallback", 0.0),
                "imitation_weight_label_pearson": loss_metrics.get("weight_label_pearson", np.nan),
                "imitation_weight_label_spearman": loss_metrics.get("weight_label_spearman", np.nan),
                "imitation_weight_neg_log_size_pearson": loss_metrics.get("weight_neg_log_size_pearson", np.nan),
                "imitation_weight_neg_log_size_spearman": loss_metrics.get("weight_neg_log_size_spearman", np.nan),
                "imitation_weighted_size_l2": loss_metrics.get("weighted_size_l2", np.nan),
                "imitation_topk_high_importance_l2": loss_metrics.get("topk_high_importance_l2", np.nan),
                "imitation_bucket_low_size_l2": loss_metrics.get("bucket_low_size_l2", np.nan),
                "imitation_bucket_high_size_l2": loss_metrics.get("bucket_high_size_l2", np.nan),
                "imitation_bucket_high_low_ratio": loss_metrics.get("bucket_high_low_ratio", np.nan),
                "imitation_top20_weighted_loss_ratio": loss_metrics.get("top20_weighted_loss_ratio", np.nan),
                "imitation_reference_q95": loss_metrics.get("reference_q95", np.nan),
                "imitation_reference_mean": loss_metrics.get("reference_mean", np.nan),
                "imitation_reference_std": loss_metrics.get("reference_std", np.nan),
                "imitation_projected_q95": loss_metrics.get("projected_q95", np.nan),
                "imitation_projected_mean": loss_metrics.get("projected_mean", np.nan),
                "imitation_projected_std": loss_metrics.get("projected_std", np.nan),
                "imitation_projection_q95_ratio": loss_metrics.get("projection_q95_ratio", np.nan),
                "imitation_reference_top20_ratio": loss_metrics.get("reference_top20_ratio", np.nan),
                "imitation_projected_top20_ratio": loss_metrics.get("projected_top20_ratio", np.nan),
                "imitation_projection_top20_ratio_delta": loss_metrics.get("projection_top20_ratio_delta", np.nan),
                "imitation_stage2_active": loss_metrics.get("stage2_active", 0.0),
                "loss_main": loss_metrics.get("loss_main", np.nan),
                "loss_expert_aux": loss_metrics.get("loss_expert_aux", 0.0),
                "loss_corr_aux": loss_metrics.get("loss_corr_aux", 0.0),
                "loss_corr_reg": loss_metrics.get("loss_corr_reg", 0.0),
                "physics_gate_mean": loss_metrics.get("gate_mean", 0.0),
                "physics_gate_std": loss_metrics.get("gate_std", 0.0),
                "physics_gate_min": loss_metrics.get("gate_min", 0.0),
                "physics_gate_max": loss_metrics.get("gate_max", 0.0),
                "physics_gate_high_importance_mean": loss_metrics.get("gate_high_importance_mean", 0.0),
                "physics_gate_low_importance_mean": loss_metrics.get("gate_low_importance_mean", 0.0),
                "physics_delta_phys_abs_mean": loss_metrics.get("delta_phys_abs_mean", 0.0),
                "physics_delta_phys_high_importance_abs_mean": loss_metrics.get("delta_phys_high_importance_abs_mean", 0.0),
                "physics_delta_phys_low_importance_abs_mean": loss_metrics.get("delta_phys_low_importance_abs_mean", 0.0),
                "physics_applied_correction_abs_mean": loss_metrics.get("applied_correction_abs_mean", 0.0),
                "physics_applied_correction_high_importance_abs_mean": loss_metrics.get("applied_correction_high_importance_abs_mean", 0.0),
                "physics_applied_correction_low_importance_abs_mean": loss_metrics.get("applied_correction_low_importance_abs_mean", 0.0),
                "expert_prior_size_l2": loss_metrics.get("expert_prior_size_l2", np.nan),
                "expert_prior_weighted_size_l2": loss_metrics.get("expert_prior_weighted_size_l2", np.nan),
                "expert_prior_topk_high_importance_l2": loss_metrics.get("expert_prior_topk_high_importance_l2", np.nan),
                "final_prediction_size_l2": loss_metrics.get("final_prediction_size_l2", np.nan),
                "final_prediction_weighted_size_l2": loss_metrics.get("final_prediction_weighted_size_l2", np.nan),
                "final_prediction_topk_high_importance_l2": loss_metrics.get("final_prediction_topk_high_importance_l2", np.nan),
            }
        # [CodeX] 通过现有训练日志通道补充 gate / correction / expert-vs-final 的关键诊断，不改动 logger 结构。
        self.training_step_outputs.append(scalars)
        return loss

    @abstractmethod
    def _training_step(self, batch, batch_idx) -> torch.Tensor:
        raise NotImplementedError

    def on_after_backward(self):
        total_norm = detach(torch.norm(torch.stack([p.grad.norm() for p in self.parameters() if p.grad is not None]), p=2))
        self.grad_norms.append(total_norm)

    def on_train_epoch_end(self) -> None:
        epoch_averages = aggregate_metrics(metrics=self.training_step_outputs)
        epoch_averages = prefix_keys(epoch_averages, prefix="metrics.train")
        epoch_averages["grad_norm"] = np.mean(self.grad_norms)
        self.log_dict(epoch_averages, on_epoch=True, prog_bar=True)
        self.training_step_outputs.clear()

    ##########################
    # Evaluation and testing #
    ##########################
    def validation_step(self, batch, batch_idx: int) -> None:
        if self.current_epoch % self.evaluation_frequency == 0:
            evaluation_dict = self._evaluate_data_point(data=batch)
            self.validation_step_outputs.append(evaluation_dict)

        if self.current_epoch % self.plot_frequency == 0 and (self.plot_initial_epoch or self.current_epoch > 0):
            if batch_idx in self.plotting_sample_idxs:
                plot_dict = self._visualize_data_point(data=batch)
                plot_dict = prefix_keys(plot_dict, prefix=f"val{batch_idx}")
                self._log_plots(plot_dict)

    def test_step(self, batch, batch_idx: int) -> None:
        evaluation_dict = self._evaluate_data_point(data=batch)
        self.test_step_outputs.append(evaluation_dict)

        if batch_idx in self.plotting_sample_idxs:
            plot_dict = self._visualize_data_point(data=batch)
            plot_dict = prefix_keys(plot_dict, prefix=f"test{batch_idx}")
            self._log_plots(plot_dict)

    def _evaluate_data_point(self, data) -> Dict:
        raise NotImplementedError

    def _visualize_data_point(self, data) -> Dict:
        raise NotImplementedError

    def on_validation_epoch_end(self):
        if len(self.validation_step_outputs) > 0:
            validation_averages = aggregate_metrics(metrics=self.validation_step_outputs)
            validation_averages = prefix_keys(validation_averages, prefix="metrics.val", separator="_")
            self.log_dict(validation_averages, on_epoch=True, prog_bar=True)
            self.validation_step_outputs.clear()

    def on_test_end(self) -> None:
        test_averages = aggregate_metrics(metrics=self.test_step_outputs)
        test_averages = prefix_keys(test_averages, prefix="metrics.test", separator="_")
        experiment_logger = self._get_experiment_logger_codex()
        if experiment_logger is not None:
            experiment_logger.log(test_averages, step=self.current_epoch)

        test_step_outputs_df = pd.DataFrame(self.test_step_outputs)
        if experiment_logger is not None:
            experiment_logger.log({"test_table": test_step_outputs_df}, step=self.current_epoch)
        self.test_step_outputs.clear()

        test_loader = self.trainer.test_dataloaders
        self._log_constant_plots(dataloader=test_loader, prefix="test")

    ############################
    # prediction/model forward #
    ############################

    def _predict(
        self,
        batch: Batch | torch.Tensor,
        is_train: bool = False,
        flatten: bool = True,
        return_details: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        batch = batch.to(self.device)
        batch = self.normalizer.normalize_inputs(batch)
        model_outputs = self.model(
            batch,
            correction_warmup_factor=self._get_correction_warmup_factor_codex(is_train=is_train),
        )
        prediction_bundle = self._build_prediction_bundle_codex(
            batch=batch,
            model_outputs=model_outputs,
            is_train=is_train,
            flatten=flatten,
        )
        return prediction_bundle if return_details else prediction_bundle["final"]

    def _build_prediction_bundle_codex(
        self,
        *,
        batch: Batch | torch.Tensor,
        model_outputs,
        is_train: bool,
        flatten: bool,
    ) -> Dict[str, torch.Tensor]:
        if hasattr(batch, "current_sizing_field"):
            current_sizing_field = batch.current_sizing_field
        else:
            current_sizing_field = None

        if isinstance(model_outputs, dict):
            expert_output = model_outputs["expert_output"]
            physics_output = model_outputs["physics_output"]
            gate = model_outputs["gate"]
            gate_logits = model_outputs["gate_logits"]
            physics_feature_available = model_outputs.get(
                "physics_feature_available",
                torch.ones_like(gate),
            )
        else:
            expert_output = model_outputs
            physics_output = torch.zeros_like(expert_output)
            gate = torch.zeros_like(expert_output)
            gate_logits = torch.zeros_like(expert_output)
            physics_feature_available = torch.zeros_like(expert_output)

        if flatten:
            expert_output = expert_output.flatten()
            physics_output = physics_output.flatten()
            gate = gate.flatten()
            gate_logits = gate_logits.flatten()
            physics_feature_available = physics_feature_available.flatten()

        semantic_expert_delta = self.normalizer.denormalize_predictions(expert_output)
        semantic_phys_delta = self._denormalize_correction_residual_codex(physics_output)
        applied_correction = gate * semantic_phys_delta
        semantic_total_delta = semantic_expert_delta + applied_correction

        final_predictions = self.prediction_transform.forward(
            semantic_total_delta,
            baseline=current_sizing_field,
            is_train=is_train,
        )
        expert_predictions = self.prediction_transform.forward(
            semantic_expert_delta,
            baseline=current_sizing_field,
            is_train=is_train,
        )
        # [CodeX] 统一返回 final / expert-only / correction 相关张量，训练和验证共用同一诊断入口。
        return {
            "final": final_predictions,
            "expert": expert_predictions,
            "semantic_expert_delta": semantic_expert_delta,
            "semantic_phys_delta": semantic_phys_delta,
            "semantic_total_delta": semantic_total_delta,
            "applied_correction": applied_correction,
            "gate": gate,
            "gate_logits": gate_logits,
            "physics_feature_available": physics_feature_available,
        }

    def _denormalize_correction_residual_codex(self, correction_output: torch.Tensor) -> torch.Tensor:
        prediction_normalizer = getattr(self.normalizer, "prediction_normalizer", None)
        if prediction_normalizer is None:
            return correction_output

        scale = torch.sqrt(prediction_normalizer.var + self.normalizer.epsilon).to(
            device=correction_output.device,
            dtype=correction_output.dtype,
        )
        if correction_output.ndim == 1:
            if scale.numel() == 1:
                return correction_output * scale.squeeze(0)
            return correction_output * scale

        view_shape = [1] * correction_output.ndim
        view_shape[-1] = -1
        return correction_output * scale.view(view_shape)

    def _get_correction_warmup_factor_codex(self, *, is_train: bool) -> float:
        warmup_epochs = int(self.config.get("correction_warmup_epochs", 0) or 0)
        if warmup_epochs <= 0:
            return 1.0
        progress_epoch = float(self.current_epoch) + 1.0
        if not is_train:
            return min(1.0, progress_epoch / float(warmup_epochs))
        return min(1.0, progress_epoch / float(warmup_epochs))

    def _clip_detach(self, sizing_field: torch.Tensor) -> np.ndarray:
        sizing_field = np.clip(
            detach(sizing_field),
            self.gmsh_kwargs.get("min_sizing_field"),
            self.gmsh_kwargs.get("max_sizing_field"),
        )
        return sizing_field

    ###########
    # logging #
    ###########

    def _get_experiment_logger_codex(self):
        # [CodeX] 在关闭 WandB 或使用不带 experiment 接口的 logger 时安全回退，避免测试与绘图阶段因日志调用崩溃。
        logger = getattr(self, "logger", None)
        return getattr(logger, "experiment", None)

    def _log_plots(self, plot_dict: Dict[str, go.Figure]):
        experiment_logger = self._get_experiment_logger_codex()
        if experiment_logger is None:
            # [CodeX] 无外部实验日志器时直接跳过图像上报，保留训练与评估主流程。
            return
        plot_dict = prefix_keys(plot_dict, prefix="figure", separator=".")
        plot_dict["epoch"] = self.current_epoch
        experiment_logger.log(plot_dict, step=int(plot_dict["epoch"]))
