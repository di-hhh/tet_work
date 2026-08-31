from __future__ import annotations

import copy
import csv
import json
from abc import ABC, abstractmethod
from pathlib import Path
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


def _safe_filename(value: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in value)


def _write_dict_rows_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_safe(row.get(key)) for key in fieldnames})


def _csv_safe(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item() if value.numel() == 1 else str(value.shape)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    value = _csv_safe(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _steady_state_timing_summary(rows: List[Dict[str, object]]) -> Dict[str, object]:
    timing_keys = (
        "inference_graph_preparation_seconds",
        "inference_gnn_forward_seconds",
        "inference_mesh_generation_seconds",
        "inference_end_to_end_inference_seconds",
    )
    steady_rows = [row for row in rows if not bool(row.get("timing_is_warmup"))]
    summary: Dict[str, object] = {
        "warmup_samples": len(rows) - len(steady_rows),
        "steady_samples": len(steady_rows),
    }
    for key in timing_keys:
        values = [
            float(row[key])
            for row in steady_rows
            if row.get(key) is not None and np.isfinite(float(row[key]))
        ]
        summary[f"{key}_mean"] = float(np.mean(values)) if values else None
        summary[f"{key}_median"] = float(np.median(values)) if values else None
    return summary


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
        self.test_sample_rows = []
        self.test_prediction_rows = []
        self.expert_only_test_step_outputs = []
        self.expert_only_sample_rows = []
        self.expert_only_prediction_rows = []
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
        sample_metadata = self._pipeline_sample_metadata(batch=batch, batch_idx=batch_idx)
        prediction_metadata = self._export_test_prediction(
            sample_metadata=sample_metadata,
            batch_idx=batch_idx,
            evaluation_variant="final",
        )
        self.test_sample_rows.append(sample_metadata | prediction_metadata | evaluation_dict)
        self.test_prediction_rows.append(sample_metadata | prediction_metadata)

        evaluation_variants = getattr(self, "local_evaluation_prediction_variants", ["final"])
        if "expert_only" in evaluation_variants:
            if not hasattr(self, "_evaluate_expert_only_data_point"):
                raise RuntimeError("expert_only evaluation was requested but the algorithm does not support it")
            expert_evaluation = self._evaluate_expert_only_data_point(data=batch)
            self.expert_only_test_step_outputs.append(expert_evaluation)
            expert_prediction_metadata = self._export_test_prediction(
                sample_metadata=sample_metadata,
                batch_idx=batch_idx,
                evaluation_variant="expert_only",
            )
            self.expert_only_sample_rows.append(
                sample_metadata | expert_prediction_metadata | expert_evaluation
            )
            self.expert_only_prediction_rows.append(sample_metadata | expert_prediction_metadata)

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
        expert_only_outputs = getattr(self, "expert_only_test_step_outputs", [])
        if hasattr(self, "_write_local_test_artifacts"):
            expert_only_averages = (
                aggregate_metrics(metrics=expert_only_outputs)
                if expert_only_outputs
                else None
            )
            self._write_local_test_artifacts(test_averages, expert_only_averages)
        self.test_step_outputs.clear()
        expert_only_outputs.clear()

        test_loader = self.trainer.test_dataloaders
        self._log_constant_plots(dataloader=test_loader, prefix="test")

    def _pipeline_sample_metadata(self, *, batch, batch_idx: int) -> Dict[str, object]:
        source_data = getattr(batch, "source_data", None)
        cache = getattr(source_data, "imitation_weight_cache", None) or {}
        run_metadata = getattr(self, "local_run_metadata", {}) or {}
        return {
            "sample_id": cache.get("sample_id") or f"test_{batch_idx:05d}",
            "geometry_id": cache.get("geometry_id"),
            "condition_id": cache.get("condition_id"),
            "pde_family": cache.get("pde_family"),
            "split": cache.get("split", "test"),
            "desired_budget": cache.get("budget"),
            "budget": cache.get("budget"),
            "quality_verdict": cache.get("quality_verdict"),
            "teacher_generation_time_seconds": cache.get("teacher_generation_time_seconds"),
            "teacher_generation_time_scope": "whole_teacher_sample_not_isolated_stage_field",
            "stage_field_generation_time_isolated": False,
            "timing_is_warmup": bool(batch_idx == 0),
            "run_id": run_metadata.get("run_id"),
            "method_id": run_metadata.get("method_id"),
            "analysis_id": run_metadata.get("analysis_id"),
            "method_role": run_metadata.get("method_role"),
            "oracle_only": run_metadata.get("oracle_only"),
            "seed": run_metadata.get("seed"),
            "checkpoint": run_metadata.get("evaluation_checkpoint"),
            "dataset_fingerprint_sha256": run_metadata.get("dataset_fingerprint_sha256"),
            "manifest_sha256": run_metadata.get("manifest_sha256"),
            "amber_code_commit": run_metadata.get("amber_code_commit"),
            "pipeline_code_commit": run_metadata.get("pipeline_code_commit"),
            "initial_elements": int(source_data.initial_mesh.num_elements) if source_data is not None else None,
            "initial_vertices": int(source_data.initial_mesh.num_vertices) if source_data is not None else None,
            "target_elements": int(source_data.expert_mesh.num_elements) if source_data is not None else None,
            "target_vertices": int(source_data.expert_mesh.num_vertices) if source_data is not None else None,
            "sizing_mse_semantics": "legacy *_size_l2 columns are mean squared errors; use *_size_mse aliases",
        }

    def _export_test_prediction(
        self,
        *,
        sample_metadata: Dict[str, object],
        batch_idx: int,
        evaluation_variant: str,
    ) -> Dict[str, object]:
        artifact_root = getattr(self, "local_artifact_root", None)
        mesh = getattr(self, "_last_evaluation_mesh", None)
        success = bool(getattr(self, "_last_evaluation_success", False))
        status = str(getattr(self, "_last_evaluation_status", "unknown"))
        if not artifact_root or mesh is None:
            return {
                "prediction_mesh_path": None,
                "evaluation_variant": evaluation_variant,
                "mesh_generation_success": success,
                "mesh_generation_status": status,
            }
        from src.mesh_util.save_mesh import save_as_vtk

        sample_id = _safe_filename(str(sample_metadata.get("sample_id") or f"test_{batch_idx:05d}"))
        prediction_dir = Path(artifact_root) / "test_predictions"
        if evaluation_variant == "expert_only":
            prediction_dir = prediction_dir / "expert_only"
        output_path = prediction_dir / f"{sample_id}.vtk"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_as_vtk(mesh, output_path)
        desired_budget = int(sample_metadata.get("desired_budget") or 0)
        predicted_elements = int(mesh.nelements)
        budget_ratio = predicted_elements / max(desired_budget, 1)
        relative_deviation = abs(predicted_elements - desired_budget) / max(desired_budget, 1)
        run_metadata = getattr(self, "local_run_metadata", {}) or {}
        close_tolerance = float(run_metadata.get("budget_close_relative_tolerance", 0.18))
        valid_min = float(run_metadata.get("budget_valid_min_ratio", 0.8))
        valid_max = float(run_metadata.get("budget_valid_max_ratio", 11000.0 / 7000.0))
        return {
            "prediction_mesh_path": output_path.relative_to(Path(artifact_root)).as_posix(),
            "evaluation_variant": evaluation_variant,
            "mesh_generation_success": success,
            "mesh_generation_status": status,
            "predicted_elements": predicted_elements,
            "predicted_vertices": int(mesh.nvertices),
            "budget_ratio": budget_ratio,
            "absolute_budget_deviation": abs(predicted_elements - desired_budget),
            "absolute_budget_relative_deviation": relative_deviation,
            "budget_close": relative_deviation <= close_tolerance,
            "budget_valid": valid_min <= budget_ratio <= valid_max,
        }

    def _write_local_test_artifacts(
        self,
        aggregate_metrics_payload: Dict[str, object],
        expert_only_aggregate_metrics_payload: Dict[str, object] | None = None,
    ) -> None:
        expert_only_sample_rows = getattr(self, "expert_only_sample_rows", [])
        expert_only_prediction_rows = getattr(self, "expert_only_prediction_rows", [])
        artifact_root = getattr(self, "local_artifact_root", None)
        if not artifact_root:
            self.test_sample_rows.clear()
            self.test_prediction_rows.clear()
            expert_only_sample_rows.clear()
            expert_only_prediction_rows.clear()
            return
        root = Path(artifact_root)
        _write_dict_rows_csv(root / "per_sample_metrics.csv", self.test_sample_rows)
        _write_dict_rows_csv(root / "test_predictions" / "prediction_manifest.csv", self.test_prediction_rows)
        if expert_only_prediction_rows:
            _write_dict_rows_csv(root / "expert_only_per_sample_metrics.csv", expert_only_sample_rows)
            _write_dict_rows_csv(
                root / "test_predictions" / "expert_only_prediction_manifest.csv",
                expert_only_prediction_rows,
            )
        success_count = sum(bool(row.get("mesh_generation_success")) for row in self.test_prediction_rows)
        payload = {
            "checkpoint": "checkpoints/last.ckpt",
            "num_samples": len(self.test_prediction_rows),
            "mesh_generation_success": success_count,
            "mesh_generation_failures": len(self.test_prediction_rows) - success_count,
            "steady_state_timing": _steady_state_timing_summary(self.test_sample_rows),
            "metrics": _json_safe(aggregate_metrics_payload),
        }
        if expert_only_aggregate_metrics_payload is not None:
            payload["expert_only"] = {
                "num_samples": len(expert_only_prediction_rows),
                "mesh_generation_success": sum(
                    bool(row.get("mesh_generation_success"))
                    for row in expert_only_prediction_rows
                ),
                "steady_state_timing": _steady_state_timing_summary(expert_only_sample_rows),
                "metrics": _json_safe(expert_only_aggregate_metrics_payload),
            }
        (root / "aggregate_metrics.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        self.test_sample_rows.clear()
        self.test_prediction_rows.clear()
        expert_only_sample_rows.clear()
        expert_only_prediction_rows.clear()

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
