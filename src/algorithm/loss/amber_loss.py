from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Batch

from src.algorithm.loss.mesh_generation_loss import MeshGenerationLoss
from src.algorithm.prediction_transform.prediction_transform import PredictionTransform
from src.algorithm.util.weighted_imitation_diagnostics_codex import (
    build_weights_from_normalized_importance_codex,
    compute_correlation_stats_codex,
    compute_distribution_stats_codex,
    compute_size_error_metrics_codex,
    should_enable_stage2_codex,
)


class AmberLoss(MeshGenerationLoss):
    def __init__(
        self,
        label_transform: PredictionTransform,
        loss_type: str = "mse",
        weighted_imitation_config: Optional[Dict] = None,
    ):
        super().__init__(label_transform=label_transform)
        self.loss_type = loss_type
        self.weighted_imitation_config = weighted_imitation_config or {}
        self._last_loss_metrics: Dict[str, float] = {}
        self._stage2_active = False

    def calculate_loss(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        graph_batch: Optional[Batch] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if graph_batch is not None and hasattr(graph_batch, "current_sizing_field"):
            baseline = graph_batch.current_sizing_field
        else:
            baseline = None
        labels = self.label_transform.inverse(labels, baseline=baseline, is_train=True)

        differences = self.get_differences(predictions=predictions, labels=labels)
        if self.loss_type == "mse":
            element_loss = differences ** 2
        elif self.loss_type == "mae":
            element_loss = differences
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        unweighted_loss = torch.mean(element_loss)
        importance = self._get_importance_signal(graph_batch=graph_batch, element_loss=element_loss)
        weights = self._get_imitation_weights(
            graph_batch=graph_batch,
            element_loss=element_loss,
            labels=labels,
            importance=importance,
        )
        epsilon = float(self.weighted_imitation_config.get("epsilon", 1.0e-8))
        weighted_loss = torch.sum(weights * element_loss) / (torch.sum(weights) + epsilon)  # [CodeX] 保持逐点损失定义不变，只替换聚合为按节点权重的加权平均。
        use_weighted_loss = self.weighted_imitation_config.get("enabled", False) or self._stage2_active
        loss = weighted_loss if use_weighted_loss else unweighted_loss

        diagnostics = self._build_loss_diagnostics(
            predictions=predictions,
            labels=labels,
            weights=weights,
            importance=importance,
            graph_batch=graph_batch,
            epsilon=epsilon,
        )

        self._last_loss_metrics = {
            "weighted_loss": float(weighted_loss.detach().item()),
            "unweighted_loss": float(unweighted_loss.detach().item()),
            "weights_loaded": self._get_imitation_weights_loaded(graph_batch=graph_batch),
            "weights_fallback": self._get_imitation_weights_fallback(graph_batch=graph_batch),
            "stage2_active": float(self._stage2_active),
            **diagnostics,
        }
        return loss, differences

    @property
    def last_loss_metrics(self) -> Dict[str, float]:
        return self._last_loss_metrics

    def set_stage_context(self, *, current_epoch: int, max_epochs: int, resumed_from_checkpoint: bool = False) -> None:
        # [CodeX] 同一套 loss 支持两种阶段二用法：单次训练后半段切换，或从已有 checkpoint 恢复后直接进入强化微调。
        config = self.weighted_imitation_config
        stage2_enable = bool(config.get("stage2_enable", False))
        stage2_epochs = int(config.get("stage2_epochs", 0))
        if not stage2_enable or stage2_epochs <= 0:
            self._stage2_active = False
            return
        self._stage2_active = should_enable_stage2_codex(
            current_epoch=current_epoch,
            max_epochs=max_epochs,
            stage2_enable=stage2_enable,
            stage2_epochs=stage2_epochs,
            resumed_from_checkpoint=resumed_from_checkpoint,
            stage2_resume_mode=bool(config.get("stage2_resume_mode", True)),
        )

    def _get_imitation_weights(
        self,
        *,
        graph_batch: Optional[Batch],
        element_loss: torch.Tensor,
        labels: torch.Tensor,
        importance: torch.Tensor,
    ) -> torch.Tensor:
        if graph_batch is not None and hasattr(graph_batch, "imitation_weights"):
            weights = graph_batch.imitation_weights.to(device=element_loss.device, dtype=element_loss.dtype)
            if weights.shape != labels.shape:
                weights = weights.reshape(labels.shape)
            if self._stage2_active:
                weights = self._apply_stage2_weights(weights=weights, importance=importance, dtype=element_loss.dtype, device=element_loss.device)
            return weights

        if self.weighted_imitation_config.get("fallback_to_ones", True):
            return torch.ones_like(element_loss)
        raise ValueError("Weighted imitation is enabled but the batch does not contain imitation weights.")

    @staticmethod
    def _get_imitation_weights_loaded(*, graph_batch: Optional[Batch]) -> float:
        if graph_batch is not None and hasattr(graph_batch, "imitation_weights_loaded"):
            return float(graph_batch.imitation_weights_loaded.float().mean().item())
        return 0.0

    @staticmethod
    def _get_imitation_weights_fallback(*, graph_batch: Optional[Batch]) -> float:
        if graph_batch is not None and hasattr(graph_batch, "imitation_weights_fallback"):
            return float(graph_batch.imitation_weights_fallback.float().mean().item())
        return 0.0

    def get_differences(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        differences = torch.abs(predictions - labels)
        return differences

    def _get_importance_signal(self, *, graph_batch: Optional[Batch], element_loss: torch.Tensor) -> torch.Tensor:
        if graph_batch is not None and hasattr(graph_batch, "imitation_normalized_importance"):
            importance = graph_batch.imitation_normalized_importance.to(device=element_loss.device, dtype=element_loss.dtype)
            if importance.shape != element_loss.shape:
                importance = importance.reshape(element_loss.shape)
            return importance
        if graph_batch is not None and hasattr(graph_batch, "imitation_weights"):
            importance = graph_batch.imitation_weights.to(device=element_loss.device, dtype=element_loss.dtype)
            if importance.shape != element_loss.shape:
                importance = importance.reshape(element_loss.shape)
            return importance
        return torch.ones_like(element_loss)

    def _apply_stage2_weights(self, *, weights: torch.Tensor, importance: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        importance_np = importance.detach().cpu().numpy()
        stage2_weights_np = build_weights_from_normalized_importance_codex(
            importance_np,
            self.weighted_imitation_config,
            mode_key="stage2_weight_mode",
            beta_key="stage2_beta",
            gamma_key="stage2_gamma",
            lambda_high_key="stage2_lambda_high",
            lambda_mid_key="stage2_lambda_mid",
            topk_percent_key="stage2_topk_percent",
            ternary_low_quantile_key="stage2_ternary_low_quantile",
            ternary_high_quantile_key="stage2_ternary_high_quantile",
        )
        stage2_weights = torch.as_tensor(stage2_weights_np, device=device, dtype=dtype)
        threshold = torch.quantile(importance, q=max(0.0, 1.0 - float(self.weighted_imitation_config.get("stage2_topk_percent", 0.2))))
        high_mask = importance >= threshold
        if bool(self.weighted_imitation_config.get("stage2_high_importance_only", False)):
            low_weight = float(self.weighted_imitation_config.get("stage2_low_weight", self.weighted_imitation_config.get("epsilon", 1.0e-8)))
            stage2_weights = torch.where(high_mask, stage2_weights, torch.full_like(stage2_weights, low_weight))
        emphasis = float(self.weighted_imitation_config.get("stage2_high_importance_emphasis", 1.0))
        if emphasis != 1.0:
            stage2_weights = torch.where(high_mask, stage2_weights * emphasis, stage2_weights)

        min_weight = float(self.weighted_imitation_config.get("clip_min", self.weighted_imitation_config.get("weight_clip_min", 1.0)))
        max_weight = float(self.weighted_imitation_config.get("clip_max", self.weighted_imitation_config.get("weight_clip_max", 10.0)))
        return torch.clamp(stage2_weights, min=min_weight, max=max_weight)

    def _build_loss_diagnostics(
        self,
        *,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
        importance: torch.Tensor,
        graph_batch: Optional[Batch],
        epsilon: float,
    ) -> Dict[str, float]:
        weights_np = weights.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        predictions_np = predictions.detach().cpu().numpy()
        importance_np = importance.detach().cpu().numpy()

        diagnostics = {
            **compute_distribution_stats_codex(weights_np, topk_percent=float(self.weighted_imitation_config.get("topk_percent", 0.2)), prefix="weight_"),
            **compute_correlation_stats_codex(weights_np, labels_np, epsilon=epsilon),
            **compute_size_error_metrics_codex(
                predictions_np,
                labels_np,
                weights_np,
                importance_np,
                epsilon=epsilon,
                topk_percent=float(self.weighted_imitation_config.get("topk_percent", 0.2)),
                bucket_count=int(self.weighted_imitation_config.get("bucket_count", 5)),
            ),
        }
        diagnostics["top20_weighted_loss_ratio"] = self._top20_weighted_loss_ratio(
            predictions=predictions_np,
            labels=labels_np,
            weights=weights_np,
            importance=importance_np,
            topk_percent=float(self.weighted_imitation_config.get("topk_percent", 0.2)),
        )

        graph_scalar_names = [
            "reference_mean",
            "reference_std",
            "reference_q95",
            "reference_top20_ratio",
            "projected_mean",
            "projected_std",
            "projected_q95",
            "projected_top20_ratio",
            "projection_q95_ratio",
            "projection_top20_ratio_delta",
        ]
        for scalar_name in graph_scalar_names:
            diagnostics[scalar_name] = self._read_graph_scalar(graph_batch=graph_batch, name=scalar_name)
        return diagnostics

    @staticmethod
    def _top20_weighted_loss_ratio(*, predictions, labels, weights, importance, topk_percent: float) -> float:
        squared_error = np.square(np.asarray(predictions, dtype=np.float64) - np.asarray(labels, dtype=np.float64))
        weighted_error = np.asarray(weights, dtype=np.float64) * squared_error
        importance = np.asarray(importance, dtype=np.float64).reshape(-1)
        if importance.size == 0:
            return float("nan")
        topk = max(1, int(np.ceil(importance.size * topk_percent)))
        top_indices = np.argsort(importance, kind="mergesort")[-topk:]
        denominator = float(np.sum(weighted_error))
        if denominator <= 0:
            return 0.0
        return float(np.sum(weighted_error[top_indices]) / denominator)

    @staticmethod
    def _read_graph_scalar(*, graph_batch: Optional[Batch], name: str) -> float:
        if graph_batch is None or not hasattr(graph_batch, name):
            return float("nan")
        value = getattr(graph_batch, name)
        value = value.float()
        return float(value.mean().item())
