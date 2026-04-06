from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def normalize_importance_codex(importance: np.ndarray, config: Optional[Dict[str, Any]] = None) -> np.ndarray:
    # [CodeX] 将物理重要性统一映射到 [0, 1]，供不同权重模式和诊断指标复用。
    config = config or {}
    epsilon = float(config.get("epsilon", 1.0e-8))
    lower_quantile = float(config.get("normalization_lower_quantile", 0.05))
    upper_quantile = float(config.get("normalization_upper_quantile", 0.95))

    clean_importance = np.asarray(importance, dtype=np.float64)
    clean_importance[~np.isfinite(clean_importance)] = 0.0
    clean_importance = np.maximum(clean_importance, 0.0)
    log_importance = np.log1p(clean_importance)

    q_low = float(np.quantile(log_importance, lower_quantile))
    q_high = float(np.quantile(log_importance, upper_quantile))
    if not np.isfinite(q_low):
        q_low = 0.0
    if not np.isfinite(q_high):
        q_high = q_low + 1.0

    if q_high <= q_low + epsilon:
        normalized = np.zeros_like(log_importance)
    else:
        normalized = (log_importance - q_low) / (q_high - q_low + epsilon)
    normalized[~np.isfinite(normalized)] = 0.0
    return np.clip(normalized, a_min=0.0, a_max=1.0)


def build_weights_from_normalized_importance_codex(
    normalized_importance: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
    *,
    mode_key: str = "weight_mode",
    beta_key: str = "beta",
    gamma_key: str = "gamma",
    lambda_high_key: str = "lambda_high",
    lambda_mid_key: str = "lambda_mid",
    topk_percent_key: str = "topk_percent",
    ternary_low_quantile_key: str = "ternary_low_quantile",
    ternary_high_quantile_key: str = "ternary_high_quantile",
) -> np.ndarray:
    # [CodeX] 用统一入口支持 linear / power / binary_topk / ternary_quantile 四种权重模式。
    config = config or {}
    epsilon = float(config.get("epsilon", 1.0e-8))
    min_weight = float(config.get("clip_min", config.get("weight_clip_min", 1.0)))
    max_weight = float(config.get("clip_max", config.get("weight_clip_max", 10.0)))
    beta = float(config.get(beta_key, 1.0))
    gamma = float(config.get(gamma_key, 2.0))
    lambda_high = float(config.get(lambda_high_key, max(1.0 + beta, 2.0)))
    lambda_mid = float(config.get(lambda_mid_key, max(1.0 + 0.5 * beta, 1.5)))
    topk_percent = float(config.get(topk_percent_key, 0.2))
    ternary_low_quantile = float(config.get(ternary_low_quantile_key, 0.5))
    ternary_high_quantile = float(config.get(ternary_high_quantile_key, 0.8))
    mode = str(config.get(mode_key, "linear"))

    normalized = np.asarray(normalized_importance, dtype=np.float64)
    normalized[~np.isfinite(normalized)] = 0.0
    normalized = np.clip(normalized, a_min=0.0, a_max=1.0)

    if mode == "linear":
        weights = 1.0 + beta * normalized
    elif mode == "power":
        weights = 1.0 + beta * np.power(normalized, gamma)
    elif mode == "binary_topk":
        threshold = _quantile_threshold_codex(normalized, 1.0 - topk_percent)
        weights = np.ones_like(normalized)
        weights[normalized >= threshold] = lambda_high
    elif mode == "ternary_quantile":
        low_threshold = _quantile_threshold_codex(normalized, ternary_low_quantile)
        high_threshold = _quantile_threshold_codex(normalized, ternary_high_quantile)
        weights = np.ones_like(normalized)
        middle_mask = normalized >= low_threshold
        high_mask = normalized >= high_threshold
        weights[middle_mask] = lambda_mid
        weights[high_mask] = lambda_high
    else:
        raise ValueError(f"Unsupported weight_mode '{mode}'")

    weights[~np.isfinite(weights)] = 1.0
    weights = np.clip(weights, a_min=min_weight, a_max=max_weight)
    return np.maximum(weights, epsilon).astype(np.float32)


def compute_distribution_stats_codex(values: np.ndarray, *, topk_percent: float = 0.2, prefix: str = "") -> Dict[str, float]:
    # [CodeX] 统一输出均值、分位数、头部占比和有效样本比，用来判断权重是否过平或过尖。
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        nan_stats = {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "q50": np.nan,
            "q80": np.nan,
            "q95": np.nan,
            "top20_ratio": np.nan,
            "effective_sample_ratio": np.nan,
        }
        return {f"{prefix}{key}": float(value) for key, value in nan_stats.items()}

    arr = arr.copy()
    arr[~np.isfinite(arr)] = 0.0
    total = float(np.sum(arr))
    sorted_arr = np.sort(arr)
    topk = max(1, int(np.ceil(arr.size * topk_percent)))
    top_ratio = float(np.sum(sorted_arr[-topk:]) / total) if total > 0 else 0.0
    denominator = float(arr.size * np.sum(arr**2))
    effective_ratio = float((total**2) / denominator) if denominator > 0 else 1.0

    stats = {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q50": float(np.quantile(arr, 0.50)),
        "q80": float(np.quantile(arr, 0.80)),
        "q95": float(np.quantile(arr, 0.95)),
        "top20_ratio": top_ratio,
        "effective_sample_ratio": effective_ratio,
    }
    return {f"{prefix}{key}": value for key, value in stats.items()}


def compute_correlation_stats_codex(weights: np.ndarray, labels: np.ndarray, *, epsilon: float = 1.0e-8) -> Dict[str, float]:
    # [CodeX] 同时记录与尺寸标签、与 -log(size) 的相关性，帮助判断物理权重是否与监督目标基本错位。
    weights_np = _clean_flat_array_codex(weights)
    labels_np = _clean_flat_array_codex(labels)
    if weights_np.size == 0 or labels_np.size == 0 or weights_np.size != labels_np.size:
        return {
            "weight_label_pearson": np.nan,
            "weight_label_spearman": np.nan,
            "weight_neg_log_size_pearson": np.nan,
            "weight_neg_log_size_spearman": np.nan,
        }

    neg_log_size = -np.log(np.maximum(labels_np, epsilon))
    return {
        "weight_label_pearson": _pearson_codex(weights_np, labels_np),
        "weight_label_spearman": _spearman_codex(weights_np, labels_np),
        "weight_neg_log_size_pearson": _pearson_codex(weights_np, neg_log_size),
        "weight_neg_log_size_spearman": _spearman_codex(weights_np, neg_log_size),
    }


def compute_size_error_metrics_codex(
    predictions: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    importance: np.ndarray,
    *,
    epsilon: float = 1.0e-8,
    topk_percent: float = 0.2,
    bucket_count: int = 5,
) -> Dict[str, float]:
    # [CodeX] 用统一的高重要区、分桶和加权指标替代单一全域 L2，直接检验误差是否被重新分配。
    pred = _clean_flat_array_codex(predictions)
    target = _clean_flat_array_codex(labels)
    sample_weights = np.maximum(_clean_flat_array_codex(weights), epsilon)
    ranking_importance = _clean_flat_array_codex(importance)

    if pred.size == 0 or target.size == 0 or pred.size != target.size:
        return {
            "weighted_size_l2": np.nan,
            "topk_high_importance_l2": np.nan,
            "bucket_low_size_l2": np.nan,
            "bucket_high_size_l2": np.nan,
            "bucket_high_low_ratio": np.nan,
        }

    if ranking_importance.size != pred.size:
        ranking_importance = sample_weights

    squared_error = np.square(pred - target)
    weighted_size_l2 = float(np.sum(sample_weights * squared_error) / (np.sum(sample_weights) + epsilon))

    topk_indices = _topk_indices_codex(ranking_importance, topk_percent=topk_percent)
    topk_high_importance_l2 = float(np.mean(squared_error[topk_indices])) if topk_indices.size > 0 else float(np.mean(squared_error))

    bucket_metrics = _bucket_error_metrics_codex(
        squared_error=squared_error,
        importance=ranking_importance,
        bucket_count=bucket_count,
        epsilon=epsilon,
    )
    return {
        "weighted_size_l2": weighted_size_l2,
        "topk_high_importance_l2": topk_high_importance_l2,
        **bucket_metrics,
    }


def compute_projection_diagnostics_codex(
    reference_importance: np.ndarray,
    projected_importance: np.ndarray,
    *,
    topk_percent: float = 0.2,
) -> Dict[str, float]:
    # [CodeX] 比较参考网格与投影后网格的重要性分布，判断高重要区域是否在投影时被压平。
    reference_stats = compute_distribution_stats_codex(reference_importance, topk_percent=topk_percent, prefix="reference_")
    projected_stats = compute_distribution_stats_codex(projected_importance, topk_percent=topk_percent, prefix="projected_")
    reference_q95 = reference_stats.get("reference_q95", np.nan)
    projected_q95 = projected_stats.get("projected_q95", np.nan)
    reference_top = reference_stats.get("reference_top20_ratio", np.nan)
    projected_top = projected_stats.get("projected_top20_ratio", np.nan)
    diagnostics = {
        **reference_stats,
        **projected_stats,
        "projection_q95_ratio": float(projected_q95 / reference_q95) if np.isfinite(reference_q95) and reference_q95 > 0 else np.nan,
        "projection_top20_ratio_delta": float(projected_top - reference_top)
        if np.isfinite(reference_top) and np.isfinite(projected_top)
        else np.nan,
    }
    return diagnostics


def should_enable_stage2_codex(
    *,
    current_epoch: int,
    max_epochs: int,
    stage2_enable: bool,
    stage2_epochs: int,
    resumed_from_checkpoint: bool,
    stage2_resume_mode: bool,
) -> bool:
    # [CodeX] 用纯函数统一阶段二切换逻辑，方便在无训练依赖环境下做单元测试。
    if not stage2_enable or stage2_epochs <= 0:
        return False
    if resumed_from_checkpoint and stage2_resume_mode:
        return True
    switch_epoch = max(int(max_epochs) - int(stage2_epochs), 0)
    return int(current_epoch) >= switch_epoch


def _bucket_error_metrics_codex(*, squared_error: np.ndarray, importance: np.ndarray, bucket_count: int, epsilon: float) -> Dict[str, float]:
    order = np.argsort(importance, kind="mergesort")
    buckets = np.array_split(order, max(int(bucket_count), 1))
    bucket_values = []
    metrics = {}
    for bucket_idx, indices in enumerate(buckets):
        if indices.size == 0:
            bucket_error = np.nan
        else:
            bucket_error = float(np.mean(squared_error[indices]))
        bucket_values.append(bucket_error)
        metrics[f"bucket_{bucket_idx}_size_l2"] = bucket_error

    low_value = bucket_values[0] if len(bucket_values) > 0 else np.nan
    high_value = bucket_values[-1] if len(bucket_values) > 0 else np.nan
    metrics["bucket_low_size_l2"] = low_value
    metrics["bucket_high_size_l2"] = high_value
    if np.isfinite(low_value) and low_value > epsilon and np.isfinite(high_value):
        metrics["bucket_high_low_ratio"] = float(high_value / low_value)
    else:
        metrics["bucket_high_low_ratio"] = np.nan
    return metrics


def _topk_indices_codex(values: np.ndarray, *, topk_percent: float) -> np.ndarray:
    arr = _clean_flat_array_codex(values)
    if arr.size == 0:
        return np.zeros(0, dtype=np.int64)
    topk = max(1, int(np.ceil(arr.size * topk_percent)))
    order = np.argsort(arr, kind="mergesort")
    return np.asarray(order[-topk:], dtype=np.int64)


def _quantile_threshold_codex(values: np.ndarray, quantile: float) -> float:
    arr = _clean_flat_array_codex(values)
    if arr.size == 0:
        return 0.0
    return float(np.quantile(arr, np.clip(quantile, 0.0, 1.0)))


def _clean_flat_array_codex(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr.copy()
    arr[~np.isfinite(arr)] = 0.0
    return arr


def _pearson_codex(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return np.nan
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std <= 1.0e-12 or y_std <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _spearman_codex(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return np.nan
    try:
        from scipy.stats import rankdata

        rank_x = rankdata(x)
        rank_y = rankdata(y)
    except Exception:
        rank_x = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort")
        rank_y = np.argsort(np.argsort(y, kind="mergesort"), kind="mergesort")
    return _pearson_codex(np.asarray(rank_x, dtype=np.float64), np.asarray(rank_y, dtype=np.float64))
