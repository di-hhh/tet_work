import warnings
from typing import Any, Dict, Optional

import numpy as np

from src.algorithm.dataloader.source_data import SourceData
from src.algorithm.util.amber_util import interpolate_vertex_field
from src.algorithm.util.console_mold_reference import (
    get_console_mold_reference_fields,
    supports_console_mold_reference,
)
from src.algorithm.util.weighted_imitation_diagnostics_codex import (
    build_weights_from_normalized_importance_codex,
    compute_distribution_stats_codex,
    compute_projection_diagnostics_codex,
    normalize_importance_codex,
)
from src.helpers.custom_types import MeshNodeType, SizingFieldInterpolationType
from src.tasks.domains.mesh_wrapper import MeshWrapper


PIPELINE_PHYSICS_SOURCES = {"pipeline_indicator", "stage_field", "stage_field_fusion"}
STAGE_FIELD_SOURCES = {"stage_field", "stage_field_fusion"}


# [CodeX] 统一入口：在不改训练主链路的前提下，为 Console/Mold 提供可投影的节点物理权重。
def get_imitation_weight_bundle(
    *,
    queried_mesh: MeshWrapper,
    source_data: SourceData,
    sizing_field_interpolation_type: SizingFieldInterpolationType,
    node_type: MeshNodeType,
    weighted_imitation_config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    config = weighted_imitation_config or {}
    if node_type == "pixel":
        return _make_fallback_bundle(queried_mesh=queried_mesh, node_type=node_type, fallback=False)

    if not config.get("enabled", False) and not config.get("metric_use_physics_weights", True):
        return _make_fallback_bundle(queried_mesh=queried_mesh, node_type=node_type, fallback=False)

    weight_mode = config.get("weight_source_mode", "console_mold_reference")
    fallback_to_ones = bool(config.get("fallback_to_ones", True))

    if weight_mode == "ones":
        return _make_fallback_bundle(queried_mesh=queried_mesh, node_type=node_type, fallback=False)

    if weight_mode in PIPELINE_PHYSICS_SOURCES:
        if not _supports_pipeline_source(source_data=source_data, source=str(weight_mode)):
            if fallback_to_ones:
                return _make_fallback_bundle(queried_mesh=queried_mesh, node_type=node_type, fallback=True)
            raise ValueError(f"Pipeline weights were requested from '{weight_mode}', but the source data is incomplete.")
        try:
            if weight_mode in STAGE_FIELD_SOURCES:
                return _bundle_from_pipeline_stage_probes(
                    queried_mesh=queried_mesh,
                    source_data=source_data,
                    source=str(weight_mode),
                    node_type=node_type,
                    config=config,
                )
            reference_fields = _get_pipeline_reference_fields(source_data=source_data, source=str(weight_mode))
            return _bundle_from_reference_fields(
                queried_mesh=queried_mesh,
                source_data=source_data,
                reference_fields=reference_fields,
                sizing_field_interpolation_type=sizing_field_interpolation_type,
                node_type=node_type,
                config=config,
            )
        except Exception as exc:
            if fallback_to_ones:
                warnings.warn(
                    f"Falling back to all-one imitation weights for pipeline sample "
                    f"'{getattr(source_data, 'data_point_path', 'unknown')}'. Reason: {exc}"
                )
                return _make_fallback_bundle(queried_mesh=queried_mesh, node_type=node_type, fallback=True)
            raise

    if weight_mode != "console_mold_reference" or not supports_console_mold_reference(
        source_data=source_data,
        weighted_imitation_config=config,
    ):
        return _make_fallback_bundle(queried_mesh=queried_mesh, node_type=node_type, fallback=True)

    try:
        reference_fields = get_console_mold_reference_fields(
            source_data=source_data,
            weighted_imitation_config=config,
        )
        return _bundle_from_reference_fields(
            queried_mesh=queried_mesh,
            source_data=source_data,
            reference_fields=reference_fields,
            sizing_field_interpolation_type=sizing_field_interpolation_type,
            node_type=node_type,
            config=config,
        )
    except Exception as exc:
        if fallback_to_ones:
            warnings.warn(
                f"Falling back to all-one imitation weights for '{getattr(source_data, 'dataset_name', 'unknown')}' "
                f"sample '{getattr(source_data, 'data_point_path', 'unknown')}'. Reason: {exc}"
            )
            return _make_fallback_bundle(queried_mesh=queried_mesh, node_type=node_type, fallback=True)
        raise


def get_imitation_weights(
    *,
    queried_mesh: MeshWrapper,
    source_data: SourceData,
    sizing_field_interpolation_type: SizingFieldInterpolationType,
    node_type: MeshNodeType,
    weighted_imitation_config: Optional[Dict[str, Any]],
) -> np.ndarray:
    return get_imitation_weight_bundle(
        queried_mesh=queried_mesh,
        source_data=source_data,
        sizing_field_interpolation_type=sizing_field_interpolation_type,
        node_type=node_type,
        weighted_imitation_config=weighted_imitation_config,
    )["weights"]


def _interpolate_reference_weights(
    *,
    queried_mesh: MeshWrapper,
    reference_mesh: MeshWrapper,
    reference_fields: Dict[str, np.ndarray],
    sizing_field_interpolation_type: SizingFieldInterpolationType,
    node_type: MeshNodeType,
) -> np.ndarray:
    matching_reference_field = _reference_field_for_matching_mesh(
        queried_mesh=queried_mesh,
        reference_mesh=reference_mesh,
        reference_fields=reference_fields,
        node_type=node_type,
    )
    if matching_reference_field is not None:
        return matching_reference_field

    if node_type == "vertex":
        if sizing_field_interpolation_type == "interpolated_vertex":
            return interpolate_vertex_field(
                from_mesh=reference_mesh,
                to_mesh=queried_mesh,
                from_scalars=reference_fields["vertex_importance"],
            )
        if sizing_field_interpolation_type == "sampled_vertex":
            correspondences = reference_mesh.find_closest_elements(queried_mesh.vertex_positions)  # [CodeX] 复用 AMBER 现有 sampled-vertex 几何对应逻辑，避免另写一套权重投影查询。
            corresponding_elements = reference_mesh.element_indices[correspondences]
            return reference_fields["vertex_importance"][corresponding_elements].mean(axis=1)
        raise ValueError(f"Unsupported interpolation type '{sizing_field_interpolation_type}' for vertex weights")

    if node_type == "element":
        from src.algorithm.util.interpolate_sizing_field import interpolate_element_field

        return interpolate_element_field(
            fine_mesh=reference_mesh,
            queried_mesh=queried_mesh,
            fine_field=reference_fields["element_importance"],
        )

    raise ValueError(f"Unsupported node_type '{node_type}'")


def _reference_field_for_matching_mesh(
    *,
    queried_mesh: MeshWrapper,
    reference_mesh: MeshWrapper,
    reference_fields: Dict[str, np.ndarray],
    node_type: MeshNodeType,
) -> np.ndarray | None:
    if not _same_mesh_geometry_and_topology(queried_mesh=queried_mesh, reference_mesh=reference_mesh):
        return None
    if node_type == "vertex":
        return np.asarray(reference_fields["vertex_importance"], dtype=np.float32)
    if node_type == "element":
        return np.asarray(reference_fields["element_importance"], dtype=np.float32)
    return None


def _same_mesh_geometry_and_topology(*, queried_mesh: MeshWrapper, reference_mesh: MeshWrapper) -> bool:
    if queried_mesh.num_vertices != reference_mesh.num_vertices:
        return False
    if queried_mesh.num_elements != reference_mesh.num_elements:
        return False
    queried_positions = np.asarray(queried_mesh.vertex_positions, dtype=np.float64)
    reference_positions = np.asarray(reference_mesh.vertex_positions, dtype=np.float64)
    if queried_positions.shape != reference_positions.shape or not np.allclose(queried_positions, reference_positions):
        return False
    queried_elements = np.asarray(queried_mesh.element_indices, dtype=np.int64)
    reference_elements = np.asarray(reference_mesh.element_indices, dtype=np.int64)
    return queried_elements.shape == reference_elements.shape and np.array_equal(queried_elements, reference_elements)


def _bundle_from_reference_fields(
    *,
    queried_mesh: MeshWrapper,
    source_data: SourceData,
    reference_fields: Dict[str, np.ndarray],
    sizing_field_interpolation_type: SizingFieldInterpolationType,
    node_type: MeshNodeType,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    interpolated_weights = _interpolate_reference_weights(
        queried_mesh=queried_mesh,
        reference_mesh=source_data.expert_mesh,
        reference_fields=reference_fields,
        sizing_field_interpolation_type=sizing_field_interpolation_type,
        node_type=node_type,
    )
    reference_importance = _get_reference_importance(reference_fields=reference_fields, node_type=node_type)
    normalized_importance = normalize_importance_codex(interpolated_weights, config=config)
    weights = build_weights_from_normalized_importance_codex(normalized_importance, config=config)
    diagnostic_scalars = {
        **compute_distribution_stats_codex(weights, topk_percent=float(config.get("topk_percent", 0.2)), prefix="imitation_weight_"),
        **compute_projection_diagnostics_codex(
            reference_importance=reference_importance,
            projected_importance=interpolated_weights,
            topk_percent=float(config.get("topk_percent", 0.2)),
        ),
    }
    return {
        "weights": weights,
        "raw_importance": np.asarray(interpolated_weights, dtype=np.float32),
        "normalized_importance": np.asarray(normalized_importance, dtype=np.float32),
        "loaded": True,
        "fallback": False,
        "diagnostic_scalars": diagnostic_scalars,
    }


def _supports_pipeline_indicator(*, source_data: SourceData) -> bool:
    cache = source_data.imitation_weight_cache or {}
    return bool(cache.get("indicator_path"))


def _supports_pipeline_source(*, source_data: SourceData, source: str) -> bool:
    cache = source_data.imitation_weight_cache or {}
    if source == "pipeline_indicator":
        return bool(cache.get("indicator_path"))
    if source in STAGE_FIELD_SOURCES:
        return bool(cache.get("stage_field_path"))
    return False


def _get_pipeline_reference_fields(*, source_data: SourceData, source: str) -> Dict[str, np.ndarray]:
    if source == "pipeline_indicator":
        return _get_pipeline_indicator_reference_fields(source_data=source_data)
    raise ValueError(f"Unsupported pipeline source '{source}'.")


def _get_pipeline_indicator_reference_fields(*, source_data: SourceData) -> Dict[str, np.ndarray]:
    cache = source_data.imitation_weight_cache or {}
    cached_fields = cache.get("_pipeline_indicator_reference_fields")
    if cached_fields is not None:
        return cached_fields

    indicator_path = cache.get("indicator_path")
    if not indicator_path:
        raise ValueError("Pipeline indicator path is missing.")
    element_importance = np.asarray(np.load(indicator_path), dtype=np.float32).reshape(-1)
    expected_elements = int(source_data.expert_mesh.num_elements)
    if element_importance.shape[0] != expected_elements:
        raise ValueError(
            f"Pipeline indicator length {element_importance.shape[0]} does not match expert mesh elements {expected_elements}."
        )
    vertex_importance = _element_values_to_vertices(
        num_vertices=int(source_data.expert_mesh.num_vertices),
        simplices=np.asarray(source_data.expert_mesh.element_indices, dtype=np.int64),
        element_values=element_importance,
    )
    reference_fields = {
        "element_importance": element_importance.astype(np.float32),
        "vertex_importance": vertex_importance.astype(np.float32),
    }
    cache["_pipeline_indicator_reference_fields"] = reference_fields
    return reference_fields


def _bundle_from_pipeline_stage_probes(
    *,
    queried_mesh: MeshWrapper,
    source_data: SourceData,
    source: str,
    node_type: MeshNodeType,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    cache = source_data.imitation_weight_cache or {}
    stage_field_path = cache.get("stage_field_path")
    if not stage_field_path:
        raise ValueError("Pipeline stage field path is missing.")
    stage_field_config = cache.get("stage_field_config") or {}
    projection_target = str(stage_field_config.get("projection_target", "learner_mesh"))
    if projection_target != "learner_mesh":
        raise ValueError(
            "Pipeline stage fields must be projected directly to the current learner mesh; "
            f"got projection_target='{projection_target}'."
        )

    cache_key = f"_pipeline_{source}_probe_field"
    cached_probe_field = cache.get(cache_key)
    if cached_probe_field is None:
        cached_probe_field = _load_stage_probe_importance(
            stage_field_path=stage_field_path,
            stage_field_config=stage_field_config,
            source=source,
        )
        cache[cache_key] = cached_probe_field
    probe_points, probe_importance = cached_probe_field

    if node_type == "vertex":
        query_points = np.asarray(queried_mesh.vertex_positions, dtype=np.float64)
    elif node_type == "element":
        vertices = np.asarray(queried_mesh.vertex_positions, dtype=np.float64)
        simplices = np.asarray(queried_mesh.element_indices, dtype=np.int64)
        query_points = vertices[simplices].mean(axis=1)
    else:
        raise ValueError(f"Unsupported node_type '{node_type}' for direct stage-field projection.")

    projected_importance = _project_probe_field_to_points(
        probe_points=probe_points,
        probe_values=probe_importance,
        query_points=query_points,
        stage_field_config=stage_field_config,
    )
    projected_importance = _normalize_stage_importance(
        projected_importance,
        stage_field_config,
    ).astype(np.float32)
    normalized_importance = normalize_importance_codex(projected_importance, config=config)
    weights = build_weights_from_normalized_importance_codex(normalized_importance, config=config)
    diagnostic_scalars = {
        **compute_distribution_stats_codex(
            weights,
            topk_percent=float(config.get("topk_percent", 0.2)),
            prefix="imitation_weight_",
        ),
        **compute_projection_diagnostics_codex(
            reference_importance=probe_importance,
            projected_importance=projected_importance,
            topk_percent=float(config.get("topk_percent", 0.2)),
        ),
    }
    return {
        "weights": weights,
        "raw_importance": projected_importance,
        "normalized_importance": np.asarray(normalized_importance, dtype=np.float32),
        "loaded": True,
        "fallback": False,
        "diagnostic_scalars": diagnostic_scalars,
    }


def _load_stage_probe_importance(
    *,
    stage_field_path: str,
    stage_field_config: Dict[str, Any],
    source: str,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(stage_field_path) as payload:
        if "probe_points" not in payload.files:
            raise ValueError(f"Pipeline stage field '{stage_field_path}' does not contain probe_points.")
        probe_points = np.asarray(payload["probe_points"], dtype=np.float64)
        if probe_points.ndim != 2:
            raise ValueError(f"Pipeline stage field probe_points must be 2D, got shape {probe_points.shape}.")
        if source == "stage_field":
            probe_importance = _stage_field_single_importance(payload=payload, stage_field_config=stage_field_config)
        elif source == "stage_field_fusion":
            mode = str(stage_field_config.get("mode", "single_field"))
            if mode == "single_field":
                probe_importance = _stage_field_single_importance(payload=payload, stage_field_config=stage_field_config)
            elif mode == "weighted_mean":
                probe_importance = _stage_field_weighted_mean_importance(payload=payload, stage_field_config=stage_field_config)
            else:
                raise ValueError(f"Unsupported stage_field.mode '{mode}'.")
        else:
            raise ValueError(f"Unsupported stage field source '{source}'.")

    if probe_importance.shape[0] != probe_points.shape[0]:
        raise ValueError(
            f"Pipeline stage field has {probe_points.shape[0]} probe points but {probe_importance.shape[0]} values."
        )
    return probe_points, probe_importance.astype(np.float32)


def _stage_field_single_importance(*, payload, stage_field_config: Dict[str, Any]) -> np.ndarray:
    field_config = stage_field_config.get("single_field", {}) or {}
    if not field_config:
        field_config = {"name": "s_pde_raw", "direction": "high_is_important", "transform": "log1p", "normalize": "robust_minmax"}
    return _stage_field_to_importance(payload=payload, field_config=field_config)


def _stage_field_weighted_mean_importance(*, payload, stage_field_config: Dict[str, Any]) -> np.ndarray:
    fusion_config = stage_field_config.get("fusion", {}) or {}
    fields = fusion_config.get("fields", []) or []
    if not fields:
        return _stage_field_single_importance(payload=payload, stage_field_config=stage_field_config)

    weighted_sum = None
    total_weight = 0.0
    for field_config in fields:
        field_weight = float(field_config.get("weight", 1.0))
        if field_weight <= 0.0:
            continue
        field_importance = _stage_field_to_importance(payload=payload, field_config=field_config)
        if weighted_sum is None:
            weighted_sum = np.zeros_like(field_importance, dtype=np.float64)
        weighted_sum += field_weight * field_importance
        total_weight += field_weight
    if weighted_sum is None or total_weight <= 0.0:
        raise ValueError("stage_field.fusion.fields must contain at least one positive-weight field.")
    fused = weighted_sum / total_weight
    if bool(fusion_config.get("renormalize", True)):
        fused = _normalize_stage_importance(fused, fusion_config)
    return fused.astype(np.float32)


def _stage_field_to_importance(*, payload, field_config: Dict[str, Any]) -> np.ndarray:
    field_name = str(field_config.get("name", "s_pde_raw"))
    if field_name not in payload.files:
        raise ValueError(f"Pipeline stage field does not contain '{field_name}'.")
    values = np.asarray(payload[field_name], dtype=np.float64).reshape(-1)
    values[~np.isfinite(values)] = 0.0
    values = _transform_stage_values(values=values, transform=str(field_config.get("transform", "identity")))

    direction = str(field_config.get("direction", "high_is_important"))
    if direction == "low_is_important":
        max_value = float(np.max(values)) if values.size > 0 else 0.0
        values = max_value - values
    elif direction != "high_is_important":
        raise ValueError(f"Unsupported stage field direction '{direction}'.")
    return _normalize_stage_importance(values, field_config)


def _transform_stage_values(*, values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        transformed = values
    elif transform == "log1p":
        transformed = np.log1p(np.maximum(values, 0.0))
    elif transform == "sqrt":
        transformed = np.sqrt(np.maximum(values, 0.0))
    else:
        raise ValueError(f"Unsupported stage field transform '{transform}'.")
    transformed = np.asarray(transformed, dtype=np.float64)
    transformed[~np.isfinite(transformed)] = 0.0
    return transformed


def _normalize_stage_importance(values: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values[~np.isfinite(values)] = 0.0
    values = np.maximum(values, 0.0)

    normalize_mode = str(config.get("normalize", "robust_minmax"))
    if normalize_mode in {"none", "identity"}:
        return values.astype(np.float32)
    if normalize_mode != "robust_minmax":
        raise ValueError(f"Unsupported stage field normalize mode '{normalize_mode}'.")

    lower_quantile = float(config.get("robust_lower_quantile", config.get("normalization_lower_quantile", 0.05)))
    upper_quantile = float(config.get("robust_upper_quantile", config.get("normalization_upper_quantile", 0.95)))
    epsilon = float(config.get("epsilon", 1.0e-8))
    q_low = float(np.quantile(values, lower_quantile)) if values.size > 0 else 0.0
    q_high = float(np.quantile(values, upper_quantile)) if values.size > 0 else 0.0
    if q_high <= q_low + epsilon:
        return np.zeros_like(values, dtype=np.float32)
    normalized = (values - q_low) / (q_high - q_low + epsilon)
    normalized[~np.isfinite(normalized)] = 0.0
    return np.clip(normalized, a_min=0.0, a_max=1.0).astype(np.float32)


def _project_probe_field_to_points(
    *,
    probe_points: np.ndarray,
    probe_values: np.ndarray,
    query_points: np.ndarray,
    stage_field_config: Dict[str, Any],
) -> np.ndarray:
    projection = str(stage_field_config.get("projection", "idw"))
    if projection != "idw":
        raise ValueError(f"Unsupported stage field projection '{projection}'.")
    if probe_points.shape[0] == 0:
        raise ValueError("Pipeline stage field contains no probe points.")

    from pykdtree.kdtree import KDTree

    k = min(int(stage_field_config.get("idw_k", 4)), int(probe_points.shape[0]))
    epsilon = float(stage_field_config.get("idw_epsilon", 1.0e-12))
    tree = KDTree(np.asarray(probe_points, dtype=np.float64))
    distances, indices = tree.query(np.asarray(query_points, dtype=np.float64), k=k)
    distances = np.asarray(distances, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    gathered_values = np.asarray(probe_values, dtype=np.float64)[indices]
    exact_mask = distances <= epsilon
    weights = 1.0 / np.maximum(distances, epsilon) ** 2
    weights[exact_mask] = 0.0
    has_exact = np.any(exact_mask, axis=1)
    if np.any(has_exact):
        first_exact = np.argmax(exact_mask[has_exact], axis=1)
        gathered_values[has_exact, 0] = gathered_values[has_exact, first_exact]
        weights[has_exact, :] = 0.0
        weights[has_exact, 0] = 1.0
    denominator = np.sum(weights, axis=1)
    denominator[denominator <= 0.0] = 1.0
    projected = np.sum(gathered_values * weights, axis=1) / denominator
    projected[~np.isfinite(projected)] = 0.0
    return projected.astype(np.float32)


def _element_values_to_vertices(*, num_vertices: int, simplices: np.ndarray, element_values: np.ndarray) -> np.ndarray:
    element_values = np.asarray(element_values, dtype=np.float32).reshape(-1)
    accum = np.zeros(num_vertices, dtype=np.float64)
    counts = np.zeros(num_vertices, dtype=np.float64)
    np.add.at(accum, simplices.reshape(-1), np.repeat(element_values, simplices.shape[1]))
    np.add.at(counts, simplices.reshape(-1), 1.0)
    return (accum / np.maximum(counts, 1.0)).astype(np.float32)


def _ones_for_mesh(mesh: MeshWrapper, node_type: MeshNodeType) -> np.ndarray:
    if node_type == "vertex":
        size = mesh.num_vertices
    elif node_type == "element":
        size = mesh.num_elements
    else:
        size = mesh.num_vertices
    return np.ones(size, dtype=np.float32)


def _make_fallback_bundle(*, queried_mesh: MeshWrapper, node_type: MeshNodeType, fallback: bool) -> Dict[str, Any]:
    weights = _ones_for_mesh(queried_mesh, node_type)
    zeros = np.zeros_like(weights, dtype=np.float32)
    diagnostic_scalars = compute_distribution_stats_codex(weights, prefix="imitation_weight_")
    diagnostic_scalars |= compute_projection_diagnostics_codex(reference_importance=zeros, projected_importance=zeros)
    return {
        "weights": weights,
        "raw_importance": zeros,
        "normalized_importance": zeros,
        "loaded": False,
        "fallback": fallback,
        "diagnostic_scalars": diagnostic_scalars,
    }


def _get_reference_importance(*, reference_fields: Dict[str, np.ndarray], node_type: MeshNodeType) -> np.ndarray:
    if node_type == "element":
        return np.asarray(reference_fields["element_importance"], dtype=np.float64)
    return np.asarray(reference_fields["vertex_importance"], dtype=np.float64)
