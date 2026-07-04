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

    if weight_mode == "pipeline_indicator":
        if not _supports_pipeline_indicator(source_data=source_data):
            if fallback_to_ones:
                return _make_fallback_bundle(queried_mesh=queried_mesh, node_type=node_type, fallback=True)
            raise ValueError("Pipeline indicator weights were requested, but the source data has no indicator path.")
        try:
            reference_fields = _get_pipeline_indicator_reference_fields(source_data=source_data)
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
    return cache.get("weight_source_mode") == "pipeline_indicator" and bool(cache.get("indicator_path"))


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
