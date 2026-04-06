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
