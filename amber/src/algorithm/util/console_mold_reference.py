from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.algorithm.util.linear_elasticity_reference_codex import (
    build_linear_elasticity_reference_fields_codex,
)


def supports_console_mold_reference(*, source_data, weighted_imitation_config: Optional[Dict[str, Any]]) -> bool:
    config = weighted_imitation_config or {}
    enabled_datasets = set(config.get("datasets", ["console", "mold"]))
    dataset_name = getattr(source_data, "dataset_name", None)
    return dataset_name in {"console", "mold"} and dataset_name in enabled_datasets


def get_console_mold_reference_fields(*, source_data, weighted_imitation_config: Optional[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    config = weighted_imitation_config or {}
    if source_data.imitation_weight_cache is None:
        source_data.imitation_weight_cache = {}

    cache_key = "console_mold_reference"
    if cache_key in source_data.imitation_weight_cache:
        return source_data.imitation_weight_cache[cache_key]

    cache_path = resolve_console_mold_cache_path(source_data=source_data, weighted_imitation_config=config)
    if not cache_path.exists():
        if bool(config.get("auto_prepare", False)):
            ensure_console_mold_reference_cache(
                source_data=source_data,
                weighted_imitation_config=config,
                overwrite=bool(config.get("overwrite_cache", False)),
            )
        else:
            raise FileNotFoundError(f"Missing weighted imitation cache at '{cache_path}'.")

    fields = _load_cached_fields(cache_path=cache_path)
    if _cache_requires_refresh(fields=fields, config=config):
        if bool(config.get("auto_prepare", False)):
            ensure_console_mold_reference_cache(
                source_data=source_data,
                weighted_imitation_config=config,
                overwrite=True,
            )  # [CodeX] 当默认参考物理从 harmonic 切换到线弹性后，自动刷新旧缓存，避免继续读到过期权重。
            fields = _load_cached_fields(cache_path=cache_path)
        else:
            warnings.warn(
                f"Weighted imitation cache '{cache_path}' does not match the requested reference physics. "
                "Using the existing cache because auto_prepare is disabled."
            )

    source_data.imitation_weight_cache[cache_key] = fields
    return fields


def ensure_console_mold_reference_cache(
    *,
    source_data,
    weighted_imitation_config: Optional[Dict[str, Any]],
    overwrite: bool = False,
) -> Path:
    config = weighted_imitation_config or {}
    cache_path = resolve_console_mold_cache_path(
        source_data=source_data,
        weighted_imitation_config=config,
    )
    if cache_path.exists() and not overwrite:
        return cache_path

    fields = build_console_mold_reference_fields(
        source_data=source_data,
        weighted_imitation_config=config,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **fields)
    return cache_path


def resolve_console_mold_cache_path(*, source_data, weighted_imitation_config: Optional[Dict[str, Any]]) -> Path:
    config = weighted_imitation_config or {}
    dataset_name = getattr(source_data, "dataset_name", None)
    data_point_path = getattr(source_data, "data_point_path", None)
    if dataset_name is None or data_point_path is None:
        raise ValueError("Console/Mold weighted imitation requires source_data.dataset_name and source_data.data_point_path.")

    cache_dir = Path(config.get("cache_dir", "data/weighted_imitation"))
    sample_path = Path(data_point_path)
    dataset_mode = sample_path.parent.name
    sample_name = sample_path.name
    return cache_dir / dataset_name / dataset_mode / f"{sample_name}.npz"


def build_console_mold_reference_fields(
    *,
    source_data,
    weighted_imitation_config: Optional[Dict[str, Any]],
) -> Dict[str, np.ndarray]:
    # [CodeX] Console/Mold 默认走三维线弹性参考求解；harmonic 仅保留为显式兼容模式。
    config = weighted_imitation_config or {}
    reference_physics_type = str(config.get("reference_physics_type", "linear_elasticity"))
    if reference_physics_type == "linear_elasticity":
        return build_linear_elasticity_reference_fields_codex(
            source_data=source_data,
            weighted_imitation_config=config,
        )
    if reference_physics_type == "harmonic":
        return _build_harmonic_reference_fields_compatibility(source_data=source_data)
    raise ValueError(f"Unsupported reference_physics_type '{reference_physics_type}'")


def _build_harmonic_reference_fields_compatibility(*, source_data) -> Dict[str, np.ndarray]:
    expert_mesh = source_data.expert_mesh
    dataset_name = getattr(source_data, "dataset_name", None)
    inlet_position = _get_inlet_position(source_data=source_data)
    source_nodes, sink_nodes = _select_reference_boundary_nodes(
        expert_mesh=expert_mesh,
        dataset_name=dataset_name,
        inlet_position=inlet_position,
    )
    solution = _solve_harmonic_reference_field(
        expert_mesh=expert_mesh,
        source_nodes=source_nodes,
        sink_nodes=sink_nodes,
    )
    element_importance = _harmonic_energy_density_compatibility(
        element_indices=expert_mesh.element_indices,
        vertex_positions=expert_mesh.vertex_positions,
        solution=solution,
    )
    vertex_importance = _element_to_vertex_importance(
        element_indices=expert_mesh.element_indices,
        element_importance=element_importance,
        element_volumes=expert_mesh.simplex_volumes,
        num_vertices=expert_mesh.num_vertices,
    )
    return {
        "solution": solution.astype(np.float32),
        "element_importance": element_importance.astype(np.float32),
        "vertex_importance": vertex_importance.astype(np.float32),
        "source_nodes": source_nodes.astype(np.int64),
        "sink_nodes": sink_nodes.astype(np.int64),
        "reference_physics_type": np.array(["harmonic"]),
        "importance_metric": np.array(["grad_u_squared"]),
    }


def _cache_requires_refresh(*, fields: Dict[str, np.ndarray], config: Dict[str, Any]) -> bool:
    expected_reference = str(config.get("reference_physics_type", "linear_elasticity"))
    expected_importance = str(config.get("importance_metric", "strain_energy_density"))
    actual_reference = _extract_cache_string(fields.get("reference_physics_type"))
    actual_importance = _extract_cache_string(fields.get("importance_metric"))
    if actual_reference is None or actual_importance is None:
        return bool(config.get("auto_prepare", False))
    return actual_reference != expected_reference or actual_importance != expected_importance


def _load_cached_fields(*, cache_path: Path) -> Dict[str, np.ndarray]:
    with np.load(cache_path, allow_pickle=True) as saved_fields:
        return {key: saved_fields[key] for key in saved_fields.files}


def _extract_cache_string(value: Optional[np.ndarray]) -> Optional[str]:
    if value is None:
        return None
    try:
        extracted = value.tolist()
        if isinstance(extracted, list):
            extracted = extracted[0]
        if isinstance(extracted, bytes):
            extracted = extracted.decode("utf-8")
        return str(extracted)
    except Exception:
        return None


def _get_inlet_position(*, source_data) -> Optional[np.ndarray]:
    feature_provider = getattr(source_data, "feature_provider", None)
    inlet_position = getattr(feature_provider, "inlet_position", None)
    if inlet_position is None:
        return None
    inlet_position = np.asarray(inlet_position, dtype=np.float64)
    if inlet_position.ndim != 1:
        return None
    return inlet_position


def _select_reference_boundary_nodes(*, expert_mesh, dataset_name: Optional[str], inlet_position: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    boundary_nodes = np.asarray(expert_mesh.mesh.boundary_nodes(), dtype=np.int64)
    if len(boundary_nodes) == 0:
        raise ValueError("The expert mesh does not expose any boundary nodes.")

    boundary_positions = np.asarray(expert_mesh.vertex_positions[boundary_nodes], dtype=np.float64)
    if dataset_name == "mold":
        return _select_mold_boundary_nodes(boundary_nodes=boundary_nodes, boundary_positions=boundary_positions, inlet_position=inlet_position)
    return _select_console_boundary_nodes(boundary_nodes=boundary_nodes, boundary_positions=boundary_positions)


def _select_console_boundary_nodes(*, boundary_nodes: np.ndarray, boundary_positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    spans = boundary_positions.max(axis=0) - boundary_positions.min(axis=0)
    dominant_axis = int(np.argmax(spans))
    axis_coordinates = boundary_positions[:, dominant_axis]

    low_quantile, high_quantile = np.quantile(axis_coordinates, [0.05, 0.95])
    source_nodes = boundary_nodes[axis_coordinates <= low_quantile]
    sink_nodes = boundary_nodes[axis_coordinates >= high_quantile]
    return _finalize_boundary_node_sets(
        source_nodes=source_nodes,
        sink_nodes=sink_nodes,
        fallback_scores=axis_coordinates,
        candidate_nodes=boundary_nodes,
    )


def _select_mold_boundary_nodes(
    *,
    boundary_nodes: np.ndarray,
    boundary_positions: np.ndarray,
    inlet_position: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    if inlet_position is None or inlet_position.shape[0] != boundary_positions.shape[1]:
        warnings.warn("Mold weighted imitation is missing a valid inlet position. Falling back to the console-style boundary setup.")
        return _select_console_boundary_nodes(boundary_nodes=boundary_nodes, boundary_positions=boundary_positions)

    distances = np.linalg.norm(boundary_positions - inlet_position[None, :], axis=1)
    order = np.argsort(distances)
    neighborhood_size = max(4, int(np.ceil(0.02 * len(boundary_nodes))))
    source_nodes = boundary_nodes[order[:neighborhood_size]]
    sink_nodes = boundary_nodes[order[-neighborhood_size:]]
    return _finalize_boundary_node_sets(
        source_nodes=source_nodes,
        sink_nodes=sink_nodes,
        fallback_scores=distances,
        candidate_nodes=boundary_nodes,
    )


def _finalize_boundary_node_sets(
    *,
    source_nodes: np.ndarray,
    sink_nodes: np.ndarray,
    fallback_scores: np.ndarray,
    candidate_nodes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    source_nodes = np.unique(np.asarray(source_nodes, dtype=np.int64))
    sink_nodes = np.unique(np.asarray(sink_nodes, dtype=np.int64))
    sink_nodes = sink_nodes[~np.isin(sink_nodes, source_nodes)]

    if len(source_nodes) == 0 or len(sink_nodes) == 0:
        sorted_indices = np.argsort(fallback_scores)
        neighborhood_size = max(1, min(8, len(candidate_nodes) // 10 if len(candidate_nodes) > 10 else 1))
        source_nodes = np.unique(candidate_nodes[sorted_indices[:neighborhood_size]])
        sink_candidates = candidate_nodes[sorted_indices[::-1]]
        sink_nodes = sink_candidates[~np.isin(sink_candidates, source_nodes)][:neighborhood_size]

    if len(source_nodes) == 0 or len(sink_nodes) == 0:
        raise ValueError("Could not construct a non-empty pair of source/sink boundary node sets.")

    return source_nodes.astype(np.int64), sink_nodes.astype(np.int64)


def _solve_harmonic_reference_field(*, expert_mesh, source_nodes: np.ndarray, sink_nodes: np.ndarray) -> np.ndarray:
    try:
        from skfem import Basis, ElementTetP1, asm, condense, solve
        from skfem.models import laplace
    except ImportError as exc:
        raise ImportError(
            "Console/Mold weighted imitation requires scikit-fem in the active environment to prepare reference weights."
        ) from exc

    basis = Basis(expert_mesh.mesh, ElementTetP1())
    stiffness = asm(laplace, basis)
    rhs = basis.zeros()
    boundary_values = basis.zeros()
    boundary_values[source_nodes] = 1.0
    boundary_values[sink_nodes] = 0.0

    fixed_nodes = np.unique(np.concatenate([source_nodes, sink_nodes]))
    interior_nodes = np.setdiff1d(np.arange(expert_mesh.num_vertices, dtype=np.int64), fixed_nodes)
    if len(interior_nodes) == 0:
        return boundary_values.astype(np.float64)

    condensed_system = condense(stiffness, rhs, x=boundary_values, I=interior_nodes)
    solution = solve(*condensed_system)
    return np.asarray(solution, dtype=np.float64)


def _element_to_vertex_importance(
    *,
    element_indices: np.ndarray,
    element_importance: np.ndarray,
    element_volumes: np.ndarray,
    num_vertices: int,
) -> np.ndarray:
    vertex_importance = np.zeros(num_vertices, dtype=np.float64)
    vertex_volume = np.zeros(num_vertices, dtype=np.float64)

    for element_idx, vertex_ids in enumerate(element_indices):
        element_volume = float(element_volumes[element_idx])
        contribution = float(element_importance[element_idx]) * element_volume
        vertex_importance[vertex_ids] += contribution
        vertex_volume[vertex_ids] += element_volume

    vertex_volume = np.clip(vertex_volume, a_min=1.0e-12, a_max=None)
    return vertex_importance / vertex_volume


def _harmonic_energy_density_compatibility(
    *,
    element_indices: np.ndarray,
    vertex_positions: np.ndarray,
    solution: np.ndarray,
) -> np.ndarray:
    element_vertex_positions = np.asarray(vertex_positions[element_indices], dtype=np.float64)
    element_solution = np.asarray(solution[element_indices], dtype=np.float64)

    ones = np.ones((element_vertex_positions.shape[0], element_vertex_positions.shape[1], 1), dtype=np.float64)
    local_system = np.concatenate([ones, element_vertex_positions], axis=2)
    coefficients = np.linalg.solve(local_system, element_solution[..., None]).squeeze(-1)
    gradients = coefficients[:, 1:]

    importance = np.sum(gradients**2, axis=1)
    importance = _clean_nonfinite(importance)
    importance = np.maximum(importance, 0.0)
    return importance


def _clean_nonfinite(values: np.ndarray) -> np.ndarray:
    clean_values = np.array(values, dtype=np.float64, copy=True)
    clean_values[~np.isfinite(clean_values)] = 0.0
    return clean_values
