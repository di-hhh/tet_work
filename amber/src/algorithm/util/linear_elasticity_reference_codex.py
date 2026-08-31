from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, eye
from scipy.sparse.linalg import spsolve


@dataclass
class LinearElasticitySetupCodex:
    # [CodeX] 记录线弹性参考问题的边界条件与载荷装配结果，供缓存式预处理复用。
    fixed_nodes: np.ndarray
    load_faces: np.ndarray
    load_vectors: np.ndarray
    body_force: Optional[np.ndarray]
    principal_axes: np.ndarray
    load_type: str


def build_linear_elasticity_reference_fields_codex(*, source_data, weighted_imitation_config: Optional[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    # [CodeX] Console/Mold 默认改为三维线弹性静力参考解；这里只负责生成可缓存的参考场与重要性。
    config = weighted_imitation_config or {}
    expert_mesh = source_data.expert_mesh
    dataset_name = getattr(source_data, "dataset_name", None)
    if dataset_name not in {"console", "mold"}:
        raise ValueError(f"Linear elasticity reference is only implemented for console/mold, got '{dataset_name}'.")

    if expert_mesh.dim() != 3:
        raise ValueError("Linear elasticity reference expects a 3D tetrahedral expert mesh.")

    inlet_position = _get_inlet_position_codex(source_data=source_data)
    surface = _get_boundary_surface_codex(expert_mesh=expert_mesh)
    principal_axes = _principal_axes_codex(vertex_positions=expert_mesh.vertex_positions)

    if dataset_name == "mold":
        setup = _build_mold_setup_codex(
            expert_mesh=expert_mesh,
            surface=surface,
            principal_axes=principal_axes,
            inlet_position=inlet_position,
            config=config,
        )
    else:
        setup = _build_console_setup_codex(
            expert_mesh=expert_mesh,
            surface=surface,
            principal_axes=principal_axes,
            config=config,
        )

    solution_bundle = solve_linear_elasticity_reference_codex(
        vertex_positions=np.asarray(expert_mesh.vertex_positions, dtype=np.float64),
        element_indices=np.asarray(expert_mesh.element_indices, dtype=np.int64),
        fixed_nodes=np.asarray(setup.fixed_nodes, dtype=np.int64),
        load_faces=np.asarray(setup.load_faces, dtype=np.int64),
        load_vectors=np.asarray(setup.load_vectors, dtype=np.float64),
        body_force=None if setup.body_force is None else np.asarray(setup.body_force, dtype=np.float64),
        young_modulus=float(config.get("young_modulus", 1.0)),
        poisson_ratio=float(config.get("poisson_ratio", 0.30)),
        importance_metric=str(config.get("importance_metric", "strain_energy_density")),
        regularization_epsilon=float(config.get("solver_regularization_epsilon", 1.0e-10)),
    )

    vertex_importance = element_to_vertex_importance_codex(
        element_indices=np.asarray(expert_mesh.element_indices, dtype=np.int64),
        element_importance=solution_bundle["element_importance"],
        element_volumes=solution_bundle["element_volumes"],
        num_vertices=expert_mesh.num_vertices,
    )
    return {
        "displacement": solution_bundle["displacement"].astype(np.float32),
        "element_importance": solution_bundle["element_importance"].astype(np.float32),
        "vertex_importance": vertex_importance.astype(np.float32),
        "fixed_nodes": np.asarray(setup.fixed_nodes, dtype=np.int64),
        "load_faces": np.asarray(setup.load_faces, dtype=np.int64),
        "load_vectors": np.asarray(setup.load_vectors, dtype=np.float32),
        "reference_physics_type": np.array(["linear_elasticity"]),
        "importance_metric": np.array([str(config.get("importance_metric", "strain_energy_density"))]),
        "young_modulus": np.array([float(config.get("young_modulus", 1.0))], dtype=np.float32),
        "poisson_ratio": np.array([float(config.get("poisson_ratio", 0.30))], dtype=np.float32),
    }


def solve_linear_elasticity_reference_codex(
    *,
    vertex_positions: np.ndarray,
    element_indices: np.ndarray,
    fixed_nodes: np.ndarray,
    load_faces: np.ndarray,
    load_vectors: np.ndarray,
    body_force: Optional[np.ndarray],
    young_modulus: float,
    poisson_ratio: float,
    importance_metric: str,
    regularization_epsilon: float,
) -> Dict[str, np.ndarray]:
    # [CodeX] 线弹性求解只在离线缓存阶段执行，不进入逐步训练循环。
    stiffness_bundle = assemble_linear_elasticity_system_codex(
        vertex_positions=vertex_positions,
        element_indices=element_indices,
        young_modulus=young_modulus,
        poisson_ratio=poisson_ratio,
    )
    stiffness = stiffness_bundle["stiffness"]
    element_b_matrices = stiffness_bundle["element_b_matrices"]
    element_stiffness = stiffness_bundle["element_stiffness"]
    element_volumes = stiffness_bundle["element_volumes"]
    constitutive_matrix = stiffness_bundle["constitutive_matrix"]

    rhs = np.zeros(3 * vertex_positions.shape[0], dtype=np.float64)
    if body_force is not None:
        rhs += assemble_body_force_codex(
            vertex_positions=vertex_positions,
            element_indices=element_indices,
            body_force=body_force,
            element_volumes=element_volumes,
        )
    if len(load_faces) > 0:
        rhs += assemble_surface_load_codex(
            vertex_positions=vertex_positions,
            load_faces=load_faces,
            load_vectors=load_vectors,
        )

    displacement = solve_constrained_linear_system_codex(
        stiffness=stiffness,
        rhs=rhs,
        fixed_nodes=fixed_nodes,
        regularization_epsilon=regularization_epsilon,
    )

    element_importance = np.zeros(len(element_indices), dtype=np.float64)
    for element_idx, element_nodes in enumerate(element_indices):
        if element_volumes[element_idx] <= 0:
            continue
        local_dofs = _node_ids_to_dofs_codex(element_nodes)
        local_displacement = displacement.reshape(-1)[local_dofs]
        strain = element_b_matrices[element_idx].dot(local_displacement)
        stress = constitutive_matrix.dot(strain)
        if importance_metric == "strain_energy_density":
            importance = 0.5 * float(strain.dot(stress))
        elif importance_metric == "von_mises":
            importance = von_mises_from_stress_codex(stress)
        else:
            raise ValueError(f"Unsupported importance_metric '{importance_metric}'")
        element_importance[element_idx] = max(float(importance), 0.0)

    return {
        "stiffness": stiffness,
        "displacement": displacement.reshape(-1, 3),
        "element_importance": _clean_nonfinite_codex(element_importance),
        "element_stiffness": element_stiffness,
        "element_volumes": element_volumes,
    }


def assemble_linear_elasticity_system_codex(
    *,
    vertex_positions: np.ndarray,
    element_indices: np.ndarray,
    young_modulus: float,
    poisson_ratio: float,
) -> Dict[str, Any]:
    # [CodeX] 使用四节点线性四面体单元装配全局刚度矩阵，单元内采用常应变近似。
    num_vertices = vertex_positions.shape[0]
    constitutive_matrix = isotropic_constitutive_matrix_codex(young_modulus=young_modulus, poisson_ratio=poisson_ratio)
    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    data: List[np.ndarray] = []
    element_b_matrices = []
    element_stiffness = []
    element_volumes = np.zeros(len(element_indices), dtype=np.float64)

    for element_idx, element_nodes in enumerate(element_indices):
        element_coordinates = np.asarray(vertex_positions[element_nodes], dtype=np.float64)
        try:
            volume, gradients = tetra_volume_and_gradients_codex(element_coordinates=element_coordinates)
        except np.linalg.LinAlgError:
            volume = 0.0
            gradients = np.zeros((4, 3), dtype=np.float64)
        element_volumes[element_idx] = volume
        b_matrix = strain_displacement_matrix_codex(gradients=gradients)
        ke = b_matrix.T.dot(constitutive_matrix).dot(b_matrix) * volume
        element_b_matrices.append(b_matrix)
        element_stiffness.append(ke)

        local_dofs = _node_ids_to_dofs_codex(element_nodes)
        rr, cc = np.meshgrid(local_dofs, local_dofs, indexing="ij")
        rows.append(rr.reshape(-1))
        cols.append(cc.reshape(-1))
        data.append(ke.reshape(-1))

    stiffness = coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(3 * num_vertices, 3 * num_vertices),
    ).tocsr()
    return {
        "stiffness": stiffness,
        "element_b_matrices": element_b_matrices,
        "element_stiffness": element_stiffness,
        "element_volumes": element_volumes,
        "constitutive_matrix": constitutive_matrix,
    }


def assemble_surface_load_codex(*, vertex_positions: np.ndarray, load_faces: np.ndarray, load_vectors: np.ndarray) -> np.ndarray:
    num_vertices = vertex_positions.shape[0]
    rhs = np.zeros(3 * num_vertices, dtype=np.float64)
    for face_idx, face_nodes in enumerate(load_faces):
        triangle = np.asarray(vertex_positions[face_nodes], dtype=np.float64)
        area = triangle_area_codex(triangle=triangle)
        local_force = np.tile(load_vectors[face_idx] * area / 3.0, 3)
        rhs[_node_ids_to_dofs_codex(face_nodes)] += local_force
    return rhs


def assemble_body_force_codex(
    *,
    vertex_positions: np.ndarray,
    element_indices: np.ndarray,
    body_force: np.ndarray,
    element_volumes: np.ndarray,
) -> np.ndarray:
    num_vertices = vertex_positions.shape[0]
    rhs = np.zeros(3 * num_vertices, dtype=np.float64)
    for element_idx, element_nodes in enumerate(element_indices):
        local_force = np.tile(body_force * element_volumes[element_idx] / 4.0, 4)
        rhs[_node_ids_to_dofs_codex(element_nodes)] += local_force
    return rhs


def solve_constrained_linear_system_codex(
    *,
    stiffness: csr_matrix,
    rhs: np.ndarray,
    fixed_nodes: np.ndarray,
    regularization_epsilon: float,
) -> np.ndarray:
    # [CodeX] 通过消元固定自由度来施加全位移约束，保证参考问题是确定的静力学线性系统。
    num_vertices = stiffness.shape[0] // 3
    fixed_dofs = np.concatenate([3 * fixed_nodes + axis for axis in range(3)]).astype(np.int64)
    fixed_dofs = np.unique(fixed_dofs)
    free_dofs = np.setdiff1d(np.arange(3 * num_vertices, dtype=np.int64), fixed_dofs)
    displacement = np.zeros(3 * num_vertices, dtype=np.float64)

    if len(free_dofs) == 0:
        return displacement

    reduced_stiffness = stiffness[free_dofs][:, free_dofs]
    reduced_rhs = rhs[free_dofs]
    diagonal_scale = float(np.mean(np.abs(reduced_stiffness.diagonal()))) if reduced_stiffness.nnz > 0 else 1.0
    if not np.isfinite(diagonal_scale) or diagonal_scale <= 0:
        diagonal_scale = 1.0

    regularized = reduced_stiffness + regularization_epsilon * diagonal_scale * eye(reduced_stiffness.shape[0], format="csr")
    solved = spsolve(regularized, reduced_rhs)
    solved = np.asarray(solved, dtype=np.float64)
    if solved.ndim == 0:
        solved = solved[None]
    if not np.all(np.isfinite(solved)):
        raise ValueError("Linear elasticity solve produced non-finite displacement values.")
    displacement[free_dofs] = solved
    return displacement


def tetra_volume_and_gradients_codex(*, element_coordinates: np.ndarray) -> Tuple[float, np.ndarray]:
    matrix = np.vstack(
        [
            np.ones(4, dtype=np.float64),
            element_coordinates[:, 0],
            element_coordinates[:, 1],
            element_coordinates[:, 2],
        ]
    )
    inverse = np.linalg.inv(matrix)
    gradients = inverse[1:, :].T
    jacobian = np.column_stack(
        [
            element_coordinates[1] - element_coordinates[0],
            element_coordinates[2] - element_coordinates[0],
            element_coordinates[3] - element_coordinates[0],
        ]
    )
    volume = abs(np.linalg.det(jacobian)) / 6.0
    if volume <= 1.0e-14:
        raise np.linalg.LinAlgError("Degenerate tetrahedron volume.")
    return volume, gradients


def strain_displacement_matrix_codex(*, gradients: np.ndarray) -> np.ndarray:
    b_matrix = np.zeros((6, 12), dtype=np.float64)
    for node_idx, gradient in enumerate(gradients):
        base = 3 * node_idx
        dndx, dndy, dndz = gradient
        b_matrix[0, base + 0] = dndx
        b_matrix[1, base + 1] = dndy
        b_matrix[2, base + 2] = dndz
        b_matrix[3, base + 0] = dndy
        b_matrix[3, base + 1] = dndx
        b_matrix[4, base + 1] = dndz
        b_matrix[4, base + 2] = dndy
        b_matrix[5, base + 0] = dndz
        b_matrix[5, base + 2] = dndx
    return b_matrix


def isotropic_constitutive_matrix_codex(*, young_modulus: float, poisson_ratio: float) -> np.ndarray:
    lame_lambda = young_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    shear_modulus = young_modulus / (2.0 * (1.0 + poisson_ratio))
    constitutive_matrix = np.array(
        [
            [lame_lambda + 2.0 * shear_modulus, lame_lambda, lame_lambda, 0.0, 0.0, 0.0],
            [lame_lambda, lame_lambda + 2.0 * shear_modulus, lame_lambda, 0.0, 0.0, 0.0],
            [lame_lambda, lame_lambda, lame_lambda + 2.0 * shear_modulus, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, shear_modulus, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, shear_modulus, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, shear_modulus],
        ],
        dtype=np.float64,
    )
    return constitutive_matrix


def von_mises_from_stress_codex(stress: np.ndarray) -> float:
    sigma_xx, sigma_yy, sigma_zz, tau_xy, tau_yz, tau_zx = stress
    von_mises = np.sqrt(
        0.5 * ((sigma_xx - sigma_yy) ** 2 + (sigma_yy - sigma_zz) ** 2 + (sigma_zz - sigma_xx) ** 2)
        + 3.0 * (tau_xy**2 + tau_yz**2 + tau_zx**2)
    )
    return float(von_mises)


def element_to_vertex_importance_codex(
    *,
    element_indices: np.ndarray,
    element_importance: np.ndarray,
    element_volumes: np.ndarray,
    num_vertices: int,
) -> np.ndarray:
    vertex_importance = np.zeros(num_vertices, dtype=np.float64)
    vertex_volume = np.zeros(num_vertices, dtype=np.float64)
    for element_idx, vertex_ids in enumerate(element_indices):
        weighted_importance = float(element_importance[element_idx]) * float(element_volumes[element_idx])
        vertex_importance[vertex_ids] += weighted_importance
        vertex_volume[vertex_ids] += float(element_volumes[element_idx])
    vertex_volume = np.clip(vertex_volume, a_min=1.0e-12, a_max=None)
    return _clean_nonfinite_codex(vertex_importance / vertex_volume)


def _build_console_setup_codex(*, expert_mesh, surface: Dict[str, np.ndarray], principal_axes: np.ndarray, config: Dict[str, Any]) -> LinearElasticitySetupCodex:
    # [CodeX] Console 默认工况：最长主轴一端全固定，对端边界施加沿第二主轴负向的分布面力。
    face_nodes = surface["face_nodes"]
    face_centers = surface["face_centers"]
    face_areas = surface["face_areas"]

    primary_axis = principal_axes[0]
    load_direction = console_load_direction_codex(principal_axes=principal_axes, vertex_positions=expert_mesh.vertex_positions)
    min_faces = int(config.get("console_min_region_faces", 8))
    support_faces = select_patch_by_projection_codex(
        face_nodes=face_nodes,
        face_centers=face_centers,
        face_areas=face_areas,
        axis=primary_axis,
        mode="min",
        quantile=float(config.get("console_support_quantile", 0.08)),
        min_faces=min_faces,
    )
    fixed_nodes = np.unique(face_nodes[support_faces].reshape(-1))

    load_faces = select_patch_by_projection_codex(
        face_nodes=face_nodes,
        face_centers=face_centers,
        face_areas=face_areas,
        axis=primary_axis,
        mode="max",
        quantile=float(config.get("console_load_quantile", 0.92)),
        min_faces=min_faces,
        excluded_nodes=fixed_nodes,
    )
    if len(load_faces) == 0:
        load_faces = select_patch_by_projection_codex(
            face_nodes=face_nodes,
            face_centers=face_centers,
            face_areas=face_areas,
            axis=primary_axis,
            mode="max",
            quantile=float(config.get("console_load_quantile", 0.92)),
            min_faces=min_faces,
        )

    load_type = str(config.get("console_load_type", "surface_traction"))
    if load_type == "surface_traction" and len(load_faces) > 0:
        traction_magnitude = float(config.get("console_surface_traction_magnitude", 1.0))
        load_vectors = np.repeat((traction_magnitude * load_direction)[None, :], len(load_faces), axis=0)
        return LinearElasticitySetupCodex(
            fixed_nodes=fixed_nodes,
            load_faces=face_nodes[load_faces],
            load_vectors=load_vectors,
            body_force=None,
            principal_axes=principal_axes,
            load_type=load_type,
        )

    body_force_magnitude = float(config.get("console_body_force_magnitude", 1.0))
    return LinearElasticitySetupCodex(
        fixed_nodes=fixed_nodes,
        load_faces=np.zeros((0, 3), dtype=np.int64),
        load_vectors=np.zeros((0, 3), dtype=np.float64),
        body_force=body_force_magnitude * load_direction,
        principal_axes=principal_axes,
        load_type="body_force",
    )


def _build_mold_setup_codex(
    *,
    expert_mesh,
    surface: Dict[str, np.ndarray],
    principal_axes: np.ndarray,
    inlet_position: Optional[np.ndarray],
    config: Dict[str, Any],
) -> LinearElasticitySetupCodex:
    # [CodeX] Mold 默认工况：浇口邻域边界受压，距浇口最远的稳定边界片全固定。
    if inlet_position is None or inlet_position.shape[0] != 3:
        warnings.warn("Mold inlet position is unavailable. Falling back to the Console-style linear-elasticity setup.")
        return _build_console_setup_codex(expert_mesh=expert_mesh, surface=surface, principal_axes=principal_axes, config=config)

    face_nodes = surface["face_nodes"]
    face_centers = surface["face_centers"]
    face_areas = surface["face_areas"]
    outward_normals = surface["outward_normals"]
    bbox_diagonal = np.linalg.norm(np.max(expert_mesh.vertex_positions, axis=0) - np.min(expert_mesh.vertex_positions, axis=0))
    inlet_radius = float(config.get("mold_inlet_radius_scale", 0.08)) * max(bbox_diagonal, 1.0e-8)
    min_faces = int(config.get("mold_min_region_faces", 8))

    distances_to_inlet = np.linalg.norm(face_centers - inlet_position[None, :], axis=1)
    load_faces = select_patch_by_distance_codex(
        face_nodes=face_nodes,
        face_centers=face_centers,
        face_areas=face_areas,
        distances=distances_to_inlet,
        mode="nearest",
        threshold=inlet_radius,
        min_faces=min_faces,
    )
    if len(load_faces) == 0:
        warnings.warn("Mold inlet pressure patch selection failed. Falling back to the Console-style linear-elasticity setup.")
        return _build_console_setup_codex(expert_mesh=expert_mesh, surface=surface, principal_axes=principal_axes, config=config)

    fixed_faces = select_patch_by_distance_codex(
        face_nodes=face_nodes,
        face_centers=face_centers,
        face_areas=face_areas,
        distances=distances_to_inlet,
        mode="farthest",
        threshold=float(config.get("mold_far_region_quantile", 0.92)),
        min_faces=min_faces,
        use_quantile=True,
        excluded_nodes=np.unique(face_nodes[load_faces].reshape(-1)),
    )
    if len(fixed_faces) == 0:
        warnings.warn("Mold far-end fixed patch selection failed. Falling back to the Console-style linear-elasticity setup.")
        return _build_console_setup_codex(expert_mesh=expert_mesh, surface=surface, principal_axes=principal_axes, config=config)

    pressure_magnitude = float(config.get("mold_pressure_magnitude", 1.0))
    load_vectors = -pressure_magnitude * outward_normals[load_faces]
    fixed_nodes = np.unique(face_nodes[fixed_faces].reshape(-1))
    return LinearElasticitySetupCodex(
        fixed_nodes=fixed_nodes,
        load_faces=face_nodes[load_faces],
        load_vectors=load_vectors,
        body_force=None,
        principal_axes=principal_axes,
        load_type="surface_pressure",
    )


def _get_boundary_surface_codex(*, expert_mesh) -> Dict[str, np.ndarray]:
    # [CodeX] 从专家四面体网格提取外边界三角面、面积和朝外法向，供面载荷与边界片筛选复用。
    boundary_facet_ids = np.asarray(expert_mesh.mesh.boundary_facets(), dtype=np.int64)
    face_nodes = np.asarray(expert_mesh.mesh.facets[:, boundary_facet_ids].T, dtype=np.int64)
    face_coordinates = np.asarray(expert_mesh.vertex_positions[face_nodes], dtype=np.float64)
    face_centers = np.mean(face_coordinates, axis=1)
    tet_ids = np.asarray(expert_mesh.mesh.f2t[0, boundary_facet_ids], dtype=np.int64)
    tet_centers = np.asarray(np.mean(expert_mesh.vertex_positions[expert_mesh.element_indices[tet_ids]], axis=1), dtype=np.float64)

    normals = np.cross(face_coordinates[:, 1] - face_coordinates[:, 0], face_coordinates[:, 2] - face_coordinates[:, 0])
    normal_norms = np.linalg.norm(normals, axis=1)
    face_areas = 0.5 * normal_norms
    safe_norms = np.clip(normal_norms, a_min=1.0e-12, a_max=None)
    outward_normals = normals / safe_norms[:, None]
    inward_vectors = tet_centers - face_centers
    inward_alignment = np.sum(outward_normals * inward_vectors, axis=1) > 0
    outward_normals[inward_alignment] *= -1.0
    return {
        "boundary_facet_ids": boundary_facet_ids,
        "face_nodes": face_nodes,
        "face_centers": face_centers,
        "face_areas": face_areas,
        "outward_normals": outward_normals,
    }


def select_patch_by_projection_codex(
    *,
    face_nodes: np.ndarray,
    face_centers: np.ndarray,
    face_areas: np.ndarray,
    axis: np.ndarray,
    mode: str,
    quantile: float,
    min_faces: int,
    excluded_nodes: Optional[np.ndarray] = None,
) -> np.ndarray:
    projections = face_centers.dot(axis)
    if mode == "min":
        threshold = np.quantile(projections, quantile)
        candidate_faces = np.where(projections <= threshold)[0]
        sorted_faces = np.argsort(projections)
    else:
        threshold = np.quantile(projections, quantile)
        candidate_faces = np.where(projections >= threshold)[0]
        sorted_faces = np.argsort(-projections)
    return select_connected_patch_codex(
        face_nodes=face_nodes,
        face_areas=face_areas,
        candidate_faces=candidate_faces,
        sorted_faces=sorted_faces,
        min_faces=min_faces,
        excluded_nodes=excluded_nodes,
    )


def select_patch_by_distance_codex(
    *,
    face_nodes: np.ndarray,
    face_centers: np.ndarray,
    face_areas: np.ndarray,
    distances: np.ndarray,
    mode: str,
    threshold: float,
    min_faces: int,
    use_quantile: bool = False,
    excluded_nodes: Optional[np.ndarray] = None,
) -> np.ndarray:
    if mode == "nearest":
        limit = np.quantile(distances, threshold) if use_quantile else threshold
        candidate_faces = np.where(distances <= limit)[0]
        sorted_faces = np.argsort(distances)
    else:
        limit = np.quantile(distances, threshold) if use_quantile else threshold
        candidate_faces = np.where(distances >= limit)[0]
        sorted_faces = np.argsort(-distances)
    return select_connected_patch_codex(
        face_nodes=face_nodes,
        face_areas=face_areas,
        candidate_faces=candidate_faces,
        sorted_faces=sorted_faces,
        min_faces=min_faces,
        excluded_nodes=excluded_nodes,
    )


def select_connected_patch_codex(
    *,
    face_nodes: np.ndarray,
    face_areas: np.ndarray,
    candidate_faces: np.ndarray,
    sorted_faces: np.ndarray,
    min_faces: int,
    excluded_nodes: Optional[np.ndarray],
) -> np.ndarray:
    if excluded_nodes is not None and len(excluded_nodes) > 0:
        candidate_mask = ~np.any(np.isin(face_nodes, excluded_nodes), axis=1)
        candidate_faces = candidate_faces[candidate_mask[candidate_faces]]

    if len(candidate_faces) < min_faces:
        expanded = []
        for face_idx in sorted_faces:
            if excluded_nodes is not None and np.any(np.isin(face_nodes[face_idx], excluded_nodes)):
                continue
            expanded.append(int(face_idx))
            if len(expanded) >= min_faces:
                break
        candidate_faces = np.unique(np.concatenate([candidate_faces, np.asarray(expanded, dtype=np.int64)])) if len(expanded) > 0 else candidate_faces

    if len(candidate_faces) == 0:
        return np.zeros(0, dtype=np.int64)

    components = boundary_face_components_codex(face_nodes=face_nodes[candidate_faces])
    if len(components) == 0:
        return np.zeros(0, dtype=np.int64)
    component_areas = [float(np.sum(face_areas[candidate_faces[component]])) for component in components]
    best_component = components[int(np.argmax(component_areas))]
    return np.asarray(candidate_faces[best_component], dtype=np.int64)


def boundary_face_components_codex(*, face_nodes: np.ndarray) -> List[np.ndarray]:
    edge_to_faces: Dict[Tuple[int, int], List[int]] = {}
    for face_idx, face in enumerate(face_nodes):
        for edge in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = tuple(sorted((int(edge[0]), int(edge[1]))))
            edge_to_faces.setdefault(key, []).append(face_idx)

    adjacency = [set() for _ in range(len(face_nodes))]
    for face_ids in edge_to_faces.values():
        for index, face_a in enumerate(face_ids):
            for face_b in face_ids[index + 1 :]:
                adjacency[face_a].add(face_b)
                adjacency[face_b].add(face_a)

    components = []
    visited = np.zeros(len(face_nodes), dtype=bool)
    for start in range(len(face_nodes)):
        if visited[start]:
            continue
        queue = [start]
        visited[start] = True
        component = []
        while queue:
            current = queue.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        components.append(np.asarray(component, dtype=np.int64))
    return components


def _principal_axes_codex(*, vertex_positions: np.ndarray) -> np.ndarray:
    centered = np.asarray(vertex_positions, dtype=np.float64) - np.mean(vertex_positions, axis=0, keepdims=True)
    covariance = centered.T.dot(centered) / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, order].T
    canonical_axes = np.stack([canonicalize_axis_codex(axis) for axis in axes], axis=0)
    return canonical_axes


def console_load_direction_codex(*, principal_axes: np.ndarray, vertex_positions: np.ndarray) -> np.ndarray:
    strategy_axis = principal_axes[1] if principal_axes.shape[0] > 1 else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if not np.all(np.isfinite(strategy_axis)) or np.linalg.norm(strategy_axis) < 1.0e-8:
        bbox_spans = np.max(vertex_positions, axis=0) - np.min(vertex_positions, axis=0)
        shortest_axis = int(np.argmin(bbox_spans))
        direction = np.zeros(3, dtype=np.float64)
        direction[shortest_axis] = -1.0
        return direction
    return -canonicalize_axis_codex(strategy_axis)


def canonicalize_axis_codex(axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm <= 1.0e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    axis = axis / norm
    dominant_index = int(np.argmax(np.abs(axis)))
    if axis[dominant_index] < 0:
        axis = -axis
    return axis


def triangle_area_codex(*, triangle: np.ndarray) -> float:
    return 0.5 * float(np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])))


def _get_inlet_position_codex(*, source_data) -> Optional[np.ndarray]:
    feature_provider = getattr(source_data, "feature_provider", None)
    inlet_position = getattr(feature_provider, "inlet_position", None)
    if inlet_position is None:
        return None
    inlet_position = np.asarray(inlet_position, dtype=np.float64)
    if inlet_position.shape != (3,):
        return None
    return inlet_position


def _node_ids_to_dofs_codex(node_ids: np.ndarray) -> np.ndarray:
    node_ids = np.asarray(node_ids, dtype=np.int64).reshape(-1)
    # [CodeX] 局部自由度必须按节点交错排列，保持与四面体单元 B 矩阵和等效结点力的标准顺序一致。
    return (3 * node_ids[:, None] + np.arange(3, dtype=np.int64)[None, :]).reshape(-1).astype(np.int64)


def _clean_nonfinite_codex(values: np.ndarray) -> np.ndarray:
    clean_values = np.array(values, dtype=np.float64, copy=True)
    clean_values[~np.isfinite(clean_values)] = 0.0
    return clean_values
