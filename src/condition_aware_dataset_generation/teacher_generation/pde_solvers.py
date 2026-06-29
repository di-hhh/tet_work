from __future__ import annotations

import time
from typing import Any

import numpy as np
from skfem import Basis, ElementTetP1, ElementTriP1, ElementVector, FacetBasis, LinearForm, asm, condense, solve
from skfem.models.elasticity import lame_parameters, linear_elasticity, plane_stress
from skfem.models.poisson import laplace

from src.condition_aware_dataset_generation.records import ConditionRecord, GeometryPreprocessRecord
from src.condition_aware_dataset_generation.runtime_controls import ComplexityLimitError


def _build_scalar_basis(mesh) -> Basis:
    element = ElementTriP1() if mesh.dim() == 2 else ElementTetP1()
    return Basis(mesh, element)


def _build_vector_basis(mesh) -> Basis:
    base_element = ElementTriP1() if mesh.dim() == 2 else ElementTetP1()
    return Basis(mesh, ElementVector(base_element))


def _selector_callable(preprocess_record: GeometryPreprocessRecord, selector_spec: dict[str, Any]):
    centroid = np.asarray(preprocess_record.centroid, dtype=float)
    principal_axes = np.asarray(preprocess_record.principal_axes, dtype=float)
    oriented_bbox_min = np.asarray(preprocess_record.oriented_bbox_min, dtype=float)
    oriented_bbox_max = np.asarray(preprocess_record.oriented_bbox_max, dtype=float)
    axis_index = int(selector_spec['axis_index'])
    band_fraction = float(selector_spec.get('band_fraction', 0.12))
    side = selector_spec['side']
    extent = oriented_bbox_max[axis_index] - oriented_bbox_min[axis_index]
    band_width = max(extent * band_fraction, 1.0e-8)

    def _selector(x: np.ndarray) -> np.ndarray:
        local = (x.T - centroid) @ principal_axes
        coordinate = local[:, axis_index]
        if side == 'min':
            return coordinate <= oriented_bbox_min[axis_index] + band_width + 1.0e-9
        return coordinate >= oriented_bbox_max[axis_index] - band_width - 1.0e-9

    return _selector


def select_boundary_facets(mesh, preprocess_record: GeometryPreprocessRecord, selector_spec: dict[str, Any]) -> np.ndarray:
    selector = _selector_callable(preprocess_record, selector_spec)
    facets = mesh.facets_satisfying(selector, boundaries_only=True)
    if len(facets) > 0:
        return np.asarray(facets, dtype=np.int32)

    centroid = np.asarray(preprocess_record.centroid, dtype=float)
    principal_axes = np.asarray(preprocess_record.principal_axes, dtype=float)
    axis_index = int(selector_spec['axis_index'])
    side = selector_spec['side']
    midpoints = mesh.p[:, mesh.facets[:, mesh.boundary_facets()]].mean(axis=1).T
    local = (midpoints - centroid) @ principal_axes
    coordinates = local[:, axis_index]
    target_value = coordinates.min() if side == 'min' else coordinates.max()
    fallback_facets = mesh.boundary_facets()[np.isclose(coordinates, target_value, atol=1.0e-8)]
    return np.asarray(fallback_facets, dtype=np.int32)


def evaluate_solution_at_points(basis: Basis, solution_vector: np.ndarray, points: np.ndarray) -> np.ndarray:
    component_values = []
    for component_vector, component_basis in basis.split(solution_vector):
        probe_matrix = component_basis.probes(points)
        values = np.asarray(probe_matrix @ component_vector).reshape(-1)
        component_values.append(values)
    return np.stack(component_values, axis=1)


def _solver_limits(solver_options: dict[str, Any] | None) -> tuple[int | None, int | None, str]:
    solver_options = solver_options or {}
    max_dofs = solver_options.get('max_dofs')
    max_matrix_nnz = solver_options.get('max_matrix_nnz')
    stage_name = str(solver_options.get('solver_stage_name', 'pde_solve'))
    return (int(max_dofs) if max_dofs is not None else None, int(max_matrix_nnz) if max_matrix_nnz is not None else None, stage_name)


def _complexity_metadata(mesh, basis: Basis, component_dim: int) -> dict[str, Any]:
    num_elements = int(mesh.t.shape[1])
    local_dofs = int((mesh.dim() + 1) * component_dim)
    return {
        'num_elements': num_elements,
        'num_vertices': int(mesh.nvertices),
        'num_dofs': int(basis.N),
        'estimated_matrix_nnz': int(num_elements * (local_dofs**2)),
        'component_dim': int(component_dim),
        'matrix_shape': [int(basis.N), int(basis.N)],
    }


def _enforce_complexity_caps(metadata: dict[str, Any], *, max_dofs: int | None, max_matrix_nnz: int | None, stage_name: str) -> None:
    if max_dofs is not None and metadata['num_dofs'] > max_dofs:
        raise ComplexityLimitError(
            f'{stage_name} exceeded the DOF cap ({metadata["num_dofs"]} > {max_dofs})',
            category='matrix_too_large',
            stage=stage_name,
            details=metadata,
        )
    if max_matrix_nnz is not None and metadata['estimated_matrix_nnz'] > max_matrix_nnz:
        raise ComplexityLimitError(
            f'{stage_name} exceeded the matrix nnz cap ({metadata["estimated_matrix_nnz"]} > {max_matrix_nnz})',
            category='matrix_too_large',
            stage=stage_name,
            details=metadata,
        )


def solve_scalar_elliptic(
    mesh,
    preprocess_record: GeometryPreprocessRecord,
    condition_record: ConditionRecord,
    solver_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    condition_spec = condition_record.condition_spec
    basis = _build_scalar_basis(mesh)
    max_dofs, max_matrix_nnz, stage_name = _solver_limits(solver_options)
    metadata = _complexity_metadata(mesh, basis, component_dim=1)
    _enforce_complexity_caps(metadata, max_dofs=max_dofs, max_matrix_nnz=max_matrix_nnz, stage_name=stage_name)

    diffusion = float(condition_spec['coefficient_spec']['diffusion'])
    assembly_start = time.perf_counter()
    stiffness = diffusion * asm(laplace, basis)
    rhs = np.zeros(stiffness.shape[0], dtype=float)

    principal_axes = np.asarray(preprocess_record.principal_axes, dtype=float)
    centroid = np.asarray(preprocess_record.centroid, dtype=float)
    source_spec = condition_spec['source_or_load_spec'].get('internal_source')
    if source_spec is not None:
        center_local = np.asarray(source_spec['center_local'], dtype=float)
        sigma = np.asarray(source_spec['sigma'], dtype=float)
        amplitude = float(source_spec['amplitude'])

        @LinearForm
        def source_form(v, w):
            positions = np.stack(w.x, axis=-1)
            local_positions = (positions - centroid) @ principal_axes
            normalized = (local_positions - center_local) / np.maximum(sigma, 1.0e-8)
            values = amplitude * np.exp(-0.5 * np.sum(normalized**2, axis=-1))
            return values * v

        rhs += asm(source_form, basis)

    dirichlet_dofs = []
    dirichlet_values = np.zeros(stiffness.shape[0], dtype=float)
    for boundary_role in condition_spec['boundary_role_spec']:
        selector = boundary_role['selector']
        selected_facets = select_boundary_facets(mesh, preprocess_record, selector)
        if boundary_role['role'].startswith('dirichlet'):
            dofs = basis.get_dofs(facets=selected_facets).flatten()
            dirichlet_dofs.extend(dofs.tolist())
            dirichlet_values[dofs] = float(boundary_role['value'])
        elif boundary_role['role'] == 'neumann_flux':
            flux_value = float(boundary_role['value'])
            facet_basis = FacetBasis(mesh, basis.elem, facets=selected_facets)

            @LinearForm
            def flux_form(v, w):
                return flux_value * v

            rhs += asm(flux_form, facet_basis)

    metadata['actual_matrix_nnz'] = int(stiffness.nnz)
    if max_matrix_nnz is not None and metadata['actual_matrix_nnz'] > max_matrix_nnz:
        raise ComplexityLimitError(
            f'{stage_name} assembled a matrix that exceeded the nnz cap ({metadata["actual_matrix_nnz"]} > {max_matrix_nnz})',
            category='matrix_too_large',
            stage=stage_name,
            details=metadata,
        )
    solve_start = time.perf_counter()
    solution_vector = solve(*condense(stiffness, rhs, D=np.unique(np.asarray(dirichlet_dofs, dtype=np.int32)), x=dirichlet_values))
    nodal_values = evaluate_solution_at_points(basis, solution_vector, mesh.p)
    metadata['assembly_time_sec'] = float(solve_start - assembly_start)
    metadata['solve_time_sec'] = float(time.perf_counter() - solve_start)
    return {
        'basis': basis,
        'solution_vector': solution_vector,
        'nodal_values': nodal_values,
        'component_dim': 1,
        'solver_metadata': metadata,
    }


def solve_linear_elasticity_problem(
    mesh,
    preprocess_record: GeometryPreprocessRecord,
    condition_record: ConditionRecord,
    solver_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    condition_spec = condition_record.condition_spec
    basis = _build_vector_basis(mesh)
    max_dofs, max_matrix_nnz, stage_name = _solver_limits(solver_options)
    metadata = _complexity_metadata(mesh, basis, component_dim=mesh.dim())
    _enforce_complexity_caps(metadata, max_dofs=max_dofs, max_matrix_nnz=max_matrix_nnz, stage_name=stage_name)

    coefficients = condition_spec['coefficient_spec']
    youngs_modulus = float(coefficients['youngs_modulus'])
    poissons_ratio = float(coefficients['poissons_ratio'])
    if mesh.dim() == 2 and coefficients.get('constitutive_model') == 'plane_stress':
        youngs_modulus, poissons_ratio = plane_stress(youngs_modulus, poissons_ratio)
    lambda_param, mu_param = lame_parameters(youngs_modulus, poissons_ratio)
    assembly_start = time.perf_counter()
    stiffness = asm(linear_elasticity(lambda_param, mu_param), basis)
    rhs = np.zeros(stiffness.shape[0], dtype=float)

    body_force = np.asarray(condition_spec['source_or_load_spec'].get('body_force', np.zeros(mesh.dim())), dtype=float)
    if np.linalg.norm(body_force) > 0.0:

        @LinearForm
        def body_force_form(v, w):
            force_components = [body_force[i] * v[i] for i in range(mesh.dim())]
            return sum(force_components)

        rhs += asm(body_force_form, basis)

    constrained_dofs = []
    constrained_values = np.zeros(stiffness.shape[0], dtype=float)
    for boundary_role in condition_spec['boundary_role_spec']:
        selected_facets = select_boundary_facets(mesh, preprocess_record, boundary_role['selector'])
        if boundary_role['role'] == 'support':
            dofs = basis.get_dofs(facets=selected_facets).flatten()
            constrained_dofs.extend(dofs.tolist())
        elif boundary_role['role'] == 'traction':
            traction_vector = np.asarray(boundary_role['vector'], dtype=float)
            facet_basis = FacetBasis(mesh, basis.elem, facets=selected_facets)

            @LinearForm
            def traction_form(v, w):
                force_components = [traction_vector[i] * v[i] for i in range(mesh.dim())]
                return sum(force_components)

            rhs += asm(traction_form, facet_basis)

    metadata['actual_matrix_nnz'] = int(stiffness.nnz)
    if max_matrix_nnz is not None and metadata['actual_matrix_nnz'] > max_matrix_nnz:
        raise ComplexityLimitError(
            f'{stage_name} assembled a matrix that exceeded the nnz cap ({metadata["actual_matrix_nnz"]} > {max_matrix_nnz})',
            category='matrix_too_large',
            stage=stage_name,
            details=metadata,
        )
    solve_start = time.perf_counter()
    solution_vector = solve(
        *condense(
            stiffness,
            rhs,
            D=np.unique(np.asarray(constrained_dofs, dtype=np.int32)),
            x=constrained_values,
        )
    )
    nodal_values = evaluate_solution_at_points(basis, solution_vector, mesh.p)
    metadata['assembly_time_sec'] = float(solve_start - assembly_start)
    metadata['solve_time_sec'] = float(time.perf_counter() - solve_start)
    return {
        'basis': basis,
        'solution_vector': solution_vector,
        'nodal_values': nodal_values,
        'component_dim': mesh.dim(),
        'solver_metadata': metadata,
    }


def solve_condition(
    mesh,
    preprocess_record: GeometryPreprocessRecord,
    condition_record: ConditionRecord,
    solver_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if condition_record.pde_family == 'scalar_elliptic':
        return solve_scalar_elliptic(mesh, preprocess_record, condition_record, solver_options=solver_options)
    if condition_record.pde_family == 'linear_elasticity':
        return solve_linear_elasticity_problem(mesh, preprocess_record, condition_record, solver_options=solver_options)
    raise ValueError(f'Unsupported PDE family: {condition_record.pde_family}')
