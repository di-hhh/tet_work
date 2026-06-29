from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from skfem import adaptive_theta

from src.condition_aware_dataset_generation.geometry_preprocessing import geometry_fn_from_path
from src.condition_aware_dataset_generation.records import (
    ConditionRecord,
    FailureRecord,
    GeometryPreprocessRecord,
    GeometryRecord,
    PrescreenRecord,
    SampleRecord,
    TeacherRecord,
)
from src.condition_aware_dataset_generation.runtime_controls import BudgetControlError, PipelineAbort, RuntimeTracker, StageTimeoutError
from src.condition_aware_dataset_generation.serialization.layout import PipelineLayout
from src.condition_aware_dataset_generation.teacher_generation.cad_meshing import (
    combine_geometry_constraints,
    evaluate_geometry_sizing,
    generate_cad_aware_mesh,
)
from src.condition_aware_dataset_generation.teacher_generation.pde_solvers import _selector_callable, evaluate_solution_at_points, solve_condition
from src.condition_aware_dataset_generation.utils import dump_json, load_json, now_iso, stable_identifier
from src.mesh_util.load_mesh import load_expert_mesh
from src.mesh_util.save_mesh import save_as_vtk
from src.tasks.domains.geometry_util import edge_length_to_volume, get_simplex_volumes_from_indices, volume_to_edge_length


def _average_to_vertices(num_vertices: int, simplices: np.ndarray, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    accum = np.zeros(num_vertices, dtype=float)
    counts = np.zeros(num_vertices, dtype=float)
    np.add.at(accum, simplices.reshape(-1), np.repeat(values, simplices.shape[1]))
    np.add.at(counts, simplices.reshape(-1), 1.0)
    return accum / np.maximum(counts, 1.0)


def _normalize_importance(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    centered = values - float(np.min(values))
    scale = max(float(np.quantile(centered, 0.95)), 1.0e-12)
    return np.clip(centered / scale, 0.0, 1.0)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _safe_quantile(values: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, quantile))


def _surface_face_count(surface_mesh: Any | None) -> int:
    if surface_mesh is None:
        return 0
    count = 0
    for block in surface_mesh.cells:
        if block.type in {'line', 'triangle'}:
            count += int(len(block.data))
    return count


def _mesh_mean_edge_length(mesh) -> float:
    simplices = np.asarray(mesh.t.T, dtype=np.int64)
    points = np.asarray(mesh.p.T, dtype=float)
    if simplices.size == 0:
        return 0.0
    edge_blocks = []
    for left in range(simplices.shape[1]):
        for right in range(left + 1, simplices.shape[1]):
            edge_blocks.append(np.sort(simplices[:, [left, right]], axis=1))
    unique_edges = np.unique(np.concatenate(edge_blocks, axis=0), axis=0)
    lengths = np.linalg.norm(points[unique_edges[:, 0]] - points[unique_edges[:, 1]], axis=1)
    return float(lengths.mean()) if len(lengths) else 0.0


def _bbox_measure(preprocess_record: GeometryPreprocessRecord) -> float:
    bbox = np.asarray(preprocess_record.bounding_box, dtype=float)
    dim = int(preprocess_record.dimension)
    mins = bbox[:dim]
    maxs = bbox[dim : 2 * dim]
    extents = np.maximum(maxs - mins, 1.0e-8)
    return float(np.prod(extents))


def _estimate_initial_dofs(mesh, pde_family: str) -> int:
    component_dim = int(mesh.dim()) if pde_family == 'linear_elasticity' else 1
    return int(mesh.nvertices * component_dim)


def _sample_probe_field(probe_points: np.ndarray, source_points: np.ndarray, source_values: np.ndarray) -> np.ndarray:
    if probe_points.size == 0:
        return np.zeros(0, dtype=float)
    if source_points.size == 0:
        return np.zeros(len(probe_points), dtype=float)
    tree = cKDTree(np.asarray(source_points, dtype=float))
    _, indices = tree.query(np.asarray(probe_points, dtype=float))
    return np.asarray(source_values, dtype=float)[indices]


def _status_priority(status: str) -> int:
    priority = {
        'success': 0,
        'success_budget_closed': 0,
        'success_near_desired_budget': 1,
        'success_partial_under_budget': 2,
        'budget_exceeded': 3,
        'budget_closure_failed': 4,
        'fail_budget_growth_stalled': 5,
        'fail_budget_growth_timeout': 6,
        'fail_budget_hard_cap_exceeded': 7,
        'early_stop': 8,
        'timeout_budget_calibration': 9,
    }
    return priority.get(status, 99)


def _is_success_status(status: str | None) -> bool:
    return str(status or '').startswith('success') or status == 'success'


def _mesh_element_sizes(mesh) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(mesh.p.T, dtype=float)
    simplices = np.asarray(mesh.t.T, dtype=np.int64)
    dim = int(mesh.dim())
    volumes = get_simplex_volumes_from_indices(points, simplices)
    return volumes, volume_to_edge_length(volumes, dim=dim)


def _element_average_from_vertex_values(simplices: np.ndarray, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    return values[np.asarray(simplices, dtype=np.int64)].mean(axis=1)


def _allocation_diagnostics_for_mesh(
    *,
    mesh,
    indicator: np.ndarray,
    hotspot_quantile: float,
    low_error_quantile: float = 0.5,
) -> dict[str, Any]:
    volumes, sizes = _mesh_element_sizes(mesh)
    indicator = np.asarray(indicator, dtype=float).reshape(-1)
    if indicator.shape[0] != sizes.shape[0]:
        fallback = float(np.mean(indicator)) if indicator.size else 1.0
        indicator = np.full(sizes.shape[0], fallback, dtype=float)
    hotspot_threshold = _safe_quantile(indicator, hotspot_quantile)
    low_threshold = _safe_quantile(indicator, low_error_quantile)
    hotspot_mask = indicator >= hotspot_threshold
    low_mask = indicator <= low_threshold
    if not np.any(hotspot_mask):
        hotspot_mask[np.argmax(indicator)] = True
    if not np.any(low_mask):
        low_mask[np.argmin(indicator)] = True
    hotspot_element_fraction = float(np.mean(hotspot_mask))
    hotspot_volume_fraction = float(volumes[hotspot_mask].sum() / max(volumes.sum(), 1.0e-12))
    hotspot_size_ratio = float(np.median(sizes[hotspot_mask]) / max(np.median(sizes[low_mask]), 1.0e-12))
    allocation_gain = float(hotspot_element_fraction / max(hotspot_volume_fraction, 1.0e-12))
    return {
        'num_elements': int(mesh.t.shape[1]),
        'num_nodes': int(mesh.nvertices),
        'hotspot_threshold': float(hotspot_threshold),
        'hotspot_size_ratio': hotspot_size_ratio,
        'final_hotspot_size_ratio': hotspot_size_ratio,
        'hotspot_element_fraction': hotspot_element_fraction,
        'final_hotspot_element_fraction': hotspot_element_fraction,
        'hotspot_volume_fraction': hotspot_volume_fraction,
        'final_hotspot_volume_fraction': hotspot_volume_fraction,
        'allocation_gain': allocation_gain,
        'final_allocation_gain': allocation_gain,
        'size_q10': float(np.quantile(sizes, 0.1)),
        'size_q50': float(np.quantile(sizes, 0.5)),
        'size_q90': float(np.quantile(sizes, 0.9)),
    }


class TeacherGenerator:
    def __init__(self, teacher_config: dict, smoke_config: dict | None = None):
        self.teacher_config = teacher_config
        self.smoke_config = smoke_config or {}
        self.initial_mesh_element_volume = float(teacher_config.get('initial_mesh_element_volume', 0.08))
        self.reference_refinement_levels = int(teacher_config.get('reference_refinement_levels', 1))
        self.max_adaptive_steps = int(self.smoke_config.get('smoke_max_refinement_steps', teacher_config.get('max_adaptive_steps', 5)))
        self.refine_theta = float(teacher_config.get('refine_theta', 0.55))
        self.store_trajectory = bool(teacher_config.get('store_trajectory', True))
        self.allow_budget_shortfall = bool(teacher_config.get('allow_budget_shortfall', True))
        self.enable_geometry_fidelity_constraints = bool(teacher_config.get('enable_geometry_fidelity_constraints', True))
        self.surface_first_meshing = bool(teacher_config.get('surface_first_meshing', True))
        self.max_geometry_retry = int(self.smoke_config.get('smoke_max_retries', teacher_config.get('max_geometry_retry', 2)))
        self.sample_timeout_seconds = _optional_float(self.smoke_config.get('smoke_max_runtime_seconds_per_sample'))
        self.max_dofs = _optional_int(self.smoke_config.get('smoke_max_dofs'))
        self.max_matrix_nnz = _optional_int(self.smoke_config.get('smoke_max_matrix_nnz'))
        self.target_num_elements = _optional_int(self.smoke_config.get('smoke_target_num_elements'))
        self.minimum_viable_budget = _optional_int(
            teacher_config.get('minimum_viable_budget', self.smoke_config.get('minimum_viable_budget'))
        )
        self.desired_budget = _optional_int(teacher_config.get('desired_budget', self.smoke_config.get('desired_budget')))
        self.hard_max_budget = _optional_int(teacher_config.get('hard_max_budget', self.smoke_config.get('hard_max_budget')))
        self.budget_status_mode = str(teacher_config.get('budget_status_mode', self.smoke_config.get('budget_status_mode', 'layered'))).lower()
        self.allow_success_partial_under_budget = bool(
            teacher_config.get(
                'allow_success_partial_under_budget',
                self.smoke_config.get('allow_success_partial_under_budget', self.allow_budget_shortfall),
            )
        )
        self.budget_growth_enable = bool(teacher_config.get('budget_growth_enable', self.smoke_config.get('budget_growth_enable', True)))
        self.budget_growth_max_steps = int(teacher_config.get('budget_growth_max_steps', self.smoke_config.get('budget_growth_max_steps', 4)))
        self.budget_growth_batch_refine_fraction = float(
            teacher_config.get('budget_growth_batch_refine_fraction', self.smoke_config.get('budget_growth_batch_refine_fraction', 0.18))
        )
        self.budget_growth_stop_on_diminishing_return = bool(
            teacher_config.get(
                'budget_growth_stop_on_diminishing_return',
                self.smoke_config.get('budget_growth_stop_on_diminishing_return', True),
            )
        )
        self.budget_growth_timeout_seconds = float(
            teacher_config.get('budget_growth_timeout_seconds', self.smoke_config.get('budget_growth_timeout_seconds', 20.0))
        )
        self.budget_growth_use_local_refine = bool(
            teacher_config.get('budget_growth_use_local_refine', self.smoke_config.get('budget_growth_use_local_refine', True))
        )
        self.budget_growth_cad_cleanup_interval = int(
            teacher_config.get('budget_growth_cad_cleanup_interval', self.smoke_config.get('budget_growth_cad_cleanup_interval', 0))
        )
        self.budget_growth_dynamic_step_enable = bool(
            teacher_config.get('budget_growth_dynamic_step_enable', self.smoke_config.get('budget_growth_dynamic_step_enable', True))
        )
        self.budget_growth_max_steps_cap = int(
            teacher_config.get('budget_growth_max_steps_cap', self.smoke_config.get('budget_growth_max_steps_cap', 18))
        )
        self.budget_growth_batch_refine_fraction_max = float(
            teacher_config.get(
                'budget_growth_batch_refine_fraction_max',
                self.smoke_config.get('budget_growth_batch_refine_fraction_max', 0.32),
            )
        )
        self.high_budget_threshold = int(teacher_config.get('high_budget_threshold', self.smoke_config.get('high_budget_threshold', 50_000)))
        self.disable_predictive_growth_caps_for_high_budget = bool(
            teacher_config.get(
                'disable_predictive_growth_caps_for_high_budget',
                self.smoke_config.get('disable_predictive_growth_caps_for_high_budget', True),
            )
        )
        self.dof_cap_guard_fraction = float(
            teacher_config.get('dof_cap_guard_fraction', self.smoke_config.get('dof_cap_guard_fraction', 0.98))
        )
        self.matrix_cap_guard_fraction = float(
            teacher_config.get('matrix_cap_guard_fraction', self.smoke_config.get('matrix_cap_guard_fraction', 0.98))
        )
        self.save_final_allocation_diagnostics = bool(
            teacher_config.get(
                'save_final_allocation_diagnostics',
                self.smoke_config.get('save_final_allocation_diagnostics', True),
            )
        )
        self.save_parent_wall_clock = bool(
            teacher_config.get('save_parent_wall_clock', self.smoke_config.get('save_parent_wall_clock', True))
        )
        self.save_worker_elapsed_separately = bool(
            teacher_config.get('save_worker_elapsed_separately', self.smoke_config.get('save_worker_elapsed_separately', True))
        )
        self.elasticity_smoke_mode = str(
            teacher_config.get('elasticity_smoke_mode', self.smoke_config.get('elasticity_smoke_mode', 'cheap_reference'))
        ).lower()
        self.elasticity_smoke_reference_level = int(
            teacher_config.get('elasticity_smoke_reference_level', self.smoke_config.get('elasticity_smoke_reference_level', 0))
        )
        self.scalar_high_budget_reference_mode = str(
            teacher_config.get(
                'scalar_high_budget_reference_mode',
                self.smoke_config.get('scalar_high_budget_reference_mode', 'cheap_reference'),
            )
        ).lower()
        self.contrast_mode = str(self.smoke_config.get('contrast_mode', 'hybrid')).lower()
        self.contrast_gamma = float(self.smoke_config.get('contrast_gamma', 1.8))
        self.hotspot_quantile = float(self.smoke_config.get('hotspot_quantile', 0.9))
        self.medium_quantile = float(self.smoke_config.get('medium_quantile', 0.7))
        self.low_importance_size_boost = float(self.smoke_config.get('low_importance_size_boost', 1.35))
        self.target_hotspot_size_ratio = float(self.smoke_config.get('target_hotspot_size_ratio', 0.75))
        self.max_budget_overrun_ratio = float(self.smoke_config.get('smoke_max_budget_overrun_ratio', 1.25))
        self.scalar_smoke_enable = bool(self.smoke_config.get('scalar_smoke_enable', True))
        self.elasticity_smoke_enable = bool(self.smoke_config.get('elasticity_smoke_enable', True))
        self.elasticity_smoke_strict_cost_gate = bool(
            self.smoke_config.get('elasticity_smoke_strict_cost_gate', self.smoke_config.get('skip_expensive_elasticity', True))
        )
        default_target = max(int(self.target_num_elements or 64), 1)
        self.initial_target_num_elements = _optional_int(
            teacher_config.get('initial_target_num_elements', max(12, int(round(default_target * 0.12))))
        )
        self.initial_target_num_surface_faces = _optional_int(
            teacher_config.get('initial_target_num_surface_faces', max(24, 2 * int(self.initial_target_num_elements or 12)))
        )
        self.initial_max_nodes = _optional_int(teacher_config.get('initial_max_nodes', max(32, 4 * int(self.initial_target_num_elements or 12))))
        self.initial_max_dofs = _optional_int(teacher_config.get('initial_max_dofs', max(64, 2 * int(self.initial_max_nodes or 32))))
        self.initial_max_runtime_seconds = float(teacher_config.get('initial_max_runtime_seconds', 20.0))
        self.initial_max_budget_fraction = float(teacher_config.get('initial_max_budget_fraction', 0.22))
        self.initial_measure_budget_factor = float(teacher_config.get('initial_measure_budget_factor', 3.0))
        self.initial_mesh_generation_mode = str(teacher_config.get('initial_mesh_generation_mode', 'amber_uniform')).lower()
        self.initial_sizing_field_scale = float(teacher_config.get('initial_sizing_field_scale', 1.0))
        self.initial_sizing_field_retry_factor = float(teacher_config.get('initial_sizing_field_retry_factor', 1.25))
        self.initial_preserve_feature_edges = bool(teacher_config.get('initial_preserve_feature_edges', True))
        self.initial_geometry_constraint_mode = str(teacher_config.get('initial_geometry_constraint_mode', 'full')).lower()
        self.initial_enable_transfinite_hole_curves = bool(teacher_config.get('initial_enable_transfinite_hole_curves', True))
        self.initial_geometry_locality_scale = float(teacher_config.get('initial_geometry_locality_scale', 0.2))
        self.initial_hole_band_distance_scale = float(teacher_config.get('initial_hole_band_distance_scale', 0.35))
        self.initial_hole_min_segments = int(teacher_config.get('initial_hole_min_segments', 6))
        self.initial_absolute_caps_scale_with_budget = bool(
            teacher_config.get(
                'initial_absolute_caps_scale_with_budget',
                self.smoke_config.get('initial_absolute_caps_scale_with_budget', True),
            )
        )
        self.reject_if_initial_mesh_too_dense = bool(teacher_config.get('reject_if_initial_mesh_too_dense', True))
        self.enable_budget_calibration = bool(teacher_config.get('enable_budget_calibration', True))
        self.budget_calibration_max_iters = int(teacher_config.get('budget_calibration_max_iters', 4))
        self.budget_calibration_tolerance = float(teacher_config.get('budget_calibration_tolerance', 0.15))
        self.budget_calibration_timeout_seconds = float(teacher_config.get('budget_calibration_timeout_seconds', 20.0))
        self.field_stage_debug_dump = bool(teacher_config.get('field_stage_debug_dump', True))
        self.separability_probe_count = int(self.smoke_config.get('separability_probe_count', 256))
        self.enable_low_importance_inflation = bool(teacher_config.get('enable_low_importance_inflation', True))
        self.condition_difference_preservation_enable = bool(teacher_config.get('condition_difference_preservation_enable', True))
        self.geometry_local_floor_gap = float(teacher_config.get('geometry_local_floor_gap', 0.12))
        self.adaptive_refinement_local_refine_for_complex_3d_enable = bool(
            teacher_config.get('adaptive_refinement_local_refine_for_complex_3d_enable', True)
        )
        self.adaptive_refinement_local_refine_complexity_threshold = float(
            teacher_config.get('adaptive_refinement_local_refine_complexity_threshold', 1.85)
        )
        self.adaptive_refinement_local_refine_reference_elements = float(
            teacher_config.get(
                'adaptive_refinement_local_refine_reference_elements',
                self.smoke_config.get('adaptive_stage_timeout_reference_elements', 2500.0),
            )
        )

    def generate(
        self,
        geometry_record: GeometryRecord,
        preprocess_record: GeometryPreprocessRecord,
        condition_record: ConditionRecord,
        layout: PipelineLayout,
        overwrite: bool = False,
        runtime_tracker: RuntimeTracker | None = None,
        prescreen_record: PrescreenRecord | None = None,
    ) -> tuple[TeacherRecord | None, list[SampleRecord], FailureRecord | None]:
        teacher_record_path = layout.teacher_record_path(geometry_record.geometry_id, condition_record.condition_id)
        if teacher_record_path.exists() and not overwrite:
            teacher_payload = load_json(teacher_record_path)
            teacher_record = TeacherRecord(**teacher_payload)
            sample_records = [
                SampleRecord(**load_json(sample_path))
                for sample_path in sorted(layout.samples_dir.glob('*.json'))
                if load_json(sample_path).get('condition_id') == condition_record.condition_id
            ]
            return teacher_record, sample_records, None

        started_at = runtime_tracker.started_at if runtime_tracker is not None else now_iso()
        wall_time_start = time.perf_counter()
        try:
            teacher_record, sample_records = self._run_teacher(
                geometry_record=geometry_record,
                preprocess_record=preprocess_record,
                condition_record=condition_record,
                layout=layout,
                started_at=started_at,
                wall_time_start=wall_time_start,
                runtime_tracker=runtime_tracker,
                prescreen_record=prescreen_record,
            )
            dump_json(teacher_record_path, teacher_record.to_dict())
            for sample_record in sample_records:
                dump_json(layout.sample_path(sample_record.sample_id), sample_record.to_dict())
            return teacher_record, sample_records, None
        except PipelineAbort as exc:
            return self._persist_failure(
                geometry_record=geometry_record,
                preprocess_record=preprocess_record,
                condition_record=condition_record,
                layout=layout,
                started_at=started_at,
                wall_time_start=wall_time_start,
                stage_where_stopped=exc.stage,
                failure_category=exc.category,
                failure_reason=str(exc),
                runtime_tracker=runtime_tracker,
            )
        except Exception as exc:
            return self._persist_failure(
                geometry_record=geometry_record,
                preprocess_record=preprocess_record,
                condition_record=condition_record,
                layout=layout,
                started_at=started_at,
                wall_time_start=wall_time_start,
                stage_where_stopped='teacher_runtime',
                failure_category='numerical_failure',
                failure_reason=str(exc),
                runtime_tracker=runtime_tracker,
            )

    def _solver_metadata(self) -> dict[str, Any]:
        return {
            'reference_refinement_levels': self.reference_refinement_levels,
            'max_adaptive_steps': self.max_adaptive_steps,
            'refine_theta': self.refine_theta,
            'initial_mesh_element_volume': self.initial_mesh_element_volume,
            'enable_geometry_fidelity_constraints': self.enable_geometry_fidelity_constraints,
            'surface_first_meshing': self.surface_first_meshing,
            'contrast_mode': self.contrast_mode,
            'contrast_gamma': self.contrast_gamma,
            'max_dofs': self.max_dofs,
            'max_matrix_nnz': self.max_matrix_nnz,
            'minimum_viable_budget': self.minimum_viable_budget,
            'desired_budget': self.desired_budget,
            'hard_max_budget': self.hard_max_budget,
            'budget_status_mode': self.budget_status_mode,
            'budget_growth_enable': self.budget_growth_enable,
            'budget_growth_max_steps': self.budget_growth_max_steps,
            'budget_growth_batch_refine_fraction': self.budget_growth_batch_refine_fraction,
            'budget_growth_stop_on_diminishing_return': self.budget_growth_stop_on_diminishing_return,
            'budget_growth_timeout_seconds': self.budget_growth_timeout_seconds,
            'budget_growth_use_local_refine': self.budget_growth_use_local_refine,
            'budget_growth_cad_cleanup_interval': self.budget_growth_cad_cleanup_interval,
            'save_final_allocation_diagnostics': self.save_final_allocation_diagnostics,
            'save_parent_wall_clock': self.save_parent_wall_clock,
            'save_worker_elapsed_separately': self.save_worker_elapsed_separately,
            'elasticity_smoke_mode': self.elasticity_smoke_mode,
            'elasticity_smoke_reference_level': self.elasticity_smoke_reference_level,
            'initial_target_num_elements': self.initial_target_num_elements,
            'initial_target_num_surface_faces': self.initial_target_num_surface_faces,
            'initial_max_nodes': self.initial_max_nodes,
            'initial_max_dofs': self.initial_max_dofs,
            'initial_max_runtime_seconds': self.initial_max_runtime_seconds,
            'initial_max_budget_fraction': self.initial_max_budget_fraction,
            'initial_mesh_generation_mode': self.initial_mesh_generation_mode,
            'initial_sizing_field_scale': self.initial_sizing_field_scale,
            'initial_sizing_field_retry_factor': self.initial_sizing_field_retry_factor,
            'enable_budget_calibration': self.enable_budget_calibration,
            'budget_calibration_max_iters': self.budget_calibration_max_iters,
            'budget_calibration_tolerance': self.budget_calibration_tolerance,
            'budget_calibration_timeout_seconds': self.budget_calibration_timeout_seconds,
            'field_stage_debug_dump': self.field_stage_debug_dump,
            'separability_probe_count': self.separability_probe_count,
            'enable_low_importance_inflation': self.enable_low_importance_inflation,
            'condition_difference_preservation_enable': self.condition_difference_preservation_enable,
            'adaptive_refinement_local_refine_for_complex_3d_enable': self.adaptive_refinement_local_refine_for_complex_3d_enable,
            'adaptive_refinement_local_refine_complexity_threshold': self.adaptive_refinement_local_refine_complexity_threshold,
            'adaptive_refinement_local_refine_reference_elements': self.adaptive_refinement_local_refine_reference_elements,
        }

    def _persist_failure(
        self,
        *,
        geometry_record: GeometryRecord,
        preprocess_record: GeometryPreprocessRecord,
        condition_record: ConditionRecord,
        layout: PipelineLayout,
        started_at: str,
        wall_time_start: float,
        stage_where_stopped: str,
        failure_category: str,
        failure_reason: str,
        runtime_tracker: RuntimeTracker | None,
    ) -> tuple[TeacherRecord, list[SampleRecord], FailureRecord]:
        teacher_dir = layout.teacher_dir(geometry_record.geometry_id, condition_record.condition_id)
        initial_mesh_path = teacher_dir / 'initial_mesh.vtk'
        trajectory_dir = teacher_dir / 'trajectory'
        trajectory_mesh_paths = [str(path) for path in sorted(trajectory_dir.glob('*.vtk'))]
        elapsed_seconds = float(time.perf_counter() - wall_time_start)
        partial_output_available = initial_mesh_path.exists() or bool(trajectory_mesh_paths)
        teacher_record = TeacherRecord(
            geometry_id=geometry_record.geometry_id,
            condition_id=condition_record.condition_id,
            pde_family=condition_record.pde_family,
            initial_mesh_path=str(initial_mesh_path),
            initial_surface_mesh_path=str(layout.teacher_surface_mesh_path(geometry_record.geometry_id, condition_record.condition_id)),
            trajectory_mesh_paths=trajectory_mesh_paths,
            solver_metadata=self._solver_metadata(),
            wall_time_sec=elapsed_seconds,
            status='failed',
            started_at=started_at,
            finished_at=now_iso(),
            elapsed_seconds=elapsed_seconds,
            stage_where_stopped=stage_where_stopped,
            failure_category=failure_category,
            partial_output_available=partial_output_available,
            failure_reason=failure_reason,
        )
        failed_samples = self._build_failed_sample_records(
            geometry_record=geometry_record,
            condition_record=condition_record,
            preprocess_record=preprocess_record,
            layout=layout,
            initial_mesh_path=initial_mesh_path,
            trajectory_mesh_paths=trajectory_mesh_paths,
            failure_reason=failure_reason,
            failure_category=failure_category,
            stage_where_stopped=stage_where_stopped,
            started_at=started_at,
            finished_at=now_iso(),
            elapsed_seconds=elapsed_seconds,
            partial_output_available=partial_output_available,
        )
        dump_json(layout.teacher_record_path(geometry_record.geometry_id, condition_record.condition_id), teacher_record.to_dict())
        for sample_record in failed_samples:
            dump_json(layout.sample_path(sample_record.sample_id), sample_record.to_dict())
        failure = FailureRecord(
            stage='teacher',
            item_id=f'{geometry_record.geometry_id}:{condition_record.condition_id}',
            source_path=geometry_record.source_path,
            reason=failure_reason,
            category=failure_category,
            started_at=started_at,
            finished_at=now_iso(),
            elapsed_seconds=elapsed_seconds,
            stage_where_stopped=stage_where_stopped,
            partial_output_available=partial_output_available,
        )
        if runtime_tracker is not None:
            runtime_tracker.fail(
                failure_reason=failure_reason,
                failure_category=failure_category,
                stage_where_stopped=stage_where_stopped,
                partial_output_available=partial_output_available,
            )
        return teacher_record, failed_samples, failure

    def _run_teacher(
        self,
        *,
        geometry_record: GeometryRecord,
        preprocess_record: GeometryPreprocessRecord,
        condition_record: ConditionRecord,
        layout: PipelineLayout,
        started_at: str,
        wall_time_start: float,
        runtime_tracker: RuntimeTracker | None,
        prescreen_record: PrescreenRecord | None,
    ) -> tuple[TeacherRecord, list[SampleRecord]]:
        if condition_record.pde_family == 'scalar_elliptic' and not self.scalar_smoke_enable:
            raise BudgetControlError('scalar smoke layer is disabled for this run', category='reject_invalid', stage='teacher_runtime')
        if condition_record.pde_family == 'linear_elasticity' and not self.elasticity_smoke_enable:
            raise BudgetControlError('elasticity smoke layer is disabled for this run', category='reject_too_expensive_prescreen', stage='teacher_runtime')

        teacher_dir = layout.teacher_dir(geometry_record.geometry_id, condition_record.condition_id)
        trajectory_dir = teacher_dir / 'trajectory'
        fields_dir = teacher_dir / 'fields'
        budgets_dir = teacher_dir / 'budgets'
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        fields_dir.mkdir(parents=True, exist_ok=True)
        budgets_dir.mkdir(parents=True, exist_ok=True)

        budgets = sorted(int(budget) for budget in condition_record.budget_or_tolerance_spec.get('budgets', []))
        if not budgets and self.target_num_elements is not None:
            budgets = [int(self.target_num_elements)]
        if not budgets:
            budgets = [max(int(self.initial_target_num_elements or 16), 1)]
        target_budget = int(min(budgets))
        primary_budget_tiers = self._budget_tiers(target_budget)

        geometry_fn = geometry_fn_from_path(geometry_record.source_path)
        initial_mesh_path = teacher_dir / 'initial_mesh.vtk'
        initial_surface_mesh_path = layout.teacher_surface_mesh_path(geometry_record.geometry_id, condition_record.condition_id)
        current_mesh, surface_quality_metrics, volume_quality_metrics, initial_mesh_diagnostics, geometry_retry_history = self._build_initial_mesh(
            geometry_fn=geometry_fn,
            preprocess_record=preprocess_record,
            condition_record=condition_record,
            target_budget=target_budget,
            surface_mesh_path=initial_surface_mesh_path,
            runtime_tracker=runtime_tracker,
        )
        save_as_vtk(current_mesh, initial_mesh_path)

        trajectory_mesh_paths: list[str] = []
        trajectory_solution_paths: list[str] = []
        trajectory_indicator_paths: list[str] = []
        adaptive_error_history: list[dict[str, Any]] = []
        adaptive_error_history_path = fields_dir / 'adaptive_error_history.json'
        final_payload: dict[str, Any] | None = None
        final_stage = 'completed'
        stop_reason: str | None = None

        for step in range(self.max_adaptive_steps + 1):
            if runtime_tracker is not None:
                runtime_tracker.check_soft_limits()
            solve_result, indicator, reference_mesh, reference_result = self._solve_and_estimate(
                current_mesh,
                preprocess_record,
                condition_record,
                geometry_fn,
                runtime_tracker=runtime_tracker,
            )
            mesh_step_path, solution_step_path, indicator_step_path = self._save_step_artifacts(
                current_mesh=current_mesh,
                solve_result=solve_result,
                indicator=indicator,
                trajectory_dir=trajectory_dir,
                fields_dir=fields_dir,
                step=step,
            )
            adaptive_error_history.append(
                self._amr_step_error_diagnostics(
                    step=step,
                    current_mesh=current_mesh,
                    indicator=indicator,
                    solve_result=solve_result,
                    reference_mesh=reference_mesh,
                    reference_result=reference_result,
                    indicator_path=indicator_step_path,
                    mesh_path=mesh_step_path,
                )
            )
            if self.store_trajectory or step == 0:
                trajectory_mesh_paths.append(str(mesh_step_path))
                trajectory_solution_paths.append(str(solution_step_path))
                trajectory_indicator_paths.append(str(indicator_step_path))
            final_payload = {
                'mesh': current_mesh,
                'solve_result': solve_result,
                'indicator': indicator,
                'reference_mesh': reference_mesh,
                'reference_result': reference_result,
            }
            if step == self.max_adaptive_steps:
                break
            if runtime_tracker is not None and runtime_tracker.should_soft_stop():
                final_stage = 'adaptive_refinement'
                stop_reason = 'early_stop'
                break
            next_mesh = self._advance_mesh(
                current_mesh=current_mesh,
                indicator=indicator,
                geometry_fn=geometry_fn,
                preprocess_record=preprocess_record,
                target_budget=target_budget,
                condition_record=condition_record,
                prescreen_record=prescreen_record,
                runtime_tracker=runtime_tracker,
            )
            if int(next_mesh.t.shape[1]) <= int(current_mesh.t.shape[1]):
                break
            if int(next_mesh.t.shape[1]) > int(primary_budget_tiers['hard_max_budget']):
                final_stage = 'adaptive_refinement'
                stop_reason = 'hard_cap_prevented_adaptive_overgrowth'
                break
            current_mesh = next_mesh

        if final_payload is None:
            raise BudgetControlError('teacher failed before producing a control mesh', category='numerical_failure', stage='teacher_runtime')
        dump_json(adaptive_error_history_path, {'steps': adaptive_error_history})

        probe_points = self._stage_probe_points(preprocess_record)
        budget_results: list[dict[str, Any]] = []
        sample_records: list[SampleRecord] = []
        for budget in budgets:
            budget_result = self._materialize_budget_result(
                geometry_record=geometry_record,
                condition_record=condition_record,
                preprocess_record=preprocess_record,
                budgets_dir=budgets_dir,
                budget=budget,
                current_mesh=final_payload['mesh'],
                solve_result=final_payload['solve_result'],
                indicator=final_payload['indicator'],
                reference_mesh=final_payload['reference_mesh'],
                reference_result=final_payload['reference_result'],
                initial_mesh_path=initial_mesh_path,
                trajectory_mesh_paths=trajectory_mesh_paths,
                geometry_fn=geometry_fn,
                runtime_tracker=runtime_tracker,
                probe_points=probe_points,
                prescreen_record=prescreen_record,
                initial_mesh_diagnostics=initial_mesh_diagnostics,
                preprocess_record_for_constraints=preprocess_record,
            )
            budget_result['adaptive_error_history'] = adaptive_error_history
            budget_result['adaptive_error_history_path'] = str(adaptive_error_history_path)
            budget_results.append(budget_result)
            sample_records.append(
                self._sample_record_from_budget_result(
                    budget_result,
                    geometry_record,
                    condition_record,
                    preprocess_record,
                    initial_mesh_path,
                    trajectory_mesh_paths,
                    layout,
                    started_at=started_at,
                    finished_at=now_iso(),
                    elapsed_seconds=time.perf_counter() - wall_time_start,
                    surface_quality_metrics=surface_quality_metrics,
                    volume_quality_metrics=volume_quality_metrics,
                    initial_mesh_diagnostics=initial_mesh_diagnostics,
                )
            )

        statuses = [result['status'] for result in budget_results]
        teacher_status = 'success_budget_closed'
        if stop_reason == 'early_stop' and any(not _is_success_status(status) for status in statuses):
            teacher_status = 'early_stop'
        elif statuses:
            teacher_status = sorted(statuses, key=_status_priority)[-1]
            if _is_success_status(teacher_status) and any(not _is_success_status(status) for status in statuses):
                teacher_status = next(status for status in statuses if not _is_success_status(status))
        if _is_success_status(teacher_status) and stop_reason == 'early_stop':
            final_stage = 'adaptive_refinement'

        elapsed_seconds = float(time.perf_counter() - wall_time_start)
        solver_metadata = self._solver_metadata()
        solver_metadata['adaptive_error_history'] = adaptive_error_history
        solver_metadata['adaptive_error_history_path'] = str(adaptive_error_history_path)
        teacher_record = TeacherRecord(
            geometry_id=geometry_record.geometry_id,
            condition_id=condition_record.condition_id,
            pde_family=condition_record.pde_family,
            initial_mesh_path=str(initial_mesh_path),
            initial_surface_mesh_path=str(initial_surface_mesh_path) if initial_surface_mesh_path.exists() else None,
            trajectory_mesh_paths=trajectory_mesh_paths,
            trajectory_solution_paths=trajectory_solution_paths,
            trajectory_indicator_paths=trajectory_indicator_paths,
            budget_results=budget_results,
            solver_metadata=solver_metadata,
            wall_time_sec=elapsed_seconds,
            status=teacher_status,
            surface_quality_metrics=surface_quality_metrics,
            volume_quality_metrics=volume_quality_metrics,
            geometry_constraint_summary=combine_geometry_constraints(
                preprocess_record.geometry_features or {},
                base_size=float(volume_to_edge_length(self.initial_mesh_element_volume, dim=int(preprocess_record.dimension))),
                config=self.teacher_config,
                attempt_index=0,
            ),
            geometry_retry_history=geometry_retry_history,
            initial_mesh_diagnostics=initial_mesh_diagnostics,
            budget_calibration_diagnostics=[result['budget_diagnostics'] for result in budget_results],
            started_at=started_at,
            finished_at=now_iso(),
            elapsed_seconds=elapsed_seconds,
            stage_where_stopped=final_stage if _is_success_status(teacher_status) else teacher_status,
            partial_output_available=True,
        )
        return teacher_record, sample_records

    def _effective_initial_element_cap(self, target_budget: int) -> int:
        budget_fraction_cap = max(int(round(target_budget * self.initial_max_budget_fraction)), 4)
        if self.initial_target_num_elements is None:
            return budget_fraction_cap
        absolute_cap = max(int(self.initial_target_num_elements), 4)
        if self.initial_absolute_caps_scale_with_budget and int(target_budget) >= self.high_budget_threshold:
            return max(budget_fraction_cap, absolute_cap)
        return max(min(budget_fraction_cap, absolute_cap), 4)

    def _effective_initial_cap_scale(self, target_budget: int) -> float:
        if self.initial_target_num_elements is None:
            return 1.0
        base_cap = max(int(self.initial_target_num_elements), 4)
        return max(float(self._effective_initial_element_cap(target_budget)) / float(base_cap), 1.0)

    def _scaled_initial_cap(self, value: int | None, target_budget: int) -> int | None:
        if value is None:
            return None
        scale = self._effective_initial_cap_scale(target_budget)
        return max(int(np.ceil(float(value) * scale)), int(value))

    def _effective_initial_surface_face_cap(self, target_budget: int) -> int | None:
        return self._scaled_initial_cap(self.initial_target_num_surface_faces, target_budget)

    def _effective_initial_node_cap(self, target_budget: int) -> int | None:
        return self._scaled_initial_cap(self.initial_max_nodes, target_budget)

    def _effective_initial_dof_cap(self, target_budget: int) -> int | None:
        return self._scaled_initial_cap(self.initial_max_dofs, target_budget)

    def _effective_budget_growth_controls(self, *, current_elements: int, desired_budget: int) -> tuple[int, float]:
        max_steps = max(int(self.budget_growth_max_steps), 0)
        batch_fraction = float(self.budget_growth_batch_refine_fraction)
        if not self.budget_growth_dynamic_step_enable or current_elements <= 0 or desired_budget <= 0:
            return max_steps, batch_fraction
        budget_ratio = max(float(desired_budget) / max(float(current_elements), 1.0), 1.0)
        if budget_ratio <= 1.0:
            return max_steps, batch_fraction
        growth_baseline = 1.0 + max(min(batch_fraction * 1.6, 0.60), 0.20)
        required_steps = int(np.ceil(np.log(budget_ratio) / np.log(growth_baseline)))
        max_steps = max(max_steps, required_steps + 2)
        if desired_budget >= self.high_budget_threshold:
            if budget_ratio >= 12.0:
                batch_fraction = max(batch_fraction, 0.28)
            elif budget_ratio >= 6.0:
                batch_fraction = max(batch_fraction, 0.25)
            elif budget_ratio >= 3.0:
                batch_fraction = max(batch_fraction, 0.22)
        batch_fraction = min(batch_fraction, self.budget_growth_batch_refine_fraction_max)
        max_steps = min(max_steps, max(self.budget_growth_max_steps_cap, self.budget_growth_max_steps))
        return max_steps, batch_fraction

    def _should_enforce_predictive_growth_caps(self, budget_tiers: dict[str, Any]) -> bool:
        if not self.disable_predictive_growth_caps_for_high_budget:
            return True
        return int(budget_tiers.get('desired_budget', 0) or 0) < self.high_budget_threshold

    def _initial_target_element_volume(self, preprocess_record: GeometryPreprocessRecord, target_budget: int) -> float:
        target_elements = self._effective_initial_element_cap(target_budget)
        geometry_measure = _bbox_measure(preprocess_record)
        return max(
            float(self.initial_mesh_element_volume),
            float(self.initial_measure_budget_factor * geometry_measure / max(target_elements, 1)),
        )

    def _amber_uniform_initial_element_volume(self, *, dimension: int, sizing_scale: float) -> float:
        base_edge_length = float(volume_to_edge_length(float(self.initial_mesh_element_volume), dim=dimension))
        scaled_edge_length = base_edge_length * max(float(sizing_scale), 1.0e-6)
        return float(edge_length_to_volume(np.asarray(scaled_edge_length), dim=dimension))

    def _budget_tiers(self, requested_budget: int) -> dict[str, Any]:
        requested_budget = max(int(requested_budget), 1)
        desired_budget = max(int(self.desired_budget or requested_budget), 1)
        minimum_viable_budget = int(self.minimum_viable_budget or max(1, round(0.35 * desired_budget)))
        minimum_viable_budget = min(max(minimum_viable_budget, 1), desired_budget)
        hard_max_budget = int(self.hard_max_budget or max(requested_budget, desired_budget, round(desired_budget * self.max_budget_overrun_ratio)))
        hard_max_budget = max(hard_max_budget, desired_budget)
        return {
            'requested_budget': requested_budget,
            'minimum_viable_budget': minimum_viable_budget,
            'desired_budget': desired_budget,
            'hard_max_budget': hard_max_budget,
            'budget_status_mode': self.budget_status_mode,
            'allow_success_partial_under_budget': bool(self.allow_success_partial_under_budget),
        }

    def _classify_budget_status(
        self,
        *,
        actual_budget: int,
        budget_tiers: dict[str, Any],
        growth_stalled: bool = False,
        timed_out: bool = False,
        hard_cap_exceeded: bool = False,
    ) -> str:
        actual_budget = int(actual_budget)
        minimum_viable = max(int(budget_tiers['minimum_viable_budget']), 1)
        desired = max(int(budget_tiers['desired_budget']), 1)
        hard_max = max(int(budget_tiers['hard_max_budget']), desired)
        ratio = float(actual_budget / max(float(desired), 1.0))
        if hard_cap_exceeded or actual_budget > hard_max:
            return 'fail_budget_hard_cap_exceeded'
        if timed_out and actual_budget < minimum_viable:
            return 'fail_budget_growth_timeout'
        if abs(ratio - 1.0) <= self.budget_calibration_tolerance:
            return 'success_budget_closed'
        near_threshold = max(0.80, 1.0 - 2.0 * self.budget_calibration_tolerance)
        if actual_budget >= int(np.ceil(desired * near_threshold)):
            return 'success_near_desired_budget'
        if actual_budget >= minimum_viable and bool(budget_tiers.get('allow_success_partial_under_budget', True)):
            return 'success_partial_under_budget'
        if timed_out:
            return 'fail_budget_growth_timeout'
        return 'fail_budget_growth_stalled' if growth_stalled or actual_budget < minimum_viable else 'budget_closure_failed'

    def _estimated_solution_dofs(self, mesh, condition_record: ConditionRecord) -> int:
        return int(mesh.nvertices * (int(mesh.dim()) if condition_record.pde_family == 'linear_elasticity' else 1))

    def _initial_meshing_config(self) -> dict[str, Any]:
        config = dict(self.teacher_config)
        config['geometry_constraint_mode'] = self.initial_geometry_constraint_mode
        config['enable_transfinite_hole_curves'] = self.initial_enable_transfinite_hole_curves
        config['geometry_constraint_locality_scale'] = self.initial_geometry_locality_scale
        config['hole_band_distance_scale'] = self.initial_hole_band_distance_scale
        if not self.initial_preserve_feature_edges:
            config['min_circle_segments'] = min(int(self.teacher_config.get('min_circle_segments', 20)), int(self.initial_hole_min_segments))
            config['hole_edge_length_ratio'] = max(float(self.teacher_config.get('hole_edge_length_ratio', 0.24)), 0.85)
            config['hole_radial_refinement_layers'] = max(1, int(self.teacher_config.get('initial_hole_radial_refinement_layers', 1)))
            config['hole_radial_growth_rate'] = max(float(self.teacher_config.get('hole_radial_growth_rate', 1.45)), 2.2)
            config['curvature_refinement_strength'] = 0.0
            config['feature_size_refinement_strength'] = 0.0
            config['max_circle_fit_error'] = max(float(self.teacher_config.get('max_circle_fit_error', 0.04)), 0.18)
            config['max_boundary_deviation'] = max(float(self.teacher_config.get('max_boundary_deviation', 0.02)), 0.16)
            config['max_normal_deviation'] = max(float(self.teacher_config.get('max_normal_deviation', 22.5)), 50.0)
            return config

        config['geometry_min_size_ratio'] = float(
            self.teacher_config.get(
                'initial_geometry_min_size_ratio',
                max(0.35, float(self.teacher_config.get('geometry_min_size_ratio', 0.015))),
            )
        )
        config['geometry_constraint_mode'] = 'full' if self.initial_geometry_constraint_mode == 'topology_only' else self.initial_geometry_constraint_mode
        config['enable_transfinite_hole_curves'] = self.initial_enable_transfinite_hole_curves
        config['geometry_constraint_locality_scale'] = max(self.initial_geometry_locality_scale, 0.65)
        config['hole_band_distance_scale'] = max(self.initial_hole_band_distance_scale, 0.65)
        config['min_circle_segments'] = max(int(self.teacher_config.get('min_circle_segments', 20)), int(self.initial_hole_min_segments))
        config['hole_edge_length_ratio'] = min(
            float(self.teacher_config.get('hole_edge_length_ratio', 0.24)),
            float(
                self.teacher_config.get(
                    'initial_hole_edge_length_ratio',
                    self.teacher_config.get('hole_edge_length_ratio', 0.24),
                )
            ),
        )
        config['hole_radial_refinement_layers'] = max(
            1,
            int(self.teacher_config.get('initial_hole_radial_refinement_layers', self.teacher_config.get('hole_radial_refinement_layers', 3))),
        )
        config['hole_radial_growth_rate'] = float(
            self.teacher_config.get('initial_hole_radial_growth_rate', self.teacher_config.get('hole_radial_growth_rate', 1.45))
        )
        config['curvature_refinement_strength'] = float(
            self.teacher_config.get(
                'initial_curvature_refinement_strength',
                0.2 * float(self.teacher_config.get('curvature_refinement_strength', 1.5)),
            )
        )
        config['feature_size_refinement_strength'] = float(
            self.teacher_config.get(
                'initial_feature_size_refinement_strength',
                0.2 * float(self.teacher_config.get('feature_size_refinement_strength', 1.5)),
            )
        )
        config['max_circle_fit_error'] = float(
            self.teacher_config.get('initial_max_circle_fit_error', self.teacher_config.get('max_circle_fit_error', 0.04))
        )
        config['max_boundary_deviation'] = float(
            self.teacher_config.get('initial_max_boundary_deviation', self.teacher_config.get('max_boundary_deviation', 0.02))
        )
        config['max_normal_deviation'] = float(
            self.teacher_config.get('initial_max_normal_deviation', self.teacher_config.get('max_normal_deviation', 22.5))
        )
        return config

    def _annotate_initial_mesh_diagnostics(
        self,
        diagnostics: dict[str, Any],
        *,
        seed_config: dict[str, Any],
        source: str,
    ) -> dict[str, Any]:
        diagnostics.update(
            {
                'initial_mesh_source': source,
                'initial_mesh_generation_mode': str(seed_config.get('initial_mesh_generation_mode', self.initial_mesh_generation_mode)),
                'initial_sizing_field_scale': float(seed_config.get('initial_sizing_field_scale', self.initial_sizing_field_scale)),
                'initial_requested_element_volume': _optional_float(
                    seed_config.get('initial_requested_element_volume', self.initial_mesh_element_volume)
                ),
                'initial_preserve_feature_edges': bool(self.initial_preserve_feature_edges),
                'initial_geometry_constraint_mode': str(seed_config.get('geometry_constraint_mode', 'unknown')),
                'initial_enable_transfinite_hole_curves': bool(seed_config.get('enable_transfinite_hole_curves', False)),
                'initial_required_min_circle_segments': int(seed_config.get('min_circle_segments', 0)),
                'initial_hole_edge_length_ratio': float(seed_config.get('hole_edge_length_ratio', 0.0)),
                'initial_geometry_min_size_ratio': float(seed_config.get('geometry_min_size_ratio', 0.0)),
                'initial_max_boundary_deviation': float(seed_config.get('max_boundary_deviation', 0.0)),
                'initial_max_normal_deviation': float(seed_config.get('max_normal_deviation', 0.0)),
            }
        )
        return diagnostics

    def _compute_initial_mesh_diagnostics(
        self,
        *,
        mesh,
        surface_mesh: Any | None,
        surface_metrics: dict[str, Any],
        elapsed_seconds: float,
        target_budget: int,
        condition_record: ConditionRecord,
        preprocess_record: GeometryPreprocessRecord,
    ) -> dict[str, Any]:
        expected_holes = int((preprocess_record.geometry_features or {}).get('statistics', {}).get('num_hole_features', 0))
        measured_holes = int(surface_metrics.get('hole_sampling', {}).get('measured_holes', expected_holes))
        hole_records = surface_metrics.get('hole_sampling', {}).get('records', [])
        hole_segments = [int(record.get('segments', 0)) for record in hole_records]
        initial_num_elements = int(mesh.t.shape[1])
        initial_num_nodes = int(mesh.nvertices)
        initial_surface_faces = _surface_face_count(surface_mesh) if surface_mesh is not None else int(len(mesh.boundary_facets()))
        initial_mean_edge_length = _mesh_mean_edge_length(mesh)
        initial_budget_fraction = float(initial_num_elements / max(float(target_budget), 1.0))
        estimated_dofs = _estimate_initial_dofs(mesh, condition_record.pde_family)
        surface_cap = self._effective_initial_surface_face_cap(target_budget)
        node_cap = self._effective_initial_node_cap(target_budget)
        dof_cap = self._effective_initial_dof_cap(target_budget)
        density_reasons = []
        if initial_num_elements > self._effective_initial_element_cap(target_budget):
            density_reasons.append('initial_target_num_elements')
        if surface_cap is not None and initial_surface_faces > int(surface_cap):
            density_reasons.append('initial_target_num_surface_faces')
        if node_cap is not None and initial_num_nodes > int(node_cap):
            density_reasons.append('initial_max_nodes')
        if dof_cap is not None and estimated_dofs > int(dof_cap):
            density_reasons.append('initial_max_dofs')
        if elapsed_seconds > self.initial_max_runtime_seconds:
            density_reasons.append('initial_max_runtime_seconds')
        if initial_budget_fraction > self.initial_max_budget_fraction:
            density_reasons.append('initial_max_budget_fraction')
        topology_preserved = measured_holes >= expected_holes
        return {
            'initial_num_elements': initial_num_elements,
            'initial_num_nodes': initial_num_nodes,
            'initial_surface_faces': initial_surface_faces,
            'initial_mean_edge_length': initial_mean_edge_length,
            'initial_hole_boundary_segments': hole_segments,
            'initial_budget_fraction': initial_budget_fraction,
            'initial_is_too_dense': bool(density_reasons),
            'initial_density_reasons': density_reasons,
            'initial_estimated_dofs': estimated_dofs,
            'initial_runtime_seconds': float(elapsed_seconds),
            'initial_target_num_elements': self._effective_initial_element_cap(target_budget),
            'initial_target_num_surface_faces': surface_cap,
            'initial_max_nodes': node_cap,
            'initial_max_dofs': dof_cap,
            'initial_max_runtime_seconds': self.initial_max_runtime_seconds,
            'initial_max_budget_fraction': self.initial_max_budget_fraction,
            'topology_preserved': bool(topology_preserved),
            'expected_holes': expected_holes,
            'measured_holes': measured_holes,
            'coarse_seed_verdict': 'too_dense' if density_reasons else ('topology_invalid' if not topology_preserved else 'coarse_ok'),
        }

    def _build_amber_uniform_initial_mesh(
        self,
        *,
        geometry_fn,
        preprocess_record: GeometryPreprocessRecord,
        condition_record: ConditionRecord,
        target_budget: int,
        runtime_tracker: RuntimeTracker | None,
        started: float,
    ):
        if int(preprocess_record.dimension) == 2:
            from src.tasks.domains.extended_mesh_tri1 import ExtendedMeshTri1

            mesh_cls = ExtendedMeshTri1
        else:
            from src.tasks.domains.extended_mesh_tet1 import ExtendedMeshTet1

            mesh_cls = ExtendedMeshTet1

        retry_history = []
        last_diagnostics: dict[str, Any] | None = None
        last_status = 'amber_uniform_failed'
        retry_factor = max(float(self.initial_sizing_field_retry_factor), 1.01)
        expected_holes = int((preprocess_record.geometry_features or {}).get('statistics', {}).get('num_hole_features', 0))
        surface_metrics = {
            'status': 'amber_uniform_geom_fn',
            'hole_sampling': {'records': [], 'measured_holes': expected_holes},
        }
        volume_metrics = {'status': 'success'}

        for attempt in range(self.max_geometry_retry + 1):
            if runtime_tracker is not None:
                runtime_tracker.enter_stage('surface_meshing', {'attempt': attempt, 'coarse_seed': True, 'mode': 'amber_uniform'})
                runtime_tracker.check_soft_limits()
            if time.perf_counter() - started > self.initial_max_runtime_seconds:
                raise StageTimeoutError(
                    'AMBER-style initial mesh generation exceeded the coarse-seed runtime cap',
                    category='timeout_surface_mesh',
                    stage='surface_meshing',
                )
            sizing_scale = max(float(self.initial_sizing_field_scale) * (retry_factor**attempt), 1.0e-6)
            attempt_volume = self._amber_uniform_initial_element_volume(
                dimension=int(preprocess_record.dimension),
                sizing_scale=sizing_scale,
            )
            mesh = mesh_cls.init_from_geom_fn(geom_fn=geometry_fn, max_element_volume=attempt_volume)
            mesh.geom_fn = geometry_fn
            diagnostics = self._compute_initial_mesh_diagnostics(
                mesh=mesh,
                surface_mesh=None,
                surface_metrics=surface_metrics,
                elapsed_seconds=float(time.perf_counter() - started),
                target_budget=target_budget,
                condition_record=condition_record,
                preprocess_record=preprocess_record,
            )
            seed_config = {
                'initial_mesh_generation_mode': 'amber_uniform',
                'initial_sizing_field_scale': sizing_scale,
                'initial_requested_element_volume': attempt_volume,
                'geometry_constraint_mode': 'amber_uniform_geom_fn',
                'enable_transfinite_hole_curves': False,
                'min_circle_segments': 0,
                'hole_edge_length_ratio': 0.0,
                'geometry_min_size_ratio': 0.0,
                'max_boundary_deviation': 0.0,
                'max_normal_deviation': 0.0,
            }
            self._annotate_initial_mesh_diagnostics(diagnostics, seed_config=seed_config, source='amber_uniform_geom_fn')
            last_diagnostics = diagnostics
            last_status = diagnostics['coarse_seed_verdict']
            retry_history.append(
                {
                    'attempt': attempt,
                    'mesh_element_volume': float(attempt_volume),
                    'sizing_field_scale': float(sizing_scale),
                    'status': last_status,
                    'mode': 'amber_uniform',
                }
            )
            if diagnostics['topology_preserved'] and not diagnostics['initial_is_too_dense']:
                return mesh, surface_metrics, volume_metrics, diagnostics, retry_history, last_status
            if not diagnostics['initial_is_too_dense']:
                break

        return None, surface_metrics, volume_metrics, last_diagnostics, retry_history, last_status

    def _build_initial_mesh(
        self,
        geometry_fn,
        preprocess_record: GeometryPreprocessRecord,
        condition_record: ConditionRecord,
        target_budget: int,
        surface_mesh_path: Path,
        runtime_tracker: RuntimeTracker | None,
    ):
        started = time.perf_counter()
        retry_history = []
        seed_config = self._initial_meshing_config()
        last_diagnostics: dict[str, Any] | None = None
        last_status = 'invalid_geometry'
        preprocess_modes = {'preprocess_coarse', 'coarse_mesh', 'coarse'}

        if self.initial_mesh_generation_mode in preprocess_modes:
            if runtime_tracker is not None:
                runtime_tracker.enter_stage('surface_meshing', {'attempt': 0, 'coarse_seed': True, 'mode': 'preprocess_coarse'})
                runtime_tracker.check_soft_limits()
            coarse_mesh = load_expert_mesh(preprocess_record.coarse_mesh_path)
            coarse_mesh.geom_fn = geometry_fn
            coarse_diagnostics = self._compute_initial_mesh_diagnostics(
                mesh=coarse_mesh,
                surface_mesh=None,
                surface_metrics={
                    'hole_sampling': {
                        'records': [],
                        'measured_holes': int((preprocess_record.geometry_features or {}).get('statistics', {}).get('num_hole_features', 0)),
                    }
                },
                elapsed_seconds=float(time.perf_counter() - started),
                target_budget=target_budget,
                condition_record=condition_record,
                preprocess_record=preprocess_record,
            )
            coarse_seed_config = {
                'initial_mesh_generation_mode': 'preprocess_coarse',
                'initial_sizing_field_scale': 1.0,
                'initial_requested_element_volume': None,
                'geometry_constraint_mode': 'preprocess_coarse',
                'enable_transfinite_hole_curves': False,
                'min_circle_segments': 0,
                'hole_edge_length_ratio': 0.0,
                'geometry_min_size_ratio': 0.0,
                'max_boundary_deviation': 0.0,
                'max_normal_deviation': 0.0,
            }
            self._annotate_initial_mesh_diagnostics(coarse_diagnostics, seed_config=coarse_seed_config, source='preprocess_coarse_seed')
            coarse_diagnostics['used_preprocess_coarse_as_initial'] = True
            retry_history.append({'attempt': 0, 'mesh_element_volume': None, 'status': coarse_diagnostics['coarse_seed_verdict'], 'mode': 'preprocess_coarse'})
            if coarse_diagnostics['initial_is_too_dense'] and self.reject_if_initial_mesh_too_dense:
                raise BudgetControlError(
                    'preprocess coarse mesh is too dense to use as the initial AMR seed under the current budget controls',
                    category='reject_bad_initial_mesh',
                    stage='surface_meshing',
                    partial_output_available=surface_mesh_path.exists(),
                    details=coarse_diagnostics,
                )
            return (
                coarse_mesh,
                {'status': 'preprocess_coarse_seed', 'hole_sampling': {'records': [], 'min_segments': 0}},
                {'status': 'success'},
                coarse_diagnostics,
                retry_history,
            )

        amber_modes = {'amber', 'amber_uniform', 'amber_uniform_then_cad'}
        cad_modes = {'cad', 'cad_aware', 'geometry_aware', 'amber_uniform_then_cad'}
        if self.initial_mesh_generation_mode in amber_modes:
            (
                amber_mesh,
                amber_surface_metrics,
                amber_volume_metrics,
                last_diagnostics,
                amber_retry_history,
                last_status,
            ) = self._build_amber_uniform_initial_mesh(
                geometry_fn=geometry_fn,
                preprocess_record=preprocess_record,
                condition_record=condition_record,
                target_budget=target_budget,
                runtime_tracker=runtime_tracker,
                started=started,
            )
            retry_history.extend(amber_retry_history)
            if amber_mesh is not None:
                return amber_mesh, amber_surface_metrics, amber_volume_metrics, last_diagnostics, retry_history

        if self.initial_mesh_generation_mode in cad_modes:
            base_volume = self._initial_target_element_volume(preprocess_record, target_budget)
        elif self.initial_mesh_generation_mode in amber_modes:
            base_volume = None
        else:
            raise BudgetControlError(
                f'Unsupported initial mesh generation mode: {self.initial_mesh_generation_mode}',
                category='reject_invalid',
                stage='surface_meshing',
                partial_output_available=surface_mesh_path.exists(),
                details={'initial_mesh_generation_mode': self.initial_mesh_generation_mode},
            )

        if self.initial_mesh_generation_mode in cad_modes and int(preprocess_record.dimension) == 2:
            from src.tasks.domains.extended_mesh_tri1 import ExtendedMeshTri1

            for attempt in range(self.max_geometry_retry + 1):
                if runtime_tracker is not None:
                    runtime_tracker.enter_stage('surface_meshing', {'attempt': attempt, 'coarse_seed': True})
                    runtime_tracker.check_soft_limits()
                attempt_volume = float(base_volume) * (0.35**attempt)
                mesh = ExtendedMeshTri1.init_from_geom_fn(geometry_fn, max_element_volume=attempt_volume)
                mesh.geom_fn = geometry_fn
                diagnostics = self._compute_initial_mesh_diagnostics(
                    mesh=mesh,
                    surface_mesh=None,
                    surface_metrics={'hole_sampling': {'records': [], 'measured_holes': 0}},
                    elapsed_seconds=float(time.perf_counter() - started),
                    target_budget=target_budget,
                    condition_record=condition_record,
                    preprocess_record=preprocess_record,
                )
                self._annotate_initial_mesh_diagnostics(diagnostics, seed_config=seed_config, source='direct_geometry_seed')
                retry_history.append({'attempt': attempt, 'mesh_element_volume': float(attempt_volume), 'status': diagnostics['coarse_seed_verdict']})
                last_diagnostics = diagnostics
                last_status = diagnostics['coarse_seed_verdict']
                if diagnostics['topology_preserved'] and not diagnostics['initial_is_too_dense']:
                    return mesh, {'status': 'success'}, {'status': 'success'}, diagnostics, retry_history
                if diagnostics['initial_is_too_dense']:
                    break
        elif self.initial_mesh_generation_mode in cad_modes:
            for attempt in range(self.max_geometry_retry + 1):
                if runtime_tracker is not None:
                    runtime_tracker.enter_stage('surface_meshing', {'attempt': attempt, 'coarse_seed': True})
                    runtime_tracker.check_soft_limits()
                if time.perf_counter() - started > self.initial_max_runtime_seconds:
                    raise StageTimeoutError(
                        'initial mesh generation exceeded the coarse-seed runtime cap',
                        category='timeout_surface_mesh',
                        stage='surface_meshing',
                    )
                attempt_volume = float(base_volume) * (0.35**attempt)
                result = generate_cad_aware_mesh(
                    geometry_fn=geometry_fn,
                    preprocess_record=preprocess_record,
                    max_element_volume=attempt_volume,
                    config=seed_config,
                    attempt_index=attempt,
                    surface_mesh_path=str(surface_mesh_path),
                )
                retry_history.append({'attempt': attempt, 'mesh_element_volume': float(attempt_volume), 'status': result['status']})
                if result['status'] != 'success':
                    last_status = result['status']
                    continue
                diagnostics = self._compute_initial_mesh_diagnostics(
                    mesh=result['mesh'],
                    surface_mesh=result.get('surface_mesh'),
                    surface_metrics=result.get('surface_metrics', {}),
                    elapsed_seconds=float(time.perf_counter() - started),
                    target_budget=target_budget,
                    condition_record=condition_record,
                    preprocess_record=preprocess_record,
                )
                self._annotate_initial_mesh_diagnostics(diagnostics, seed_config=seed_config, source='cad_aware_seed')
                last_diagnostics = diagnostics
                last_status = diagnostics['coarse_seed_verdict']
                if diagnostics['topology_preserved'] and not diagnostics['initial_is_too_dense']:
                    return result['mesh'], result.get('surface_metrics', {}), result.get('volume_metrics', {}), diagnostics, retry_history
                if diagnostics['initial_is_too_dense']:
                    break

        if last_diagnostics is None or not last_diagnostics.get('initial_is_too_dense') or not self.reject_if_initial_mesh_too_dense:
            fallback_mesh = load_expert_mesh(preprocess_record.coarse_mesh_path)
            fallback_mesh.geom_fn = geometry_fn
            fallback_diagnostics = self._compute_initial_mesh_diagnostics(
                mesh=fallback_mesh,
                surface_mesh=None,
                surface_metrics={'hole_sampling': {'records': [], 'measured_holes': int((preprocess_record.geometry_features or {}).get('statistics', {}).get('num_hole_features', 0))}},
                elapsed_seconds=float(time.perf_counter() - started),
                target_budget=target_budget,
                condition_record=condition_record,
                preprocess_record=preprocess_record,
            )
            self._annotate_initial_mesh_diagnostics(fallback_diagnostics, seed_config=seed_config, source='preprocess_coarse_fallback')
            fallback_diagnostics['used_preprocess_coarse_fallback'] = True
            retry_history.append({'attempt': 'preprocess_fallback', 'mesh_element_volume': None, 'status': 'fallback_preprocess_coarse'})
            if not fallback_diagnostics.get('initial_is_too_dense') or not self.reject_if_initial_mesh_too_dense:
                return fallback_mesh, {'status': 'fallback_preprocess_coarse', 'hole_sampling': {'records': [], 'min_segments': 0}}, {'status': 'success'}, fallback_diagnostics, retry_history
            last_diagnostics = fallback_diagnostics
        if last_diagnostics is not None and last_diagnostics.get('initial_is_too_dense') and self.reject_if_initial_mesh_too_dense:
            raise BudgetControlError(
                'initial mesh consumed too much of the target budget before teacher refinement started',
                category='reject_bad_initial_mesh',
                stage='surface_meshing',
                partial_output_available=surface_mesh_path.exists(),
                details=last_diagnostics,
            )
        raise BudgetControlError(
            f'Geometry-aware meshing failed after minimal coarse retries ({last_status})',
            category='invalid_geometry',
            stage='volume_meshing',
            partial_output_available=surface_mesh_path.exists(),
            details=last_diagnostics or {},
        )

    def _stage_probe_points(self, preprocess_record: GeometryPreprocessRecord) -> np.ndarray:
        coarse_mesh = load_expert_mesh(preprocess_record.coarse_mesh_path)
        points = np.asarray(coarse_mesh.p.T, dtype=float)
        if len(points) <= self.separability_probe_count:
            return points
        indices = np.linspace(0, len(points) - 1, num=self.separability_probe_count, dtype=int)
        return points[indices]

    def _build_stage_fields(
        self,
        *,
        current_mesh,
        indicator: np.ndarray,
        preprocess_record: GeometryPreprocessRecord,
        budget: int,
        prescreen_record: PrescreenRecord | None,
    ) -> dict[str, Any]:
        points = np.asarray(current_mesh.p.T, dtype=float)
        simplices = np.asarray(current_mesh.t.T, dtype=np.int64)
        dim = int(current_mesh.dim())
        simplex_sizes = volume_to_edge_length(get_simplex_volumes_from_indices(points, simplices), dim=dim)
        current_vertex_size = _average_to_vertices(current_mesh.nvertices, simplices, simplex_sizes)
        vertex_indicator = _average_to_vertices(current_mesh.nvertices, simplices, indicator)
        importance = _normalize_importance(vertex_indicator)
        current_elements = max(int(current_mesh.t.shape[1]), 1)
        global_scale = np.power(current_elements / max(float(budget), 1.0), 1.0 / max(dim, 1))
        base_budget_size = float(np.quantile(current_vertex_size, 0.65)) * global_scale
        h_base_budget = np.full_like(current_vertex_size, base_budget_size, dtype=float)
        contrast_boost = float((prescreen_record.metrics if prescreen_record else {}).get('contrast_boost', 1.0))
        h_pde_only = self._contrast_enhanced_sizes(
            base_budget_sizes=h_base_budget,
            vertex_indicator=vertex_indicator,
            contrast_boost=contrast_boost,
        )
        geometry_constraint_summary = combine_geometry_constraints(
            preprocess_record.geometry_features or {},
            base_size=float(np.quantile(h_base_budget, 0.65)),
            config=self.teacher_config,
            attempt_index=0,
        )
        geometry_sizes = evaluate_geometry_sizing(
            points=points,
            geometry_features=preprocess_record.geometry_features,
            constraint_summary=geometry_constraint_summary,
            base_size=float(np.quantile(h_base_budget, 0.65)),
            config=self.teacher_config,
        )
        geometry_local_mask = geometry_sizes <= h_base_budget * (1.0 - self.geometry_local_floor_gap)
        hotspot_mask = vertex_indicator >= _safe_quantile(vertex_indicator, self.hotspot_quantile)
        low_mask = vertex_indicator <= _safe_quantile(vertex_indicator, 0.5)
        h_after_geometry_fusion = np.asarray(h_pde_only, dtype=float).copy()
        h_after_geometry_fusion[geometry_local_mask] = np.minimum(h_after_geometry_fusion[geometry_local_mask], geometry_sizes[geometry_local_mask])
        hotspot_cap = h_base_budget * max(self.target_hotspot_size_ratio * 0.82, 0.10)
        h_after_geometry_fusion[hotspot_mask] = np.minimum(h_after_geometry_fusion[hotspot_mask], hotspot_cap[hotspot_mask])
        if self.enable_low_importance_inflation:
            h_after_geometry_fusion[low_mask] = np.maximum(
                h_after_geometry_fusion[low_mask],
                h_base_budget[low_mask] * (1.0 + 0.20 * max(self.low_importance_size_boost - 1.0, 0.0)),
            )
        protected_cap = h_after_geometry_fusion * (0.55 + 0.35 * importance)
        protected_cap[geometry_local_mask] = h_after_geometry_fusion[geometry_local_mask]
        protected_cap[hotspot_mask] = h_after_geometry_fusion[hotspot_mask]
        protected_cap = np.minimum(protected_cap, h_after_geometry_fusion)
        return {
            'points': points,
            's_pde_raw': vertex_indicator,
            'h_pde_only': h_pde_only,
            'h_after_geometry_fusion': h_after_geometry_fusion,
            'protected_cap': protected_cap,
            'importance': importance,
            'hotspot_mask': hotspot_mask,
            'low_mask': low_mask,
            'geometry_local_mask': geometry_local_mask,
            'geometry_constraint_summary': geometry_constraint_summary,
            'geometry_sizes': geometry_sizes,
            'base_budget_size': h_base_budget,
        }

    def _contrast_enhanced_sizes(self, *, base_budget_sizes: np.ndarray, vertex_indicator: np.ndarray, contrast_boost: float) -> np.ndarray:
        base_budget_sizes = np.asarray(base_budget_sizes, dtype=float)
        vertex_indicator = np.asarray(vertex_indicator, dtype=float)
        importance = _normalize_importance(vertex_indicator)
        hot_threshold = _safe_quantile(vertex_indicator, self.hotspot_quantile)
        medium_threshold = _safe_quantile(vertex_indicator, self.medium_quantile)
        top_threshold = _safe_quantile(vertex_indicator, 0.98)
        low_threshold = _safe_quantile(vertex_indicator, 0.45)
        hot_mask = vertex_indicator >= hot_threshold
        top_mask = vertex_indicator >= top_threshold
        medium_mask = (vertex_indicator >= medium_threshold) & ~hot_mask
        low_mask = vertex_indicator <= low_threshold
        scale = np.ones_like(base_budget_sizes, dtype=float)

        if self.contrast_mode in {'quantile_bucket', 'hybrid', 'topk'}:
            scale[medium_mask] = np.minimum(scale[medium_mask], max(0.42, 1.0 / (1.0 + self.contrast_gamma * contrast_boost * 0.9)))
            scale[hot_mask] = np.minimum(scale[hot_mask], max(0.22, 1.0 / (1.0 + self.contrast_gamma * contrast_boost * 1.8)))
            scale[top_mask] = np.minimum(scale[top_mask], max(0.12, 1.0 / (1.0 + self.contrast_gamma * contrast_boost * 2.8)))
        if self.contrast_mode in {'power_law', 'hybrid'}:
            normalized = np.maximum(vertex_indicator / max(hot_threshold, 1.0e-12), 1.0e-12)
            power_scale = np.power(1.0 + normalized, -(self.contrast_gamma * contrast_boost))
            power_scale = np.clip(power_scale, 0.08, self.low_importance_size_boost * 1.5)
            scale = np.minimum(scale, power_scale)
        if self.enable_low_importance_inflation:
            scale[low_mask] = np.maximum(
                scale[low_mask],
                self.low_importance_size_boost * (1.0 + 0.25 * (1.0 - importance[low_mask])),
            )
        if self.condition_difference_preservation_enable:
            tail_mask = importance <= _safe_quantile(importance, 0.35)
            scale[tail_mask] = np.maximum(
                scale[tail_mask],
                self.low_importance_size_boost * (1.0 + 0.20 * (1.0 - importance[tail_mask])),
            )
        candidate_sizes = base_budget_sizes * np.clip(scale, 0.06, self.low_importance_size_boost * 1.9)
        for _ in range(4):
            if not np.any(hot_mask) or not np.any(low_mask):
                break
            hotspot_ratio = np.median(candidate_sizes[hot_mask]) / max(np.median(candidate_sizes[low_mask]), 1.0e-12)
            if hotspot_ratio <= self.target_hotspot_size_ratio:
                break
            candidate_sizes[hot_mask] *= 0.78
            if self.enable_low_importance_inflation:
                candidate_sizes[low_mask] *= 1.10
        return candidate_sizes

    def _prefer_local_refine_for_complex_3d_step(self, preprocess_record: GeometryPreprocessRecord) -> bool:
        if not self.adaptive_refinement_local_refine_for_complex_3d_enable:
            return False
        if int(preprocess_record.dimension) < 3:
            return False
        validation = dict(preprocess_record.validation or {})
        coarse_elements = max(float(preprocess_record.coarse_mesh_num_elements or 0), 1.0)
        reference_elements = max(float(self.adaptive_refinement_local_refine_reference_elements), 1.0)
        boundary_patches = float(validation.get('num_boundary_patches', 0.0) or 0.0)
        sharp_edges = float(validation.get('num_sharp_edges', 0.0) or 0.0)
        hole_features = float(validation.get('num_hole_features', 0.0) or 0.0)
        complexity_score = 1.0
        complexity_score += 0.65
        complexity_score += max(np.sqrt(coarse_elements / reference_elements) - 1.0, 0.0)
        complexity_score += max(boundary_patches - 8.0, 0.0) * 0.035
        complexity_score += max(sharp_edges - 10.0, 0.0) * 0.015
        complexity_score += hole_features * 0.20
        return complexity_score >= self.adaptive_refinement_local_refine_complexity_threshold

    def _advance_mesh(
        self,
        *,
        current_mesh,
        indicator: np.ndarray,
        geometry_fn,
        preprocess_record: GeometryPreprocessRecord,
        target_budget: int,
        condition_record: ConditionRecord,
        prescreen_record: PrescreenRecord | None,
        runtime_tracker: RuntimeTracker | None,
    ):
        if runtime_tracker is not None:
            runtime_tracker.enter_stage('adaptive_refinement', {'target_budget': target_budget})
            runtime_tracker.check_soft_limits()
        stage_fields = self._build_stage_fields(
            current_mesh=current_mesh,
            indicator=indicator,
            preprocess_record=preprocess_record,
            budget=target_budget,
            prescreen_record=prescreen_record,
        )
        remesh_sizes = np.asarray(stage_fields['h_after_geometry_fusion'], dtype=float)
        prefer_local_refine = self._prefer_local_refine_for_complex_3d_step(preprocess_record)
        if self.enable_geometry_fidelity_constraints and not prefer_local_refine:
            dim = int(current_mesh.dim())
            representative_size = float(np.quantile(remesh_sizes, 0.65))
            remesh_result = generate_cad_aware_mesh(
                geometry_fn=geometry_fn,
                preprocess_record=preprocess_record,
                max_element_volume=float(edge_length_to_volume(np.asarray([max(representative_size, 1.0e-8)]), dim=dim)[0]),
                config=self.teacher_config,
                attempt_index=0,
                additional_background={'positions': current_mesh.p.T, 'sizes': remesh_sizes},
            )
            if remesh_result['status'] == 'success' and int(remesh_result['mesh'].t.shape[1]) > int(current_mesh.t.shape[1]):
                return remesh_result['mesh']
        marked = adaptive_theta(indicator, theta=self.refine_theta)
        if len(marked) == 0:
            marked = np.asarray([int(np.argmax(indicator))], dtype=np.int32)
        refined = current_mesh.refined(marked)
        refined.geom_fn = geometry_fn
        return refined

    def _transfer_indicator_to_mesh(self, *, old_mesh, old_indicator: np.ndarray, new_mesh) -> np.ndarray:
        old_indicator = np.asarray(old_indicator, dtype=float).reshape(-1)
        if old_indicator.size == 0:
            return np.ones(int(new_mesh.t.shape[1]), dtype=float)
        new_centroids = np.asarray(new_mesh.p[:, new_mesh.t].mean(axis=1), dtype=float)
        fallback_values = old_indicator
        try:
            old_ids = old_mesh.element_finder()(*new_centroids)
            transferred = np.full(int(new_mesh.t.shape[1]), float(np.mean(old_indicator)), dtype=float)
            valid = (old_ids >= 0) & (old_ids < old_indicator.shape[0])
            transferred[valid] = old_indicator[old_ids[valid]]
            if np.any(~valid):
                old_centroids = np.asarray(old_mesh.p[:, old_mesh.t].mean(axis=1).T, dtype=float)
                _, nearest = cKDTree(old_centroids).query(new_centroids.T[~valid])
                transferred[~valid] = fallback_values[nearest]
            return np.maximum(transferred, 1.0e-12)
        except Exception:
            old_centroids = np.asarray(old_mesh.p[:, old_mesh.t].mean(axis=1).T, dtype=float)
            _, nearest = cKDTree(old_centroids).query(new_centroids.T)
            return np.maximum(fallback_values[nearest], 1.0e-12)

    def _budget_growth_step_scores(
        self,
        *,
        mesh,
        indicator: np.ndarray,
        stage_fields: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        simplices = np.asarray(mesh.t.T, dtype=np.int64)
        _, element_sizes = _mesh_element_sizes(mesh)
        desired_vertex_sizes = np.asarray(stage_fields['h_after_geometry_fusion'], dtype=float)
        desired_element_sizes = _element_average_from_vertex_values(simplices, desired_vertex_sizes)
        importance = _element_average_from_vertex_values(simplices, np.asarray(stage_fields['importance'], dtype=float))
        hotspot_threshold = _safe_quantile(indicator, self.hotspot_quantile)
        very_hot_threshold = _safe_quantile(indicator, max(self.hotspot_quantile, 0.95))
        low_threshold = _safe_quantile(indicator, 0.45)
        mismatch = element_sizes / np.maximum(desired_element_sizes, 1.0e-12)
        normalized_indicator = _normalize_importance(indicator)
        hotspot_mask = indicator >= hotspot_threshold
        very_hot_mask = indicator >= very_hot_threshold
        low_mask = indicator <= low_threshold
        score = np.maximum(mismatch - 1.0, 0.0)
        score *= 1.0 + 1.75 * importance
        score += 0.65 * normalized_indicator
        score[hotspot_mask] += 1.0
        score[very_hot_mask] += 1.5
        if self.enable_low_importance_inflation:
            score[low_mask] *= 0.03
        return score, {
            'mean_desired_size_mismatch': float(np.mean(mismatch)),
            'p90_desired_size_mismatch': float(np.quantile(mismatch, 0.9)),
            'hotspot_candidate_fraction': float(np.mean(hotspot_mask)),
            'very_hot_candidate_fraction': float(np.mean(very_hot_mask)),
            'low_importance_protected_fraction': float(np.mean(low_mask)),
        }

    def _maybe_cad_cleanup_growth_mesh(
        self,
        *,
        mesh,
        geometry_fn,
        preprocess_record: GeometryPreprocessRecord,
        stage_fields: dict[str, Any],
        hard_max_budget: int,
    ) -> tuple[Any, dict[str, Any]]:
        dim = int(mesh.dim())
        candidate_sizes = np.asarray(stage_fields['h_after_geometry_fusion'], dtype=float)
        representative_size = float(np.quantile(candidate_sizes, 0.65))
        cleanup_record: dict[str, Any] = {'attempted': True, 'accepted': False}
        remesh_result = generate_cad_aware_mesh(
            geometry_fn=geometry_fn,
            preprocess_record=preprocess_record,
            max_element_volume=float(edge_length_to_volume(np.asarray([max(representative_size, 1.0e-8)]), dim=dim)[0]),
            config=self.teacher_config,
            attempt_index=0,
            additional_background={'positions': mesh.p.T, 'sizes': candidate_sizes},
        )
        cleanup_record['status'] = remesh_result.get('status')
        if remesh_result.get('status') == 'success':
            candidate_mesh = remesh_result['mesh']
            candidate_elements = int(candidate_mesh.t.shape[1])
            cleanup_record['candidate_num_elements'] = candidate_elements
            if candidate_elements <= int(hard_max_budget) and candidate_elements >= max(1, int(0.75 * mesh.t.shape[1])):
                cleanup_record['accepted'] = True
                return candidate_mesh, cleanup_record
        return mesh, cleanup_record

    def _run_budget_growth_loop(
        self,
        *,
        current_mesh,
        indicator: np.ndarray,
        geometry_fn,
        preprocess_record: GeometryPreprocessRecord,
        condition_record: ConditionRecord,
        budget_tiers: dict[str, Any],
        prescreen_record: PrescreenRecord | None,
        runtime_tracker: RuntimeTracker | None,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        desired_budget = int(budget_tiers['desired_budget'])
        minimum_viable_budget = int(budget_tiers['minimum_viable_budget'])
        hard_max_budget = int(budget_tiers['hard_max_budget'])
        history: list[dict[str, Any]] = []
        mesh = current_mesh
        mesh.geom_fn = geometry_fn
        working_indicator = np.asarray(indicator, dtype=float).reshape(-1)
        max_growth_steps, batch_refine_fraction = self._effective_budget_growth_controls(
            current_elements=int(current_mesh.t.shape[1]),
            desired_budget=desired_budget,
        )
        enforce_predictive_caps = self._should_enforce_predictive_growth_caps(budget_tiers)
        status = 'disabled'
        stop_reason = 'budget_growth_disabled'
        stalled = False
        timed_out = False
        hard_cap_exceeded = False
        previous_growth_fraction: float | None = None

        def _stage_fields_for_working_mesh():
            return self._build_stage_fields(
                current_mesh=mesh,
                indicator=working_indicator,
                preprocess_record=preprocess_record,
                budget=desired_budget,
                prescreen_record=prescreen_record,
            )

        if not self.budget_growth_enable or not self.budget_growth_use_local_refine:
            stage_fields = _stage_fields_for_working_mesh()
            final_status = self._classify_budget_status(
                actual_budget=int(mesh.t.shape[1]),
                budget_tiers=budget_tiers,
                growth_stalled=False,
            )
            return {
                'mesh': mesh,
                'indicator': working_indicator,
                'stage_fields': stage_fields,
                'diagnostics': {
                    'enabled': bool(self.budget_growth_enable),
                    'use_local_refine': bool(self.budget_growth_use_local_refine),
                    'status': final_status,
                    'stop_reason': stop_reason,
                    'history': history,
                    'elapsed_seconds': float(time.perf_counter() - start),
                    'budget_tiers': budget_tiers,
                    'max_growth_steps': int(max_growth_steps),
                    'batch_refine_fraction': float(batch_refine_fraction),
                    'predictive_caps_enforced': bool(enforce_predictive_caps),
                },
            }

        for step in range(max(max_growth_steps, 0)):
            step_start = time.perf_counter()
            if runtime_tracker is not None:
                runtime_tracker.enter_stage(
                    'budget_growth',
                    {
                        'step': int(step),
                        'num_elements': int(mesh.t.shape[1]),
                        'desired_budget': desired_budget,
                        'hard_max_budget': hard_max_budget,
                    },
                )
                runtime_tracker.check_soft_limits()
            if self.budget_growth_timeout_seconds and time.perf_counter() - start >= self.budget_growth_timeout_seconds:
                timed_out = True
                status = 'fail_budget_growth_timeout'
                stop_reason = 'budget_growth_timeout_seconds'
                break
            if runtime_tracker is not None and runtime_tracker.should_soft_stop():
                timed_out = True
                status = 'fail_budget_growth_timeout'
                stop_reason = 'sample_soft_timeout_near'
                break

            current_elements = int(mesh.t.shape[1])
            if current_elements >= hard_max_budget:
                hard_cap_exceeded = current_elements > hard_max_budget
                stop_reason = 'hard_max_budget_reached'
                break
            if current_elements >= int(np.ceil(desired_budget * max(0.80, 1.0 - 2.0 * self.budget_calibration_tolerance))):
                stop_reason = 'near_desired_budget'
                break
            if enforce_predictive_caps:
                if self.max_dofs is not None and self._estimated_solution_dofs(mesh, condition_record) >= int(self.dof_cap_guard_fraction * self.max_dofs):
                    stop_reason = 'dof_cap_near'
                    break
                if self.max_matrix_nnz is not None:
                    local_dofs = (int(mesh.dim()) + 1) * (int(mesh.dim()) if condition_record.pde_family == 'linear_elasticity' else 1)
                    estimated_nnz = int(current_elements * (local_dofs**2))
                    if estimated_nnz >= int(self.matrix_cap_guard_fraction * self.max_matrix_nnz):
                        stop_reason = 'matrix_cap_near'
                        break

            stage_fields = _stage_fields_for_working_mesh()
            before_allocation = _allocation_diagnostics_for_mesh(mesh=mesh, indicator=working_indicator, hotspot_quantile=self.hotspot_quantile)
            scores, score_diagnostics = self._budget_growth_step_scores(mesh=mesh, indicator=working_indicator, stage_fields=stage_fields)
            positive = np.flatnonzero(scores > 1.0e-10)
            if positive.size == 0:
                stalled = True
                stop_reason = 'no_positive_desired_size_mismatch'
                break

            dim = int(mesh.dim())
            expected_children = 4 if dim == 2 else 8
            remaining_budget = max(hard_max_budget - current_elements, 0)
            max_mark_by_cap = max(1, remaining_budget // max(expected_children - 1, 1))
            batch_count = max(1, int(np.ceil(current_elements * batch_refine_fraction)))
            batch_count = min(batch_count, int(positive.size), int(max_mark_by_cap))
            ranked = positive[np.argsort(scores[positive])[::-1]]
            accepted_mesh = None
            accepted_marked: np.ndarray | None = None
            for shrink in (1.0, 0.5, 0.25, 0.1):
                trial_count = max(1, int(np.floor(batch_count * shrink)))
                marked = np.asarray(ranked[:trial_count], dtype=np.int32)
                refined = mesh.refined(marked)
                refined.geom_fn = geometry_fn
                refined_elements = int(refined.t.shape[1])
                if refined_elements <= hard_max_budget and refined_elements > current_elements:
                    accepted_mesh = refined
                    accepted_marked = marked
                    break
            if accepted_mesh is None or accepted_marked is None:
                stalled = True
                stop_reason = 'local_refine_would_exceed_hard_max_budget'
                break

            old_mesh = mesh
            old_indicator = working_indicator
            mesh = accepted_mesh
            working_indicator = self._transfer_indicator_to_mesh(old_mesh=old_mesh, old_indicator=old_indicator, new_mesh=mesh)

            cleanup_record: dict[str, Any] | None = None
            if self.budget_growth_cad_cleanup_interval > 0 and (step + 1) % self.budget_growth_cad_cleanup_interval == 0:
                cleanup_stage_fields = _stage_fields_for_working_mesh()
                before_cleanup_mesh = mesh
                mesh, cleanup_record = self._maybe_cad_cleanup_growth_mesh(
                    mesh=mesh,
                    geometry_fn=geometry_fn,
                    preprocess_record=preprocess_record,
                    stage_fields=cleanup_stage_fields,
                    hard_max_budget=hard_max_budget,
                )
                if mesh is not before_cleanup_mesh:
                    working_indicator = self._transfer_indicator_to_mesh(
                        old_mesh=before_cleanup_mesh,
                        old_indicator=working_indicator,
                        new_mesh=mesh,
                    )

            new_elements = int(mesh.t.shape[1])
            growth_fraction = float((new_elements - current_elements) / max(float(current_elements), 1.0))
            after_allocation = _allocation_diagnostics_for_mesh(mesh=mesh, indicator=working_indicator, hotspot_quantile=self.hotspot_quantile)
            history.append(
                {
                    'step': int(step),
                    'num_elements_before': current_elements,
                    'num_elements_after': new_elements,
                    'num_nodes_after': int(mesh.nvertices),
                    'budget_ratio_before': float(current_elements / max(float(desired_budget), 1.0)),
                    'budget_ratio_after': float(new_elements / max(float(desired_budget), 1.0)),
                    'minimum_viable_ratio_after': float(new_elements / max(float(minimum_viable_budget), 1.0)),
                    'hard_max_ratio_after': float(new_elements / max(float(hard_max_budget), 1.0)),
                    'marked_elements': int(len(accepted_marked)),
                    'growth_fraction': growth_fraction,
                    'hotspot_element_fraction': after_allocation['hotspot_element_fraction'],
                    'hotspot_size_ratio': after_allocation['hotspot_size_ratio'],
                    'allocation_gain': after_allocation['allocation_gain'],
                    'allocation_gain_before': before_allocation['allocation_gain'],
                    'step_elapsed_seconds': float(time.perf_counter() - step_start),
                    'continue_growth': bool(new_elements < desired_budget and new_elements < hard_max_budget),
                    'score_diagnostics': score_diagnostics,
                    'cad_cleanup': cleanup_record or {'attempted': False},
                }
            )
            if self.budget_growth_stop_on_diminishing_return and previous_growth_fraction is not None:
                if growth_fraction < 0.03 and previous_growth_fraction < 0.05 and new_elements < minimum_viable_budget:
                    stalled = True
                    stop_reason = 'diminishing_return_before_minimum_viable'
                    break
            previous_growth_fraction = growth_fraction

        else:
            stop_reason = 'max_growth_steps'

        final_stage_fields = _stage_fields_for_working_mesh()
        actual_budget = int(mesh.t.shape[1])
        final_status = self._classify_budget_status(
            actual_budget=actual_budget,
            budget_tiers=budget_tiers,
            growth_stalled=stalled or stop_reason in {'max_growth_steps', 'no_positive_desired_size_mismatch'},
            timed_out=timed_out,
            hard_cap_exceeded=hard_cap_exceeded,
        )
        if status == 'fail_budget_growth_timeout' and actual_budget < minimum_viable_budget:
            final_status = status
        return {
            'mesh': mesh,
            'indicator': working_indicator,
            'stage_fields': final_stage_fields,
            'diagnostics': {
                'enabled': True,
                'use_local_refine': True,
                'status': final_status,
                'stop_reason': stop_reason,
                'history': history,
                'num_steps': len(history),
                'initial_num_elements': int(current_mesh.t.shape[1]),
                'final_num_elements': actual_budget,
                'minimum_viable_budget': minimum_viable_budget,
                'desired_budget': desired_budget,
                'hard_max_budget': hard_max_budget,
                'minimum_viable_reached': bool(actual_budget >= minimum_viable_budget),
                'desired_budget_reached': bool(actual_budget >= desired_budget),
                'elapsed_seconds': float(time.perf_counter() - start),
                'max_growth_steps': int(max_growth_steps),
                'batch_refine_fraction': float(batch_refine_fraction),
                'predictive_caps_enforced': bool(enforce_predictive_caps),
                'final_allocation': _allocation_diagnostics_for_mesh(
                    mesh=mesh,
                    indicator=working_indicator,
                    hotspot_quantile=self.hotspot_quantile,
                ),
                'budget_tiers': budget_tiers,
            },
        }

    def _budget_calibration_limiter(self, stage_fields: dict[str, Any], budget_ratio: float) -> str | None:
        if budget_ratio <= 1.0 + self.budget_calibration_tolerance:
            return None
        if float(np.mean(stage_fields['geometry_local_mask'])) > 0.05:
            return 'geometry_floor'
        if float(np.mean(stage_fields['hotspot_mask'])) > 0.05:
            return 'hotspot_floor'
        return 'mesher_behavior'

    def _calibrate_budget_mesh(
        self,
        *,
        current_mesh,
        geometry_fn,
        preprocess_record: GeometryPreprocessRecord,
        budget: int,
        budget_tiers: dict[str, Any],
        stage_fields: dict[str, Any],
        runtime_tracker: RuntimeTracker | None,
    ) -> dict[str, Any]:
        dim = int(current_mesh.dim())
        start = time.perf_counter()
        base_field = np.asarray(stage_fields['h_after_geometry_fusion'], dtype=float)
        protected_cap = np.asarray(stage_fields['protected_cap'], dtype=float)
        current_elements = max(int(current_mesh.t.shape[1]), 1)
        target_budget = max(int(budget_tiers['desired_budget']), 1)
        lambda_guess = float(np.clip(np.power(current_elements / float(target_budget), 1.0 / max(dim, 1)), 0.35, 4.5))
        history: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        timed_out = False
        accepted_shortfall = False
        calibration_strategy = 'iterative_search'

        def _timeout_reached() -> bool:
            if self.budget_calibration_timeout_seconds and (time.perf_counter() - start) > self.budget_calibration_timeout_seconds:
                return True
            if runtime_tracker is not None and runtime_tracker.should_soft_stop():
                return True
            return False

        def _evaluate_lambda(lambda_value: float, iteration: int) -> dict[str, Any]:
            nonlocal best, timed_out
            if runtime_tracker is not None:
                runtime_tracker.enter_stage('budget_calibration', {'budget': target_budget, 'iteration': iteration, 'lambda': float(lambda_value)})
                runtime_tracker.check_soft_limits()
            if _timeout_reached():
                timed_out = True
                raise StageTimeoutError(
                    'budget calibration exceeded its runtime cap',
                    category='timeout_budget_calibration',
                    stage='budget_calibration',
                )
            candidate_sizes = protected_cap + float(lambda_value) * np.maximum(base_field - protected_cap, 1.0e-9)
            representative_size = float(np.quantile(candidate_sizes, 0.65))
            remesh_result = generate_cad_aware_mesh(
                geometry_fn=geometry_fn,
                preprocess_record=preprocess_record,
                max_element_volume=float(edge_length_to_volume(np.asarray([max(representative_size, 1.0e-8)]), dim=dim)[0]),
                config=self.teacher_config,
                attempt_index=0,
                additional_background={'positions': current_mesh.p.T, 'sizes': candidate_sizes},
            )
            if remesh_result['status'] != 'success':
                record = {
                    'lambda': float(lambda_value),
                    'iteration': int(iteration),
                    'status': 'meshing_failed',
                    'actual_budget': None,
                    'budget_ratio': None,
                }
                history.append(record)
                return record
            candidate_mesh = remesh_result['mesh']
            actual_budget = int(candidate_mesh.t.shape[1])
            budget_ratio = float(actual_budget / max(float(target_budget), 1.0))
            record = {
                'lambda': float(lambda_value),
                'iteration': int(iteration),
                'status': 'success',
                'actual_budget': actual_budget,
                'budget_ratio': budget_ratio,
                'mesh': candidate_mesh,
                'field': candidate_sizes,
            }
            history.append(
                {
                    'lambda': float(lambda_value),
                    'iteration': int(iteration),
                    'status': 'success',
                    'actual_budget': actual_budget,
                    'budget_ratio': budget_ratio,
                }
            )
            if best is None or abs(budget_ratio - 1.0) < abs(float(best['budget_ratio']) - 1.0):
                best = record
            return record

        pre_calibration_status = self._classify_budget_status(
            actual_budget=int(current_mesh.t.shape[1]),
            budget_tiers=budget_tiers,
            growth_stalled=False,
        )
        if self.budget_growth_enable and _is_success_status(pre_calibration_status):
            accepted_shortfall = int(current_mesh.t.shape[1]) < target_budget
            calibration_strategy = 'cheap_growth_final_mesh'
            best = {
                'lambda': 1.0,
                'iteration': 0,
                'status': 'success',
                'actual_budget': int(current_mesh.t.shape[1]),
                'budget_ratio': float(current_mesh.t.shape[1] / max(float(target_budget), 1.0)),
                'mesh': current_mesh,
                'field': base_field,
            }
            history.append(
                {
                    'lambda': 1.0,
                    'iteration': 0,
                    'status': 'accepted_cheap_growth_final_mesh',
                    'actual_budget': int(current_mesh.t.shape[1]),
                    'budget_ratio': best['budget_ratio'],
                    'budget_status': pre_calibration_status,
                }
            )
        elif pre_calibration_status == 'fail_budget_hard_cap_exceeded':
            calibration_strategy = 'hard_cap_current_mesh'
            best = {
                'lambda': 1.0,
                'iteration': 0,
                'status': 'hard_cap_exceeded_before_calibration',
                'actual_budget': int(current_mesh.t.shape[1]),
                'budget_ratio': float(current_mesh.t.shape[1] / max(float(target_budget), 1.0)),
                'mesh': current_mesh,
                'field': base_field,
            }
            history.append(
                {
                    'lambda': 1.0,
                    'iteration': 0,
                    'status': 'hard_cap_exceeded_before_calibration',
                    'actual_budget': int(current_mesh.t.shape[1]),
                    'budget_ratio': best['budget_ratio'],
                    'budget_status': pre_calibration_status,
                }
            )
        elif not self.enable_budget_calibration:
            calibration_strategy = 'disabled'
            best = {
                'lambda': 1.0,
                'iteration': 0,
                'status': 'success',
                'actual_budget': int(current_mesh.t.shape[1]),
                'budget_ratio': float(current_mesh.t.shape[1] / max(float(target_budget), 1.0)),
                'mesh': current_mesh,
                'field': base_field,
            }
            history.append({'lambda': 1.0, 'iteration': 0, 'status': 'disabled', 'actual_budget': int(current_mesh.t.shape[1]), 'budget_ratio': best['budget_ratio']})
        elif (
            self.allow_budget_shortfall
            and self._prefer_local_refine_for_complex_3d_step(preprocess_record)
            and int(current_mesh.t.shape[1]) <= target_budget
        ):
            accepted_shortfall = int(current_mesh.t.shape[1]) < target_budget
            calibration_strategy = 'local_growth_current_mesh'
            best = {
                'lambda': 1.0,
                'iteration': 0,
                'status': 'success',
                'actual_budget': int(current_mesh.t.shape[1]),
                'budget_ratio': float(current_mesh.t.shape[1] / max(float(target_budget), 1.0)),
                'mesh': current_mesh,
                'field': base_field,
            }
            history.append(
                {
                    'lambda': 1.0,
                    'iteration': 0,
                    'status': 'accepted_local_growth_current_mesh',
                    'actual_budget': int(current_mesh.t.shape[1]),
                    'budget_ratio': best['budget_ratio'],
                }
            )
        else:
            lo_lambda = max(0.20, lambda_guess * 0.55)
            hi_lambda = min(6.0, lambda_guess * 1.80)
            try:
                lo_eval = _evaluate_lambda(lo_lambda, 0)
                hi_eval = _evaluate_lambda(hi_lambda, 1)
                if lo_eval.get('status') == 'success' and hi_eval.get('status') == 'success':
                    if lo_eval['actual_budget'] < target_budget and lo_lambda > 0.20:
                        lo_lambda = max(0.12, lo_lambda * 0.6)
                        lo_eval = _evaluate_lambda(lo_lambda, 2)
                    elif hi_eval['actual_budget'] > target_budget and hi_lambda < 6.0:
                        hi_lambda = min(6.0, hi_lambda * 1.45)
                        hi_eval = _evaluate_lambda(hi_lambda, 2)
                iterations_used = len(history)
                while iterations_used < max(self.budget_calibration_max_iters, len(history)):
                    if _timeout_reached():
                        timed_out = True
                        break
                    mid_lambda = float(np.sqrt(lo_lambda * hi_lambda))
                    mid_eval = _evaluate_lambda(mid_lambda, iterations_used)
                    iterations_used = len(history)
                    if mid_eval.get('status') != 'success':
                        break
                    if abs(float(mid_eval['budget_ratio']) - 1.0) <= self.budget_calibration_tolerance:
                        best = mid_eval
                        break
                    if mid_eval['actual_budget'] > target_budget:
                        lo_lambda = mid_lambda
                    else:
                        hi_lambda = mid_lambda
            except StageTimeoutError:
                timed_out = True

        if best is None:
            best = {
                'lambda': 1.0,
                'iteration': -1,
                'status': 'fallback',
                'actual_budget': int(current_mesh.t.shape[1]),
                'budget_ratio': float(current_mesh.t.shape[1] / max(float(target_budget), 1.0)),
                'mesh': current_mesh,
                'field': base_field,
            }
        converged = abs(float(best['budget_ratio']) - 1.0) <= self.budget_calibration_tolerance
        hard_cap_exceeded = int(best['actual_budget']) > int(budget_tiers['hard_max_budget'])
        status = self._classify_budget_status(
            actual_budget=int(best['actual_budget']),
            budget_tiers=budget_tiers,
            growth_stalled=not converged,
            timed_out=timed_out,
            hard_cap_exceeded=hard_cap_exceeded,
        )
        limiter = self._budget_calibration_limiter(stage_fields, float(best['budget_ratio']))
        stage_fields = dict(stage_fields)
        stage_fields['h_after_budget_calibration'] = np.asarray(best['field'], dtype=float)
        diagnostics = {
            'target_budget': int(budget),
            'requested_budget': int(budget_tiers['requested_budget']),
            'minimum_viable_budget': int(budget_tiers['minimum_viable_budget']),
            'desired_budget': int(budget_tiers['desired_budget']),
            'hard_max_budget': int(budget_tiers['hard_max_budget']),
            'actual_budget': int(best['actual_budget']),
            'budget_ratio': float(best['budget_ratio']),
            'desired_budget_ratio': float(int(best['actual_budget']) / max(float(budget_tiers['desired_budget']), 1.0)),
            'minimum_viable_budget_ratio': float(int(best['actual_budget']) / max(float(budget_tiers['minimum_viable_budget']), 1.0)),
            'hard_max_budget_ratio': float(int(best['actual_budget']) / max(float(budget_tiers['hard_max_budget']), 1.0)),
            'lambda': float(best['lambda']),
            'calibration_iters': len(history),
            'calibration_converged': bool(converged),
            'calibration_iterations': history,
            'budget_closure_limiter': limiter,
            'shortfall_accepted': bool(accepted_shortfall),
            'budget_calibration_strategy': calibration_strategy,
            'budget_status': status,
            'status': status,
        }
        return {
            'mesh': best['mesh'],
            'stage_fields': stage_fields,
            'budget_diagnostics': diagnostics,
            'status': status,
        }

    def _solve_and_estimate(
        self,
        current_mesh,
        preprocess_record: GeometryPreprocessRecord,
        condition_record: ConditionRecord,
        geometry_fn,
        runtime_tracker: RuntimeTracker | None,
    ):
        if runtime_tracker is not None:
            runtime_tracker.enter_stage('pde_solve')
            runtime_tracker.check_soft_limits()
        solve_result = solve_condition(
            current_mesh,
            preprocess_record,
            condition_record,
            solver_options={'max_dofs': self.max_dofs, 'max_matrix_nnz': self.max_matrix_nnz, 'solver_stage_name': 'pde_solve'},
        )
        reference_mesh = current_mesh
        if (
            condition_record.pde_family == 'scalar_elliptic'
            and self.desired_budget is not None
            and self.desired_budget >= self.high_budget_threshold
            and self.scalar_high_budget_reference_mode in {'cheap', 'cheap_reference', 'coarse_reference', 'reduced_order'}
        ):
            indicator = self._compute_cheap_scalar_indicator(current_mesh, solve_result, preprocess_record, condition_record)
            reference_result = dict(solve_result)
            reference_result['solver_metadata'] = dict(reference_result.get('solver_metadata', {}))
            reference_result['solver_metadata']['scalar_reference_mode'] = self.scalar_high_budget_reference_mode
            reference_result['solver_metadata']['scalar_reference_level'] = 0
            return solve_result, indicator, reference_mesh, reference_result
        reference_levels = self.reference_refinement_levels
        if condition_record.pde_family == 'linear_elasticity' and self.elasticity_smoke_mode in {'cheap', 'cheap_reference', 'coarse_reference', 'reduced_order'}:
            reference_levels = max(0, min(reference_levels, self.elasticity_smoke_reference_level))
        if condition_record.pde_family == 'linear_elasticity' and self.elasticity_smoke_mode in {'cheap', 'cheap_reference', 'coarse_reference', 'reduced_order'} and reference_levels <= 0:
            indicator = self._compute_cheap_elasticity_indicator(current_mesh, solve_result, preprocess_record, condition_record)
            reference_result = dict(solve_result)
            reference_result['solver_metadata'] = dict(reference_result.get('solver_metadata', {}))
            reference_result['solver_metadata']['elasticity_smoke_mode'] = self.elasticity_smoke_mode
            reference_result['solver_metadata']['elasticity_smoke_reference_level'] = 0
            return solve_result, indicator, reference_mesh, reference_result
        for _ in range(reference_levels):
            reference_mesh = reference_mesh.refined()
            reference_mesh.geom_fn = geometry_fn
        if runtime_tracker is not None:
            runtime_tracker.enter_stage('reference_solve')
            runtime_tracker.check_soft_limits()
        reference_result = solve_condition(
            reference_mesh,
            preprocess_record,
            condition_record,
            solver_options={'max_dofs': self.max_dofs, 'max_matrix_nnz': self.max_matrix_nnz, 'solver_stage_name': 'reference_solve'},
        )
        indicator = self._compute_indicator(current_mesh, solve_result, reference_mesh, reference_result)
        return solve_result, indicator, reference_mesh, reference_result

    def _compute_cheap_elasticity_indicator(
        self,
        mesh,
        solve_result: dict[str, Any],
        preprocess_record: GeometryPreprocessRecord,
        condition_record: ConditionRecord,
    ) -> np.ndarray:
        simplices = np.asarray(mesh.t.T, dtype=np.int64)
        points = np.asarray(mesh.p.T, dtype=float)
        nodal_values = np.asarray(solve_result['nodal_values'], dtype=float)
        displacement_norm = np.linalg.norm(nodal_values, axis=1)
        element_disp = displacement_norm[simplices]
        variation = element_disp.max(axis=1) - element_disp.min(axis=1)
        mean_disp = element_disp.mean(axis=1)
        indicator = variation + 0.20 * mean_disp

        condition_spec = condition_record.condition_spec
        centroids = points[simplices].mean(axis=1)
        for boundary_role in condition_spec.get('boundary_role_spec', []):
            if boundary_role.get('role') != 'traction':
                continue
            selector = boundary_role.get('selector')
            if not selector:
                continue
            selector_fn = _selector_callable(preprocess_record, selector)
            try:
                traction_mask = selector_fn(centroids.T)
            except Exception:
                traction_mask = np.zeros(len(centroids), dtype=bool)
            traction_norm = float(np.linalg.norm(np.asarray(boundary_role.get('vector', np.zeros(mesh.dim())), dtype=float)))
            indicator[traction_mask] += max(traction_norm, 1.0e-12)

        body_force = np.asarray(condition_spec.get('source_or_load_spec', {}).get('body_force', np.zeros(mesh.dim())), dtype=float)
        if np.linalg.norm(body_force) > 0.0:
            principal_axes = np.asarray(preprocess_record.principal_axes, dtype=float)
            centroid = np.asarray(preprocess_record.centroid, dtype=float)
            local = (centroids - centroid) @ principal_axes
            radius = np.linalg.norm(local, axis=1)
            indicator += 0.05 * np.linalg.norm(body_force) / np.maximum(1.0 + radius, 1.0e-12)
        return np.maximum(indicator, 1.0e-12)

    def _compute_cheap_scalar_indicator(
        self,
        mesh,
        solve_result: dict[str, Any],
        preprocess_record: GeometryPreprocessRecord,
        condition_record: ConditionRecord,
    ) -> np.ndarray:
        simplices = np.asarray(mesh.t.T, dtype=np.int64)
        points = np.asarray(mesh.p.T, dtype=float)
        nodal_values = np.asarray(solve_result['nodal_values'], dtype=float).reshape(-1)
        element_values = nodal_values[simplices]
        variation = element_values.max(axis=1) - element_values.min(axis=1)
        mean_values = np.abs(element_values.mean(axis=1))
        indicator = variation + 0.20 * mean_values

        condition_spec = condition_record.condition_spec
        centroids = points[simplices].mean(axis=1)
        for boundary_role in condition_spec.get('boundary_role_spec', []):
            role = str(boundary_role.get('role', '')).lower()
            selector = boundary_role.get('selector')
            if role not in {'dirichlet', 'flux', 'robin', 'neumann'} or not selector:
                continue
            selector_fn = _selector_callable(preprocess_record, selector)
            try:
                boundary_mask = np.asarray(selector_fn(centroids.T), dtype=bool)
            except Exception:
                boundary_mask = np.zeros(len(centroids), dtype=bool)
            role_strength = float(np.linalg.norm(np.atleast_1d(boundary_role.get('value', boundary_role.get('vector', 1.0)))))
            indicator[boundary_mask] += max(role_strength, 1.0e-12)

        source_spec = dict(condition_spec.get('source_or_load_spec', {}))
        source_strength = abs(float(source_spec.get('source_amplitude', 0.0) or 0.0))
        if source_strength > 0.0:
            principal_axes = np.asarray(preprocess_record.principal_axes, dtype=float)
            centroid = np.asarray(preprocess_record.centroid, dtype=float)
            local = (centroids - centroid) @ principal_axes
            radius = np.linalg.norm(local, axis=1)
            indicator += 0.05 * source_strength / np.maximum(1.0 + radius, 1.0e-12)
        return np.maximum(indicator, 1.0e-12)

    def _compute_indicator(self, coarse_mesh, coarse_result: dict[str, Any], reference_mesh, reference_result: dict[str, Any]) -> np.ndarray:
        coarse_values_on_reference = evaluate_solution_at_points(coarse_result['basis'], coarse_result['solution_vector'], reference_mesh.p)
        point_errors = np.linalg.norm(reference_result['nodal_values'] - coarse_values_on_reference, axis=1)
        coarse_element_ids = coarse_mesh.element_finder()(*reference_mesh.p)
        valid = coarse_element_ids >= 0
        indicator = np.zeros(coarse_mesh.t.shape[1], dtype=float)
        counts = np.zeros(coarse_mesh.t.shape[1], dtype=float)
        np.add.at(indicator, coarse_element_ids[valid], point_errors[valid] ** 2)
        np.add.at(counts, coarse_element_ids[valid], 1.0)
        indicator = np.sqrt(indicator / np.maximum(counts, 1.0))
        return np.maximum(indicator, 1.0e-12)

    def _amr_step_error_diagnostics(
        self,
        *,
        step: int,
        current_mesh,
        indicator: np.ndarray,
        solve_result: dict[str, Any],
        reference_mesh,
        reference_result: dict[str, Any],
        indicator_path: Path,
        mesh_path: Path,
    ) -> dict[str, Any]:
        indicator = np.asarray(indicator, dtype=float).reshape(-1)
        volumes, sizes = _mesh_element_sizes(current_mesh)
        if indicator.shape[0] != volumes.shape[0]:
            fallback = float(np.mean(indicator)) if indicator.size else 1.0
            indicator = np.full(volumes.shape[0], fallback, dtype=float)
        error_mass = indicator * volumes
        solve_metadata = dict(solve_result.get('solver_metadata', {}))
        reference_metadata = dict(reference_result.get('solver_metadata', {}))
        if reference_metadata.get('elasticity_smoke_reference_level') == 0:
            reference_mode = 'cheap_elasticity_indicator'
        elif reference_metadata.get('scalar_reference_level') == 0:
            reference_mode = 'cheap_scalar_indicator'
        else:
            reference_mode = 'reference_projection_error'
        return {
            'step': int(step),
            'mesh_path': str(mesh_path),
            'indicator_path': str(indicator_path),
            'indicator_semantics': reference_mode,
            'num_elements': int(current_mesh.t.shape[1]),
            'num_nodes': int(current_mesh.nvertices),
            'reference_num_elements': int(reference_mesh.t.shape[1]),
            'reference_num_nodes': int(reference_mesh.nvertices),
            'solution_num_dofs': int(solve_metadata.get('num_dofs', 0) or 0),
            'reference_num_dofs': int(reference_metadata.get('num_dofs', 0) or 0),
            'matrix_nnz': int(solve_metadata.get('actual_matrix_nnz', solve_metadata.get('estimated_matrix_nnz', 0)) or 0),
            'reference_matrix_nnz': int(
                reference_metadata.get('actual_matrix_nnz', reference_metadata.get('estimated_matrix_nnz', 0)) or 0
            ),
            'indicator_min': float(np.min(indicator)) if indicator.size else 0.0,
            'indicator_mean': float(np.mean(indicator)) if indicator.size else 0.0,
            'indicator_rms': float(np.sqrt(np.mean(indicator**2))) if indicator.size else 0.0,
            'indicator_q50': _safe_quantile(indicator, 0.5),
            'indicator_q90': _safe_quantile(indicator, 0.9),
            'indicator_q95': _safe_quantile(indicator, 0.95),
            'indicator_q99': _safe_quantile(indicator, 0.99),
            'indicator_max': float(np.max(indicator)) if indicator.size else 0.0,
            'volume_weighted_error_mass': float(np.sum(error_mass)),
            'volume_weighted_error_mean': float(np.sum(error_mass) / max(float(np.sum(volumes)), 1.0e-12)),
            'element_size_q10': _safe_quantile(sizes, 0.1),
            'element_size_q50': _safe_quantile(sizes, 0.5),
            'element_size_q90': _safe_quantile(sizes, 0.9),
        }

    def _save_step_artifacts(self, current_mesh, solve_result: dict[str, Any], indicator: np.ndarray, trajectory_dir: Path, fields_dir: Path, step: int):
        mesh_step_path = trajectory_dir / f'mesh_step_{step:03d}.vtk'
        solution_step_path = fields_dir / f'solution_step_{step:03d}.npz'
        indicator_step_path = fields_dir / f'indicator_step_{step:03d}.npy'
        save_as_vtk(current_mesh, mesh_step_path)
        np.savez(solution_step_path, points=current_mesh.p.T, connectivity=current_mesh.t.T, values=solve_result['nodal_values'])
        np.save(indicator_step_path, indicator)
        return mesh_step_path, solution_step_path, indicator_step_path

    def _save_stage_fields(
        self,
        *,
        budget_dir: Path,
        probe_points: np.ndarray,
        stage_fields: dict[str, Any],
    ) -> tuple[Path, Path]:
        stage_probe_points_path = budget_dir / 'stage_probe_points.npy'
        stage_field_path = budget_dir / 'stage_fields.npz'
        points = np.asarray(stage_fields['points'], dtype=float)
        payload = {
            'probe_points': np.asarray(probe_points, dtype=float),
            's_pde_raw': _sample_probe_field(probe_points, points, stage_fields['s_pde_raw']),
            'h_pde_only': _sample_probe_field(probe_points, points, stage_fields['h_pde_only']),
            'h_after_geometry_fusion': _sample_probe_field(probe_points, points, stage_fields['h_after_geometry_fusion']),
            'h_after_budget_calibration': _sample_probe_field(probe_points, points, stage_fields['h_after_budget_calibration']),
        }
        np.save(stage_probe_points_path, payload['probe_points'])
        np.savez(stage_field_path, **payload)
        return stage_field_path, stage_probe_points_path

    def _materialize_budget_result(
        self,
        *,
        geometry_record: GeometryRecord,
        condition_record: ConditionRecord,
        preprocess_record: GeometryPreprocessRecord,
        budgets_dir: Path,
        budget: int,
        current_mesh,
        solve_result: dict[str, Any],
        indicator: np.ndarray,
        reference_mesh,
        reference_result: dict[str, Any],
        initial_mesh_path: Path,
        trajectory_mesh_paths: list[str],
        geometry_fn,
        runtime_tracker: RuntimeTracker | None,
        probe_points: np.ndarray,
        prescreen_record: PrescreenRecord | None,
        initial_mesh_diagnostics: dict[str, Any],
        preprocess_record_for_constraints: GeometryPreprocessRecord,
    ) -> dict[str, Any]:
        budget_dir = budgets_dir / f'budget_{budget:06d}'
        budget_dir.mkdir(parents=True, exist_ok=True)
        budget_tiers = self._budget_tiers(budget)
        growth = self._run_budget_growth_loop(
            current_mesh=current_mesh,
            indicator=indicator,
            geometry_fn=geometry_fn,
            preprocess_record=preprocess_record_for_constraints,
            condition_record=condition_record,
            budget_tiers=budget_tiers,
            prescreen_record=prescreen_record,
            runtime_tracker=runtime_tracker,
        )
        growth_mesh = growth['mesh']
        growth_indicator = growth['indicator']
        stage_fields = growth['stage_fields']
        calibration = self._calibrate_budget_mesh(
            current_mesh=growth_mesh,
            geometry_fn=geometry_fn,
            preprocess_record=preprocess_record,
            budget=budget,
            budget_tiers=budget_tiers,
            stage_fields=stage_fields,
            runtime_tracker=runtime_tracker,
        )
        calibration['budget_diagnostics']['budget_growth'] = growth['diagnostics']
        target_mesh = calibration['mesh']
        if target_mesh is current_mesh:
            budget_solve_result = solve_result
            budget_indicator = indicator
            budget_reference_mesh = reference_mesh
            budget_reference_result = reference_result
        elif target_mesh is growth_mesh:
            budget_solve_result, budget_indicator, budget_reference_mesh, budget_reference_result = self._solve_and_estimate(
                target_mesh,
                preprocess_record,
                condition_record,
                geometry_fn,
                runtime_tracker=runtime_tracker,
            )
        else:
            budget_solve_result, budget_indicator, budget_reference_mesh, budget_reference_result = self._solve_and_estimate(
                target_mesh,
                preprocess_record,
                condition_record,
                geometry_fn,
                runtime_tracker=runtime_tracker,
            )
        target_mesh_path = budget_dir / 'target_mesh.vtk'
        reference_solution_path = budget_dir / 'reference_solution.npz'
        indicator_path = budget_dir / 'error_indicator.npy'
        save_as_vtk(target_mesh, target_mesh_path)
        np.savez(reference_solution_path, points=budget_reference_mesh.p.T, connectivity=budget_reference_mesh.t.T, values=budget_reference_result['nodal_values'])
        np.save(indicator_path, budget_indicator)
        stage_field_path, stage_probe_points_path = self._save_stage_fields(
            budget_dir=budget_dir,
            probe_points=probe_points,
            stage_fields=calibration['stage_fields'],
        )
        final_allocation_diagnostics = _allocation_diagnostics_for_mesh(
            mesh=target_mesh,
            indicator=budget_indicator,
            hotspot_quantile=self.hotspot_quantile,
        )
        final_allocation_diagnostics.update(
            {
                'budget_progress_minimum_viable': float(target_mesh.t.shape[1] / max(float(budget_tiers['minimum_viable_budget']), 1.0)),
                'budget_progress_desired': float(target_mesh.t.shape[1] / max(float(budget_tiers['desired_budget']), 1.0)),
                'budget_progress_hard_max': float(target_mesh.t.shape[1] / max(float(budget_tiers['hard_max_budget']), 1.0)),
            }
        )
        final_allocation_diagnostics_path = budget_dir / 'final_allocation_diagnostics.json'
        if self.save_final_allocation_diagnostics:
            dump_json(final_allocation_diagnostics_path, final_allocation_diagnostics)
        sample_id = stable_identifier(prefix=f'sample_{budget}', text=f'{geometry_record.geometry_id}::{condition_record.condition_id}::{budget}')
        achieved_num_elements = int(target_mesh.t.shape[1])
        budget_ratio = float(achieved_num_elements / max(float(budget_tiers['desired_budget']), 1.0))
        status = calibration['status']
        shortfall_accepted = bool(calibration['budget_diagnostics'].get('shortfall_accepted', False))
        if status == 'success' and abs(budget_ratio - 1.0) > self.budget_calibration_tolerance and not shortfall_accepted:
            status = self._classify_budget_status(actual_budget=achieved_num_elements, budget_tiers=budget_tiers, growth_stalled=True)
        return {
            'sample_id': sample_id,
            'budget': budget,
            'achieved_num_elements': achieved_num_elements,
            'budget_overrun_ratio': budget_ratio,
            'target_mesh_path': str(target_mesh_path),
            'reference_solution_path': str(reference_solution_path),
            'indicator_path': str(indicator_path),
            'stage_field_path': str(stage_field_path),
            'stage_probe_points_path': str(stage_probe_points_path),
            'normalization_metadata': self._normalization_metadata(budget_solve_result['nodal_values'], budget_indicator),
            'status': status,
            'initial_mesh_path': str(initial_mesh_path),
            'trajectory_mesh_paths': list(trajectory_mesh_paths),
            'budget_diagnostics': calibration['budget_diagnostics'],
            'initial_mesh_diagnostics': initial_mesh_diagnostics,
            'budget_growth_diagnostics': growth['diagnostics'],
            'final_allocation_diagnostics': final_allocation_diagnostics,
            'final_allocation_diagnostics_path': str(final_allocation_diagnostics_path) if self.save_final_allocation_diagnostics else None,
            'budget_tiers': budget_tiers,
        }

    def _normalization_metadata(self, nodal_values: np.ndarray, indicator: np.ndarray) -> dict[str, Any]:
        value_mean = nodal_values.mean(axis=0)
        value_std = nodal_values.std(axis=0)
        return {
            'solution_mean': np.asarray(value_mean).tolist(),
            'solution_std': np.maximum(np.asarray(value_std), 1.0e-12).tolist(),
            'indicator_mean': float(np.mean(indicator)),
            'indicator_std': float(max(np.std(indicator), 1.0e-12)),
        }

    def _sample_record_from_budget_result(
        self,
        budget_result: dict[str, Any],
        geometry_record: GeometryRecord,
        condition_record: ConditionRecord,
        preprocess_record: GeometryPreprocessRecord,
        initial_mesh_path: Path,
        trajectory_mesh_paths: list[str],
        layout: PipelineLayout,
        *,
        started_at: str,
        finished_at: str,
        elapsed_seconds: float,
        surface_quality_metrics: dict[str, Any] | None = None,
        volume_quality_metrics: dict[str, Any] | None = None,
        initial_mesh_diagnostics: dict[str, Any] | None = None,
    ) -> SampleRecord:
        geometry_artifact_paths = {
            'source_path': geometry_record.source_path,
            'geometry_record_path': str(layout.geometry_record_path(geometry_record.geometry_id)),
            'preprocess_record_path': str(layout.preprocess_record_path(geometry_record.geometry_id)),
            'coarse_mesh_path': preprocess_record.coarse_mesh_path,
        }
        if preprocess_record.geometry_feature_metadata_path:
            geometry_artifact_paths['geometry_feature_metadata_path'] = preprocess_record.geometry_feature_metadata_path
        failure_category = budget_result['status'] if not _is_success_status(budget_result['status']) else None
        return SampleRecord(
            sample_id=budget_result['sample_id'],
            geometry_id=geometry_record.geometry_id,
            condition_id=condition_record.condition_id,
            pde_family=condition_record.pde_family,
            budget=int(budget_result['budget']),
            condition_spec=condition_record.condition_spec,
            geometry_artifact_paths=geometry_artifact_paths,
            initial_mesh_path=str(initial_mesh_path),
            optional_intermediate_mesh_paths=list(trajectory_mesh_paths),
            final_target_mesh_path=budget_result['target_mesh_path'],
            optional_reference_solution_path=budget_result['reference_solution_path'],
            optional_error_indicator_path=budget_result['indicator_path'],
            optional_stage_field_path=budget_result.get('stage_field_path'),
            optional_stage_probe_points_path=budget_result.get('stage_probe_points_path'),
            normalization_metadata=budget_result['normalization_metadata'],
            source=geometry_record.source_name,
            status=budget_result['status'],
            teacher_metadata={
                'achieved_num_elements': budget_result['achieved_num_elements'],
                'budget_overrun_ratio': budget_result['budget_overrun_ratio'],
                'budget_status': budget_result['status'],
                'budget_tiers': budget_result.get('budget_tiers', {}),
                'budget_diagnostics': budget_result.get('budget_diagnostics', {}),
                'budget_growth_diagnostics': budget_result.get('budget_growth_diagnostics', {}),
                'final_allocation_diagnostics': budget_result.get('final_allocation_diagnostics', {}),
                'final_allocation_diagnostics_path': budget_result.get('final_allocation_diagnostics_path'),
                'adaptive_error_history': budget_result.get('adaptive_error_history', []),
                'adaptive_error_history_path': budget_result.get('adaptive_error_history_path'),
                'initial_mesh_diagnostics': budget_result.get('initial_mesh_diagnostics', initial_mesh_diagnostics or {}),
                'surface_quality_metrics': surface_quality_metrics or {},
                'volume_quality_metrics': volume_quality_metrics or {},
                'stage_field_path': budget_result.get('stage_field_path'),
                'stage_probe_points_path': budget_result.get('stage_probe_points_path'),
            },
            started_at=started_at,
            finished_at=finished_at,
            elapsed_seconds=float(elapsed_seconds),
            stage_where_stopped='completed' if _is_success_status(budget_result['status']) else budget_result['status'],
            failure_category=failure_category,
            partial_output_available=True,
            failure_reason=None if failure_category is None else budget_result['status'],
        )

    def _build_failed_sample_records(
        self,
        *,
        geometry_record: GeometryRecord,
        condition_record: ConditionRecord,
        preprocess_record: GeometryPreprocessRecord,
        layout: PipelineLayout,
        initial_mesh_path: Path,
        trajectory_mesh_paths: list[str],
        failure_reason: str,
        failure_category: str,
        stage_where_stopped: str,
        started_at: str,
        finished_at: str,
        elapsed_seconds: float,
        partial_output_available: bool,
    ) -> list[SampleRecord]:
        geometry_artifact_paths = {
            'source_path': geometry_record.source_path,
            'geometry_record_path': str(layout.geometry_record_path(geometry_record.geometry_id)),
            'preprocess_record_path': str(layout.preprocess_record_path(geometry_record.geometry_id)),
            'coarse_mesh_path': preprocess_record.coarse_mesh_path,
        }
        if preprocess_record.geometry_feature_metadata_path:
            geometry_artifact_paths['geometry_feature_metadata_path'] = preprocess_record.geometry_feature_metadata_path
        failed_samples = []
        for budget in condition_record.budget_or_tolerance_spec.get('budgets', []):
            sample_id = stable_identifier(prefix=f'sample_{budget}', text=f'{geometry_record.geometry_id}::{condition_record.condition_id}::{budget}')
            failed_samples.append(
                SampleRecord(
                    sample_id=sample_id,
                    geometry_id=geometry_record.geometry_id,
                    condition_id=condition_record.condition_id,
                    pde_family=condition_record.pde_family,
                    budget=int(budget),
                    condition_spec=condition_record.condition_spec,
                    geometry_artifact_paths=geometry_artifact_paths,
                    initial_mesh_path=str(initial_mesh_path),
                    optional_intermediate_mesh_paths=list(trajectory_mesh_paths),
                    final_target_mesh_path='',
                    optional_reference_solution_path=None,
                    optional_error_indicator_path=None,
                    optional_stage_field_path=None,
                    optional_stage_probe_points_path=None,
                    normalization_metadata={},
                    source=geometry_record.source_name,
                    status='failed',
                    started_at=started_at,
                    finished_at=finished_at,
                    elapsed_seconds=float(elapsed_seconds),
                    stage_where_stopped=stage_where_stopped,
                    failure_category=failure_category,
                    partial_output_available=partial_output_available,
                    failure_reason=failure_reason,
                )
            )
        return failed_samples


