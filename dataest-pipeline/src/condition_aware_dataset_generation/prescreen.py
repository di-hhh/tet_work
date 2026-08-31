# 生成时间：2026-04-09 19:32:38 +08:00（北京时间）
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from src.condition_aware_dataset_generation.records import ConditionRecord, FailureRecord, GeometryPreprocessRecord, GeometryRecord, PrescreenRecord
from src.condition_aware_dataset_generation.runtime_controls import (
    ComplexityLimitError,
    InvalidConditionError,
    PipelineAbort,
    RuntimeTracker,
)
from src.condition_aware_dataset_generation.teacher_generation.pde_solvers import evaluate_solution_at_points, solve_condition
from src.condition_aware_dataset_generation.utils import dump_json, load_json
from src.mesh_util.load_mesh import load_expert_mesh
from src.tasks.domains.geometry_util import get_simplex_volumes_from_indices


class ConditionPrescreener:
    def __init__(self, prescreen_config: dict | None = None, smoke_config: dict | None = None):
        self.prescreen_config = prescreen_config or {}
        self.smoke_config = smoke_config or {}
        self.enable_prescreen = bool(self.prescreen_config.get('enable_prescreen', False))
        self.max_elements = int(self.prescreen_config.get('prescreen_max_elements', 4000))
        self.max_runtime_seconds = float(self.prescreen_config.get('prescreen_max_runtime_seconds', 30.0))
        self.probe_count = int(self.prescreen_config.get('prescreen_probe_count', 512))
        self.hotspot_quantile = float(self.prescreen_config.get('prescreen_hotspot_quantile', 0.9))
        self.condition_overlap_threshold = float(self.prescreen_config.get('prescreen_condition_overlap_threshold', 0.65))
        self.min_hotspot_concentration = float(self.prescreen_config.get('prescreen_min_hotspot_concentration', 0.35))
        self.min_allocation_gain = float(self.prescreen_config.get('prescreen_min_allocation_gain', 1.02))
        self.max_dofs = int(self.smoke_config.get('smoke_max_dofs', self.prescreen_config.get('prescreen_max_dofs', 80000)))
        self.max_matrix_nnz = int(
            self.smoke_config.get('smoke_max_matrix_nnz', self.prescreen_config.get('prescreen_max_matrix_nnz', 10_000_000))
        )
        self.elasticity_smoke_enable = bool(self.smoke_config.get('elasticity_smoke_enable', True))

    def evaluate_condition(
        self,
        *,
        geometry_record: GeometryRecord,
        preprocess_record: GeometryPreprocessRecord,
        condition_record: ConditionRecord,
        layout,
        overwrite: bool = False,
        runtime_tracker: RuntimeTracker | None = None,
    ) -> tuple[PrescreenRecord, FailureRecord | None]:
        record_path = layout.prescreen_record_path(geometry_record.geometry_id, condition_record.condition_id)
        if record_path.exists() and not overwrite:
            return PrescreenRecord(**load_json(record_path)), None

        prescreen_dir = layout.prescreen_dir(geometry_record.geometry_id)
        probe_points_path = prescreen_dir / f'{condition_record.condition_id}_probe_points.npy'
        probe_field_path = prescreen_dir / f'{condition_record.condition_id}_probe_field.npy'
        started_at = runtime_tracker.started_at if runtime_tracker is not None else None
        wall_time_start = time.perf_counter()

        try:
            if runtime_tracker is not None:
                runtime_tracker.start({'geometry_id': geometry_record.geometry_id, 'condition_id': condition_record.condition_id})
                runtime_tracker.enter_stage('prescreen_solve')
            if condition_record.pde_family == 'linear_elasticity' and not self.elasticity_smoke_enable:
                record = self._build_record(
                    geometry_record=geometry_record,
                    condition_record=condition_record,
                    coarse_mesh_path=preprocess_record.coarse_mesh_path,
                    label='reject_too_expensive',
                    status='success',
                    metrics={},
                    solve_cost_estimate={'reason': 'elasticity_smoke_disabled'},
                    probe_points_path=None,
                    probe_field_path=None,
                    started_at=started_at,
                    elapsed_seconds=time.perf_counter() - wall_time_start,
                    finished_at=_now_iso(),
                    selected_reason='elasticity smoke layer is disabled for this run',
                )
                dump_json(record_path, record.to_dict())
                if runtime_tracker is not None:
                    runtime_tracker.finish('success', {'label': record.label})
                return record, None

            if preprocess_record.coarse_mesh_num_elements > self.max_elements:
                record = self._build_record(
                    geometry_record=geometry_record,
                    condition_record=condition_record,
                    coarse_mesh_path=preprocess_record.coarse_mesh_path,
                    label='reject_too_expensive',
                    status='success',
                    metrics={},
                    solve_cost_estimate={
                        'coarse_mesh_num_elements': preprocess_record.coarse_mesh_num_elements,
                        'prescreen_max_elements': self.max_elements,
                    },
                    probe_points_path=None,
                    probe_field_path=None,
                    started_at=started_at,
                    elapsed_seconds=time.perf_counter() - wall_time_start,
                    finished_at=_now_iso(),
                    selected_reason='preprocess coarse mesh already exceeds the prescreen element cap',
                )
                dump_json(record_path, record.to_dict())
                if runtime_tracker is not None:
                    runtime_tracker.finish('success', {'label': record.label})
                return record, None

            mesh = load_expert_mesh(preprocess_record.coarse_mesh_path)
            solve_result = solve_condition(
                mesh,
                preprocess_record,
                condition_record,
                solver_options={
                    'max_dofs': self.max_dofs,
                    'max_matrix_nnz': self.max_matrix_nnz,
                    'solver_stage_name': 'prescreen_solve',
                },
            )
            reference_mesh = mesh.refined()
            reference_result = solve_condition(
                reference_mesh,
                preprocess_record,
                condition_record,
                solver_options={
                    'max_dofs': self.max_dofs,
                    'max_matrix_nnz': self.max_matrix_nnz,
                    'solver_stage_name': 'prescreen_solve',
                },
            )
            indicator = _compute_indicator(mesh, solve_result, reference_mesh, reference_result)
            volumes = get_simplex_volumes_from_indices(mesh.p.T, mesh.t.T)
            error_mass = indicator * volumes
            hotspot_threshold = float(np.quantile(indicator, self.hotspot_quantile))
            hotspot_mask = indicator >= hotspot_threshold
            hotspot_concentration = float(error_mass[hotspot_mask].sum() / max(error_mass.sum(), 1.0e-12))
            hotspot_volume_fraction = float(volumes[hotspot_mask].sum() / max(volumes.sum(), 1.0e-12))
            hotspot_element_fraction = float(np.mean(hotspot_mask))
            allocation_gain = float(hotspot_element_fraction / max(hotspot_volume_fraction, 1.0e-12))
            probe_points, probe_field = _probe_field(mesh, indicator, probe_count=self.probe_count)
            np.save(probe_points_path, probe_points)
            np.save(probe_field_path, probe_field)

            solver_metadata = dict(solve_result.get('solver_metadata', {}))
            solver_metadata['reference_dofs'] = int(reference_result.get('solver_metadata', {}).get('num_dofs', 0))
            solver_metadata['reference_estimated_matrix_nnz'] = int(
                reference_result.get('solver_metadata', {}).get('estimated_matrix_nnz', 0)
            )
            contrast_boost = float(
                np.clip(1.0 + 0.8 * hotspot_concentration + 0.25 * max(allocation_gain - 1.0, 0.0), 1.0, 2.5)
            )
            metrics = {
                'hotspot_concentration': hotspot_concentration,
                'hotspot_element_fraction': hotspot_element_fraction,
                'hotspot_volume_fraction': hotspot_volume_fraction,
                'allocation_gain': allocation_gain,
                'indicator_mean': float(np.mean(indicator)),
                'indicator_q90': float(np.quantile(indicator, 0.9)),
                'hotspot_quantile': self.hotspot_quantile,
                'contrast_boost': contrast_boost,
            }
            label = 'accept_for_smoke'
            selected_reason = 'passed the prescreen concentration and cost filters'
            if hotspot_concentration < self.min_hotspot_concentration or allocation_gain <= self.min_allocation_gain:
                label = 'reject_low_contrast'
                selected_reason = 'hotspot concentration is too diffuse for strong teacher contrast'
            record = self._build_record(
                geometry_record=geometry_record,
                condition_record=condition_record,
                coarse_mesh_path=preprocess_record.coarse_mesh_path,
                label=label,
                status='success',
                metrics=metrics,
                solve_cost_estimate=solver_metadata,
                probe_points_path=str(probe_points_path),
                probe_field_path=str(probe_field_path),
                started_at=started_at,
                elapsed_seconds=time.perf_counter() - wall_time_start,
                finished_at=_now_iso(),
                selected_for_teacher=label == 'accept_for_smoke',
                selected_reason=selected_reason,
            )
            dump_json(record_path, record.to_dict())
            if runtime_tracker is not None:
                runtime_tracker.finish('success', {'label': record.label})
            return record, None
        except PipelineAbort as exc:
            record = self._build_record(
                geometry_record=geometry_record,
                condition_record=condition_record,
                coarse_mesh_path=preprocess_record.coarse_mesh_path,
                label='reject_too_expensive' if exc.category == 'matrix_too_large' else 'reject_invalid',
                status='failed',
                metrics={},
                solve_cost_estimate={},
                probe_points_path=str(probe_points_path) if probe_points_path.exists() else None,
                probe_field_path=str(probe_field_path) if probe_field_path.exists() else None,
                started_at=started_at,
                elapsed_seconds=time.perf_counter() - wall_time_start,
                finished_at=_now_iso(),
                stage_where_stopped=exc.stage,
                failure_reason=str(exc),
                failure_category=exc.category,
                partial_output_available=probe_field_path.exists(),
                selected_reason='prescreen aborted before the condition could be accepted',
            )
            dump_json(record_path, record.to_dict())
            failure = FailureRecord(
                stage='prescreen',
                item_id=f'{geometry_record.geometry_id}:{condition_record.condition_id}',
                source_path=geometry_record.source_path,
                reason=str(exc),
                category=exc.category,
                started_at=started_at,
                finished_at=_now_iso(),
                elapsed_seconds=time.perf_counter() - wall_time_start,
                stage_where_stopped=exc.stage,
                partial_output_available=probe_field_path.exists(),
            )
            if runtime_tracker is not None:
                runtime_tracker.fail(
                    failure_reason=str(exc),
                    failure_category=exc.category,
                    stage_where_stopped=exc.stage,
                    partial_output_available=probe_field_path.exists(),
                )
            return record, failure
        except Exception as exc:
            record = self._build_record(
                geometry_record=geometry_record,
                condition_record=condition_record,
                coarse_mesh_path=preprocess_record.coarse_mesh_path,
                label='reject_invalid',
                status='failed',
                metrics={},
                solve_cost_estimate={},
                probe_points_path=str(probe_points_path) if probe_points_path.exists() else None,
                probe_field_path=str(probe_field_path) if probe_field_path.exists() else None,
                started_at=started_at,
                elapsed_seconds=time.perf_counter() - wall_time_start,
                finished_at=_now_iso(),
                stage_where_stopped='prescreen_solve',
                failure_reason=str(exc),
                failure_category='reject_invalid',
                partial_output_available=probe_field_path.exists(),
                selected_reason='unexpected numerical failure during prescreen',
            )
            dump_json(record_path, record.to_dict())
            failure = FailureRecord(
                stage='prescreen',
                item_id=f'{geometry_record.geometry_id}:{condition_record.condition_id}',
                source_path=geometry_record.source_path,
                reason=str(exc),
                category='reject_invalid',
                started_at=started_at,
                finished_at=_now_iso(),
                elapsed_seconds=time.perf_counter() - wall_time_start,
                stage_where_stopped='prescreen_solve',
                partial_output_available=probe_field_path.exists(),
            )
            if runtime_tracker is not None:
                runtime_tracker.fail(
                    failure_reason=str(exc),
                    failure_category='reject_invalid',
                    stage_where_stopped='prescreen_solve',
                    partial_output_available=probe_field_path.exists(),
                )
            return record, failure

    def finalize_geometry_records(self, records: list[PrescreenRecord], layout) -> list[PrescreenRecord]:
        updated_records = [PrescreenRecord(**record.to_dict()) for record in records]
        pairwise = self._pairwise_metrics(updated_records)
        for record in updated_records:
            record.pairwise_metrics = pairwise.get(record.condition_id, {})

        accepted: list[PrescreenRecord] = []
        ordered = sorted(
            updated_records,
            key=lambda record: (
                record.label != 'accept_for_smoke',
                record.pde_family != 'scalar_elliptic',
                -float(record.metrics.get('hotspot_concentration', 0.0)),
                float(record.solve_cost_estimate.get('num_dofs', 1.0e12)),
            ),
        )
        for record in ordered:
            if record.label != 'accept_for_smoke':
                record.selected_for_teacher = False
                continue
            similar_to = None
            for accepted_record in accepted:
                metrics = pairwise.get(record.condition_id, {}).get(accepted_record.condition_id, {})
                if not metrics:
                    continue
                if (
                    metrics['hotspot_jaccard'] >= self.condition_overlap_threshold
                    and metrics['probe_spearman'] >= 0.80
                ):
                    similar_to = accepted_record.condition_id
                    break
            if similar_to is not None:
                record.label = 'reject_low_contrast'
                record.selected_for_teacher = False
                record.selected_reason = f'prescreen field is too similar to {similar_to}'
                continue
            record.selected_for_teacher = True
            if record.selected_reason is None:
                record.selected_reason = 'accepted after geometry-level separability filtering'
            accepted.append(record)

        for record in updated_records:
            dump_json(layout.prescreen_record_path(record.geometry_id, record.condition_id), record.to_dict())
        return updated_records

    def _pairwise_metrics(self, records: list[PrescreenRecord]) -> dict[str, dict[str, dict[str, float]]]:
        payload: dict[str, dict[str, dict[str, float]]] = {record.condition_id: {} for record in records}
        fields = {
            record.condition_id: np.asarray(np.load(record.probe_field_path), dtype=float)
            for record in records
            if record.probe_field_path
        }
        condition_ids = list(fields)
        for left_index, left_condition in enumerate(condition_ids):
            for right_condition in condition_ids[left_index + 1 :]:
                left_field = fields[left_condition]
                right_field = fields[right_condition]
                left_hot = left_field >= np.quantile(left_field, self.hotspot_quantile)
                right_hot = right_field >= np.quantile(right_field, self.hotspot_quantile)
                union = np.logical_or(left_hot, right_hot).sum()
                metric = {
                    'probe_pearson': float(np.corrcoef(left_field, right_field)[0, 1]),
                    'probe_spearman': float(_spearman(left_field, right_field)),
                    'probe_relative_l2_difference': float(np.linalg.norm(left_field - right_field) / max(np.linalg.norm(left_field), np.linalg.norm(right_field), 1.0e-12)),
                    'hotspot_jaccard': float(np.logical_and(left_hot, right_hot).sum() / max(union, 1)),
                }
                payload[left_condition][right_condition] = metric
                payload[right_condition][left_condition] = metric
        return payload

    def _build_record(
        self,
        *,
        geometry_record: GeometryRecord,
        condition_record: ConditionRecord,
        coarse_mesh_path: str,
        label: str,
        status: str,
        metrics: dict[str, Any],
        solve_cost_estimate: dict[str, Any],
        probe_points_path: str | None,
        probe_field_path: str | None,
        started_at: str | None,
        finished_at: str | None,
        elapsed_seconds: float,
        selected_for_teacher: bool = False,
        selected_reason: str | None = None,
        stage_where_stopped: str | None = None,
        failure_reason: str | None = None,
        failure_category: str | None = None,
        partial_output_available: bool = False,
    ) -> PrescreenRecord:
        return PrescreenRecord(
            geometry_id=geometry_record.geometry_id,
            condition_id=condition_record.condition_id,
            pde_family=condition_record.pde_family,
            source_name=geometry_record.source_name,
            label=label,
            status=status,
            coarse_mesh_path=str(coarse_mesh_path),
            probe_points_path=probe_points_path,
            probe_field_path=probe_field_path,
            metrics=metrics,
            solve_cost_estimate=solve_cost_estimate,
            selected_for_teacher=selected_for_teacher,
            selected_reason=selected_reason,
            started_at=started_at,
            finished_at=finished_at,
            elapsed_seconds=float(elapsed_seconds),
            stage_where_stopped=stage_where_stopped,
            failure_reason=failure_reason,
            failure_category=failure_category,
            partial_output_available=partial_output_available,
        )


def _compute_indicator(coarse_mesh, coarse_result: dict[str, Any], reference_mesh, reference_result: dict[str, Any]) -> np.ndarray:
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


def _probe_field(mesh, indicator: np.ndarray, probe_count: int) -> tuple[np.ndarray, np.ndarray]:
    vertex_indicator = np.zeros(mesh.nvertices, dtype=float)
    counts = np.zeros(mesh.nvertices, dtype=float)
    np.add.at(vertex_indicator, mesh.t.reshape(-1), np.repeat(indicator, mesh.t.shape[0]))
    np.add.at(counts, mesh.t.reshape(-1), 1.0)
    vertex_indicator = vertex_indicator / np.maximum(counts, 1.0)
    if mesh.p.shape[1] <= probe_count:
        return mesh.p.T, vertex_indicator
    indices = np.linspace(0, mesh.p.shape[1] - 1, num=probe_count, dtype=int)
    return mesh.p.T[indices], vertex_indicator[indices]


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = np.argsort(np.argsort(left))
    right_rank = np.argsort(np.argsort(right))
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _now_iso() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime())

