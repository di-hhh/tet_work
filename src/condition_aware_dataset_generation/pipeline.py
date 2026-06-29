from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from omegaconf import OmegaConf

from src.condition_aware_dataset_generation.condition_sampling import ConditionSampler
from src.condition_aware_dataset_generation.geometry_preprocessing import GeometryPreprocessor
from src.condition_aware_dataset_generation.geometry_sources import build_geometry_source
from src.condition_aware_dataset_generation.prescreen import ConditionPrescreener
from src.condition_aware_dataset_generation.records import (
    ConditionRecord,
    FailureRecord,
    GeometryPreprocessRecord,
    GeometryRecord,
    PrescreenRecord,
    SampleRecord,
    TeacherRecord,
)
from src.condition_aware_dataset_generation.runtime_controls import run_worker_subprocess
from src.condition_aware_dataset_generation.serialization.layout import PipelineLayout
from src.condition_aware_dataset_generation.smoke_analysis import build_smoke_report as build_smoke_report_payload
from src.condition_aware_dataset_generation.teacher_generation import TeacherGenerator
from src.condition_aware_dataset_generation.utils import configure_logging, dump_json, dump_jsonl, load_json, now_iso


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]


def _dict_to_geometry_record(payload: dict[str, Any]) -> GeometryRecord:
    return GeometryRecord(**payload)


def _dict_to_preprocess_record(payload: dict[str, Any]) -> GeometryPreprocessRecord:
    return GeometryPreprocessRecord(**payload)


def _dict_to_condition_record(payload: dict[str, Any]) -> ConditionRecord:
    return ConditionRecord(**payload)


def _is_success_status(status: str | None) -> bool:
    return str(status or '').startswith('success') or status == 'success'


class ConditionAwareDatasetPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.layout = PipelineLayout(config['output_root'])
        self.overwrite = bool(config.get('overwrite', False))
        self.workers = int(config.get('workers', 1))
        self.prescreen_config = dict(config.get('prescreen', {}))
        self.smoke_config = dict(config.get('smoke', {}))
        configure_logging(config.get('log_level', 'INFO'))
        self.preprocessor = GeometryPreprocessor(config.get('preprocessing', {}))
        self.condition_sampler = ConditionSampler(config.get('condition_sampling', {}))
        self.prescreener = ConditionPrescreener(self.prescreen_config, self.smoke_config)
        self.teacher_generator = TeacherGenerator(config.get('teacher', {}), self.smoke_config)
        OmegaConf.save(config=OmegaConf.create(config), f=str(self.layout.output_root / 'config_snapshot.yaml'))

    def ingest_geometries(self) -> dict[str, Any]:
        source = build_geometry_source(self.config['geometry_source'])
        records, failures = source.ingest()
        limit = self.config.get('limit_geometries')
        if limit is not None:
            records = records[: int(limit)]
        for record in records:
            cache_path = self.layout.geometry_record_path(record.geometry_id)
            if cache_path.exists() and not self.overwrite:
                continue
            dump_json(cache_path, record.to_dict())
        dump_jsonl(self.layout.failure_log_path('ingest'), [failure.to_dict() for failure in failures])
        LOGGER.info('Ingested %s geometries with %s failures', len(records), len(failures))
        return {'num_geometries': len(records), 'num_failures': len(failures)}

    def preprocess_geometries(self) -> dict[str, Any]:
        geometry_records = self._load_geometry_records()
        failures: list[FailureRecord] = []

        def _work(record: GeometryRecord):
            return self._run_preprocess_worker(record)

        for record, preprocess_record, failure in self._maybe_parallel_map(_work, geometry_records):
            if preprocess_record is not None:
                record.preprocessing_status = preprocess_record.status
                dump_json(self.layout.geometry_record_path(record.geometry_id), record.to_dict())
            if failure is not None:
                record.preprocessing_status = 'failed'
                dump_json(self.layout.geometry_record_path(record.geometry_id), record.to_dict())
                failures.append(failure)

        dump_jsonl(self.layout.failure_log_path('preprocess'), [failure.to_dict() for failure in failures])
        success_count = len([record for record in self._load_preprocess_records() if record.status == 'success'])
        LOGGER.info('Preprocessed %s geometries successfully with %s failures', success_count, len(failures))
        return {'num_success': success_count, 'num_failures': len(failures)}

    def sample_conditions(self) -> dict[str, Any]:
        seed = int(self.config.get('random_seed', 0))
        generated = 0
        for geometry_record, preprocess_record in self._load_successful_preprocess_pairs():
            condition_dir = self.layout.condition_dir(geometry_record.geometry_id)
            if any(condition_dir.glob('*.json')) and not self.overwrite:
                continue
            for condition_record in self.condition_sampler.sample_for_geometry(geometry_record, preprocess_record, seed):
                dump_json(self.layout.condition_record_path(geometry_record.geometry_id, condition_record.condition_id), condition_record.to_dict())
                generated += 1
        LOGGER.info('Generated %s condition specifications', generated)
        return {'num_conditions': generated}
    def prescreen_conditions(self) -> dict[str, Any]:
        condition_records = self._load_condition_records()
        geometry_lookup = {record.geometry_id: record for record in self._load_geometry_records()}
        preprocess_lookup = {record.geometry_id: record for record in self._load_preprocess_records()}
        failures: list[FailureRecord] = []

        if not bool(self.prescreen_config.get('enable_prescreen', False)):
            records = []
            for condition_record in condition_records:
                preprocess_record = preprocess_lookup.get(condition_record.geometry_id)
                if preprocess_record is None:
                    continue
                record = PrescreenRecord(
                    geometry_id=condition_record.geometry_id,
                    condition_id=condition_record.condition_id,
                    pde_family=condition_record.pde_family,
                    source_name=condition_record.source_name,
                    label='accept_for_smoke',
                    status='success',
                    coarse_mesh_path=preprocess_record.coarse_mesh_path,
                    selected_for_teacher=True,
                    selected_reason='prescreen disabled; accepted by default',
                    started_at=now_iso(),
                    finished_at=now_iso(),
                    elapsed_seconds=0.0,
                    stage_where_stopped='prescreen_disabled',
                )
                dump_json(self.layout.prescreen_record_path(record.geometry_id, record.condition_id), record.to_dict())
                records.append(record)
            return self._prescreen_summary(records, failures)

        work_items = []
        for condition_record in condition_records:
            preprocess_record = preprocess_lookup.get(condition_record.geometry_id)
            geometry_record = geometry_lookup.get(condition_record.geometry_id)
            if preprocess_record is None or geometry_record is None:
                continue
            work_items.append((geometry_record, preprocess_record, condition_record))

        staged_records: list[PrescreenRecord] = []

        def _work(item: tuple[GeometryRecord, GeometryPreprocessRecord, ConditionRecord]):
            return self._run_prescreen_worker(*item)

        for prescreen_record, failure in self._maybe_parallel_map(_work, work_items):
            staged_records.append(prescreen_record)
            if failure is not None:
                failures.append(failure)

        by_geometry: dict[str, list[PrescreenRecord]] = {}
        for record in staged_records:
            by_geometry.setdefault(record.geometry_id, []).append(record)

        finalized: list[PrescreenRecord] = []
        for _, records in sorted(by_geometry.items()):
            finalized.extend(self.prescreener.finalize_geometry_records(records, self.layout))

        finalized = self._apply_elasticity_gate(finalized)
        dump_jsonl(self.layout.failure_log_path('prescreen'), [failure.to_dict() for failure in failures])
        summary = self._prescreen_summary(finalized, failures)
        LOGGER.info('Prescreen summary: %s', summary)
        return summary

    def generate_teacher_targets(self) -> dict[str, Any]:
        failures: list[FailureRecord] = []
        generated_samples = 0
        generated_conditions = 0
        skipped_conditions = 0

        geometry_lookup = {record.geometry_id: record for record in self._load_geometry_records()}
        preprocess_lookup = {record.geometry_id: record for record in self._load_preprocess_records()}
        prescreen_lookup = {(record.geometry_id, record.condition_id): record for record in self._load_prescreen_records()}
        work_items = []

        for condition_record in self._load_condition_records():
            geometry_record = geometry_lookup.get(condition_record.geometry_id)
            preprocess_record = preprocess_lookup.get(condition_record.geometry_id)
            if geometry_record is None or preprocess_record is None:
                skipped_conditions += 1
                continue
            prescreen_record = prescreen_lookup.get((condition_record.geometry_id, condition_record.condition_id))
            if not self._should_run_teacher(condition_record, prescreen_record):
                skipped_conditions += 1
                continue
            work_items.append((geometry_record, preprocess_record, condition_record, prescreen_record))

        def _work(item: tuple[GeometryRecord, GeometryPreprocessRecord, ConditionRecord, PrescreenRecord | None]):
            return self._run_teacher_worker(*item)

        for teacher_record, sample_records, failure in self._maybe_parallel_map(_work, work_items):
            generated_conditions += 1
            generated_samples += len(sample_records)
            if failure is not None:
                failures.append(failure)

        dump_jsonl(self.layout.failure_log_path('teacher'), [failure.to_dict() for failure in failures])
        summary = {
            'num_conditions_run': generated_conditions,
            'num_conditions_skipped': skipped_conditions,
            'num_samples': generated_samples,
            'num_failures': len(failures),
        }
        LOGGER.info('Teacher generation summary: %s', summary)
        return summary

    def build_dataset_manifest(self) -> dict[str, Any]:
        geometry_records = self._load_geometry_records()
        preprocess_records = self._load_preprocess_records()
        condition_records = self._load_condition_records()
        prescreen_records = self._load_prescreen_records()
        teacher_records = self._load_teacher_records()
        sample_records = self._load_sample_records()
        split_manifest = self._assign_geometry_level_splits(sample_records)

        dump_jsonl(self.layout.manifest_path('geometry_records'), [record.to_dict() for record in geometry_records])
        dump_jsonl(self.layout.manifest_path('preprocess_records'), [record.to_dict() for record in preprocess_records])
        dump_jsonl(self.layout.manifest_path('condition_records'), [record.to_dict() for record in condition_records])
        dump_jsonl(self.layout.manifest_path('prescreen_records'), [record.to_dict() for record in prescreen_records])
        dump_jsonl(self.layout.manifest_path('teacher_records'), [record.to_dict() for record in teacher_records])
        dump_jsonl(self.layout.manifest_path('sample_manifest'), [record.to_dict() for record in sample_records])
        dump_json(self.layout.split_manifest_path, split_manifest)

        summary = {
            'num_geometries': len(geometry_records),
            'num_preprocessed': len(preprocess_records),
            'num_conditions': len(condition_records),
            'num_prescreen_records': len(prescreen_records),
            'num_prescreen_accepted': len([record for record in prescreen_records if record.selected_for_teacher]),
            'num_prescreen_rejected': len([record for record in prescreen_records if not record.selected_for_teacher]),
            'num_teacher_records': len(teacher_records),
            'num_samples': len(sample_records),
            'num_successful_samples': len([record for record in sample_records if _is_success_status(record.status)]),
            'num_failed_samples': len([record for record in sample_records if not _is_success_status(record.status)]),
        }
        LOGGER.info('Built manifest summary: %s', summary)
        return summary

    def build_smoke_report(self) -> dict[str, Any]:
        report_path = self.layout.report_path('smoke_report')
        report = build_smoke_report_payload(
            report_path=report_path,
            sample_records=self._load_sample_records(),
            prescreen_records=self._load_prescreen_records(),
            hotspot_quantile=float(self.smoke_config.get('hotspot_quantile', 0.9)),
            target_hotspot_size_ratio=float(self.smoke_config.get('target_hotspot_size_ratio', 0.75)),
            overlap_threshold=float(self.prescreen_config.get('prescreen_condition_overlap_threshold', 0.65)),
            scalar_smoke_enable=bool(self.smoke_config.get('scalar_smoke_enable', True)),
            elasticity_smoke_enable=bool(self.smoke_config.get('elasticity_smoke_enable', True)),
        )
        LOGGER.info('Built smoke report at %s', report_path)
        return report

    def run_full_pipeline(self) -> dict[str, Any]:
        return {
            'ingest': self.ingest_geometries(),
            'preprocess': self.preprocess_geometries(),
            'sample_conditions': self.sample_conditions(),
            'prescreen': self.prescreen_conditions(),
            'teacher': self.generate_teacher_targets(),
            'manifest': self.build_dataset_manifest(),
            'smoke_report': self.build_smoke_report(),
        }
    def _run_preprocess_worker(self, geometry_record: GeometryRecord) -> tuple[GeometryRecord, GeometryPreprocessRecord | None, FailureRecord | None]:
        input_path = self.layout.worker_input_path('preprocess', geometry_record.geometry_id)
        output_path = self.layout.worker_output_path('preprocess', geometry_record.geometry_id)
        status_path = self.layout.preprocess_runtime_status_path(geometry_record.geometry_id)
        dump_json(
            input_path,
            {
                'layout_root': str(self.layout.output_root),
                'geometry_record': geometry_record.to_dict(),
                'preprocess_config': self.config.get('preprocessing', {}),
                'overwrite': self.overwrite,
                'status_path': str(status_path),
                'sample_timeout_seconds': self._geometry_preprocess_timeout(),
                'stage_timeout_seconds': self._stage_timeouts(),
            },
        )
        result = run_worker_subprocess(
            worker_kind='preprocess',
            input_path=input_path,
            output_path=output_path,
            status_path=status_path,
            workdir=REPO_ROOT,
            sample_timeout_seconds=self._geometry_preprocess_timeout(),
            stage_timeout_seconds=self._stage_timeouts(),
        )
        if result.get('timed_out'):
            failure = self._timeout_failure(
                stage='preprocess',
                item_id=geometry_record.geometry_id,
                source_path=geometry_record.source_path,
                timeout_result=result,
                fallback_stage='geometry_preprocessing',
            )
            return geometry_record, None, failure
        output = result.get('output', {})
        preprocess_payload = output.get('preprocess_record')
        failure_payload = output.get('failure')
        preprocess_record = _dict_to_preprocess_record(preprocess_payload) if preprocess_payload else None
        failure = FailureRecord(**failure_payload) if failure_payload else None
        if result.get('returncode', 0) not in (0, None) and preprocess_record is None and failure is None:
            failure = self._subprocess_failure('preprocess', geometry_record.geometry_id, geometry_record.source_path, 'geometry_preprocessing')
        return geometry_record, preprocess_record, failure

    def _run_prescreen_worker(
        self,
        geometry_record: GeometryRecord,
        preprocess_record: GeometryPreprocessRecord,
        condition_record: ConditionRecord,
    ) -> tuple[PrescreenRecord, FailureRecord | None]:
        input_path = self.layout.worker_input_path('prescreen', geometry_record.geometry_id, condition_record.condition_id)
        output_path = self.layout.worker_output_path('prescreen', geometry_record.geometry_id, condition_record.condition_id)
        status_path = self.layout.prescreen_runtime_status_path(geometry_record.geometry_id, condition_record.condition_id)
        timeout_seconds = float(self.prescreen_config.get('prescreen_max_runtime_seconds', 30.0))
        dump_json(
            input_path,
            {
                'layout_root': str(self.layout.output_root),
                'geometry_record': geometry_record.to_dict(),
                'preprocess_record': preprocess_record.to_dict(),
                'condition_record': condition_record.to_dict(),
                'prescreen_config': self.prescreen_config,
                'smoke_config': self.smoke_config,
                'overwrite': self.overwrite,
                'status_path': str(status_path),
                'sample_timeout_seconds': timeout_seconds,
                'stage_timeout_seconds': {'prescreen_solve': timeout_seconds},
            },
        )
        result = run_worker_subprocess(
            worker_kind='prescreen',
            input_path=input_path,
            output_path=output_path,
            status_path=status_path,
            workdir=REPO_ROOT,
            sample_timeout_seconds=timeout_seconds,
            stage_timeout_seconds={'prescreen_solve': timeout_seconds},
        )
        if result.get('timed_out'):
            prescreen_record = self._timeout_prescreen_record(geometry_record, preprocess_record, condition_record, result)
            failure = self._timeout_failure(
                stage='prescreen',
                item_id=f'{geometry_record.geometry_id}:{condition_record.condition_id}',
                source_path=geometry_record.source_path,
                timeout_result=result,
                fallback_stage='prescreen_solve',
            )
            dump_json(self.layout.prescreen_record_path(geometry_record.geometry_id, condition_record.condition_id), prescreen_record.to_dict())
            return prescreen_record, failure
        output = result.get('output', {})
        prescreen_payload = output.get('prescreen_record')
        failure_payload = output.get('failure')
        if prescreen_payload is None:
            prescreen_record = self._timeout_prescreen_record(
                geometry_record,
                preprocess_record,
                condition_record,
                {'status': {}, 'failure_category': 'numerical_failure', 'stage_where_stopped': 'prescreen_solve'},
            )
        else:
            prescreen_record = PrescreenRecord(**prescreen_payload)
        failure = FailureRecord(**failure_payload) if failure_payload else None
        if result.get('returncode', 0) not in (0, None) and failure is None and prescreen_payload is None:
            failure = self._subprocess_failure(
                'prescreen',
                f'{geometry_record.geometry_id}:{condition_record.condition_id}',
                geometry_record.source_path,
                'prescreen_solve',
            )
        return prescreen_record, failure

    def _run_teacher_worker(
        self,
        geometry_record: GeometryRecord,
        preprocess_record: GeometryPreprocessRecord,
        condition_record: ConditionRecord,
        prescreen_record: PrescreenRecord | None,
    ) -> tuple[TeacherRecord, list[SampleRecord], FailureRecord | None]:
        input_path = self.layout.worker_input_path('teacher', geometry_record.geometry_id, condition_record.condition_id)
        output_path = self.layout.worker_output_path('teacher', geometry_record.geometry_id, condition_record.condition_id)
        status_path = self.layout.teacher_runtime_status_path(geometry_record.geometry_id, condition_record.condition_id)
        runtime_limits = self._teacher_runtime_limits(preprocess_record, condition_record)
        dump_json(
            input_path,
            {
                'layout_root': str(self.layout.output_root),
                'geometry_record': geometry_record.to_dict(),
                'preprocess_record': preprocess_record.to_dict(),
                'condition_record': condition_record.to_dict(),
                'prescreen_record': prescreen_record.to_dict() if prescreen_record is not None else None,
                'teacher_config': self.config.get('teacher', {}),
                'smoke_config': self.smoke_config,
                'overwrite': self.overwrite,
                'status_path': str(status_path),
                'sample_timeout_seconds': runtime_limits['sample_timeout_seconds'],
                'stage_timeout_seconds': runtime_limits['stage_timeout_seconds'],
            },
        )
        result = run_worker_subprocess(
            worker_kind='teacher',
            input_path=input_path,
            output_path=output_path,
            status_path=status_path,
            workdir=REPO_ROOT,
            sample_timeout_seconds=runtime_limits['sample_timeout_seconds'],
            stage_timeout_seconds=runtime_limits['stage_timeout_seconds'],
        )
        if result.get('timed_out'):
            return self._timeout_teacher_payload(geometry_record, preprocess_record, condition_record, result, runtime_limits)
        output = result.get('output', {})
        teacher_payload = output.get('teacher_record')
        sample_payloads = output.get('sample_records', [])
        failure_payload = output.get('failure')
        teacher_record = TeacherRecord(**teacher_payload) if teacher_payload else self._synthesized_teacher_failure(
            geometry_record,
            preprocess_record,
            condition_record,
            failure_category='numerical_failure',
            stage_where_stopped='teacher_runtime',
            failure_reason='teacher worker exited without producing a teacher record',
            elapsed_seconds=0.0,
            started_at=now_iso(),
        )
        teacher_record.solver_metadata = dict(teacher_record.solver_metadata or {})
        teacher_record.solver_metadata['runtime_limits'] = runtime_limits
        dump_json(self.layout.teacher_record_path(geometry_record.geometry_id, condition_record.condition_id), teacher_record.to_dict())
        sample_records = [SampleRecord(**payload) for payload in sample_payloads]
        if not sample_records:
            sample_records = self._failed_samples_for_condition(
                geometry_record=geometry_record,
                preprocess_record=preprocess_record,
                condition_record=condition_record,
                initial_mesh_path=Path(teacher_record.initial_mesh_path),
                trajectory_mesh_paths=list(teacher_record.trajectory_mesh_paths),
                failure_category=teacher_record.failure_category or 'numerical_failure',
                failure_reason=teacher_record.failure_reason or 'teacher worker exited without producing samples',
                stage_where_stopped=teacher_record.stage_where_stopped or 'teacher_runtime',
                started_at=teacher_record.started_at or now_iso(),
                finished_at=teacher_record.finished_at or now_iso(),
                elapsed_seconds=float(teacher_record.elapsed_seconds),
                partial_output_available=teacher_record.partial_output_available,
            )
            for sample_record in sample_records:
                dump_json(self.layout.sample_path(sample_record.sample_id), sample_record.to_dict())
        failure = FailureRecord(**failure_payload) if failure_payload else None
        if failure is not None:
            failure.details = dict(failure.details or {})
            failure.details.setdefault('runtime_limits', runtime_limits)
        if result.get('returncode', 0) not in (0, None) and failure is None and not teacher_payload:
            failure = self._subprocess_failure('teacher', f'{geometry_record.geometry_id}:{condition_record.condition_id}', geometry_record.source_path, 'teacher_runtime')
            failure.details = {'runtime_limits': runtime_limits}
        return teacher_record, sample_records, failure
    def _apply_elasticity_gate(self, records: list[PrescreenRecord]) -> list[PrescreenRecord]:
        elasticity_enabled = bool(self.smoke_config.get('elasticity_smoke_enable', True))
        elasticity_cap = int(self.smoke_config.get('elasticity_smoke_max_samples', 1))
        skip_expensive = bool(self.smoke_config.get('elasticity_smoke_strict_cost_gate', self.smoke_config.get('skip_expensive_elasticity', True)))
        accepted_elasticity: list[PrescreenRecord] = []

        for record in sorted(
            records,
            key=lambda item: (
                item.pde_family != 'linear_elasticity',
                float(item.solve_cost_estimate.get('num_dofs', 1.0e12)),
                -float(item.metrics.get('hotspot_concentration', 0.0)),
            ),
        ):
            if record.pde_family != 'linear_elasticity':
                continue
            if not elasticity_enabled and record.label == 'accept_for_smoke':
                record.label = 'reject_too_expensive'
                record.selected_for_teacher = False
                record.selected_reason = 'elasticity smoke layer is disabled'
            elif record.label == 'accept_for_smoke' and skip_expensive and self._is_expensive_elasticity(record):
                record.label = 'reject_too_expensive'
                record.selected_for_teacher = False
                record.selected_reason = 'elasticity prescreen cost estimate exceeds the smoke budget gate'
            elif record.label == 'accept_for_smoke':
                if len(accepted_elasticity) >= elasticity_cap:
                    record.label = 'reject_too_expensive'
                    record.selected_for_teacher = False
                    record.selected_reason = f'elasticity smoke cap reached ({elasticity_cap})'
                else:
                    accepted_elasticity.append(record)
            dump_json(self.layout.prescreen_record_path(record.geometry_id, record.condition_id), record.to_dict())
        return records

    def _is_expensive_elasticity(self, record: PrescreenRecord) -> bool:
        max_dofs = float(self.smoke_config.get('smoke_max_dofs', 0) or 0)
        max_matrix_nnz = float(self.smoke_config.get('smoke_max_matrix_nnz', 0) or 0)
        cheap_mode = str(self.smoke_config.get('elasticity_smoke_mode', 'cheap_reference')).lower() in {
            'cheap',
            'cheap_reference',
            'coarse_reference',
            'reduced_order',
        }
        if cheap_mode:
            dofs = float(record.solve_cost_estimate.get('num_dofs', 0.0) or 0.0)
            nnz = float(record.solve_cost_estimate.get('estimated_matrix_nnz', 0.0) or 0.0)
            threshold = 0.95
        else:
            dofs = float(record.solve_cost_estimate.get('reference_dofs', record.solve_cost_estimate.get('num_dofs', 0.0)) or 0.0)
            nnz = float(
                record.solve_cost_estimate.get('reference_estimated_matrix_nnz', record.solve_cost_estimate.get('estimated_matrix_nnz', 0.0))
                or 0.0
            )
            threshold = 0.7
        if max_dofs > 0 and dofs >= threshold * max_dofs:
            return True
        if max_matrix_nnz > 0 and nnz >= threshold * max_matrix_nnz:
            return True
        return False

    def _should_run_teacher(self, condition_record: ConditionRecord, prescreen_record: PrescreenRecord | None) -> bool:
        if prescreen_record is None:
            return not bool(self.prescreen_config.get('enable_prescreen', False))
        return prescreen_record.selected_for_teacher and prescreen_record.label == 'accept_for_smoke'

    def _timeout_prescreen_record(
        self,
        geometry_record: GeometryRecord,
        preprocess_record: GeometryPreprocessRecord,
        condition_record: ConditionRecord,
        timeout_result: dict[str, Any],
    ) -> PrescreenRecord:
        status = dict(timeout_result.get('status') or {})
        started_at = timeout_result.get('parent_started_at') or status.get('started_at') or now_iso()
        finished_at = timeout_result.get('parent_finish_at') or now_iso()
        elapsed_seconds = float(
            timeout_result.get(
                'parent_observed_elapsed_seconds',
                status.get('elapsed_seconds', self.prescreen_config.get('prescreen_max_runtime_seconds', 0.0)),
            )
        )
        return PrescreenRecord(
            geometry_id=geometry_record.geometry_id,
            condition_id=condition_record.condition_id,
            pde_family=condition_record.pde_family,
            source_name=geometry_record.source_name,
            label='reject_too_expensive' if timeout_result.get('failure_category') == 'matrix_too_large' else 'reject_invalid',
            status='failed',
            coarse_mesh_path=preprocess_record.coarse_mesh_path,
            selected_for_teacher=False,
            selected_reason='prescreen timed out before the condition could be accepted',
            started_at=started_at,
            finished_at=finished_at,
            elapsed_seconds=elapsed_seconds,
            stage_where_stopped=timeout_result.get('stage_where_stopped') or 'prescreen_solve',
            failure_reason='prescreen worker exceeded its internal runtime limit',
            failure_category=timeout_result.get('failure_category', 'timeout_prescreen'),
            partial_output_available=False,
        )

    def _timeout_teacher_payload(
        self,
        geometry_record: GeometryRecord,
        preprocess_record: GeometryPreprocessRecord,
        condition_record: ConditionRecord,
        timeout_result: dict[str, Any],
        runtime_limits: dict[str, Any],
    ) -> tuple[TeacherRecord, list[SampleRecord], FailureRecord]:
        status = dict(timeout_result.get('status') or {})
        runtime_observation = self._runtime_observation(timeout_result)
        started_at = runtime_observation.get('parent_started_at') or status.get('started_at') or now_iso()
        elapsed_seconds = float(runtime_observation.get('parent_observed_elapsed_seconds', self._teacher_timeout() or 0.0))
        stage_where_stopped = timeout_result.get('stage_where_stopped') or status.get('stage') or 'teacher_runtime'
        failure_category = timeout_result.get('failure_category', 'timeout_sample')
        teacher_record = self._synthesized_teacher_failure(
            geometry_record,
            preprocess_record,
            condition_record,
            failure_category=failure_category,
            stage_where_stopped=stage_where_stopped,
            failure_reason='teacher worker exceeded its internal runtime limit',
            elapsed_seconds=elapsed_seconds,
            started_at=started_at,
            runtime_observation=runtime_observation,
        )
        teacher_record.solver_metadata = dict(teacher_record.solver_metadata or {})
        teacher_record.solver_metadata['runtime_limits'] = runtime_limits
        teacher_record.solver_metadata['runtime_observation'] = runtime_observation
        sample_records = self._failed_samples_for_condition(
            geometry_record=geometry_record,
            preprocess_record=preprocess_record,
            condition_record=condition_record,
            initial_mesh_path=Path(teacher_record.initial_mesh_path),
            trajectory_mesh_paths=list(teacher_record.trajectory_mesh_paths),
            failure_category=failure_category,
            failure_reason=teacher_record.failure_reason or 'teacher worker exceeded its internal runtime limit',
            stage_where_stopped=stage_where_stopped,
            started_at=started_at,
            finished_at=teacher_record.finished_at or now_iso(),
            elapsed_seconds=elapsed_seconds,
            partial_output_available=teacher_record.partial_output_available,
            runtime_observation=runtime_observation,
        )
        dump_json(self.layout.teacher_record_path(geometry_record.geometry_id, condition_record.condition_id), teacher_record.to_dict())
        for sample_record in sample_records:
            dump_json(self.layout.sample_path(sample_record.sample_id), sample_record.to_dict())
        failure = FailureRecord(
            stage='teacher',
            item_id=f'{geometry_record.geometry_id}:{condition_record.condition_id}',
            source_path=geometry_record.source_path,
            reason=teacher_record.failure_reason or 'teacher worker exceeded its internal runtime limit',
            details={'runtime_limits': runtime_limits, 'runtime_observation': runtime_observation},
            category=failure_category,
            started_at=started_at,
            finished_at=teacher_record.finished_at,
            elapsed_seconds=elapsed_seconds,
            stage_where_stopped=stage_where_stopped,
            partial_output_available=teacher_record.partial_output_available,
        )
        return teacher_record, sample_records, failure

    def _synthesized_teacher_failure(
        self,
        geometry_record: GeometryRecord,
        preprocess_record: GeometryPreprocessRecord,
        condition_record: ConditionRecord,
        *,
        failure_category: str,
        stage_where_stopped: str,
        failure_reason: str,
        elapsed_seconds: float,
        started_at: str,
        runtime_observation: dict[str, Any] | None = None,
    ) -> TeacherRecord:
        teacher_dir = self.layout.teacher_dir(geometry_record.geometry_id, condition_record.condition_id)
        initial_mesh_path = teacher_dir / 'initial_mesh.vtk'
        trajectory_dir = teacher_dir / 'trajectory'
        trajectory_mesh_paths = [str(path) for path in sorted(trajectory_dir.glob('*.vtk'))]
        partial_output_available = initial_mesh_path.exists() or bool(trajectory_mesh_paths)
        record = TeacherRecord(
            geometry_id=geometry_record.geometry_id,
            condition_id=condition_record.condition_id,
            pde_family=condition_record.pde_family,
            initial_mesh_path=str(initial_mesh_path),
            initial_surface_mesh_path=str(self.layout.teacher_surface_mesh_path(geometry_record.geometry_id, condition_record.condition_id)),
            trajectory_mesh_paths=trajectory_mesh_paths,
            solver_metadata=self.teacher_generator._solver_metadata(),
            wall_time_sec=float(elapsed_seconds),
            status='failed',
            started_at=started_at,
            finished_at=now_iso(),
            elapsed_seconds=float(elapsed_seconds),
            stage_where_stopped=stage_where_stopped,
            failure_category=failure_category,
            partial_output_available=partial_output_available,
            failure_reason=failure_reason,
        )
        if runtime_observation:
            record.solver_metadata = dict(record.solver_metadata or {})
            record.solver_metadata['runtime_observation'] = runtime_observation
        return record

    def _failed_samples_for_condition(
        self,
        *,
        geometry_record: GeometryRecord,
        preprocess_record: GeometryPreprocessRecord,
        condition_record: ConditionRecord,
        initial_mesh_path: Path,
        trajectory_mesh_paths: list[str],
        failure_category: str,
        failure_reason: str,
        stage_where_stopped: str,
        started_at: str,
        finished_at: str,
        elapsed_seconds: float,
        partial_output_available: bool,
        runtime_observation: dict[str, Any] | None = None,
    ) -> list[SampleRecord]:
        geometry_artifact_paths = {
            'source_path': geometry_record.source_path,
            'geometry_record_path': str(self.layout.geometry_record_path(geometry_record.geometry_id)),
            'preprocess_record_path': str(self.layout.preprocess_record_path(geometry_record.geometry_id)),
            'coarse_mesh_path': preprocess_record.coarse_mesh_path,
        }
        if preprocess_record.geometry_feature_metadata_path:
            geometry_artifact_paths['geometry_feature_metadata_path'] = preprocess_record.geometry_feature_metadata_path
        sample_records: list[SampleRecord] = []
        for budget in condition_record.budget_or_tolerance_spec.get('budgets', []):
            sample_id = f'{condition_record.condition_id}_{int(budget)}'
            sample_records.append(
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
                    normalization_metadata={},
                    source=geometry_record.source_name,
                    status='failed',
                    teacher_metadata={'runtime_observation': runtime_observation or {}},
                    started_at=started_at,
                    finished_at=finished_at,
                    elapsed_seconds=float(elapsed_seconds),
                    stage_where_stopped=stage_where_stopped,
                    failure_category=failure_category,
                    partial_output_available=partial_output_available,
                    failure_reason=failure_reason,
                )
            )
        return sample_records

    def _runtime_observation(self, result: dict[str, Any]) -> dict[str, Any]:
        status = dict(result.get('status') or {})
        return {
            'parent_started_at': result.get('parent_started_at'),
            'parent_kill_at': result.get('parent_kill_at'),
            'parent_finish_at': result.get('parent_finish_at'),
            'parent_observed_elapsed_seconds': result.get('parent_observed_elapsed_seconds'),
            'worker_reported_elapsed_seconds': result.get('worker_reported_elapsed_seconds', status.get('elapsed_seconds')),
            'worker_started_at': status.get('started_at'),
            'worker_finished_at': status.get('finished_at'),
            'worker_stage': status.get('stage'),
        }

    def _timeout_failure(
        self,
        *,
        stage: str,
        item_id: str,
        source_path: str | None,
        timeout_result: dict[str, Any],
        fallback_stage: str,
    ) -> FailureRecord:
        status = dict(timeout_result.get('status') or {})
        runtime_observation = self._runtime_observation(timeout_result)
        return FailureRecord(
            stage=stage,
            item_id=item_id,
            source_path=source_path,
            reason=f'{stage} worker exceeded its internal runtime limit',
            details={'runtime_observation': runtime_observation},
            category=timeout_result.get('failure_category', 'timeout_sample'),
            started_at=runtime_observation.get('parent_started_at') or status.get('started_at') or now_iso(),
            finished_at=runtime_observation.get('parent_finish_at') or now_iso(),
            elapsed_seconds=float(runtime_observation.get('parent_observed_elapsed_seconds') or status.get('elapsed_seconds', 0.0)),
            stage_where_stopped=timeout_result.get('stage_where_stopped') or status.get('stage') or fallback_stage,
            partial_output_available=False,
        )

    def _subprocess_failure(self, stage: str, item_id: str, source_path: str | None, stage_where_stopped: str) -> FailureRecord:
        return FailureRecord(
            stage=stage,
            item_id=item_id,
            source_path=source_path,
            reason=f'{stage} worker exited unexpectedly',
            category='numerical_failure',
            started_at=now_iso(),
            finished_at=now_iso(),
            elapsed_seconds=0.0,
            stage_where_stopped=stage_where_stopped,
            partial_output_available=False,
        )

    def _prescreen_summary(self, records: list[PrescreenRecord], failures: list[FailureRecord]) -> dict[str, Any]:
        label_counts: dict[str, int] = {}
        for record in records:
            label_counts[record.label] = label_counts.get(record.label, 0) + 1
        return {
            'num_conditions': len(records),
            'num_accept': len([record for record in records if record.selected_for_teacher]),
            'num_reject': len([record for record in records if not record.selected_for_teacher]),
            'label_counts': label_counts,
            'num_failures': len(failures),
        }

    def _geometry_preprocess_timeout(self) -> float | None:
        stage_timeouts = self._stage_timeouts()
        return float(stage_timeouts.get('geometry_preprocessing')) if 'geometry_preprocessing' in stage_timeouts else None

    def _teacher_timeout(self) -> float | None:
        value = self.smoke_config.get('smoke_max_runtime_seconds_per_sample')
        return None if value is None else float(value)

    def _stage_timeouts(self) -> dict[str, float]:
        raw = self.smoke_config.get('smoke_max_runtime_seconds_per_stage', {})
        if isinstance(raw, (int, float)):
            return {'*': float(raw)}
        return {str(key): float(value) for key, value in dict(raw).items()}

    def _teacher_stage_timeout_names(self) -> tuple[str, ...]:
        return ('surface_meshing', 'volume_meshing', 'budget_growth', 'budget_calibration', 'pde_solve', 'reference_solve', 'adaptive_refinement')

    def _adaptive_teacher_timeout_biases(self) -> dict[str, float]:
        biases = {
            'surface_meshing': 0.75,
            'volume_meshing': 0.9,
            'budget_calibration': 0.6,
            'pde_solve': 0.65,
            'reference_solve': 0.75,
            'adaptive_refinement': 1.35,
        }
        raw = self.smoke_config.get('adaptive_stage_timeout_stage_bias', {})
        if isinstance(raw, dict):
            for key, value in raw.items():
                biases[str(key)] = float(value)
        return biases

    def _teacher_timeout_complexity_context(
        self,
        preprocess_record: GeometryPreprocessRecord,
        condition_record: ConditionRecord,
    ) -> dict[str, Any]:
        adaptive_enabled = bool(self.smoke_config.get('adaptive_stage_timeouts_enable', True))
        validation = dict(preprocess_record.validation or {})
        coarse_elements = max(float(preprocess_record.coarse_mesh_num_elements or 0), 1.0)
        reference_elements = max(float(self.smoke_config.get('adaptive_stage_timeout_reference_elements', 2500.0)), 1.0)
        boundary_patches = float(validation.get('num_boundary_patches', 0.0) or 0.0)
        sharp_edges = float(validation.get('num_sharp_edges', 0.0) or 0.0)
        hole_features = float(validation.get('num_hole_features', 0.0) or 0.0)
        patch_baseline = float(self.smoke_config.get('adaptive_stage_timeout_boundary_patch_baseline', 8.0))
        sharp_baseline = float(self.smoke_config.get('adaptive_stage_timeout_sharp_edge_baseline', 10.0))
        dim_bonus = float(self.smoke_config.get('adaptive_stage_timeout_3d_bonus', 0.65)) if int(preprocess_record.dimension) >= 3 else 0.0
        elasticity_bonus = (
            float(self.smoke_config.get('adaptive_stage_timeout_elasticity_bonus', 0.15))
            if condition_record.pde_family == 'linear_elasticity'
            else 0.0
        )
        element_term = max(np.sqrt(coarse_elements / reference_elements) - 1.0, 0.0)
        patch_term = max(boundary_patches - patch_baseline, 0.0) * float(self.smoke_config.get('adaptive_stage_timeout_boundary_patch_weight', 0.035))
        sharp_term = max(sharp_edges - sharp_baseline, 0.0) * float(self.smoke_config.get('adaptive_stage_timeout_sharp_edge_weight', 0.015))
        hole_term = hole_features * float(self.smoke_config.get('adaptive_stage_timeout_hole_weight', 0.20))
        raw_multiplier = 1.0 + dim_bonus + elasticity_bonus + element_term + patch_term + sharp_term + hole_term
        min_multiplier = max(float(self.smoke_config.get('adaptive_stage_timeout_min_multiplier', 1.0)), 1.0)
        max_multiplier = max(float(self.smoke_config.get('adaptive_stage_timeout_max_multiplier', 4.0)), min_multiplier)
        complexity_multiplier = float(np.clip(raw_multiplier if adaptive_enabled else 1.0, min_multiplier, max_multiplier))
        return {
            'adaptive_stage_timeouts_enable': adaptive_enabled,
            'dimension': int(preprocess_record.dimension),
            'pde_family': condition_record.pde_family,
            'coarse_mesh_num_elements': int(preprocess_record.coarse_mesh_num_elements),
            'num_boundary_patches': int(boundary_patches),
            'num_sharp_edges': int(sharp_edges),
            'num_hole_features': int(hole_features),
            'reference_elements': reference_elements,
            'raw_multiplier': float(raw_multiplier),
            'complexity_multiplier': complexity_multiplier,
            'factor_breakdown': {
                'dimension_bonus': float(dim_bonus),
                'elasticity_bonus': float(elasticity_bonus),
                'element_term': float(element_term),
                'boundary_patch_term': float(patch_term),
                'sharp_edge_term': float(sharp_term),
                'hole_term': float(hole_term),
            },
        }

    def _teacher_runtime_limits(
        self,
        preprocess_record: GeometryPreprocessRecord,
        condition_record: ConditionRecord,
    ) -> dict[str, Any]:
        base_stage_timeouts = self._stage_timeouts()
        context = self._teacher_timeout_complexity_context(preprocess_record, condition_record)
        stage_timeouts = dict(base_stage_timeouts)
        stage_multipliers: dict[str, float] = {}
        if context['adaptive_stage_timeouts_enable']:
            complexity_multiplier = float(context['complexity_multiplier'])
            biases = self._adaptive_teacher_timeout_biases()
            for stage in self._teacher_stage_timeout_names():
                if stage not in stage_timeouts:
                    continue
                bias = max(float(biases.get(stage, 1.0)), 0.0)
                stage_multiplier = 1.0 + max(complexity_multiplier - 1.0, 0.0) * bias
                stage_multipliers[stage] = float(stage_multiplier)
                stage_timeouts[stage] = float(stage_timeouts[stage]) * stage_multiplier
            if '*' in stage_timeouts:
                star_multiplier = 1.0 + max(float(context['complexity_multiplier']) - 1.0, 0.0)
                stage_multipliers['*'] = float(star_multiplier)
                stage_timeouts['*'] = float(stage_timeouts['*']) * star_multiplier

        base_sample_timeout = self._teacher_timeout()
        sample_timeout = base_sample_timeout
        if context['adaptive_stage_timeouts_enable'] and stage_timeouts:
            peak_stage_timeout = max(float(stage_timeouts.get(stage, 0.0) or 0.0) for stage in self._teacher_stage_timeout_names())
            if peak_stage_timeout <= 0.0 and '*' in stage_timeouts:
                peak_stage_timeout = float(stage_timeouts['*'])
            sample_scale = max(float(self.smoke_config.get('adaptive_stage_timeout_sample_scale', 2.25)), 1.0)
            sample_slack = max(float(self.smoke_config.get('adaptive_stage_timeout_sample_slack_seconds', 20.0)), 0.0)
            default_cap = max(float(base_sample_timeout or 0.0), 300.0)
            max_sample_timeout = max(float(self.smoke_config.get('adaptive_stage_timeout_max_sample_seconds', default_cap)), float(base_sample_timeout or 0.0))
            suggested_sample_timeout = peak_stage_timeout * sample_scale + sample_slack if peak_stage_timeout > 0.0 else base_sample_timeout
            if suggested_sample_timeout is not None:
                if sample_timeout is None:
                    sample_timeout = suggested_sample_timeout
                else:
                    sample_timeout = max(float(sample_timeout), float(suggested_sample_timeout))
                sample_timeout = min(float(sample_timeout), max_sample_timeout)

        return {
            'adaptive_stage_timeouts_enable': context['adaptive_stage_timeouts_enable'],
            'sample_timeout_seconds': None if sample_timeout is None else float(sample_timeout),
            'base_sample_timeout_seconds': None if base_sample_timeout is None else float(base_sample_timeout),
            'stage_timeout_seconds': {str(key): float(value) for key, value in stage_timeouts.items()},
            'base_stage_timeout_seconds': {str(key): float(value) for key, value in base_stage_timeouts.items()},
            'stage_timeout_multiplier': stage_multipliers,
            'complexity_multiplier': float(context['complexity_multiplier']),
            'complexity_context': context,
        }

    def _maybe_parallel_map(self, fn: Callable, items: Iterable):
        items = list(items)
        if self.workers <= 1 or len(items) <= 1:
            return [fn(item) for item in items]
        results = []
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_to_index = {executor.submit(fn, item): index for index, item in enumerate(items)}
            ordered_results = [None] * len(items)
            for future in as_completed(future_to_index):
                ordered_results[future_to_index[future]] = future.result()
        for result in ordered_results:
            results.append(result)
        return results

    def _load_geometry_records(self) -> list[GeometryRecord]:
        return [GeometryRecord(**load_json(path)) for path in sorted(self.layout.geometries_dir.glob('*/geometry_record.json'))]

    def _load_preprocess_records(self) -> list[GeometryPreprocessRecord]:
        records = []
        for path in sorted(self.layout.geometries_dir.glob('*/preprocess_record.json')):
            payload = load_json(path)
            if payload is not None and payload.get('status') == 'success':
                records.append(GeometryPreprocessRecord(**payload))
        return records

    def _load_successful_preprocess_pairs(self) -> list[tuple[GeometryRecord, GeometryPreprocessRecord]]:
        geometry_lookup = {record.geometry_id: record for record in self._load_geometry_records()}
        return [(geometry_lookup[record.geometry_id], record) for record in self._load_preprocess_records() if record.geometry_id in geometry_lookup]

    def _load_condition_records(self) -> list[ConditionRecord]:
        return [ConditionRecord(**load_json(path)) for path in sorted(self.layout.conditions_dir.glob('**/*.json'))]

    def _load_prescreen_records(self) -> list[PrescreenRecord]:
        return [PrescreenRecord(**load_json(path)) for path in sorted(self.layout.prescreens_dir.glob('**/*.json'))]

    def _load_teacher_records(self) -> list[TeacherRecord]:
        return [TeacherRecord(**load_json(path)) for path in sorted(self.layout.teachers_dir.glob('**/teacher_record.json'))]

    def _load_sample_records(self) -> list[SampleRecord]:
        return [SampleRecord(**load_json(path)) for path in sorted(self.layout.samples_dir.glob('*.json'))]

    def _assign_geometry_level_splits(self, sample_records: list[SampleRecord]) -> dict[str, Any]:
        split_config = self.config.get('split', {})
        ratios = split_config.get('ratios', {'train': 0.7, 'val': 0.15, 'test': 0.15})
        seed = int(split_config.get('seed', self.config.get('random_seed', 0)))
        geometry_ids = sorted({record.geometry_id for record in sample_records})
        rng = np.random.RandomState(seed)
        shuffled = list(geometry_ids)
        rng.shuffle(shuffled)

        total = len(shuffled)
        train_cutoff = int(round(total * float(ratios.get('train', 0.7))))
        val_cutoff = train_cutoff + int(round(total * float(ratios.get('val', 0.15))))
        geometry_to_split = {}
        for index, geometry_id in enumerate(shuffled):
            if index < train_cutoff:
                geometry_to_split[geometry_id] = 'train'
            elif index < val_cutoff:
                geometry_to_split[geometry_id] = 'val'
            else:
                geometry_to_split[geometry_id] = 'test'

        updated_records = []
        for record in sample_records:
            record.split = geometry_to_split.get(record.geometry_id, 'train')
            dump_json(self.layout.sample_path(record.sample_id), record.to_dict())
            updated_records.append(record)

        return {
            'seed': seed,
            'ratios': ratios,
            'geometry_to_split': geometry_to_split,
            'evaluation_subsets': {
                'same_geometry_new_condition': [],
                'new_geometry_new_condition': sorted(record.sample_id for record in updated_records if record.split in {'val', 'test'}),
            },
        }

