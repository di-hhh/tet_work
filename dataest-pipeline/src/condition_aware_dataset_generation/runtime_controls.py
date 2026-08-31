# 生成时间：2026-04-09 22:05:00 +08:00（北京时间）
from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.condition_aware_dataset_generation.utils import dump_json, load_json


TIMEOUT_CATEGORY_BY_STAGE = {
    'geometry_preprocessing': 'timeout_preprocess',
    'prescreen_solve': 'timeout_prescreen',
    'surface_meshing': 'timeout_surface_mesh',
    'volume_meshing': 'timeout_volume_mesh',
    'budget_calibration': 'timeout_budget_calibration',
    'budget_growth': 'timeout_budget_growth',
    'pde_solve': 'timeout_solver',
    'reference_solve': 'timeout_solver',
    'adaptive_refinement': 'timeout_solver',
}


class PipelineAbort(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        category: str,
        stage: str,
        partial_output_available: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.stage = stage
        self.partial_output_available = partial_output_available
        self.details = details or {}


class StageTimeoutError(PipelineAbort):
    pass


class ComplexityLimitError(PipelineAbort):
    pass


class ContrastRejectError(PipelineAbort):
    pass


class BudgetControlError(PipelineAbort):
    pass


class InvalidConditionError(PipelineAbort):
    pass


@dataclass
class RuntimeTracker:
    status_path: Path
    task_kind: str
    sample_timeout_seconds: float | None = None
    stage_timeout_seconds: dict[str, float] = field(default_factory=dict)
    soft_stop_fraction: float = 0.85
    started_epoch: float = field(default_factory=time.time)
    started_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat(timespec='seconds'))
    current_stage: str = 'pending'
    stage_started_epoch: float = field(default_factory=time.time)
    stage_started_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat(timespec='seconds'))

    def start(self, extra: dict[str, Any] | None = None) -> None:
        self._write_status('running', extra=extra or {})

    def enter_stage(self, stage: str, extra: dict[str, Any] | None = None) -> None:
        self.current_stage = stage
        self.stage_started_epoch = time.time()
        self.stage_started_at = datetime.now().astimezone().isoformat(timespec='seconds')
        self._write_status('running', extra=extra or {})

    def elapsed_seconds(self) -> float:
        return float(time.time() - self.started_epoch)

    def stage_elapsed_seconds(self) -> float:
        return float(time.time() - self.stage_started_epoch)

    def should_soft_stop(self) -> bool:
        if self.sample_timeout_seconds is None:
            return False
        return self.elapsed_seconds() >= self.sample_timeout_seconds * self.soft_stop_fraction

    def check_soft_limits(self) -> None:
        if self.sample_timeout_seconds is not None and self.elapsed_seconds() > self.sample_timeout_seconds:
            raise StageTimeoutError(
                f'{self.task_kind} exceeded per-sample timeout ({self.sample_timeout_seconds:.1f}s)',
                category='timeout_sample',
                stage=self.current_stage,
            )
        stage_limit = stage_timeout_for(self.stage_timeout_seconds, self.current_stage)
        if stage_limit is not None and self.stage_elapsed_seconds() > stage_limit:
            raise StageTimeoutError(
                f'{self.current_stage} exceeded stage timeout ({stage_limit:.1f}s)',
                category=timeout_category_for_stage(self.current_stage),
                stage=self.current_stage,
            )

    def finish(self, status: str, extra: dict[str, Any] | None = None) -> None:
        self._write_status(status, extra=extra or {}, finished_at=datetime.now().astimezone().isoformat(timespec='seconds'))

    def fail(
        self,
        *,
        failure_reason: str,
        failure_category: str,
        stage_where_stopped: str | None = None,
        partial_output_available: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> None:
        payload = dict(extra or {})
        payload.update(
            {
                'failure_reason': failure_reason,
                'failure_category': failure_category,
                'stage_where_stopped': stage_where_stopped or self.current_stage,
                'partial_output_available': bool(partial_output_available),
            }
        )
        self._write_status('failed', extra=payload, finished_at=datetime.now().astimezone().isoformat(timespec='seconds'))

    def _write_status(self, status: str, *, extra: dict[str, Any], finished_at: str | None = None) -> None:
        payload = {
            'task_kind': self.task_kind,
            'status': status,
            'started_at': self.started_at,
            'started_epoch': self.started_epoch,
            'stage': self.current_stage,
            'stage_started_at': self.stage_started_at,
            'stage_started_epoch': self.stage_started_epoch,
            'elapsed_seconds': self.elapsed_seconds(),
            'finished_at': finished_at,
        }
        payload.update(extra)
        dump_json(self.status_path, payload)


def timeout_category_for_stage(stage: str | None) -> str:
    return TIMEOUT_CATEGORY_BY_STAGE.get(stage or '', 'timeout_sample')


def stage_timeout_for(stage_timeouts: dict[str, float] | None, stage: str | None) -> float | None:
    if not stage_timeouts:
        return None
    if stage and stage in stage_timeouts:
        return float(stage_timeouts[stage])
    if '*' in stage_timeouts:
        return float(stage_timeouts['*'])
    return None


def normalize_stage_timeouts(raw: Any) -> dict[str, float]:
    if raw is None:
        return {}
    if isinstance(raw, (int, float)):
        return {'*': float(raw)}
    return {str(key): float(value) for key, value in dict(raw).items()}


def read_status(path: Path) -> dict[str, Any]:
    payload = load_json(path, default={})
    return payload or {}


def run_worker_subprocess(
    *,
    worker_kind: str,
    input_path: Path,
    output_path: Path,
    status_path: Path,
    workdir: str | Path,
    sample_timeout_seconds: float | None,
    stage_timeout_seconds: dict[str, float] | None,
    poll_interval_seconds: float = 0.1,
) -> dict[str, Any]:
    parent_started_epoch = time.time()
    parent_started_at = datetime.now().astimezone().isoformat(timespec='seconds')
    command = [
        sys.executable,
        '-m',
        'src.condition_aware_dataset_generation.worker_entry',
        worker_kind,
        '--input',
        str(input_path),
        '--output',
        str(output_path),
    ]
    process = subprocess.Popen(
        command,
        cwd=str(workdir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    started_epoch = parent_started_epoch
    stage_timeout_seconds = normalize_stage_timeouts(stage_timeout_seconds)
    timeout_payload: dict[str, Any] | None = None

    while process.poll() is None:
        status = read_status(status_path)
        now_epoch = time.time()
        parent_observed_elapsed_seconds = float(now_epoch - parent_started_epoch)
        if sample_timeout_seconds is not None and now_epoch - started_epoch > sample_timeout_seconds:
            timeout_payload = {
                'timed_out': True,
                'failure_category': 'timeout_sample',
                'stage_where_stopped': status.get('stage'),
                'status': status,
                'parent_started_at': parent_started_at,
                'parent_kill_at': datetime.now().astimezone().isoformat(timespec='seconds'),
                'parent_observed_elapsed_seconds': parent_observed_elapsed_seconds,
                'worker_reported_elapsed_seconds': status.get('elapsed_seconds'),
            }
            break
        stage_limit = stage_timeout_for(stage_timeout_seconds, status.get('stage'))
        stage_started_epoch = float(status.get('stage_started_epoch', started_epoch))
        if stage_limit is not None and now_epoch - stage_started_epoch > stage_limit:
            timeout_payload = {
                'timed_out': True,
                'failure_category': timeout_category_for_stage(status.get('stage')),
                'stage_where_stopped': status.get('stage'),
                'status': status,
                'parent_started_at': parent_started_at,
                'parent_kill_at': datetime.now().astimezone().isoformat(timespec='seconds'),
                'parent_observed_elapsed_seconds': parent_observed_elapsed_seconds,
                'worker_reported_elapsed_seconds': status.get('elapsed_seconds'),
            }
            break
        time.sleep(poll_interval_seconds)

    if timeout_payload is not None:
        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)
        timeout_payload['returncode'] = process.returncode
        timeout_payload['parent_finish_at'] = datetime.now().astimezone().isoformat(timespec='seconds')
        timeout_payload['parent_observed_elapsed_seconds'] = float(time.time() - parent_started_epoch)
        return timeout_payload

    parent_finished_at = datetime.now().astimezone().isoformat(timespec='seconds')
    status = read_status(status_path)
    return {
        'timed_out': False,
        'returncode': process.returncode,
        'status': status,
        'output': load_json(output_path, default={}) or {},
        'parent_started_at': parent_started_at,
        'parent_finish_at': parent_finished_at,
        'parent_observed_elapsed_seconds': float(time.time() - parent_started_epoch),
        'worker_reported_elapsed_seconds': status.get('elapsed_seconds'),
    }
