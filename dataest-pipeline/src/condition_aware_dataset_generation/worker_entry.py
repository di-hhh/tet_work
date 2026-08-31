# 生成时间：2026-04-09 19:32:38 +08:00（北京时间）
from __future__ import annotations

import argparse
from pathlib import Path

from src.condition_aware_dataset_generation.geometry_preprocessing import GeometryPreprocessor
from src.condition_aware_dataset_generation.pipeline import _dict_to_condition_record, _dict_to_geometry_record, _dict_to_preprocess_record
from src.condition_aware_dataset_generation.prescreen import ConditionPrescreener
from src.condition_aware_dataset_generation.records import PrescreenRecord
from src.condition_aware_dataset_generation.runtime_controls import RuntimeTracker, normalize_stage_timeouts
from src.condition_aware_dataset_generation.serialization.layout import PipelineLayout
from src.condition_aware_dataset_generation.teacher_generation import TeacherGenerator
from src.condition_aware_dataset_generation.utils import dump_json, load_json


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='condition_aware_worker')
    parser.add_argument('task', choices=['preprocess', 'prescreen', 'teacher'])
    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--output', required=True, type=str)
    return parser


def _runtime_tracker(payload: dict, status_path: Path, task_kind: str) -> RuntimeTracker:
    return RuntimeTracker(
        status_path=status_path,
        task_kind=task_kind,
        sample_timeout_seconds=payload.get('sample_timeout_seconds'),
        stage_timeout_seconds=normalize_stage_timeouts(payload.get('stage_timeout_seconds')),
    )


def run_preprocess(payload: dict) -> dict:
    layout = PipelineLayout(payload['layout_root'])
    geometry_record = _dict_to_geometry_record(payload['geometry_record'])
    tracker = _runtime_tracker(payload, Path(payload['status_path']), 'preprocess')
    tracker.start({'geometry_id': geometry_record.geometry_id})
    tracker.enter_stage('geometry_preprocessing')
    preprocessor = GeometryPreprocessor(payload['preprocess_config'])
    preprocess_record, failure = preprocessor.preprocess(
        geometry_record,
        layout,
        overwrite=bool(payload.get('overwrite', False)),
        runtime_tracker=tracker,
    )
    if failure is None:
        tracker.finish('success', {'geometry_id': geometry_record.geometry_id})
    else:
        tracker.fail(
            failure_reason=failure.reason,
            failure_category=failure.category or 'invalid_geometry',
            stage_where_stopped=failure.stage_where_stopped or 'geometry_preprocessing',
            partial_output_available=False,
        )
    return {
        'preprocess_record': preprocess_record.to_dict() if preprocess_record is not None else None,
        'failure': failure.to_dict() if failure is not None else None,
    }


def run_prescreen(payload: dict) -> dict:
    layout = PipelineLayout(payload['layout_root'])
    geometry_record = _dict_to_geometry_record(payload['geometry_record'])
    preprocess_record = _dict_to_preprocess_record(payload['preprocess_record'])
    condition_record = _dict_to_condition_record(payload['condition_record'])
    tracker = _runtime_tracker(payload, Path(payload['status_path']), 'prescreen')
    prescreener = ConditionPrescreener(payload.get('prescreen_config', {}), payload.get('smoke_config', {}))
    prescreen_record, failure = prescreener.evaluate_condition(
        geometry_record=geometry_record,
        preprocess_record=preprocess_record,
        condition_record=condition_record,
        layout=layout,
        overwrite=bool(payload.get('overwrite', False)),
        runtime_tracker=tracker,
    )
    if failure is None:
        tracker.finish('success', {'condition_id': condition_record.condition_id, 'label': prescreen_record.label})
    return {
        'prescreen_record': prescreen_record.to_dict(),
        'failure': failure.to_dict() if failure is not None else None,
    }


def run_teacher(payload: dict) -> dict:
    layout = PipelineLayout(payload['layout_root'])
    geometry_record = _dict_to_geometry_record(payload['geometry_record'])
    preprocess_record = _dict_to_preprocess_record(payload['preprocess_record'])
    condition_record = _dict_to_condition_record(payload['condition_record'])
    prescreen_payload = payload.get('prescreen_record')
    prescreen_record = PrescreenRecord(**prescreen_payload) if prescreen_payload else None
    tracker = _runtime_tracker(payload, Path(payload['status_path']), 'teacher')
    teacher = TeacherGenerator(payload.get('teacher_config', {}), payload.get('smoke_config', {}))
    teacher_record, sample_records, failure = teacher.generate(
        geometry_record=geometry_record,
        preprocess_record=preprocess_record,
        condition_record=condition_record,
        layout=layout,
        overwrite=bool(payload.get('overwrite', False)),
        runtime_tracker=tracker,
        prescreen_record=prescreen_record,
    )
    if failure is None:
        tracker.finish('success', {'condition_id': condition_record.condition_id, 'status': teacher_record.status if teacher_record else 'success'})
    return {
        'teacher_record': teacher_record.to_dict() if teacher_record is not None else None,
        'sample_records': [record.to_dict() for record in sample_records],
        'failure': failure.to_dict() if failure is not None else None,
    }


def main() -> int:
    args = build_argument_parser().parse_args()
    payload = load_json(Path(args.input))
    if args.task == 'preprocess':
        result = run_preprocess(payload)
    elif args.task == 'prescreen':
        result = run_prescreen(payload)
    else:
        result = run_teacher(payload)
    dump_json(Path(args.output), result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
