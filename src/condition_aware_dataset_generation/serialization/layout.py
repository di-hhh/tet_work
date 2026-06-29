from __future__ import annotations

from pathlib import Path

from src.condition_aware_dataset_generation.utils import ensure_directory


class PipelineLayout:
    def __init__(self, output_root: str | Path):
        self.output_root = Path(output_root).resolve()
        self.geometries_dir = ensure_directory(self.output_root / 'geometries')
        self.conditions_dir = ensure_directory(self.output_root / 'conditions')
        self.prescreens_dir = ensure_directory(self.output_root / 'prescreens')
        self.teachers_dir = ensure_directory(self.output_root / 'teachers')
        self.samples_dir = ensure_directory(self.output_root / 'samples')
        self.manifests_dir = ensure_directory(self.output_root / 'manifests')
        self.failures_dir = ensure_directory(self.output_root / 'failures')
        self.logs_dir = ensure_directory(self.output_root / 'logs')
        self.reports_dir = ensure_directory(self.output_root / 'reports')
        self.runtime_dir = ensure_directory(self.output_root / 'runtime')

    def geometry_dir(self, geometry_id: str) -> Path:
        return ensure_directory(self.geometries_dir / geometry_id)

    def condition_dir(self, geometry_id: str) -> Path:
        return ensure_directory(self.conditions_dir / geometry_id)

    def prescreen_dir(self, geometry_id: str) -> Path:
        return ensure_directory(self.prescreens_dir / geometry_id)

    def teacher_dir(self, geometry_id: str, condition_id: str) -> Path:
        return ensure_directory(self.teachers_dir / geometry_id / condition_id)

    def runtime_status_dir(self, geometry_id: str) -> Path:
        return ensure_directory(self.runtime_dir / geometry_id)

    def sample_path(self, sample_id: str) -> Path:
        return self.samples_dir / f'{sample_id}.json'

    def geometry_record_path(self, geometry_id: str) -> Path:
        return self.geometry_dir(geometry_id) / 'geometry_record.json'

    def preprocess_record_path(self, geometry_id: str) -> Path:
        return self.geometry_dir(geometry_id) / 'preprocess_record.json'

    def geometry_mesh_path(self, geometry_id: str) -> Path:
        return self.geometry_dir(geometry_id) / 'coarse_mesh.vtk'

    def geometry_feature_metadata_path(self, geometry_id: str) -> Path:
        return self.geometry_dir(geometry_id) / 'geometry_features.json'

    def condition_record_path(self, geometry_id: str, condition_id: str) -> Path:
        return self.condition_dir(geometry_id) / f'{condition_id}.json'

    def prescreen_record_path(self, geometry_id: str, condition_id: str) -> Path:
        return self.prescreen_dir(geometry_id) / f'{condition_id}.json'

    def teacher_record_path(self, geometry_id: str, condition_id: str) -> Path:
        return self.teacher_dir(geometry_id, condition_id) / 'teacher_record.json'

    def teacher_surface_mesh_path(self, geometry_id: str, condition_id: str) -> Path:
        return self.teacher_dir(geometry_id, condition_id) / 'initial_surface_mesh.vtk'

    def preprocess_runtime_status_path(self, geometry_id: str) -> Path:
        return self.runtime_status_dir(geometry_id) / 'preprocess_status.json'

    def prescreen_runtime_status_path(self, geometry_id: str, condition_id: str) -> Path:
        return self.runtime_status_dir(geometry_id) / f'{condition_id}_prescreen_status.json'

    def teacher_runtime_status_path(self, geometry_id: str, condition_id: str) -> Path:
        return self.runtime_status_dir(geometry_id) / f'{condition_id}_teacher_status.json'

    def worker_input_path(self, task_kind: str, geometry_id: str, condition_id: str | None = None) -> Path:
        suffix = f'_{condition_id}' if condition_id else ''
        return self.runtime_status_dir(geometry_id) / f'{task_kind}{suffix}_input.json'

    def worker_output_path(self, task_kind: str, geometry_id: str, condition_id: str | None = None) -> Path:
        suffix = f'_{condition_id}' if condition_id else ''
        return self.runtime_status_dir(geometry_id) / f'{task_kind}{suffix}_output.json'

    def failure_log_path(self, stage: str) -> Path:
        return self.failures_dir / f'{stage}_failures.jsonl'

    def manifest_path(self, name: str) -> Path:
        return self.manifests_dir / f'{name}.jsonl'

    def report_path(self, name: str) -> Path:
        return self.reports_dir / f'{name}.json'

    @property
    def split_manifest_path(self) -> Path:
        return self.manifests_dir / 'split_manifest.json'
