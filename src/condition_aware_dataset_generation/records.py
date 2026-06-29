from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class JsonDataclass:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GeometryRecord(JsonDataclass):
    geometry_id: str
    source_name: str
    source_path: str
    relative_source_path: str
    metadata: dict[str, Any] = field(default_factory=dict)
    preprocessing_status: str = 'pending'


@dataclass
class GeometryPreprocessRecord(JsonDataclass):
    geometry_id: str
    source_path: str
    dimension: int
    bounding_box: list[float]
    centroid: list[float]
    principal_axes: list[list[float]]
    oriented_bbox_min: list[float]
    oriented_bbox_max: list[float]
    boundary_patches: list[dict[str, Any]]
    validation: dict[str, Any]
    coarse_mesh_path: str
    coarse_mesh_num_vertices: int
    coarse_mesh_num_elements: int
    status: str
    geometry_feature_metadata_path: str | None = None
    geometry_features: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConditionRecord(JsonDataclass):
    condition_id: str
    geometry_id: str
    pde_family: str
    condition_index: int
    condition_spec: dict[str, Any]
    budget_or_tolerance_spec: dict[str, Any]
    source_name: str
    status: str = 'success'


@dataclass
class PrescreenRecord(JsonDataclass):
    geometry_id: str
    condition_id: str
    pde_family: str
    source_name: str
    label: str = 'pending'
    status: str = 'pending'
    coarse_mesh_path: str | None = None
    probe_points_path: str | None = None
    probe_field_path: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    pairwise_metrics: dict[str, Any] = field(default_factory=dict)
    solve_cost_estimate: dict[str, Any] = field(default_factory=dict)
    selected_for_teacher: bool = False
    selected_reason: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_seconds: float = 0.0
    stage_where_stopped: str | None = None
    failure_reason: str | None = None
    failure_category: str | None = None
    taxonomy_category: str | None = None
    partial_output_available: bool = False


@dataclass
class TeacherRecord(JsonDataclass):
    geometry_id: str
    condition_id: str
    pde_family: str
    initial_mesh_path: str
    initial_surface_mesh_path: str | None = None
    trajectory_mesh_paths: list[str] = field(default_factory=list)
    trajectory_solution_paths: list[str] = field(default_factory=list)
    trajectory_indicator_paths: list[str] = field(default_factory=list)
    budget_results: list[dict[str, Any]] = field(default_factory=list)
    solver_metadata: dict[str, Any] = field(default_factory=dict)
    wall_time_sec: float = 0.0
    status: str = 'success'
    surface_quality_metrics: dict[str, Any] = field(default_factory=dict)
    volume_quality_metrics: dict[str, Any] = field(default_factory=dict)
    geometry_constraint_summary: dict[str, Any] = field(default_factory=dict)
    geometry_retry_history: list[dict[str, Any]] = field(default_factory=list)
    initial_mesh_diagnostics: dict[str, Any] = field(default_factory=dict)
    budget_calibration_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_seconds: float = 0.0
    stage_where_stopped: str | None = None
    failure_category: str | None = None
    partial_output_available: bool = False
    failure_reason: str | None = None


@dataclass
class SampleRecord(JsonDataclass):
    sample_id: str
    geometry_id: str
    condition_id: str
    pde_family: str
    budget: int
    condition_spec: dict[str, Any]
    geometry_artifact_paths: dict[str, str]
    initial_mesh_path: str
    optional_intermediate_mesh_paths: list[str]
    final_target_mesh_path: str
    optional_reference_solution_path: str | None
    optional_error_indicator_path: str | None
    optional_stage_field_path: str | None = None
    optional_stage_probe_points_path: str | None = None
    normalization_metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ''
    split: str = 'unassigned'
    status: str = 'success'
    teacher_metadata: dict[str, Any] = field(default_factory=dict)
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_seconds: float = 0.0
    stage_where_stopped: str | None = None
    failure_category: str | None = None
    partial_output_available: bool = False
    failure_reason: str | None = None


@dataclass
class FailureRecord(JsonDataclass):
    stage: str
    item_id: str
    source_path: str | None
    reason: str
    details: dict[str, Any] = field(default_factory=dict)
    category: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_seconds: float = 0.0
    stage_where_stopped: str | None = None
    partial_output_available: bool = False
