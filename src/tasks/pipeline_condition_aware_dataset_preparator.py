from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Tuple

import meshio
import numpy as np
from omegaconf import DictConfig, ListConfig
from skfem import MeshTet, MeshTri
from skfem.io import from_meshio

from src.algorithm.dataloader.source_data import SourceData
from src.tasks.dataset_preparator import DatasetPreparator
from src.tasks.domains.extended_mesh_tet1 import ExtendedMeshTet1
from src.tasks.domains.extended_mesh_tri1 import ExtendedMeshTri1
from src.tasks.domains.gmsh_util import geom_fn_from_file
from src.tasks.domains.mesh_wrapper import MeshWrapper


STEP_SUFFIXES = {".step", ".stp", ".brep", ".iges", ".igs"}


class PipelineConditionAwareDatasetPreparator(DatasetPreparator):
    """Adapts condition-aware pipeline manifests to AMBER SourceData objects."""

    def __init__(self, algorithm_config: DictConfig, task_config: DictConfig):
        super().__init__(algorithm_config=algorithm_config, task_config=task_config)
        self.pipeline_output_root = Path(str(task_config.pipeline_output_root)).resolve()
        self.manifest_name = str(task_config.manifest_name)
        self.split_source = str(task_config.split_source)
        self.empty_split_policy = str(task_config.empty_split_policy)
        self.input_mesh_mode = str(task_config.input_mesh_mode)
        self.target_mode = str(task_config.target_mode)
        self.allowed_statuses = set(_as_list(task_config.allowed_statuses))
        self.budget_filter = _optional_set(task_config.budget_filter)
        self.require_single_budget = bool(task_config.require_single_budget)
        self.pde_family_filter = _optional_set(task_config.pde_family_filter)
        self.geometry_id_filter = _optional_set(task_config.geometry_id_filter)
        self.condition_id_filter = _optional_set(task_config.condition_id_filter)
        self.one_condition_per_geometry = bool(task_config.one_condition_per_geometry)
        self.mesh_cell_type = str(task_config.mesh_cell_type)
        self.cell_type_policy = str(task_config.cell_type_policy)
        self.min_initial_elements = _optional_int(task_config.min_initial_elements)
        self.max_initial_elements = _optional_int(task_config.max_initial_elements)
        self.min_target_elements = _optional_int(task_config.min_target_elements)
        self.max_target_elements = _optional_int(task_config.max_target_elements)
        self.over_limit_policy = str(task_config.over_limit_policy)
        self.require_indicator = bool(task_config.require_indicator)
        self.physics_weight_source = str(task_config.physics_weight_source)
        self.physics_feature_source = str(task_config.physics_feature_source)
        self.condition_spec_mode = str(task_config.condition_spec_mode)
        self.required_splits = list(_as_list(task_config.required_splits))

        if self.target_mode != "final_target_mesh":
            raise ValueError(f"Unsupported pipeline target_mode '{self.target_mode}'.")
        if self.input_mesh_mode not in {"initial_mesh", "coarse_mesh"}:
            raise ValueError(f"Unsupported pipeline input_mesh_mode '{self.input_mesh_mode}'.")
        if self.physics_weight_source != "pipeline_indicator":
            raise ValueError(f"Unsupported physics_weight_source '{self.physics_weight_source}'.")
        if self.physics_feature_source != "pipeline_indicator":
            raise ValueError(f"Unsupported physics_feature_source '{self.physics_feature_source}'.")
        if self.condition_spec_mode != "metadata_only":
            raise ValueError("The current adapter stores condition_spec as metadata only.")

        self._records_by_split = self._load_records_by_split()

    def get_dataset(self, dataset_mode: str):
        if dataset_mode not in {"train", "val", "test"}:
            raise ValueError(f"dataset_mode '{dataset_mode}' not recognized")
        records = self._records_by_split.get(dataset_mode, [])
        if not records and self.empty_split_policy == "fail" and dataset_mode in self.required_splits:
            raise ValueError(f"Pipeline split '{dataset_mode}' is empty after filtering.")
        data_points = [
            self.prepare_data_point(data_idx=data_idx, dataset_mode=dataset_mode)
            for data_idx in range(len(records))
        ]
        return self.dataset_class(algorithm_config=self.algorithm_config, persistent_data=data_points)

    def _prepare_source_and_mesh(self, data_idx: int, dataset_mode: str) -> Tuple[SourceData, MeshWrapper]:
        record = self._records_by_split[dataset_mode][data_idx]
        initial_mesh_path = self._input_mesh_path(record)
        target_mesh_path = self._resolve_path(record["final_target_mesh_path"])
        source_path = self._resolve_path(record["geometry_artifact_paths"]["source_path"])
        indicator_path = self._resolve_path(record.get("optional_error_indicator_path"))

        geometry_fn = _geometry_fn_from_path(source_path)
        initial_mesh = self._load_mesh(initial_mesh_path)
        expert_mesh = self._load_mesh(target_mesh_path)
        initial_mesh.geom_fn = geometry_fn
        expert_mesh.geom_fn = geometry_fn

        source_data = SourceData(
            expert_mesh=MeshWrapper(expert_mesh),
            initial_mesh=MeshWrapper(initial_mesh),
            feature_provider=None,
            dataset_name=str(self.task_config.name),
            data_point_path=str(target_mesh_path),
            imitation_weight_cache={
                "weight_source_mode": self.physics_weight_source,
                "indicator_path": str(indicator_path) if indicator_path is not None else None,
                "sample_id": record.get("sample_id"),
                "geometry_id": record.get("geometry_id"),
                "condition_id": record.get("condition_id"),
                "budget": record.get("budget"),
                "pde_family": record.get("pde_family"),
                "condition_spec": record.get("condition_spec"),
            },
        )
        source_data.expert_mesh.source_data = source_data
        source_data.expert_mesh.weighted_imitation_config = self.algorithm_config.get("weighted_imitation") or {}
        return source_data, source_data.initial_mesh

    def _load_records_by_split(self) -> dict[str, list[dict[str, Any]]]:
        records = self._read_manifest()
        split_lookup = self._read_split_lookup() if self.split_source == "split_manifest" else None
        filtered: list[dict[str, Any]] = []
        for record in records:
            split = self._record_split(record, split_lookup)
            if split not in {"train", "val", "test"}:
                continue
            if not self._record_passes_metadata_filters(record):
                continue
            record = dict(record)
            record["split"] = split
            if not self._record_passes_mesh_filters(record):
                continue
            filtered.append(record)
        if self.one_condition_per_geometry:
            filtered = self._keep_one_condition_per_geometry(filtered)
        if self.require_single_budget:
            budgets = {int(record.get("budget")) for record in filtered}
            if len(budgets) > 1:
                raise ValueError(
                    f"Pipeline adapter requires a single budget, but found {sorted(budgets)}. "
                    "Set task.budget_filter or disable task.require_single_budget."
                )
        records_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in filtered:
            records_by_split[str(record["split"])].append(record)
        for split_records in records_by_split.values():
            split_records.sort(key=lambda rec: str(rec.get("sample_id", "")))
        return dict(records_by_split)

    def _read_manifest(self) -> list[dict[str, Any]]:
        manifest_path = self.pipeline_output_root / "manifests" / f"{self.manifest_name}.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Pipeline manifest not found: {manifest_path}")
        records = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def _read_split_lookup(self) -> dict[str, str]:
        split_path = self.pipeline_output_root / "manifests" / "split_manifest.json"
        if not split_path.exists():
            raise FileNotFoundError(f"Pipeline split manifest not found: {split_path}")
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        return dict(payload.get("geometry_to_split", {}))

    def _record_split(self, record: dict[str, Any], split_lookup: dict[str, str] | None) -> str:
        if self.split_source == "sample_manifest":
            return str(record.get("split", "unassigned"))
        if self.split_source == "split_manifest":
            return str(split_lookup.get(record.get("geometry_id"), "unassigned")) if split_lookup is not None else "unassigned"
        raise ValueError(f"Unsupported split_source '{self.split_source}'.")

    def _record_passes_metadata_filters(self, record: dict[str, Any]) -> bool:
        if self.allowed_statuses and str(record.get("status")) not in self.allowed_statuses:
            return False
        if self.budget_filter is not None and int(record.get("budget")) not in self.budget_filter:
            return False
        if self.pde_family_filter is not None and str(record.get("pde_family")) not in self.pde_family_filter:
            return False
        if self.geometry_id_filter is not None and str(record.get("geometry_id")) not in self.geometry_id_filter:
            return False
        if self.condition_id_filter is not None and str(record.get("condition_id")) not in self.condition_id_filter:
            return False
        if not record.get("final_target_mesh_path"):
            return False
        if self.require_indicator and not record.get("optional_error_indicator_path"):
            return False
        return True

    def _record_passes_mesh_filters(self, record: dict[str, Any]) -> bool:
        try:
            initial_counts = self._mesh_counts(self._input_mesh_path(record))
            target_counts = self._mesh_counts(self._resolve_path(record["final_target_mesh_path"]))
        except Exception:
            if self.over_limit_policy == "fail":
                raise
            return False
        return self._check_count(initial_counts, self.min_initial_elements, self.max_initial_elements, "initial") and self._check_count(
            target_counts,
            self.min_target_elements,
            self.max_target_elements,
            "target",
        )

    def _check_count(self, count: int, minimum: int | None, maximum: int | None, label: str) -> bool:
        if minimum is not None and count < minimum:
            if self.over_limit_policy == "fail":
                raise ValueError(f"Pipeline {label} mesh has {count} elements, below minimum {minimum}.")
            return False
        if maximum is not None and count > maximum:
            if self.over_limit_policy == "fail":
                raise ValueError(f"Pipeline {label} mesh has {count} elements, above maximum {maximum}.")
            return False
        return True

    def _keep_one_condition_per_geometry(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        best_by_geometry: dict[str, dict[str, Any]] = {}
        for record in sorted(records, key=lambda rec: (_status_priority(str(rec.get("status"))), str(rec.get("sample_id", "")))):
            best_by_geometry.setdefault(str(record.get("geometry_id")), record)
        return list(best_by_geometry.values())

    def _input_mesh_path(self, record: dict[str, Any]) -> Path:
        if self.input_mesh_mode == "initial_mesh":
            return self._resolve_path(record["initial_mesh_path"])
        return self._resolve_path(record["geometry_artifact_paths"]["coarse_mesh_path"])

    def _resolve_path(self, path_value: str | None) -> Path | None:
        if path_value in {None, ""}:
            return None
        path = Path(str(path_value))
        if path.is_absolute():
            return path
        return (self.pipeline_output_root / path).resolve()

    def _mesh_counts(self, mesh_path: Path | None) -> int:
        if mesh_path is None:
            raise ValueError("Missing mesh path")
        mesh = meshio.read(str(mesh_path))
        return int(sum(len(block.data) for block in mesh.cells if block.type == self.mesh_cell_type))

    def _load_mesh(self, mesh_path: Path | None) -> ExtendedMeshTri1 | ExtendedMeshTet1:
        if mesh_path is None:
            raise ValueError("Missing mesh path")
        mesh = meshio.read(str(mesh_path))
        cells = _select_cells(mesh, cell_type=self.mesh_cell_type, policy=self.cell_type_policy, path=mesh_path)
        converted = from_meshio(meshio.Mesh(points=mesh.points, cells=[(self.mesh_cell_type, cells)]))
        if isinstance(converted, MeshTet):
            return ExtendedMeshTet1(converted.p, converted.t)
        if isinstance(converted, MeshTri):
            return ExtendedMeshTri1(converted.p, converted.t)
        raise ValueError(f"Unsupported mesh type {type(converted)} loaded from {mesh_path}")


def _select_cells(mesh: meshio.Mesh, *, cell_type: str, policy: str, path: Path) -> np.ndarray:
    matching = [block.data for block in mesh.cells if block.type == cell_type]
    if not matching:
        raise ValueError(f"Mesh '{path}' does not contain '{cell_type}' cells.")
    if policy == "strict_tetra_only":
        nonempty_other = [block.type for block in mesh.cells if block.type != cell_type and len(block.data) > 0]
        if nonempty_other:
            raise ValueError(f"Mesh '{path}' contains non-{cell_type} cells: {sorted(set(nonempty_other))}")
    elif policy != "filter_tetra_then_fail_if_empty":
        raise ValueError(f"Unsupported cell_type_policy '{policy}'.")
    return np.concatenate(matching, axis=0)


def _geometry_fn_from_path(path: Path):
    suffix = path.suffix.lower()
    if suffix not in STEP_SUFFIXES:
        raise ValueError(f"Unsupported pipeline geometry suffix '{suffix}' for {path}")
    return geom_fn_from_file(str(path))


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, ListConfig)):
        return list(value)
    return [value]


def _optional_set(value: Any) -> set | None:
    values = _as_list(value)
    if not values:
        return None
    return set(values)


def _optional_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)


def _status_priority(status: str) -> int:
    priority = {
        "success_budget_closed": 0,
        "success_near_desired_budget": 1,
        "success_partial_under_budget": 2,
    }
    return priority.get(status, 99)
