from __future__ import annotations

import json
import hashlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable

import meshio
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf

from src.algorithm.util.fem_imitation_weights import _load_stage_probe_importance


AMBER_REPO_ROOT = Path(__file__).resolve().parents[2]
TET_WORK_ROOT = AMBER_REPO_ROOT.parent
PATH_PROTOCOL_SCHEMA_VERSION = 1
STAGE_FIELD_SOURCES = {"stage_field", "stage_field_fusion"}


class PipelineDatasetAuditError(ValueError):
    pass


@dataclass(frozen=True)
class AuditIssue:
    category: str
    sample_id: str
    field: str
    message: str


@dataclass
class PipelineDatasetAuditResult:
    schema_version: int
    pipeline_output_root: str
    geometry_source_root: str
    counts: dict[str, int]
    raw_split_counts: dict[str, int]
    split_counts: dict[str, int]
    geometry_ids_by_split: dict[str, list[str]]
    geometry_overlap: dict[str, list[str]]
    status_distribution: dict[str, int]
    quality_distribution: dict[str, int]
    pde_family_distribution: dict[str, int]
    retained_pde_family_distribution: dict[str, int]
    element_budget: dict[str, Any]
    importance_diagnostics: dict[str, int]
    records_by_split: dict[str, list[dict[str, Any]]] = field(repr=False)
    issues: list[AuditIssue] = field(default_factory=list)
    consumed_paths: list[str] = field(default_factory=list)

    @property
    def retained_count(self) -> int:
        return int(self.counts.get("retained", 0))

    def to_dict(self, *, include_records: bool = False) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "pipeline_output_root": self.pipeline_output_root,
            "geometry_source_root": self.geometry_source_root,
            "counts": self.counts,
            "raw_split_counts": self.raw_split_counts,
            "split_counts": self.split_counts,
            "geometry_ids_by_split": self.geometry_ids_by_split,
            "geometry_overlap": self.geometry_overlap,
            "status_distribution": self.status_distribution,
            "quality_distribution": self.quality_distribution,
            "pde_family_distribution": self.pde_family_distribution,
            "retained_pde_family_distribution": self.retained_pde_family_distribution,
            "element_budget": self.element_budget,
            "importance_diagnostics": self.importance_diagnostics,
            "issues": [asdict(issue) for issue in self.issues],
            "consumed_paths": self.consumed_paths,
        }
        if include_records:
            payload["retained_records"] = [
                record
                for split in ("train", "val", "test")
                for record in self.records_by_split.get(split, [])
            ]
        return payload

    def failure_message(self) -> str:
        examples = self.issues[:3]
        example_text = "; ".join(
            f"{item.sample_id}:{item.field}: {item.message}" for item in examples
        )
        return (
            "Pipeline dataset structural audit failed. "
            f"original_records={self.counts.get('original_records', 0)}, "
            f"retained={self.retained_count}, counts={self.counts}. "
            f"First {len(examples)} issue(s): {example_text}"
        )

    def raise_for_errors(self) -> None:
        if self.issues:
            raise PipelineDatasetAuditError(self.failure_message())


class DatasetPathResolver:
    """Resolve the two declared anchors plus explicit legacy relocation."""

    def __init__(self, task_config: dict[str, Any]):
        self.pipeline_output_root = _anchor_amber_path(task_config["pipeline_output_root"])
        geometry_root_value = task_config.get("geometry_source_root")
        if geometry_root_value in {None, ""}:
            geometry_root_value = TET_WORK_ROOT / "dataest-pipeline" / "data" / "mold"
        self.geometry_source_root = _anchor_amber_path(geometry_root_value)
        self.relocation = dict(task_config.get("path_relocation", {}) or {})

    def resolve(
        self,
        value: str | Path | None,
        *,
        anchor: str = "pipeline_output",
        must_exist: bool = True,
    ) -> Path | None:
        if value in {None, ""}:
            return None
        raw = str(value)
        root = self.geometry_source_root if anchor == "geometry_source" else self.pipeline_output_root
        if not _is_absolute(raw):
            relative = Path(raw.replace("\\", "/"))
            resolved = (root / relative).resolve()
            _ensure_under_anchor(resolved, root, raw)
        else:
            resolved = Path(raw).resolve()
            if not resolved.exists():
                relocated = self._relocate_legacy_absolute(raw)
                if relocated is not None:
                    resolved = relocated
        if must_exist and not resolved.exists():
            raise FileNotFoundError(
                f"Declared {anchor} path does not exist: value='{raw}', resolved='{resolved}'"
            )
        return resolved

    def _relocate_legacy_absolute(self, raw: str) -> Path | None:
        if not bool(self.relocation.get("enabled", False)):
            return None
        old_root = self.relocation.get("old_root")
        new_root = self.relocation.get("new_root")
        if not old_root or not new_root:
            raise ValueError("path_relocation.enabled requires both old_root and new_root")
        raw_normalized = _normalized_path_text(raw)
        old_normalized = _normalized_path_text(str(old_root)).rstrip("/")
        raw_cmp = raw_normalized.casefold()
        old_cmp = old_normalized.casefold()
        if raw_cmp != old_cmp and not raw_cmp.startswith(old_cmp + "/"):
            return None
        suffix = raw_normalized[len(old_normalized) :].lstrip("/")
        relocated_root = _anchor_amber_path(new_root)
        return (relocated_root / Path(suffix)).resolve()


def audit_pipeline_dataset(task_config: DictConfig | dict[str, Any]) -> PipelineDatasetAuditResult:
    config = _plain_container(task_config)
    resolver = DatasetPathResolver(config)
    manifest_path = resolver.pipeline_output_root / "manifests" / f"{config['manifest_name']}.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Pipeline manifest not found: {manifest_path}")

    records = _read_jsonl(manifest_path)
    evaluation_reference_lookup, evaluation_reference_manifest_path = _read_evaluation_reference_lookup(
        config, resolver
    )
    quality_lookup, quality_distribution, quality_report_path = _read_quality_lookup(config, resolver)
    split_lookup, split_path = _read_split_lookup(config, resolver)
    counts: Counter[str] = Counter(original_records=len(records))
    issues: list[AuditIssue] = []
    retained: list[dict[str, Any]] = []
    consumed_paths: set[Path] = {manifest_path.resolve()}
    if quality_report_path is not None:
        consumed_paths.add(quality_report_path.resolve())
    if split_path is not None:
        consumed_paths.add(split_path.resolve())
    if evaluation_reference_manifest_path is not None:
        consumed_paths.add(evaluation_reference_manifest_path.resolve())
    config_snapshot = resolver.pipeline_output_root / "config_snapshot.yaml"
    if config_snapshot.exists():
        consumed_paths.add(config_snapshot.resolve())

    status_distribution = Counter(str(record.get("status")) for record in records)
    pde_family_distribution = Counter(str(record.get("pde_family")) for record in records)
    raw_split_counts = Counter(
        _record_split(record, split_lookup, str(config.get("split_source", "sample_manifest")))
        for record in records
    )
    seen_sample_ids: set[str] = set()
    target_counts: list[int] = []
    target_budgets: list[int] = []
    importance_diagnostics: Counter[str] = Counter()

    for raw_record in records:
        sample_id = str(raw_record.get("sample_id", "<missing-sample-id>"))
        if sample_id in seen_sample_ids:
            _issue(issues, counts, "duplicate_sample_id", sample_id, "sample_id", "duplicate sample ID")
            continue
        seen_sample_ids.add(sample_id)

        split = _record_split(raw_record, split_lookup, str(config.get("split_source", "sample_manifest")))
        if split not in {"train", "val", "test"}:
            counts["split_filtered"] += 1
            continue
        try:
            metadata_filter = _metadata_filter_reason(raw_record, config, quality_lookup)
        except Exception as exc:
            _issue(
                issues,
                counts,
                "metadata_schema_error",
                sample_id,
                "metadata",
                str(exc),
            )
            continue
        if metadata_filter is not None:
            counts[metadata_filter] += 1
            continue

        record = dict(raw_record)
        evaluation_reference = evaluation_reference_lookup.get(sample_id)
        if evaluation_reference is not None:
            record["optional_evaluation_reference_path"] = evaluation_reference.get(
                "evaluation_reference_path"
            )
            record["evaluation_reference_metadata"] = evaluation_reference
        elif (
            bool(config.get("require_evaluation_reference", False))
            and split in set(_as_list(config.get("evaluation_reference_required_splits", ["test"])))
        ):
            _issue(
                issues,
                counts,
                "missing_evaluation_reference",
                sample_id,
                "optional_evaluation_reference_path",
                "retained sample has no strong evaluation reference",
            )
            continue
        record["split"] = split
        try:
            resolved_paths = _audit_record_structure(
                record=record,
                config=config,
                resolver=resolver,
                consumed_paths=consumed_paths,
                counts=counts,
                target_counts=target_counts,
                target_budgets=target_budgets,
                importance_diagnostics=importance_diagnostics,
            )
        except _RecordAuditFailure as exc:
            _issue(issues, counts, exc.category, sample_id, exc.field, str(exc))
            continue
        if resolved_paths is None:
            # A declared min/max size filter intentionally skipped the sample.
            continue
        record["_resolved_paths"] = resolved_paths
        retained.append(record)

    if bool(config.get("one_condition_per_geometry", False)):
        retained, removed = _keep_one_condition_per_geometry(retained)
        counts["one_condition_filtered"] += removed

    if bool(config.get("require_single_budget", False)):
        budgets = sorted({int(record.get("budget")) for record in retained})
        if len(budgets) > 1:
            _issue(
                issues,
                counts,
                "multiple_budgets",
                "<dataset>",
                "budget",
                f"single budget required, found {budgets}",
            )

    records_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in retained:
        records_by_split[str(record["split"])].append(record)
    for split_records in records_by_split.values():
        split_records.sort(key=lambda item: str(item.get("sample_id", "")))

    split_counts = {split: len(records_by_split.get(split, [])) for split in ("train", "val", "test")}
    retained_pde_family_distribution = Counter(
        str(record.get("pde_family")) for record in retained
    )
    geometry_ids_by_split = {
        split: sorted({str(record.get("geometry_id")) for record in records_by_split.get(split, [])})
        for split in ("train", "val", "test")
    }
    overlap = _geometry_overlap(geometry_ids_by_split)
    for pair, geometry_ids in overlap.items():
        if geometry_ids:
            _issue(
                issues,
                counts,
                "geometry_split_overlap",
                "<dataset>",
                pair,
                f"geometries occur in both splits: {geometry_ids[:3]}",
            )

    counts["retained"] = sum(split_counts.values())
    required_splits = set(_as_list(config.get("required_splits", [])))
    for split in sorted(required_splits):
        if split_counts.get(str(split), 0) == 0:
            _issue(
                issues,
                counts,
                "empty_required_split",
                "<dataset>",
                str(split),
                f"required split '{split}' is empty after filtering",
            )

    return PipelineDatasetAuditResult(
        schema_version=PATH_PROTOCOL_SCHEMA_VERSION,
        pipeline_output_root=str(resolver.pipeline_output_root),
        geometry_source_root=str(resolver.geometry_source_root),
        counts=dict(sorted(counts.items())),
        raw_split_counts={
            split: int(raw_split_counts.get(split, 0)) for split in ("train", "val", "test")
        },
        split_counts=split_counts,
        geometry_ids_by_split=geometry_ids_by_split,
        geometry_overlap=overlap,
        status_distribution=dict(sorted(status_distribution.items())),
        quality_distribution=dict(sorted(quality_distribution.items())),
        pde_family_distribution=dict(sorted(pde_family_distribution.items())),
        retained_pde_family_distribution=dict(sorted(retained_pde_family_distribution.items())),
        element_budget=_element_budget_summary(target_counts, target_budgets),
        importance_diagnostics=dict(sorted(importance_diagnostics.items())),
        records_by_split=dict(records_by_split),
        issues=issues,
        consumed_paths=sorted(str(path) for path in consumed_paths),
    )


class _RecordAuditFailure(ValueError):
    def __init__(self, category: str, field: str, message: str):
        super().__init__(message)
        self.category = category
        self.field = field


def _audit_record_structure(
    *,
    record: dict[str, Any],
    config: dict[str, Any],
    resolver: DatasetPathResolver,
    consumed_paths: set[Path],
    counts: Counter[str],
    target_counts: list[int],
    target_budgets: list[int],
    importance_diagnostics: Counter[str],
) -> dict[str, str] | None:
    sample_id = str(record.get("sample_id"))
    geometry_artifacts = record.get("geometry_artifact_paths")
    if not isinstance(geometry_artifacts, dict):
        raise _RecordAuditFailure("schema_error", "geometry_artifact_paths", "must be an object")

    resolved: dict[str, str] = {}
    source = _resolve_declared(
        resolver,
        geometry_artifacts.get("source_path"),
        anchor="geometry_source",
        field="geometry_artifact_paths.source_path",
    )
    if source.suffix.lower() not in {".step", ".stp", ".brep", ".iges", ".igs"}:
        raise _RecordAuditFailure(
            "schema_error",
            "geometry_artifact_paths.source_path",
            f"unsupported geometry suffix '{source.suffix}'",
        )
    resolved["source_path"] = str(source)
    consumed_paths.add(source)

    input_mode = str(config.get("input_mesh_mode", "initial_mesh"))
    if input_mode == "initial_mesh":
        input_value = record.get("initial_mesh_path")
        input_field = "initial_mesh_path"
    elif input_mode == "coarse_mesh":
        input_value = geometry_artifacts.get("coarse_mesh_path")
        input_field = "geometry_artifact_paths.coarse_mesh_path"
    else:
        raise _RecordAuditFailure("schema_error", "input_mesh_mode", f"unsupported mode '{input_mode}'")

    input_path = _resolve_declared(resolver, input_value, field=input_field)
    target_path = _resolve_declared(resolver, record.get("final_target_mesh_path"), field="final_target_mesh_path")
    input_count = _read_tetra_count(input_path, config, input_field)
    target_count = _read_tetra_count(target_path, config, "final_target_mesh_path")
    consumed_paths.update({input_path, target_path})
    resolved["input_mesh_path"] = str(input_path)
    resolved["target_mesh_path"] = str(target_path)

    if not _passes_size_filter(input_count, config.get("min_initial_elements"), config.get("max_initial_elements")):
        counts["initial_size_out_of_range"] += 1
        if str(config.get("over_limit_policy", "fail")) == "fail":
            raise _RecordAuditFailure(
                "initial_size_out_of_range",
                input_field,
                f"initial tetra count {input_count} is outside configured range",
            )
        return None
    if not _passes_size_filter(target_count, config.get("min_target_elements"), config.get("max_target_elements")):
        counts["target_size_out_of_range"] += 1
        if str(config.get("over_limit_policy", "fail")) == "fail":
            raise _RecordAuditFailure(
                "target_size_out_of_range",
                "final_target_mesh_path",
                f"target tetra count {target_count} is outside configured range",
            )
        return None

    target_counts.append(target_count)
    target_budgets.append(int(record.get("budget", 0) or 0))

    require_indicator = bool(config.get("require_indicator", False))
    indicator_value = record.get("optional_error_indicator_path")
    if require_indicator and not indicator_value:
        raise _RecordAuditFailure("missing_indicator", "optional_error_indicator_path", "required indicator is missing")
    if indicator_value:
        indicator_path = _resolve_declared(
            resolver,
            indicator_value,
            field="optional_error_indicator_path",
        )
        try:
            indicator = np.asarray(np.load(indicator_path, mmap_mode="r"), dtype=np.float64).reshape(-1)
        except Exception as exc:
            raise _RecordAuditFailure(
                "indicator_shape_error",
                "optional_error_indicator_path",
                f"cannot load indicator: {exc}",
            ) from exc
        if indicator.shape[0] != target_count:
            raise _RecordAuditFailure(
                "indicator_shape_error",
                "optional_error_indicator_path",
                f"indicator length {indicator.shape[0]} != target tetra count {target_count}",
            )
        if not np.all(np.isfinite(indicator)):
            raise _RecordAuditFailure(
                "indicator_shape_error",
                "optional_error_indicator_path",
                "indicator contains non-finite values",
            )
        if np.allclose(indicator, 0.0):
            importance_diagnostics["all_zero"] += 1
        if indicator.size and np.allclose(indicator, indicator[0]):
            importance_diagnostics["constant"] += 1
        if indicator.size and np.mean(indicator >= np.quantile(indicator, 0.99)) > 0.5:
            importance_diagnostics["extreme_saturation"] += 1
        resolved["indicator_path"] = str(indicator_path)
        consumed_paths.add(indicator_path)

    requires_stage = (
        str(config.get("physics_weight_source")) in STAGE_FIELD_SOURCES
        or str(config.get("physics_feature_source")) in STAGE_FIELD_SOURCES
    )
    stage_value = record.get("optional_stage_field_path")
    if requires_stage and not stage_value:
        raise _RecordAuditFailure("missing_stage_field", "optional_stage_field_path", "required stage field is missing")
    if stage_value:
        stage_path = _resolve_declared(resolver, stage_value, field="optional_stage_field_path")
        stage_config = dict(config.get("stage_field", {}) or {})
        stage_source = str(config.get("physics_feature_source"))
        if stage_source not in STAGE_FIELD_SOURCES:
            stage_source = str(config.get("physics_weight_source"))
        if stage_source not in STAGE_FIELD_SOURCES:
            stage_source = "stage_field"
        try:
            probe_points, probe_importance = _load_stage_probe_importance(
                stage_field_path=str(stage_path),
                stage_field_config=stage_config,
                source=stage_source,
            )
        except Exception as exc:
            raise _RecordAuditFailure(
                "stage_field_shape_error",
                "optional_stage_field_path",
                f"invalid stage field: {exc}",
            ) from exc
        if probe_points.shape[0] == 0 or probe_points.shape[0] != probe_importance.shape[0]:
            raise _RecordAuditFailure(
                "stage_field_shape_error",
                "optional_stage_field_path",
                "stage probe/value arrays are empty or length-mismatched",
            )
        resolved["stage_field_path"] = str(stage_path)
        consumed_paths.add(stage_path)

    for key in ("optional_reference_solution_path", "optional_evaluation_reference_path"):
        value = record.get(key)
        if not value:
            continue
        reference_path = _resolve_declared(resolver, value, field=key)
        reference_info = _validate_reference_solution(reference_path, key)
        resolved[key] = str(reference_path)
        consumed_paths.add(reference_path)
        if key == "optional_evaluation_reference_path":
            metadata = dict(record.get("evaluation_reference_metadata", {}) or {})
            expected_condition_hash = _json_sha256(record.get("condition_spec", {}))
            if str(metadata.get("condition_id")) != str(record.get("condition_id")):
                raise _RecordAuditFailure(
                    "reference_condition_mismatch",
                    key,
                    "evaluation reference condition_id does not match sample",
                )
            if str(metadata.get("condition_sha256")) != expected_condition_hash:
                raise _RecordAuditFailure(
                    "reference_condition_mismatch",
                    key,
                    "evaluation reference condition hash does not match sample condition_spec",
                )
            if int(metadata.get("uniform_refinement_level", -1)) != 1:
                raise _RecordAuditFailure(
                    "reference_strength_error",
                    key,
                    "evaluation reference is not the frozen one-level uniform refinement",
                )
            if int(reference_info["num_elements"]) <= target_count:
                raise _RecordAuditFailure(
                    "reference_strength_error",
                    key,
                    f"evaluation reference has {reference_info['num_elements']} elements, target has {target_count}",
                )
            metadata_value = metadata.get("metadata_path")
            if metadata_value:
                metadata_path = _resolve_declared(
                    resolver,
                    metadata_value,
                    field="evaluation_reference_metadata.metadata_path",
                )
                resolved["evaluation_reference_metadata_path"] = str(metadata_path)
                consumed_paths.add(metadata_path)

    for key in ("geometry_record_path", "preprocess_record_path"):
        value = geometry_artifacts.get(key)
        if not value:
            continue
        metadata_path = _resolve_declared(resolver, value, field=f"geometry_artifact_paths.{key}")
        resolved[key] = str(metadata_path)
        consumed_paths.add(metadata_path)

    return resolved


def _resolve_declared(
    resolver: DatasetPathResolver,
    value: Any,
    *,
    field: str,
    anchor: str = "pipeline_output",
) -> Path:
    if value in {None, ""}:
        category = "missing_indicator" if "indicator" in field else "path_not_found"
        raise _RecordAuditFailure(category, field, "required path value is missing")
    try:
        path = resolver.resolve(value, anchor=anchor, must_exist=True)
    except Exception as exc:
        raise _RecordAuditFailure("path_not_found", field, str(exc)) from exc
    assert path is not None
    return path


def _read_tetra_count(path: Path, config: dict[str, Any], field: str) -> int:
    try:
        mesh = meshio.read(str(path))
    except Exception as exc:
        raise _RecordAuditFailure("mesh_read_error", field, f"cannot read mesh '{path}': {exc}") from exc
    cell_type = str(config.get("mesh_cell_type", "tetra"))
    matching = [block.data for block in mesh.cells if block.type == cell_type]
    if not matching or sum(len(cells) for cells in matching) == 0:
        raise _RecordAuditFailure("no_tetra", field, f"mesh '{path}' has no '{cell_type}' cells")
    policy = str(config.get("cell_type_policy", "filter_tetra_then_fail_if_empty"))
    if policy == "strict_tetra_only":
        other = sorted({block.type for block in mesh.cells if block.type != cell_type and len(block.data)})
        if other:
            raise _RecordAuditFailure("mesh_read_error", field, f"mesh has non-{cell_type} cells: {other}")
    elif policy != "filter_tetra_then_fail_if_empty":
        raise _RecordAuditFailure("schema_error", "cell_type_policy", f"unsupported policy '{policy}'")
    return int(sum(len(cells) for cells in matching))


def _validate_reference_solution(path: Path, field: str) -> dict[str, int]:
    try:
        with np.load(path, mmap_mode="r") as payload:
            missing = {"points", "connectivity", "values"}.difference(payload.files)
            if missing:
                raise ValueError(f"missing arrays {sorted(missing)}")
            if payload["connectivity"].ndim != 2 or payload["values"].shape[0] != payload["points"].shape[0]:
                raise ValueError("points/connectivity/values shapes are inconsistent")
            return {
                "num_points": int(payload["points"].shape[0]),
                "num_elements": int(payload["connectivity"].shape[0]),
            }
    except Exception as exc:
        raise _RecordAuditFailure("reference_shape_error", field, f"invalid reference solution: {exc}") from exc


def _json_sha256(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _metadata_filter_reason(
    record: dict[str, Any],
    config: dict[str, Any],
    quality_lookup: dict[str, str],
) -> str | None:
    allowed_statuses = set(_as_list(config.get("allowed_statuses", [])))
    if allowed_statuses and str(record.get("status")) not in allowed_statuses:
        return "status_filtered"
    filters = (
        ("budget_filter", "budget", int, "budget_filtered"),
        ("pde_family_filter", "pde_family", str, "pde_family_filtered"),
        ("geometry_id_filter", "geometry_id", str, "geometry_filtered"),
        ("condition_id_filter", "condition_id", str, "condition_filtered"),
        ("sample_id_filter", "sample_id", str, "sample_filtered"),
    )
    for config_key, record_key, cast, reason in filters:
        allowed = _optional_set(config.get(config_key))
        if allowed is not None and cast(record.get(record_key)) not in {cast(item) for item in allowed}:
            return reason
    quality_config = dict(config.get("quality_filter", {}) or {})
    if bool(quality_config.get("enabled", False)):
        sample_id = str(record.get("sample_id"))
        verdict = quality_lookup.get(sample_id)
        if verdict is None:
            policy = str(quality_config.get("missing_sample_policy", "fail"))
            if policy == "allow":
                return None
            if policy == "skip":
                return "quality_filtered"
            raise PipelineDatasetAuditError(f"Quality report does not contain sample '{sample_id}'")
        allowed = set(_as_list(quality_config.get("allowed_verdicts", [])))
        if allowed and verdict not in allowed:
            return "quality_filtered"
    return None


def _read_quality_lookup(
    config: dict[str, Any], resolver: DatasetPathResolver
) -> tuple[dict[str, str], Counter[str], Path | None]:
    quality_config = dict(config.get("quality_filter", {}) or {})
    if not bool(quality_config.get("enabled", False)):
        return {}, Counter(), None
    relative = str(quality_config.get("report_relative_path", "reports/smoke_report.json"))
    report_path = resolver.resolve(relative, must_exist=False)
    assert report_path is not None
    if not report_path.exists():
        if str(quality_config.get("missing_report_policy", "fail")) == "ignore":
            return {}, Counter(), None
        raise FileNotFoundError(f"Pipeline quality report not found: {report_path}")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    lookup: dict[str, str] = {}
    distribution: Counter[str] = Counter()
    for geometry_report in _iter_quality_geometry_reports(payload):
        verdict = geometry_report.get("verdict")
        if not verdict:
            continue
        for sample_metrics in geometry_report.get("sample_metrics", []):
            sample_id = sample_metrics.get("sample_id")
            if not sample_id:
                continue
            sample_id = str(sample_id)
            existing = lookup.get(sample_id)
            if existing is not None and existing != verdict:
                raise PipelineDatasetAuditError(
                    f"Quality report assigns sample '{sample_id}' conflicting verdicts: {existing}, {verdict}"
                )
            lookup[sample_id] = str(verdict)
    distribution.update(lookup.values())
    return lookup, distribution, report_path


def _read_split_lookup(
    config: dict[str, Any], resolver: DatasetPathResolver
) -> tuple[dict[str, str] | None, Path | None]:
    if str(config.get("split_source", "sample_manifest")) != "split_manifest":
        split_path = resolver.pipeline_output_root / "manifests" / "split_manifest.json"
        return None, split_path if split_path.exists() else None
    split_path = resolver.pipeline_output_root / "manifests" / "split_manifest.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Pipeline split manifest not found: {split_path}")
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    return dict(payload.get("geometry_to_split", {})), split_path


def _read_evaluation_reference_lookup(
    config: dict[str, Any], resolver: DatasetPathResolver
) -> tuple[dict[str, dict[str, Any]], Path | None]:
    manifest_name = str(
        config.get("evaluation_reference_manifest_name", "evaluation_reference_manifest")
    )
    manifest_path = resolver.pipeline_output_root / "manifests" / f"{manifest_name}.jsonl"
    if not manifest_path.exists():
        return {}, None
    lookup: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(manifest_path):
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            raise PipelineDatasetAuditError(
                f"Evaluation reference manifest row has no sample_id: {row}"
            )
        if sample_id in lookup:
            raise PipelineDatasetAuditError(
                f"Evaluation reference manifest contains duplicate sample '{sample_id}'"
            )
        lookup[sample_id] = row
    return lookup, manifest_path


def _record_split(record: dict[str, Any], split_lookup: dict[str, str] | None, source: str) -> str:
    if source == "sample_manifest":
        return str(record.get("split", "unassigned"))
    if source == "split_manifest":
        return str((split_lookup or {}).get(record.get("geometry_id"), "unassigned"))
    raise PipelineDatasetAuditError(f"Unsupported split_source '{source}'")


def _keep_one_condition_per_geometry(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    priority = {
        "success_budget_closed": 0,
        "success_near_desired_budget": 1,
        "success_partial_under_budget": 2,
    }
    best: dict[str, dict[str, Any]] = {}
    for record in sorted(
        records,
        key=lambda item: (priority.get(str(item.get("status")), 99), str(item.get("sample_id", ""))),
    ):
        best.setdefault(str(record.get("geometry_id")), record)
    return list(best.values()), len(records) - len(best)


def _geometry_overlap(geometry_ids_by_split: dict[str, list[str]]) -> dict[str, list[str]]:
    sets = {key: set(value) for key, value in geometry_ids_by_split.items()}
    return {
        "train_val": sorted(sets["train"] & sets["val"]),
        "train_test": sorted(sets["train"] & sets["test"]),
        "val_test": sorted(sets["val"] & sets["test"]),
    }


def _element_budget_summary(element_counts: list[int], budgets: list[int]) -> dict[str, Any]:
    if not element_counts:
        return {"count": 0}
    elements = np.asarray(element_counts, dtype=np.float64)
    desired = np.asarray(budgets, dtype=np.float64)
    valid = desired > 0
    deviations = (elements[valid] - desired[valid]) / desired[valid] if np.any(valid) else np.asarray([])
    return {
        "count": int(elements.size),
        "min_elements": int(np.min(elements)),
        "max_elements": int(np.max(elements)),
        "mean_elements": float(np.mean(elements)),
        "mean_relative_budget_deviation": float(np.mean(deviations)) if deviations.size else None,
    }


def _passes_size_filter(count: int, minimum: Any, maximum: Any) -> bool:
    if minimum not in {None, ""} and count < int(minimum):
        return False
    if maximum not in {None, ""} and count > int(maximum):
        return False
    return True


def _issue(
    issues: list[AuditIssue],
    counts: Counter[str],
    category: str,
    sample_id: str,
    field: str,
    message: str,
) -> None:
    counts[category] += 1
    issues.append(AuditIssue(category=category, sample_id=sample_id, field=field, message=message))


def _anchor_amber_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = AMBER_REPO_ROOT / path
    return path.resolve()


def _ensure_under_anchor(path: Path, root: Path, raw: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Relative path '{raw}' escapes declared anchor '{root}'") from exc


def _is_absolute(value: str) -> bool:
    return Path(value).is_absolute() or PureWindowsPath(value).is_absolute()


def _normalized_path_text(value: str) -> str:
    return value.replace("\\", "/").rstrip("/")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PipelineDatasetAuditError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise PipelineDatasetAuditError(f"Manifest row {line_number} is not an object")
            rows.append(row)
    return rows


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, ListConfig)):
        return list(value)
    return [value]


def _optional_set(value: Any) -> set[Any] | None:
    values = _as_list(value)
    return set(values) if values else None


def _plain_container(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return dict(value)


def _iter_quality_geometry_reports(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    seen_report_ids: set[int] = set()
    candidates = [
        payload.get("geometry_reports", []),
        payload.get("scalar_smoke", {}).get("geometry_reports", []),
        payload.get("elasticity_smoke", {}).get("geometry_reports", []),
    ]
    for reports in candidates:
        if not isinstance(reports, list):
            continue
        for report in reports:
            if not isinstance(report, dict):
                continue
            report_id = id(report)
            if report_id in seen_report_ids:
                continue
            seen_report_ids.add(report_id)
            yield report
