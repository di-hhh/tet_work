from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

from src.tasks.pipeline_dataset_audit import PipelineDatasetAuditResult


FINGERPRINT_SCHEMA_VERSION = 1
HASH_CHUNK_BYTES = 1024 * 1024
VIEW_CONFIG_KEYS = (
    "manifest_name",
    "evaluation_reference_manifest_name",
    "require_evaluation_reference",
    "evaluation_reference_required_splits",
    "split_source",
    "allowed_statuses",
    "budget_filter",
    "require_single_budget",
    "pde_family_filter",
    "geometry_id_filter",
    "condition_id_filter",
    "sample_id_filter",
    "one_condition_per_geometry",
    "input_mesh_mode",
    "target_mode",
    "require_indicator",
    "quality_filter",
    "stage_field",
    "mesh_cell_type",
    "cell_type_policy",
    "min_initial_elements",
    "max_initial_elements",
    "min_target_elements",
    "max_target_elements",
    "over_limit_policy",
)


class DatasetFingerprintMismatch(ValueError):
    pass


def build_dataset_fingerprint(
    *,
    audit_result: PipelineDatasetAuditResult,
    task_config: DictConfig | dict[str, Any],
) -> dict[str, Any]:
    audit_result.raise_for_errors()
    config = _plain_container(task_config)
    unique_paths = sorted({Path(path).resolve() for path in audit_result.consumed_paths})
    file_entries = [
        {
            "path": _portable_display_path(path, audit_result),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in unique_paths
    ]
    view_config = {key: config.get(key) for key in VIEW_CONFIG_KEYS}
    identity_payload = {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "files": file_entries,
        "view_config": view_config,
        "split_counts": audit_result.split_counts,
        "geometry_ids_by_split": audit_result.geometry_ids_by_split,
        "filter_counts": audit_result.counts,
    }
    canonical = json.dumps(identity_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return {
        **identity_payload,
        "dataset_fingerprint_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def verify_dataset_fingerprint(
    *,
    frozen_payload: dict[str, Any],
    audit_result: PipelineDatasetAuditResult,
    task_config: DictConfig | dict[str, Any],
) -> dict[str, Any]:
    current = build_dataset_fingerprint(audit_result=audit_result, task_config=task_config)
    expected = str(frozen_payload.get("dataset_fingerprint_sha256", ""))
    actual = str(current["dataset_fingerprint_sha256"])
    if expected != actual:
        expected_files = {entry["path"]: entry["sha256"] for entry in frozen_payload.get("files", [])}
        current_files = {entry["path"]: entry["sha256"] for entry in current.get("files", [])}
        changed = sorted(
            path
            for path in set(expected_files) | set(current_files)
            if expected_files.get(path) != current_files.get(path)
        )
        raise DatasetFingerprintMismatch(
            f"Dataset fingerprint mismatch: expected={expected}, actual={actual}, "
            f"changed_or_missing_files={changed[:10]}"
        )
    return current


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _portable_display_path(path: Path, audit_result: PipelineDatasetAuditResult) -> str:
    output_root = Path(audit_result.pipeline_output_root).resolve()
    geometry_root = Path(audit_result.geometry_source_root).resolve()
    try:
        return f"pipeline_output/{path.relative_to(output_root).as_posix()}"
    except ValueError:
        pass
    try:
        return f"geometry_source/{path.relative_to(geometry_root).as_posix()}"
    except ValueError:
        return f"external/{path.name}"


def _plain_container(value: Any) -> dict[str, Any]:
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return dict(value)
