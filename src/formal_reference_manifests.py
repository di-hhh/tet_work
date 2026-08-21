from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import meshio


REFERENCE_DEFINITIONS = {
    "R0-Initial": {
        "path_key": "input_mesh_path",
        "role": "low_budget_sanity_reference",
        "claim_boundary": "Not a budget-matched uniform baseline.",
    },
    "R1-Teacher": {
        "path_key": "target_mesh_path",
        "role": "frozen_teacher_target_reference",
        "claim_boundary": "Reference/oracle mesh, not a strict performance upper bound.",
    },
}


def build_static_reference_manifests(
    *,
    audit_result,
    output_dir: str | Path,
    protocol_id: str,
    dataset_fingerprint_sha256: str,
    manifest_sha256: str | None,
    code_versions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    test_records = sorted(
        audit_result.records_by_split.get("test", []),
        key=lambda row: str(row.get("sample_id")),
    )
    outputs: dict[str, Any] = {}
    for analysis_id, definition in REFERENCE_DEFINITIONS.items():
        rows = [
            _reference_row(
                record=record,
                analysis_id=analysis_id,
                definition=definition,
                protocol_id=protocol_id,
                dataset_fingerprint_sha256=dataset_fingerprint_sha256,
                manifest_sha256=manifest_sha256,
                code_versions=code_versions,
            )
            for record in test_records
        ]
        path = destination / f"{analysis_id.lower().replace('-', '_')}_prediction_manifest.csv"
        _write_rows(path, rows)
        outputs[analysis_id] = {
            **definition,
            "num_samples": len(rows),
            "prediction_manifest": str(path),
        }
    metadata_path = destination / "reference_manifest_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "protocol_id": protocol_id,
                "dataset_fingerprint_sha256": dataset_fingerprint_sha256,
                "references": outputs,
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {"metadata_path": str(metadata_path), "references": outputs}


def _reference_row(
    *,
    record: dict[str, Any],
    analysis_id: str,
    definition: dict[str, str],
    protocol_id: str,
    dataset_fingerprint_sha256: str,
    manifest_sha256: str | None,
    code_versions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    mesh_path = Path(record["_resolved_paths"][definition["path_key"]]).resolve()
    points, num_elements = _tetra_counts(mesh_path)
    desired_budget = int(record.get("budget", 0) or 0)
    budget_ratio = num_elements / max(desired_budget, 1)
    absolute_deviation = abs(num_elements - desired_budget)
    return {
        "run_id": f"{protocol_id}:{analysis_id}:static",
        "protocol_id": protocol_id,
        "method_id": analysis_id.split("-", 1)[0],
        "analysis_id": analysis_id,
        "method_role": "reference",
        "reference_role": definition["role"],
        "claim_boundary": definition["claim_boundary"],
        "seed": -1,
        "checkpoint": "static_reference",
        "sample_id": record.get("sample_id"),
        "geometry_id": record.get("geometry_id"),
        "condition_id": record.get("condition_id"),
        "pde_family": record.get("pde_family"),
        "split": "test",
        "desired_budget": desired_budget,
        "prediction_mesh_path": str(mesh_path),
        "evaluation_variant": "final",
        "mesh_generation_success": True,
        "mesh_generation_status": "static_reference",
        "predicted_elements": num_elements,
        "predicted_vertices": points,
        "budget_ratio": budget_ratio,
        "absolute_budget_deviation": absolute_deviation,
        "absolute_budget_relative_deviation": absolute_deviation / max(desired_budget, 1),
        "budget_close": absolute_deviation / max(desired_budget, 1) <= 0.18,
        "budget_valid": 0.8 <= budget_ratio <= 11000.0 / 7000.0,
        "dataset_fingerprint_sha256": dataset_fingerprint_sha256,
        "manifest_sha256": manifest_sha256,
        "amber_code_commit": code_versions.get("amber", {}).get("commit"),
        "pipeline_code_commit": code_versions.get("dataest-pipeline", {}).get("commit"),
    }


def _tetra_counts(path: Path) -> tuple[int, int]:
    mesh = meshio.read(path)
    num_elements = sum(len(block.data) for block in mesh.cells if block.type == "tetra")
    if num_elements <= 0:
        raise ValueError(f"Static reference mesh contains no tetra cells: {path}")
    return int(len(mesh.points)), int(num_elements)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No retained test rows were available for static references")
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
