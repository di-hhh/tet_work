import csv
import json
from pathlib import Path

import meshio
import numpy as np
from skfem import Basis

from src.condition_aware_dataset_generation.evaluation.pde_evaluator import (
    build_evaluation_references,
    evaluate_prediction_manifest,
    volume_weighted_relative_l2,
)


def test_volume_weighted_l2_self_is_zero():
    points = _points()
    connectivity = np.array([[0, 1, 2, 3]], dtype=np.int64)
    values = np.array([[0.0], [1.0], [0.2], [0.3]])
    metrics = volume_weighted_relative_l2(
        points=points,
        connectivity=connectivity,
        predicted=values,
        reference=values,
    )
    assert metrics["absolute_l2"] == 0.0
    assert metrics["relative_l2"] == 0.0


def test_reference_build_does_not_probe_its_own_basis_nodes(case_root: Path, monkeypatch):
    output_root = _write_dataset(case_root)
    stale_failure_report = output_root / "reports" / "evaluation_reference_failures.json"
    stale_failure_report.parent.mkdir(parents=True)
    stale_failure_report.write_text("{}", encoding="utf-8")

    def reject_point_location(*_args, **_kwargs):
        raise AssertionError("basis-node values must not use geometric point location")

    monkeypatch.setattr(Basis, "probes", reject_point_location)

    summary = build_evaluation_references(
        pipeline_output_root=output_root,
        geometry_source_root=case_root,
        reuse_audited_scalar_reference=False,
    )

    assert summary["num_references"] == 2
    assert not stale_failure_report.exists()


def test_scalar_and_elasticity_reference_self_evaluation_is_deterministic(case_root: Path):
    output_root = _write_dataset(case_root)
    summary = build_evaluation_references(
        pipeline_output_root=output_root,
        geometry_source_root=case_root,
        reuse_audited_scalar_reference=False,
    )
    assert summary["num_references"] == 2

    prediction_rows = []
    for reference_row in _read_jsonl(output_root / "manifests" / "evaluation_reference_manifest.jsonl"):
        reference_path = output_root / reference_row["evaluation_reference_path"]
        with np.load(reference_path) as payload:
            mesh_path = output_root / "test_predictions" / f"{reference_row['sample_id']}.vtk"
            mesh_path.parent.mkdir(parents=True, exist_ok=True)
            meshio.write(
                mesh_path,
                meshio.Mesh(
                    points=np.asarray(payload["points"]),
                    cells=[("tetra", np.asarray(payload["connectivity"], dtype=np.int32))],
                ),
            )
        prediction_rows.append(
            {
                "sample_id": reference_row["sample_id"],
                "prediction_mesh_path": mesh_path.relative_to(output_root).as_posix(),
                "method_id": "SELF",
                "seed": 0,
                "checkpoint": "last.ckpt",
            }
        )
    prediction_manifest = output_root / "test_predictions" / "prediction_manifest.csv"
    _write_csv(prediction_manifest, prediction_rows)

    first = evaluate_prediction_manifest(
        pipeline_output_root=output_root,
        geometry_source_root=case_root,
        prediction_manifest_path=prediction_manifest,
        fail_on_any_error=True,
    )
    second = evaluate_prediction_manifest(
        pipeline_output_root=output_root,
        geometry_source_root=case_root,
        prediction_manifest_path=prediction_manifest,
        fail_on_any_error=True,
    )

    assert len(first["rows"]) == 2
    assert all(row["solver_success"] for row in first["rows"])
    assert all(row["solution_l2_relative"] < 1.0e-10 for row in first["rows"])
    assert all(row["qoi_relative_error"] < 1.0e-10 for row in first["rows"])
    assert [row["solution_l2_relative"] for row in first["rows"]] == [
        row["solution_l2_relative"] for row in second["rows"]
    ]
    assert {row["pde_family"] for row in first["rows"]} == {
        "scalar_elliptic",
        "linear_elasticity",
    }


def test_scalar_coarse_prediction_has_more_error_than_reference_self(case_root: Path):
    output_root = _write_dataset(case_root, families=("scalar_elliptic",))
    build_evaluation_references(
        pipeline_output_root=output_root,
        geometry_source_root=case_root,
        reuse_audited_scalar_reference=False,
    )
    manifest = output_root / "test_predictions" / "prediction_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(
        manifest,
        [
            {
                "sample_id": "sample_scalar",
                "prediction_mesh_path": "target.vtk",
                "method_id": "COARSE",
                "seed": 0,
                "checkpoint": "last.ckpt",
            }
        ],
    )

    result = evaluate_prediction_manifest(
        pipeline_output_root=output_root,
        geometry_source_root=case_root,
        prediction_manifest_path=manifest,
        fail_on_any_error=True,
    )

    assert result["rows"][0]["solution_l2_relative"] > 1.0e-8


def test_solver_or_mesh_failure_is_recorded_without_fake_metrics(case_root: Path):
    output_root = _write_dataset(case_root, families=("scalar_elliptic",))
    build_evaluation_references(
        pipeline_output_root=output_root,
        geometry_source_root=case_root,
        reuse_audited_scalar_reference=False,
    )
    invalid_mesh = output_root / "predictions" / "surface_only.vtk"
    invalid_mesh.parent.mkdir(parents=True)
    meshio.write(
        invalid_mesh,
        meshio.Mesh(points=_points(), cells=[("triangle", np.array([[0, 1, 2]], dtype=np.int32))]),
    )
    manifest = output_root / "predictions" / "invalid.csv"
    result_csv = output_root / "predictions" / "failed_pde_metrics.csv"
    _write_csv(
        manifest,
        [{"sample_id": "sample_scalar", "prediction_mesh_path": invalid_mesh.name}],
    )
    result = evaluate_prediction_manifest(
        pipeline_output_root=output_root,
        geometry_source_root=case_root,
        prediction_manifest_path=manifest,
        prediction_root=invalid_mesh.parent,
        output_csv_path=result_csv,
        fail_on_any_error=False,
    )
    row = result["rows"][0]
    assert row["solver_success"] is False
    assert row["failure_category"]
    assert "solution_l2_relative" not in row
    header = result_csv.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "solution_l2_relative" in header
    assert "qoi_relative_error" in header


def test_reference_condition_mismatch_is_a_recorded_failure(case_root: Path):
    output_root = _write_dataset(case_root, families=("scalar_elliptic",))
    build_evaluation_references(
        pipeline_output_root=output_root,
        geometry_source_root=case_root,
        reuse_audited_scalar_reference=False,
    )
    reference_manifest = output_root / "manifests" / "evaluation_reference_manifest.jsonl"
    reference_rows = _read_jsonl(reference_manifest)
    reference_rows[0]["condition_sha256"] = "stale-reference"
    reference_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in reference_rows),
        encoding="utf-8",
    )
    prediction_manifest = output_root / "test_predictions" / "prediction_manifest.csv"
    prediction_manifest.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(
        prediction_manifest,
        [{"sample_id": "sample_scalar", "prediction_mesh_path": "target.vtk"}],
    )

    result = evaluate_prediction_manifest(
        pipeline_output_root=output_root,
        geometry_source_root=case_root,
        prediction_manifest_path=prediction_manifest,
        fail_on_any_error=False,
    )

    assert result["rows"][0]["failure_category"] == "ValueError"
    assert "condition_sha256" in result["rows"][0]["failure_reason"]


def _write_dataset(tmp_path: Path, families=("scalar_elliptic", "linear_elasticity")) -> Path:
    output_root = tmp_path / "pipeline_output"
    (output_root / "manifests").mkdir(parents=True)
    (output_root / "geometries" / "g0").mkdir(parents=True)
    target_path = output_root / "target.vtk"
    meshio.write(
        target_path,
        meshio.Mesh(points=_points(), cells=[("tetra", np.array([[0, 1, 2, 3]], dtype=np.int32))]),
    )
    preprocess_path = output_root / "geometries" / "g0" / "preprocess_record.json"
    preprocess = {
        "geometry_id": "g0",
        "source_path": str(tmp_path / "source.step"),
        "dimension": 3,
        "bounding_box": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        "centroid": [0.0, 0.0, 0.0],
        "principal_axes": np.eye(3).tolist(),
        "oriented_bbox_min": [0.0, 0.0, 0.0],
        "oriented_bbox_max": [1.0, 1.0, 1.0],
        "boundary_patches": [],
        "validation": {},
        "coarse_mesh_path": target_path.relative_to(output_root).as_posix(),
        "coarse_mesh_num_vertices": 4,
        "coarse_mesh_num_elements": 1,
        "status": "success",
        "geometry_feature_metadata_path": None,
        "geometry_features": {},
    }
    preprocess_path.write_text(json.dumps(preprocess), encoding="utf-8")

    records = []
    for family in families:
        suffix = "scalar" if family == "scalar_elliptic" else "elasticity"
        records.append(
            {
                "sample_id": f"sample_{suffix}",
                "geometry_id": "g0",
                "condition_id": f"condition_{suffix}",
                "pde_family": family,
                "budget": 8,
                "condition_spec": _condition_spec(family),
                "geometry_artifact_paths": {
                    "preprocess_record_path": preprocess_path.relative_to(output_root).as_posix(),
                    "source_path": "source.step",
                },
                "initial_mesh_path": target_path.relative_to(output_root).as_posix(),
                "final_target_mesh_path": target_path.relative_to(output_root).as_posix(),
                "optional_reference_solution_path": None,
                "optional_error_indicator_path": None,
                "split": "test",
                "status": "success_budget_closed",
            }
        )
    with (output_root / "manifests" / "sample_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return output_root


def _condition_spec(family: str) -> dict:
    selector_min = {"axis_index": 0, "side": "min", "band_fraction": 0.25}
    selector_max = {"axis_index": 0, "side": "max", "band_fraction": 0.25}
    if family == "scalar_elliptic":
        return {
            "pde_family": family,
            "boundary_role_spec": [
                {"role": "dirichlet_low", "selector": selector_min, "value": 0.0},
                {"role": "dirichlet_high", "selector": selector_max, "value": 1.0},
            ],
            "coefficient_spec": {"diffusion": 1.0},
            "source_or_load_spec": {
                "internal_source": {
                    "type": "gaussian",
                    "center_local": [0.25, 0.25, 0.25],
                    "sigma": [0.2, 0.2, 0.2],
                    "amplitude": 0.2,
                }
            },
            "budget_or_tolerance_spec": {"budgets": [8]},
            "qoi_spec": {"type": "boundary_average", "selector": selector_max},
        }
    return {
        "pde_family": family,
        "boundary_role_spec": [
            {"role": "support", "selector": selector_min, "components": [0, 1, 2], "value": 0.0},
            {"role": "traction", "selector": selector_max, "vector": [1.0, 0.0, 0.0]},
        ],
        "coefficient_spec": {
            "youngs_modulus": 100.0,
            "poissons_ratio": 0.3,
            "constitutive_model": "linear_elasticity",
        },
        "source_or_load_spec": {"body_force": [0.0, 0.0, 0.0]},
        "budget_or_tolerance_spec": {"budgets": [8]},
        "qoi_spec": {"type": "mean_displacement_norm", "selector": selector_max},
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _points() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
