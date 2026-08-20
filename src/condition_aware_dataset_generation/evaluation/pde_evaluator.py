from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import meshio
import numpy as np
from skfem import MeshTet
from skfem.io import from_meshio

from src.condition_aware_dataset_generation.records import ConditionRecord, GeometryPreprocessRecord
from src.condition_aware_dataset_generation.teacher_generation.pde_solvers import (
    select_boundary_facets,
    solve_condition,
)
from src.condition_aware_dataset_generation.utils import dump_json, dump_jsonl, read_jsonl


EVALUATION_PROTOCOL_VERSION = 2
MAX_EXTRAPOLATED_POINT_FRACTION = 0.10
MAX_EXTRAPOLATION_DISTANCE_RATIO = 0.03
DEFAULT_ALLOWED_STATUSES = {"success_budget_closed", "success_near_desired_budget"}
DEFAULT_ALLOWED_VERDICTS = {"PASS_STRONG", "PASS_WEAK"}
PDE_RESULT_FIELDS = (
    "protocol_version",
    "sample_id",
    "geometry_id",
    "condition_id",
    "pde_family",
    "method_id",
    "seed",
    "checkpoint",
    "dataset_fingerprint_sha256",
    "mesh_generation_success",
    "mesh_generation_status",
    "solver_success",
    "failure_category",
    "failure_reason",
    "extrapolated_point_count",
    "extrapolated_point_fraction",
    "max_extrapolation_distance",
    "max_extrapolation_distance_ratio",
    "solution_l2_absolute",
    "solution_l2_relative",
    "qoi_predicted",
    "qoi_reference",
    "qoi_absolute_error",
    "qoi_relative_error",
    "predicted_elements",
    "desired_budget",
    "budget_ratio",
    "budget_deviation",
    "reference_elements",
    "reference_strength_ratio",
    "runtime_seconds",
)


class EvaluationReferenceBuildError(RuntimeError):
    pass


class PdeEvaluationError(RuntimeError):
    pass


def build_evaluation_references(
    *,
    pipeline_output_root: str | Path,
    geometry_source_root: str | Path,
    sample_ids: set[str] | None = None,
    uniform_refinement_level: int = 1,
    reuse_audited_scalar_reference: bool = True,
    max_dofs: int | None = None,
    max_matrix_nnz: int | None = None,
) -> dict[str, Any]:
    """Build versioned strong references for retained test samples.

    The existing teacher ``reference_solution.npz`` is never overwritten.  A
    sidecar manifest is published only if every selected sample succeeds.
    """
    output_root = Path(pipeline_output_root).resolve()
    geometry_root = Path(geometry_source_root).resolve()
    if uniform_refinement_level != 1:
        raise ValueError("The frozen protocol requires uniform_refinement_level=1")
    records = _retained_test_records(output_root, sample_ids=sample_ids)
    if not records:
        raise EvaluationReferenceBuildError("No retained test samples were selected")
    reference_root = output_root / "evaluation_references" / f"v{EVALUATION_PROTOCOL_VERSION}"
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for record in records:
        sample_id = str(record["sample_id"])
        started = time.perf_counter()
        try:
            target_path = _resolve_output_path(record["final_target_mesh_path"], output_root)
            preprocess_path = _resolve_output_path(
                record["geometry_artifact_paths"]["preprocess_record_path"], output_root
            )
            target_mesh = load_tetra_mesh(target_path)
            preprocess_record = GeometryPreprocessRecord(**_load_json(preprocess_path))
            condition_record = condition_record_from_sample(record)
            destination_dir = reference_root / sample_id
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination_path = destination_dir / "reference_solution.npz"
            source_kind = "uniform_refinement_solve"
            solver_metadata: dict[str, Any]

            existing_value = record.get("optional_reference_solution_path")
            if (
                reuse_audited_scalar_reference
                and condition_record.pde_family == "scalar_elliptic"
                and existing_value
            ):
                existing_path = _resolve_output_path(existing_value, output_root)
                audit = _audit_existing_strong_reference(existing_path, target_mesh)
            else:
                existing_path = None
                audit = None

            if audit is not None:
                shutil.copy2(existing_path, destination_path)
                reference_elements = int(audit["reference_elements"])
                solver_metadata = {
                    "reused_teacher_reference": True,
                    "audit": audit,
                }
                source_kind = "audited_teacher_reference_copy"
            else:
                reference_mesh = target_mesh
                for _ in range(uniform_refinement_level):
                    reference_mesh = reference_mesh.refined()
                result = solve_condition(
                    reference_mesh,
                    preprocess_record,
                    condition_record,
                    solver_options={
                        "max_dofs": max_dofs,
                        "max_matrix_nnz": max_matrix_nnz,
                        "solver_stage_name": "evaluation_reference_solve",
                    },
                )
                np.savez(
                    destination_path,
                    points=reference_mesh.p.T,
                    connectivity=reference_mesh.t.T,
                    values=result["nodal_values"],
                )
                reference_elements = int(reference_mesh.t.shape[1])
                solver_metadata = dict(result.get("solver_metadata", {}))

            condition_sha256 = _json_sha256(record["condition_spec"])
            metadata = {
                "protocol_version": EVALUATION_PROTOCOL_VERSION,
                "sample_id": sample_id,
                "geometry_id": record.get("geometry_id"),
                "condition_id": record.get("condition_id"),
                "pde_family": record.get("pde_family"),
                "condition_sha256": condition_sha256,
                "uniform_refinement_level": uniform_refinement_level,
                "source_kind": source_kind,
                "target_mesh_sha256": _sha256_file(target_path),
                "target_elements": int(target_mesh.t.shape[1]),
                "reference_elements": reference_elements,
                "reference_strength_ratio": reference_elements / max(int(target_mesh.t.shape[1]), 1),
                "solver_metadata": solver_metadata,
                "elapsed_seconds": float(time.perf_counter() - started),
            }
            dump_json(destination_dir / "metadata.json", metadata)
            rows.append(
                {
                    **metadata,
                    "evaluation_reference_path": destination_path.relative_to(output_root).as_posix(),
                    "metadata_path": (destination_dir / "metadata.json").relative_to(output_root).as_posix(),
                    "status": "success",
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "sample_id": sample_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )

    failure_path = output_root / "reports" / "evaluation_reference_failures.json"
    if failures:
        dump_json(
            failure_path,
            {
                "protocol_version": EVALUATION_PROTOCOL_VERSION,
                "num_selected": len(records),
                "num_failures": len(failures),
                "failures": failures,
            },
        )
        raise EvaluationReferenceBuildError(
            f"Strong reference build failed for {len(failures)}/{len(records)} samples; "
            f"first failures={failures[:3]}"
        )

    failure_path.unlink(missing_ok=True)
    manifest_path = output_root / "manifests" / "evaluation_reference_manifest.jsonl"
    dump_jsonl(manifest_path, sorted(rows, key=lambda row: str(row["sample_id"])))
    return {
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "num_references": len(rows),
        "manifest_path": str(manifest_path),
        "uniform_refinement_level": uniform_refinement_level,
    }


def evaluate_prediction_manifest(
    *,
    pipeline_output_root: str | Path,
    geometry_source_root: str | Path,
    prediction_manifest_path: str | Path,
    prediction_root: str | Path | None = None,
    output_csv_path: str | Path | None = None,
    aggregate_json_path: str | Path | None = None,
    max_dofs: int | None = None,
    max_matrix_nnz: int | None = None,
    fail_on_any_error: bool = True,
) -> dict[str, Any]:
    output_root = Path(pipeline_output_root).resolve()
    _ = Path(geometry_source_root).resolve()  # Reserved second anchor; PDE solve consumes preprocess metadata.
    prediction_manifest = Path(prediction_manifest_path).resolve()
    if prediction_root:
        prediction_base = Path(prediction_root).resolve()
    elif prediction_manifest.parent.name == "test_predictions":
        prediction_base = prediction_manifest.parent.parent
    else:
        prediction_base = prediction_manifest.parent
    sample_lookup = {
        str(record["sample_id"]): record
        for record in read_jsonl(output_root / "manifests" / "sample_manifest.jsonl")
    }
    reference_lookup = {
        str(record["sample_id"]): record
        for record in read_jsonl(output_root / "manifests" / "evaluation_reference_manifest.jsonl")
    }
    predictions = _read_prediction_manifest(prediction_manifest)
    rows: list[dict[str, Any]] = []

    for prediction in predictions:
        sample_id = str(prediction.get("sample_id", ""))
        started = time.perf_counter()
        base_row = {
            "protocol_version": EVALUATION_PROTOCOL_VERSION,
            "sample_id": sample_id,
            "method_id": prediction.get("method_id"),
            "seed": prediction.get("seed"),
            "checkpoint": prediction.get("checkpoint"),
            "dataset_fingerprint_sha256": prediction.get("dataset_fingerprint_sha256"),
            "mesh_generation_success": (
                _as_bool(prediction.get("mesh_generation_success"))
                if prediction.get("mesh_generation_success") not in {None, ""}
                else None
            ),
            "mesh_generation_status": prediction.get("mesh_generation_status"),
            "solver_success": False,
            "failure_category": None,
            "failure_reason": None,
        }
        try:
            sample = sample_lookup[sample_id]
            desired_budget = int(sample.get("budget", 0) or 0)
            base_row.update(
                {
                    "geometry_id": sample.get("geometry_id"),
                    "condition_id": sample.get("condition_id"),
                    "pde_family": sample.get("pde_family"),
                    "desired_budget": desired_budget,
                }
            )
            reference_row = reference_lookup[sample_id]
            _validate_reference_row(sample, reference_row)
            prediction_path = _resolve_prediction_path(
                prediction.get("prediction_mesh_path"), prediction_base
            )
            predicted_mesh = load_tetra_mesh(prediction_path)
            predicted_elements = int(predicted_mesh.t.shape[1])
            base_row.update(
                {
                    "predicted_elements": predicted_elements,
                    "budget_ratio": predicted_elements / max(desired_budget, 1),
                    "budget_deviation": (predicted_elements - desired_budget)
                    / max(desired_budget, 1),
                }
            )
            preprocess_path = _resolve_output_path(
                sample["geometry_artifact_paths"]["preprocess_record_path"], output_root
            )
            preprocess_record = GeometryPreprocessRecord(**_load_json(preprocess_path))
            condition_record = condition_record_from_sample(sample)
            solve_result = solve_condition(
                predicted_mesh,
                preprocess_record,
                condition_record,
                solver_options={
                    "max_dofs": max_dofs,
                    "max_matrix_nnz": max_matrix_nnz,
                    "solver_stage_name": "offline_prediction_solve",
                },
            )
            reference_path = _resolve_output_path(
                reference_row["evaluation_reference_path"], output_root
            )
            with np.load(reference_path) as payload:
                reference_points = np.asarray(payload["points"], dtype=np.float64)
                reference_connectivity = np.asarray(payload["connectivity"], dtype=np.int64)
                reference_values = np.asarray(payload["values"], dtype=np.float64)
            predicted_on_reference, extrapolation = evaluate_p1_field_on_reference_points(
                mesh=predicted_mesh,
                nodal_values=solve_result["nodal_values"],
                reference_points=reference_points,
            )
            solution_metrics = volume_weighted_relative_l2(
                points=reference_points,
                connectivity=reference_connectivity,
                predicted=predicted_on_reference,
                reference=reference_values,
            )
            reference_mesh = MeshTet(reference_points.T, reference_connectivity.T)
            qoi_metrics = condition_qoi_errors(
                mesh=reference_mesh,
                preprocess_record=preprocess_record,
                condition_spec=sample["condition_spec"],
                predicted_values=predicted_on_reference,
                reference_values=reference_values,
            )
            rows.append(
                {
                    **base_row,
                    **extrapolation,
                    "solver_success": True,
                    "solution_l2_absolute": solution_metrics["absolute_l2"],
                    "solution_l2_relative": solution_metrics["relative_l2"],
                    "qoi_predicted": qoi_metrics["predicted"],
                    "qoi_reference": qoi_metrics["reference"],
                    "qoi_absolute_error": qoi_metrics["absolute_error"],
                    "qoi_relative_error": qoi_metrics["relative_error"],
                    "reference_elements": int(reference_connectivity.shape[0]),
                    "reference_strength_ratio": int(reference_connectivity.shape[0]) / max(predicted_elements, 1),
                    "runtime_seconds": float(time.perf_counter() - started),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    **base_row,
                    "solver_success": False,
                    "failure_category": type(exc).__name__,
                    "failure_reason": str(exc),
                    "runtime_seconds": float(time.perf_counter() - started),
                }
            )

    if output_csv_path is not None:
        _write_rows_csv(Path(output_csv_path), rows)
    aggregate = aggregate_pde_rows(rows)
    if aggregate_json_path is not None:
        dump_json(Path(aggregate_json_path), aggregate)
    failures = [row for row in rows if not bool(row.get("solver_success"))]
    if failures and fail_on_any_error:
        raise PdeEvaluationError(
            f"Offline PDE evaluation failed for {len(failures)}/{len(rows)} samples; "
            f"first failures={failures[:3]}"
        )
    return {"rows": rows, "aggregate": aggregate}


def evaluate_p1_field_on_reference_points(
    *,
    mesh: MeshTet,
    nodal_values: np.ndarray,
    reference_points: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Evaluate a P1 field on a fixed reference domain with bounded trace extension.

    A finer faceted CAD boundary can lie just outside a coarser prediction mesh.
    Interior points use their containing tetrahedron.  Outside points are
    projected to the nearest prediction boundary triangle and evaluated with
    that triangle's P1 barycentric weights.  Large or widespread extensions
    remain hard failures.
    """
    points = np.asarray(reference_points, dtype=np.float64)
    values = _as_component_matrix(nodal_values)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"reference_points must have shape (n, 3), got {points.shape}")
    if values.shape[0] != mesh.nvertices:
        raise ValueError(
            f"nodal_values length {values.shape[0]} != mesh vertices {mesh.nvertices}"
        )
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(values)):
        raise ValueError("reference points and nodal values must be finite")

    evaluated = np.empty((points.shape[0], values.shape[1]), dtype=np.float64)
    finder = mesh.element_finder()
    outside_indices: list[int] = []
    for point_index, point in enumerate(points):
        try:
            element_index = int(finder(*point.reshape(3, 1))[0])
        except ValueError:
            outside_indices.append(point_index)
            continue
        vertex_ids = mesh.t[:, element_index]
        tetra = mesh.p[:, vertex_ids].T
        local = np.linalg.solve((tetra[1:] - tetra[0]).T, point - tetra[0])
        barycentric = np.concatenate(([1.0 - float(np.sum(local))], local))
        evaluated[point_index] = barycentric @ values[vertex_ids]

    extrapolated_fraction = len(outside_indices) / max(points.shape[0], 1)
    if extrapolated_fraction > MAX_EXTRAPOLATED_POINT_FRACTION:
        raise ValueError(
            "Reference-domain boundary extension fraction exceeds protocol limit: "
            f"{extrapolated_fraction:.6f} > {MAX_EXTRAPOLATED_POINT_FRACTION:.6f}"
        )

    distances: list[float] = []
    if outside_indices:
        boundary_facets = mesh.boundary_facets()
        if boundary_facets.size == 0:
            raise ValueError("Prediction mesh has no boundary facets for reference-domain extension")
        boundary_vertex_ids = mesh.facets[:, boundary_facets].T
        boundary_triangles = mesh.p.T[boundary_vertex_ids]
        for point_index in outside_indices:
            triangle_index, barycentric, distance = _closest_triangle_projection(
                points[point_index], boundary_triangles
            )
            evaluated[point_index] = (
                barycentric @ values[boundary_vertex_ids[triangle_index]]
            )
            distances.append(distance)

    bbox_diagonal = float(np.linalg.norm(np.ptp(mesh.p.T, axis=0)))
    if bbox_diagonal <= 0.0 or not math.isfinite(bbox_diagonal):
        raise ValueError("Prediction mesh has an invalid bounding-box diagonal")
    max_distance = max(distances, default=0.0)
    max_distance_ratio = max_distance / bbox_diagonal
    if max_distance_ratio > MAX_EXTRAPOLATION_DISTANCE_RATIO:
        raise ValueError(
            "Reference-domain boundary extension distance ratio exceeds protocol limit: "
            f"{max_distance_ratio:.6f} > {MAX_EXTRAPOLATION_DISTANCE_RATIO:.6f}"
        )
    if not np.all(np.isfinite(evaluated)):
        raise ValueError("Reference-domain field evaluation produced non-finite values")
    return evaluated, {
        "extrapolated_point_count": len(outside_indices),
        "extrapolated_point_fraction": float(extrapolated_fraction),
        "max_extrapolation_distance": float(max_distance),
        "max_extrapolation_distance_ratio": float(max_distance_ratio),
    }


def _closest_triangle_projection(
    point: np.ndarray,
    triangles: np.ndarray,
) -> tuple[int, np.ndarray, float]:
    """Return the exact closest point encoded as triangle barycentric weights."""
    point = np.asarray(point, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.float64)
    a = triangles[:, 0]
    b = triangles[:, 1]
    c = triangles[:, 2]
    ab = b - a
    ac = c - a
    ap = point - a
    d00 = np.einsum("ij,ij->i", ab, ab)
    d01 = np.einsum("ij,ij->i", ab, ac)
    d11 = np.einsum("ij,ij->i", ac, ac)
    d20 = np.einsum("ij,ij->i", ap, ab)
    d21 = np.einsum("ij,ij->i", ap, ac)
    denominator = d00 * d11 - d01 * d01

    best_distance_squared = np.full(triangles.shape[0], np.inf, dtype=np.float64)
    best_weights = np.zeros((triangles.shape[0], 3), dtype=np.float64)

    valid_plane = np.abs(denominator) > np.finfo(np.float64).eps
    v = np.zeros_like(denominator)
    w = np.zeros_like(denominator)
    np.divide(d11 * d20 - d01 * d21, denominator, out=v, where=valid_plane)
    np.divide(d00 * d21 - d01 * d20, denominator, out=w, where=valid_plane)
    u = 1.0 - v - w
    plane_inside = valid_plane & (u >= -1.0e-12) & (v >= -1.0e-12) & (w >= -1.0e-12)
    plane_points = u[:, None] * a + v[:, None] * b + w[:, None] * c
    plane_distance_squared = np.sum((plane_points - point) ** 2, axis=1)
    best_distance_squared[plane_inside] = plane_distance_squared[plane_inside]
    best_weights[plane_inside] = np.column_stack((u, v, w))[plane_inside]

    segment_specs = (
        (a, b, (0, 1)),
        (b, c, (1, 2)),
        (c, a, (2, 0)),
    )
    for start, end, (start_index, end_index) in segment_specs:
        delta = end - start
        length_squared = np.einsum("ij,ij->i", delta, delta)
        parameter = np.zeros_like(length_squared)
        np.divide(
            np.einsum("ij,ij->i", point - start, delta),
            length_squared,
            out=parameter,
            where=length_squared > np.finfo(np.float64).eps,
        )
        parameter = np.clip(parameter, 0.0, 1.0)
        segment_points = start + parameter[:, None] * delta
        distance_squared = np.sum((segment_points - point) ** 2, axis=1)
        improves = distance_squared < best_distance_squared
        if not np.any(improves):
            continue
        weights = np.zeros_like(best_weights)
        weights[:, start_index] = 1.0 - parameter
        weights[:, end_index] = parameter
        best_distance_squared[improves] = distance_squared[improves]
        best_weights[improves] = weights[improves]

    triangle_index = int(np.argmin(best_distance_squared))
    distance_squared = float(best_distance_squared[triangle_index])
    if not math.isfinite(distance_squared):
        raise ValueError("Could not project an outside reference point to the prediction boundary")
    barycentric = np.clip(best_weights[triangle_index], 0.0, 1.0)
    barycentric /= np.sum(barycentric)
    return triangle_index, barycentric, math.sqrt(max(distance_squared, 0.0))


def volume_weighted_relative_l2(
    *,
    points: np.ndarray,
    connectivity: np.ndarray,
    predicted: np.ndarray,
    reference: np.ndarray,
    epsilon: float = 1.0e-14,
) -> dict[str, float]:
    points = np.asarray(points, dtype=np.float64)
    connectivity = np.asarray(connectivity, dtype=np.int64)
    predicted = _as_component_matrix(predicted)
    reference = _as_component_matrix(reference)
    if predicted.shape != reference.shape or predicted.shape[0] != points.shape[0]:
        raise ValueError(
            f"solution shapes must match reference points: predicted={predicted.shape}, "
            f"reference={reference.shape}, points={points.shape}"
        )
    weights = lumped_tetra_vertex_weights(points, connectivity)
    squared_error = np.sum((predicted - reference) ** 2, axis=1)
    squared_reference = np.sum(reference**2, axis=1)
    error_norm = math.sqrt(float(np.sum(weights * squared_error)))
    reference_norm = math.sqrt(float(np.sum(weights * squared_reference)))
    if reference_norm <= epsilon:
        relative = 0.0 if error_norm <= epsilon else math.inf
    else:
        relative = error_norm / reference_norm
    return {
        "absolute_l2": float(error_norm),
        "relative_l2": float(relative),
        "reference_l2": float(reference_norm),
    }


def lumped_tetra_vertex_weights(points: np.ndarray, connectivity: np.ndarray) -> np.ndarray:
    if connectivity.ndim != 2 or connectivity.shape[1] != 4:
        raise ValueError(f"Expected tetra connectivity with shape (n, 4), got {connectivity.shape}")
    tetra = points[connectivity]
    volumes = np.abs(
        np.einsum(
            "ij,ij->i",
            np.cross(tetra[:, 1] - tetra[:, 0], tetra[:, 2] - tetra[:, 0]),
            tetra[:, 3] - tetra[:, 0],
        )
    ) / 6.0
    if not np.all(np.isfinite(volumes)) or np.any(volumes <= 0.0):
        raise ValueError("Reference mesh contains non-positive or non-finite tetra volumes")
    weights = np.zeros(points.shape[0], dtype=np.float64)
    for local_index in range(4):
        np.add.at(weights, connectivity[:, local_index], volumes / 4.0)
    return weights


def condition_qoi_errors(
    *,
    mesh: MeshTet,
    preprocess_record: GeometryPreprocessRecord,
    condition_spec: dict[str, Any],
    predicted_values: np.ndarray,
    reference_values: np.ndarray,
    epsilon: float = 1.0e-14,
) -> dict[str, float]:
    qoi_spec = dict(condition_spec.get("qoi_spec", {}) or {})
    qoi_type = str(qoi_spec.get("type", ""))
    selector = qoi_spec.get("selector")
    if not selector:
        raise ValueError("condition_spec.qoi_spec.selector is required")
    facets = select_boundary_facets(mesh, preprocess_record, selector)
    if facets.size == 0:
        raise ValueError("QoI selector matched no boundary facets")
    facet_vertices = mesh.facets[:, facets].T
    facet_points = mesh.p.T[facet_vertices]
    areas = 0.5 * np.linalg.norm(
        np.cross(facet_points[:, 1] - facet_points[:, 0], facet_points[:, 2] - facet_points[:, 0]),
        axis=1,
    )
    if np.sum(areas) <= epsilon:
        raise ValueError("QoI boundary facets have zero total area")
    predicted = _as_component_matrix(predicted_values)
    reference = _as_component_matrix(reference_values)
    if qoi_type == "boundary_average":
        predicted_facet = predicted[facet_vertices, 0].mean(axis=1)
        reference_facet = reference[facet_vertices, 0].mean(axis=1)
    elif qoi_type == "mean_displacement_norm":
        predicted_facet = np.linalg.norm(predicted[facet_vertices], axis=2).mean(axis=1)
        reference_facet = np.linalg.norm(reference[facet_vertices], axis=2).mean(axis=1)
    else:
        raise ValueError(f"Unsupported QoI type '{qoi_type}'")
    predicted_qoi = float(np.sum(areas * predicted_facet) / np.sum(areas))
    reference_qoi = float(np.sum(areas * reference_facet) / np.sum(areas))
    absolute_error = abs(predicted_qoi - reference_qoi)
    relative_error = absolute_error / max(abs(reference_qoi), epsilon)
    return {
        "predicted": predicted_qoi,
        "reference": reference_qoi,
        "absolute_error": float(absolute_error),
        "relative_error": float(relative_error),
    }


def aggregate_pde_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("pde_family") or "unknown")].append(row)
    return {
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "num_samples": len(rows),
        "num_success": sum(_as_bool(row.get("solver_success")) for row in rows),
        "num_failures": sum(not _as_bool(row.get("solver_success")) for row in rows),
        "failure_rate": (
            sum(not _as_bool(row.get("solver_success")) for row in rows) / len(rows) if rows else None
        ),
        "by_pde_family": {
            family: _aggregate_successful_group(group) for family, group in sorted(groups.items())
        },
    }


def load_tetra_mesh(path: str | Path) -> MeshTet:
    mesh = meshio.read(str(path))
    tetra_blocks = [block.data for block in mesh.cells if block.type == "tetra"]
    if not tetra_blocks:
        raise ValueError(f"Mesh '{path}' contains no tetra cells")
    converted = from_meshio(
        meshio.Mesh(points=mesh.points, cells=[("tetra", np.concatenate(tetra_blocks, axis=0))])
    )
    if not isinstance(converted, MeshTet):
        raise TypeError(f"Expected MeshTet for '{path}', got {type(converted)!r}")
    return converted


def condition_record_from_sample(record: dict[str, Any]) -> ConditionRecord:
    condition_spec = dict(record["condition_spec"])
    return ConditionRecord(
        condition_id=str(record["condition_id"]),
        geometry_id=str(record["geometry_id"]),
        pde_family=str(record["pde_family"]),
        condition_index=int(record.get("condition_index", 0) or 0),
        condition_spec=condition_spec,
        budget_or_tolerance_spec=dict(
            condition_spec.get("budget_or_tolerance_spec", {"budgets": [int(record.get("budget", 0))]})
        ),
        source_name=str(record.get("source", "pipeline_manifest")),
        status="success",
    )


def _retained_test_records(output_root: Path, sample_ids: set[str] | None) -> list[dict[str, Any]]:
    records = read_jsonl(output_root / "manifests" / "sample_manifest.jsonl")
    quality_lookup = _quality_lookup(output_root / "reports" / "smoke_report.json")
    retained = []
    for record in records:
        sample_id = str(record.get("sample_id"))
        if sample_ids is not None and sample_id not in sample_ids:
            continue
        if str(record.get("split")) != "test":
            continue
        if str(record.get("status")) not in DEFAULT_ALLOWED_STATUSES:
            continue
        if quality_lookup and quality_lookup.get(sample_id) not in DEFAULT_ALLOWED_VERDICTS:
            continue
        retained.append(record)
    if sample_ids is not None:
        missing = sorted(sample_ids.difference(str(record.get("sample_id")) for record in retained))
        if missing:
            raise EvaluationReferenceBuildError(
                f"Requested sample IDs are not retained test samples: {missing[:10]}"
            )
    return sorted(retained, key=lambda record: str(record["sample_id"]))


def _audit_existing_strong_reference(path: Path, target_mesh: MeshTet) -> dict[str, Any] | None:
    try:
        with np.load(path, mmap_mode="r") as payload:
            points = payload["points"]
            connectivity = payload["connectivity"]
            values = payload["values"]
            reference_elements = int(connectivity.shape[0])
            if connectivity.ndim != 2 or connectivity.shape[1] != 4:
                return None
            if values.shape[0] != points.shape[0]:
                return None
            expected_elements = 8 * int(target_mesh.t.shape[1])
            if reference_elements != expected_elements:
                return None
            if not np.all(np.isfinite(values)):
                return None
            return {
                "reference_elements": reference_elements,
                "target_elements": int(target_mesh.t.shape[1]),
                "expected_one_level_elements": expected_elements,
                "reference_strength_ratio": reference_elements / int(target_mesh.t.shape[1]),
                "arrays_valid": True,
            }
    except Exception:
        return None


def _quality_lookup(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    payload = _load_json(path)
    lookup: dict[str, str] = {}
    candidates = [
        payload.get("geometry_reports", []),
        payload.get("scalar_smoke", {}).get("geometry_reports", []),
        payload.get("elasticity_smoke", {}).get("geometry_reports", []),
    ]
    for reports in candidates:
        for report in reports if isinstance(reports, list) else []:
            verdict = report.get("verdict")
            for metrics in report.get("sample_metrics", []):
                if metrics.get("sample_id") and verdict:
                    lookup[str(metrics["sample_id"])] = str(verdict)
    return lookup


def _read_prediction_manifest(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extras = sorted({key for row in rows for key in row}.difference(PDE_RESULT_FIELDS))
    fieldnames = [*PDE_RESULT_FIELDS, *extras]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _validate_reference_row(sample: dict[str, Any], reference: dict[str, Any]) -> None:
    expected_hash = _json_sha256(sample.get("condition_spec", {}))
    checks = {
        "sample_id": (str(reference.get("sample_id")), str(sample.get("sample_id"))),
        "condition_id": (str(reference.get("condition_id")), str(sample.get("condition_id"))),
        "pde_family": (str(reference.get("pde_family")), str(sample.get("pde_family"))),
        "condition_sha256": (str(reference.get("condition_sha256")), expected_hash),
        "status": (str(reference.get("status")), "success"),
    }
    mismatches = {
        key: {"reference": actual, "expected": expected}
        for key, (actual, expected) in checks.items()
        if actual != expected
    }
    if int(reference.get("uniform_refinement_level", -1)) != 1:
        mismatches["uniform_refinement_level"] = {
            "reference": reference.get("uniform_refinement_level"),
            "expected": 1,
        }
    if mismatches:
        raise ValueError(f"Evaluation reference metadata mismatch: {mismatches}")


def _aggregate_successful_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if _as_bool(row.get("solver_success"))]
    metrics = (
        "solution_l2_relative",
        "qoi_absolute_error",
        "qoi_relative_error",
        "predicted_elements",
        "budget_ratio",
        "extrapolated_point_fraction",
        "max_extrapolation_distance_ratio",
        "runtime_seconds",
    )
    payload = {
        "num_samples": len(rows),
        "num_success": len(successful),
        "num_failures": len(rows) - len(successful),
    }
    for metric in metrics:
        values = [float(row[metric]) for row in successful if row.get(metric) not in {None, ""}]
        payload[f"mean_{metric}"] = float(np.mean(values)) if values else None
    return payload


def _resolve_output_path(value: str | Path, output_root: Path) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = output_root / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _resolve_prediction_path(value: Any, prediction_root: Path) -> Path:
    if value in {None, ""}:
        raise ValueError("prediction_mesh_path is missing")
    path = Path(str(value))
    if not path.is_absolute():
        path = prediction_root / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _as_component_matrix(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[:, None] if array.ndim == 1 else array


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_sha256(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)
