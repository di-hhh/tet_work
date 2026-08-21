from __future__ import annotations

import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import meshio
import numpy as np
from skfem import MeshTet
from skfem.io import from_meshio

from src.formal_experiment_plan import FormalPlanError, iter_run_specs, resolve_amber_path
from src.tasks.domains.extended_mesh_tet1 import ExtendedMeshTet1
from src.tasks.domains.mesh_wrapper import MeshWrapper


PDE_METRICS = (
    "solution_l2_relative",
    "qoi_absolute_error",
    "qoi_relative_error",
)
SIZING_METRICS = (
    "projected_l2_error_symmetric",
    "physics_weighted_projected_l2_error",
    "weighted_size_mse",
    "topk_high_importance_mse",
    "bucket_low_size_mse",
    "bucket_high_size_mse",
)
BUDGET_METRICS = (
    "predicted_elements",
    "predicted_vertices",
    "budget_ratio",
    "absolute_budget_deviation",
    "absolute_budget_relative_deviation",
)
QUALITY_METRICS = (
    "tetra_quality_min",
    "tetra_quality_q05",
    "tetra_quality_median",
    "tetra_invalid_count",
    "tetra_inverted_count",
)
TIMING_METRICS = (
    "inference_graph_preparation_seconds",
    "inference_gnn_forward_seconds",
    "inference_mesh_generation_seconds",
    "inference_end_to_end_inference_seconds",
    "solver_runtime_seconds",
    "runtime_seconds",
)
MECHANISM_METRICS = (
    "gate_mean",
    "gate_std",
    "gate_min",
    "gate_max",
    "gate_high_importance_mean",
    "gate_low_importance_mean",
    "applied_correction_abs_mean",
)
ALL_NUMERIC_METRICS = (
    *PDE_METRICS,
    *SIZING_METRICS,
    *BUDGET_METRICS,
    *QUALITY_METRICS,
    *TIMING_METRICS,
    *MECHANISM_METRICS,
)
CONTRAST_METRICS = (
    "solution_l2_relative",
    "worst_condition_solution_l2_relative",
    "qoi_absolute_error",
    "qoi_relative_error",
    "projected_l2_error_symmetric",
    "physics_weighted_projected_l2_error",
    "weighted_size_mse",
    "topk_high_importance_mse",
    "absolute_budget_relative_deviation",
    "joint_success_rate",
)


def analyze_formal_protocol(*, plan: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    rows = collect_protocol_rows(plan)
    _write_csv(destination / "unified_per_sample_metrics.csv", rows)

    geometry_rows = aggregate_geometry_rows(rows)
    _write_csv(destination / "geometry_metrics.csv", geometry_rows)
    seed_rows = aggregate_seed_rows(geometry_rows)
    _write_csv(destination / "seed_summary.csv", seed_rows)
    method_seed_rows = aggregate_method_seed_rows(seed_rows)
    _write_csv(destination / "method_seed_summary.csv", method_seed_rows)

    contrast_rows = compute_all_contrasts(plan=plan, geometry_rows=geometry_rows)
    _write_csv(destination / "paired_geometry_contrasts.csv", contrast_rows)

    responsiveness = compute_condition_responsiveness(plan=plan)
    _write_csv(destination / "condition_responsiveness_pairs.csv", responsiveness["pairs"])
    _write_csv(destination / "condition_responsiveness_geometry.csv", responsiveness["geometries"])

    summary = {
        "protocol_id": plan["protocol_id"],
        "dataset_fingerprint_sha256": plan["dataset"]["fingerprint_sha256"],
        "reference_wording": plan["metrics"]["reference_wording"],
        "primary_pde_metric": plan["metrics"]["primary_pde_metric"],
        "energy_or_h1_metric": None,
        "legacy_sizing_l2_columns_are_mse": True,
        "independent_unit": "geometry_id",
        "failure_policy": "all rows retained; primary PDE values require joint success for every condition in a geometry group",
        "num_sample_rows": len(rows),
        "num_geometry_rows": len(geometry_rows),
        "num_method_seed_summary_rows": len(method_seed_rows),
        "num_contrast_rows": len(contrast_rows),
        "condition_responsiveness": responsiveness["summary"],
    }
    (destination / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def collect_protocol_rows(plan: dict[str, Any]) -> list[dict[str, Any]]:
    expected_samples = int(plan["dataset"]["expected_split_samples"]["test"])
    expected_fingerprint = str(plan["dataset"]["fingerprint_sha256"])
    rows: list[dict[str, Any]] = []
    diagnostics = set(plan.get("diagnostics", {}).get("same_geometry_condition_shuffle", []))
    for spec in iter_run_specs(plan):
        rows.extend(
            _merge_metric_files(
                amber_path=spec.run_root / "per_sample_metrics.csv",
                pde_path=spec.run_root / "pde_metrics.csv",
                analysis_id=spec.analysis_id,
                expected_samples=expected_samples,
                expected_fingerprint=expected_fingerprint,
                evaluation_variant="final",
            )
        )
        if spec.analysis_id == "M4":
            rows.extend(
                _merge_metric_files(
                    amber_path=spec.run_root / "expert_only_per_sample_metrics.csv",
                    pde_path=spec.run_root / "expert_only_pde_metrics.csv",
                    analysis_id="M4-EXPERT",
                    expected_samples=expected_samples,
                    expected_fingerprint=expected_fingerprint,
                    evaluation_variant="expert_only",
                )
            )
        if spec.analysis_id in diagnostics:
            diagnostic = spec.run_root / "diagnostics" / "condition_shuffle"
            rows.extend(
                _merge_metric_files(
                    amber_path=diagnostic / "per_sample_metrics.csv",
                    pde_path=diagnostic / "pde_metrics.csv",
                    analysis_id=f"{spec.analysis_id}-SHUFFLED",
                    expected_samples=expected_samples,
                    expected_fingerprint=expected_fingerprint,
                    evaluation_variant="condition_shuffle",
                )
            )

    reference_dir = resolve_amber_path(plan["execution"]["formal_root"]) / "static_references"
    for analysis_id, stem in (("R0-Initial", "r0_initial"), ("R1-Teacher", "r1_teacher")):
        pde_rows = _read_csv(reference_dir / f"{stem}_pde_metrics.csv")
        if len(pde_rows) != expected_samples:
            raise FormalPlanError(
                f"{analysis_id} PDE rows={len(pde_rows)}, expected={expected_samples}"
            )
        for row in pde_rows:
            normalized = _normalize_sample_row(row)
            normalized["analysis_id"] = analysis_id
            normalized["evaluation_variant"] = "final"
            _validate_sample_identity(normalized, expected_fingerprint=expected_fingerprint)
            rows.append(normalized)
    return rows


def _merge_metric_files(
    *,
    amber_path: Path,
    pde_path: Path,
    analysis_id: str,
    expected_samples: int,
    expected_fingerprint: str,
    evaluation_variant: str,
) -> list[dict[str, Any]]:
    amber_rows = _unique_by_sample(_read_csv(amber_path), path=amber_path)
    pde_rows = _unique_by_sample(_read_csv(pde_path), path=pde_path)
    if set(amber_rows) != set(pde_rows):
        raise FormalPlanError(
            f"AMBER/PDE sample sets differ for {analysis_id}: "
            f"amber_only={sorted(set(amber_rows).difference(pde_rows))[:5]}, "
            f"pde_only={sorted(set(pde_rows).difference(amber_rows))[:5]}"
        )
    if len(amber_rows) != expected_samples:
        raise FormalPlanError(
            f"{analysis_id} has {len(amber_rows)} test samples; expected {expected_samples}"
        )
    merged = []
    for sample_id in sorted(amber_rows):
        row = {**amber_rows[sample_id], **pde_rows[sample_id]}
        row["analysis_id"] = analysis_id
        row["evaluation_variant"] = evaluation_variant
        normalized = _normalize_sample_row(row)
        _validate_sample_identity(normalized, expected_fingerprint=expected_fingerprint)
        merged.append(normalized)
    return merged


def _normalize_sample_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    for metric in (*SIZING_METRICS, *QUALITY_METRICS, *MECHANISM_METRICS):
        for last_key in (f"last/{metric}", f"last_{metric}"):
            if normalized.get(metric) in {None, ""} and normalized.get(last_key) not in {None, ""}:
                normalized[metric] = normalized[last_key]
                break
    normalized["mesh_generation_success"] = _as_bool(normalized.get("mesh_generation_success"))
    normalized["solver_success"] = _as_bool(normalized.get("solver_success"))
    normalized["joint_success"] = (
        normalized["mesh_generation_success"] and normalized["solver_success"]
    )
    normalized["budget_close"] = _as_bool(normalized.get("budget_close"))
    normalized["budget_valid"] = _as_bool(normalized.get("budget_valid"))
    normalized["failure_row_retained"] = True
    for metric in ALL_NUMERIC_METRICS:
        normalized[metric] = _as_float(normalized.get(metric))
    return normalized


def _validate_sample_identity(row: dict[str, Any], *, expected_fingerprint: str) -> None:
    if str(row.get("dataset_fingerprint_sha256")) != expected_fingerprint:
        raise FormalPlanError(
            f"Dataset fingerprint mismatch in row {row.get('sample_id')}: "
            f"{row.get('dataset_fingerprint_sha256')}"
        )
    if str(row.get("split", "test")) != "test":
        raise FormalPlanError(f"Non-test row in formal evaluation: {row.get('sample_id')}")


def aggregate_geometry_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base_groups: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("analysis_id")),
            _as_int(row.get("seed"), default=-1),
            str(row.get("pde_family")),
            str(row.get("geometry_id")),
        )
        base_groups[key].append(row)

    output = [
        _aggregate_one_geometry(key=key, rows=group)
        for key, group in sorted(base_groups.items())
    ]

    joint_groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("analysis_id")),
            _as_int(row.get("seed"), default=-1),
            str(row.get("geometry_id")),
        )
        joint_groups[key].append(row)
    for (analysis_id, seed, geometry_id), group in sorted(joint_groups.items()):
        output.append(
            _aggregate_one_geometry(
                key=(analysis_id, seed, "joint", geometry_id),
                rows=group,
            )
        )
    return output


def _aggregate_one_geometry(
    *,
    key: tuple[str, int, str, str],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    analysis_id, seed, pde_family, geometry_id = key
    mesh_success = [_as_bool(row.get("mesh_generation_success")) for row in rows]
    joint_success = [_as_bool(row.get("joint_success")) for row in rows]
    payload: dict[str, Any] = {
        "analysis_id": analysis_id,
        "seed": seed,
        "pde_family": pde_family,
        "geometry_id": geometry_id,
        "num_conditions": len(rows),
        "mesh_success_rate": float(np.mean(mesh_success)) if rows else None,
        "mesh_success_all": bool(all(mesh_success)),
        "joint_success_rate": float(np.mean(joint_success)) if rows else None,
        "joint_success_all": bool(all(joint_success)),
    }
    for metric in ALL_NUMERIC_METRICS:
        if metric in PDE_METRICS and not payload["joint_success_all"]:
            payload[metric] = None
            continue
        if metric in (*SIZING_METRICS, *QUALITY_METRICS) and not payload["mesh_success_all"]:
            payload[metric] = None
            continue
        values = [_as_float(row.get(metric)) for row in rows]
        finite = [value for value in values if value is not None and np.isfinite(value)]
        payload[metric] = float(np.mean(finite)) if len(finite) == len(rows) and finite else None
    solution_values = [_as_float(row.get("solution_l2_relative")) for row in rows]
    finite_solution = [value for value in solution_values if value is not None and np.isfinite(value)]
    payload["worst_condition_solution_l2_relative"] = (
        float(max(finite_solution))
        if payload["joint_success_all"] and len(finite_solution) == len(rows) and finite_solution
        else None
    )
    return payload


def aggregate_seed_rows(geometry_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in geometry_rows:
        groups[(str(row["analysis_id"]), int(row["seed"]), str(row["pde_family"]))].append(row)
    output = []
    for (analysis_id, seed, pde_family), group in sorted(groups.items()):
        payload: dict[str, Any] = {
            "analysis_id": analysis_id,
            "seed": seed,
            "pde_family": pde_family,
            "num_geometries": len(group),
            "geometry_joint_success_rate": float(
                np.mean([float(row["joint_success_all"]) for row in group])
            ),
            "geometry_mesh_success_rate": float(
                np.mean([float(row["mesh_success_all"]) for row in group])
            ),
        }
        for metric in (*PDE_METRICS, *SIZING_METRICS, *BUDGET_METRICS):
            values = [_as_float(row.get(metric)) for row in group]
            finite = [value for value in values if value is not None and np.isfinite(value)]
            payload[f"valid_geometries_{metric}"] = len(finite)
            payload[f"mean_{metric}"] = float(np.mean(finite)) if finite else None
            payload[f"median_{metric}"] = float(np.median(finite)) if finite else None
        output.append(payload)
    return output


def aggregate_method_seed_rows(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-seed geometry summaries as method-level seed mean ± sample std."""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        groups[(str(row["analysis_id"]), str(row["pde_family"]))].append(row)

    source_metrics = (
        "geometry_joint_success_rate",
        "geometry_mesh_success_rate",
        *(f"mean_{metric}" for metric in (*PDE_METRICS, *SIZING_METRICS, *BUDGET_METRICS)),
    )
    output: list[dict[str, Any]] = []
    for (analysis_id, pde_family), group in sorted(groups.items()):
        payload: dict[str, Any] = {
            "analysis_id": analysis_id,
            "pde_family": pde_family,
            "num_seeds_total": len(group),
            "seeds": ",".join(str(int(row["seed"])) for row in sorted(group, key=lambda row: int(row["seed"]))),
        }
        for metric in source_metrics:
            values = [_as_float(row.get(metric)) for row in group]
            finite = [value for value in values if value is not None and np.isfinite(value)]
            payload[f"num_valid_seeds_{metric}"] = len(finite)
            payload[f"seed_mean_{metric}"] = float(np.mean(finite)) if finite else None
            payload[f"seed_std_{metric}"] = (
                float(np.std(finite, ddof=1))
                if len(finite) > 1
                else 0.0 if len(finite) == 1 else None
            )
        output.append(payload)
    return output


def compute_all_contrasts(
    *,
    plan: dict[str, Any],
    geometry_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    contrast_pairs = [
        (str(row["left"]), str(row["right"]), str(row["claim"]))
        for row in plan["statistics"]["planned_contrasts"]
    ]
    contrast_pairs.extend(
        [
            ("M4", "M4-EXPERT", "final_vs_same_checkpoint_expert_only"),
            ("M2", "M2-SHUFFLED", "correct_vs_same_geometry_shuffled_stage_field"),
            ("M4", "M4-SHUFFLED", "correct_vs_same_geometry_shuffled_stage_field"),
        ]
    )
    output = []
    for left, right, claim in contrast_pairs:
        for pde_family in ("scalar_elliptic", "linear_elasticity", "joint"):
            for metric in CONTRAST_METRICS:
                output.append(
                    compute_paired_geometry_contrast(
                        geometry_rows=geometry_rows,
                        left=left,
                        right=right,
                        claim=claim,
                        pde_family=pde_family,
                        metric=metric,
                        bootstrap_iterations=int(plan["statistics"]["bootstrap_iterations"]),
                        bootstrap_seed=int(plan["statistics"]["bootstrap_seed"]),
                        confidence_level=float(plan["statistics"]["confidence_level"]),
                    )
                )
    _apply_holm_correction(output)
    return output


def compute_paired_geometry_contrast(
    *,
    geometry_rows: list[dict[str, Any]],
    left: str,
    right: str,
    claim: str,
    pde_family: str,
    metric: str,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    index = {
        (str(row["analysis_id"]), int(row["seed"]), str(row["pde_family"]), str(row["geometry_id"])): row
        for row in geometry_rows
    }
    left_keys = {(seed, geometry) for analysis, seed, family, geometry in index if analysis == left and family == pde_family}
    right_keys = {(seed, geometry) for analysis, seed, family, geometry in index if analysis == right and family == pde_family}
    candidate_keys = sorted(left_keys.intersection(right_keys))
    common_seeds = sorted({seed for seed, _ in candidate_keys})
    geometries = sorted({geometry for _, geometry in candidate_keys})
    by_geometry: dict[str, list[float]] = defaultdict(list)
    seed_differences: dict[int, list[float]] = defaultdict(list)
    missing_pairs = 0
    for seed, geometry in candidate_keys:
        left_value = _as_float(index[(left, seed, pde_family, geometry)].get(metric))
        right_value = _as_float(index[(right, seed, pde_family, geometry)].get(metric))
        if left_value is None or right_value is None or not np.isfinite(left_value) or not np.isfinite(right_value):
            missing_pairs += 1
            continue
        difference = left_value - right_value
        by_geometry[geometry].append(difference)
        seed_differences[seed].append(difference)

    complete_geometry_differences = np.asarray(
        [
            np.mean(by_geometry[geometry])
            for geometry in geometries
            if len(by_geometry[geometry]) == len(common_seeds) and common_seeds
        ],
        dtype=np.float64,
    )
    lower_is_better = metric != "joint_success_rate"
    if complete_geometry_differences.size:
        ci_low, ci_high = paired_geometry_bootstrap_ci(
            complete_geometry_differences,
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
            confidence_level=confidence_level,
        )
        win_rate = float(
            np.mean(complete_geometry_differences < 0.0)
            if lower_is_better
            else np.mean(complete_geometry_differences > 0.0)
        )
        wilcoxon_p = _wilcoxon_pvalue(complete_geometry_differences)
    else:
        ci_low, ci_high, win_rate, wilcoxon_p = None, None, None, None
    per_seed_means = [
        float(np.mean(values)) for _, values in sorted(seed_differences.items()) if values
    ]
    return {
        "left": left,
        "right": right,
        "claim": claim,
        "pde_family": pde_family,
        "metric": metric,
        "direction": "lower_is_better" if lower_is_better else "higher_is_better",
        "candidate_seed_geometry_pairs": len(candidate_keys),
        "missing_seed_geometry_pairs": missing_pairs,
        "candidate_geometries": len(geometries),
        "paired_complete_geometries": int(complete_geometry_differences.size),
        "mean_paired_difference": (
            float(np.mean(complete_geometry_differences)) if complete_geometry_differences.size else None
        ),
        "median_paired_difference": (
            float(np.median(complete_geometry_differences)) if complete_geometry_differences.size else None
        ),
        "worst_geometry_difference": (
            float(
                np.max(complete_geometry_differences)
                if lower_is_better
                else np.min(complete_geometry_differences)
            )
            if complete_geometry_differences.size
            else None
        ),
        "geometry_win_rate": win_rate,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "seed_mean_difference": float(np.mean(per_seed_means)) if per_seed_means else None,
        "seed_std_difference": float(np.std(per_seed_means, ddof=1)) if len(per_seed_means) > 1 else 0.0 if per_seed_means else None,
        "wilcoxon_p_raw": wilcoxon_p,
        "wilcoxon_p_holm": None,
    }


def paired_geometry_bootstrap_ci(
    differences: np.ndarray,
    *,
    iterations: int,
    seed: int,
    confidence_level: float,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("paired bootstrap requires at least one geometry")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, values.size, size=(iterations, values.size))
    means = values[indices].mean(axis=1)
    alpha = 1.0 - confidence_level
    return float(np.quantile(means, alpha / 2.0)), float(np.quantile(means, 1.0 - alpha / 2.0))


def compute_condition_responsiveness(plan: dict[str, Any]) -> dict[str, Any]:
    methods = set(plan.get("diagnostics", {}).get("condition_responsiveness", []))
    reference_dir = resolve_amber_path(plan["execution"]["formal_root"]) / "static_references"
    teacher_rows = _unique_by_sample(
        _read_csv(reference_dir / "r1_teacher_prediction_manifest.csv"),
        path=reference_dir / "r1_teacher_prediction_manifest.csv",
    )
    teacher_mesh_cache: dict[str, MeshWrapper] = {}
    teacher_pair_cache: dict[tuple[str, str], float] = {}
    pair_rows: list[dict[str, Any]] = []
    for spec in iter_run_specs(plan):
        if spec.analysis_id not in methods:
            continue
        manifest_path = spec.run_root / "test_predictions" / "prediction_manifest.csv"
        predictions = _read_csv(manifest_path)
        by_geometry: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in predictions:
            by_geometry[str(row.get("geometry_id"))].append(row)
        prediction_mesh_cache: dict[str, MeshWrapper] = {}
        for geometry_id, group in sorted(by_geometry.items()):
            for left, right in combinations(sorted(group, key=lambda row: str(row["sample_id"])), 2):
                left_id, right_id = str(left["sample_id"]), str(right["sample_id"])
                predicted_left = _cached_mesh(
                    prediction_mesh_cache,
                    left_id,
                    _resolve_prediction_path(left, manifest_path),
                )
                predicted_right = _cached_mesh(
                    prediction_mesh_cache,
                    right_id,
                    _resolve_prediction_path(right, manifest_path),
                )
                predicted_diversity = _symmetric_projected_sizing_error(
                    predicted_left,
                    predicted_right,
                )
                pair_key = tuple(sorted((left_id, right_id)))
                if pair_key not in teacher_pair_cache:
                    teacher_left = _cached_mesh(
                        teacher_mesh_cache,
                        left_id,
                        Path(str(teacher_rows[left_id]["prediction_mesh_path"])),
                    )
                    teacher_right = _cached_mesh(
                        teacher_mesh_cache,
                        right_id,
                        Path(str(teacher_rows[right_id]["prediction_mesh_path"])),
                    )
                    teacher_pair_cache[pair_key] = _symmetric_projected_sizing_error(
                        teacher_left,
                        teacher_right,
                    )
                teacher_diversity = teacher_pair_cache[pair_key]
                pair_rows.append(
                    {
                        "analysis_id": spec.analysis_id,
                        "seed": spec.seed,
                        "geometry_id": geometry_id,
                        "left_sample_id": left_id,
                        "right_sample_id": right_id,
                        "left_pde_family": left.get("pde_family"),
                        "right_pde_family": right.get("pde_family"),
                        "predicted_pair_diversity": predicted_diversity,
                        "teacher_pair_diversity": teacher_diversity,
                        "diversity_gap": predicted_diversity - teacher_diversity,
                        "responsiveness_ratio": (
                            predicted_diversity / teacher_diversity
                            if teacher_diversity > 1.0e-12
                            else None
                        ),
                    }
                )

    geometry_groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        geometry_groups[(str(row["analysis_id"]), int(row["seed"]), str(row["geometry_id"]))].append(row)
    geometry_rows = []
    for (analysis_id, seed, geometry_id), group in sorted(geometry_groups.items()):
        predicted = np.asarray([float(row["predicted_pair_diversity"]) for row in group])
        teacher = np.asarray([float(row["teacher_pair_diversity"]) for row in group])
        geometry_rows.append(
            {
                "analysis_id": analysis_id,
                "seed": seed,
                "geometry_id": geometry_id,
                "num_condition_pairs": len(group),
                "mean_predicted_pair_diversity": float(np.mean(predicted)),
                "mean_teacher_pair_diversity": float(np.mean(teacher)),
                "responsiveness_ratio": float(np.mean(predicted) / np.mean(teacher)) if np.mean(teacher) > 1.0e-12 else None,
                "pair_diversity_spearman": _spearman(predicted, teacher),
            }
        )
    summary_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in geometry_rows:
        summary_groups[str(row["analysis_id"])].append(row)
    summary = {
        analysis_id: {
            "num_seed_geometry_rows": len(group),
            "mean_responsiveness_ratio": _mean_finite(row.get("responsiveness_ratio") for row in group),
            "mean_pair_diversity_spearman": _mean_finite(row.get("pair_diversity_spearman") for row in group),
        }
        for analysis_id, group in sorted(summary_groups.items())
    }
    return {"pairs": pair_rows, "geometries": geometry_rows, "summary": summary}


def _symmetric_projected_sizing_error(left: MeshWrapper, right: MeshWrapper) -> float:
    from src.algorithm.util.amber_util import interpolate_vertex_field
    from src.mesh_util.sizing_field_util import get_sizing_field

    left_size = get_sizing_field(left, mesh_node_type="vertex")
    right_size = get_sizing_field(right, mesh_node_type="vertex")
    left_on_right = interpolate_vertex_field(left, right, left_size)
    right_on_left = interpolate_vertex_field(right, left, right_size)
    forward = np.linalg.norm(right_size - left_on_right) / (np.linalg.norm(left_on_right) + 1.0e-10)
    reverse = np.linalg.norm(left_size - right_on_left) / (np.linalg.norm(right_on_left) + 1.0e-10)
    return float(0.5 * (forward + reverse))


def _load_wrapped_tetra(path: Path) -> MeshWrapper:
    mesh = meshio.read(path)
    tetra = [block.data for block in mesh.cells if block.type == "tetra"]
    if not tetra:
        raise ValueError(f"Mesh contains no tetra cells: {path}")
    converted = from_meshio(meshio.Mesh(points=mesh.points, cells=[("tetra", np.concatenate(tetra))]))
    if not isinstance(converted, MeshTet):
        raise TypeError(f"Expected MeshTet, got {type(converted)!r}")
    return MeshWrapper(ExtendedMeshTet1(converted.p, converted.t))


def _cached_mesh(cache: dict[str, MeshWrapper], key: str, path: Path) -> MeshWrapper:
    if key not in cache:
        cache[key] = _load_wrapped_tetra(path)
    return cache[key]


def _resolve_prediction_path(row: dict[str, Any], manifest_path: Path) -> Path:
    path = Path(str(row["prediction_mesh_path"]))
    if path.is_absolute():
        return path.resolve()
    base = manifest_path.parent.parent if manifest_path.parent.name == "test_predictions" else manifest_path.parent
    return (base / path).resolve()


def _apply_holm_correction(rows: list[dict[str, Any]]) -> None:
    indexed = [
        (index, float(row["wilcoxon_p_raw"]))
        for index, row in enumerate(rows)
        if row.get("wilcoxon_p_raw") is not None and np.isfinite(float(row["wilcoxon_p_raw"]))
    ]
    indexed.sort(key=lambda item: item[1])
    running = 0.0
    total = len(indexed)
    for rank, (index, pvalue) in enumerate(indexed):
        adjusted = min(1.0, (total - rank) * pvalue)
        running = max(running, adjusted)
        rows[index]["wilcoxon_p_holm"] = running


def _wilcoxon_pvalue(differences: np.ndarray) -> float | None:
    values = np.asarray(differences, dtype=np.float64)
    if values.size < 2:
        return None
    if np.allclose(values, 0.0):
        return 1.0
    from scipy.stats import wilcoxon

    return float(wilcoxon(values, alternative="two-sided", zero_method="wilcox").pvalue)


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    from scipy.stats import spearmanr

    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else None


def _mean_finite(values: Iterable[Any]) -> float | None:
    parsed = [_as_float(value) for value in values]
    finite = [value for value in parsed if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _unique_by_sample(rows: list[dict[str, Any]], *, path: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in output:
            raise FormalPlanError(f"Missing or duplicate sample_id in {path}: {sample_id!r}")
        output[sample_id] = row
    return output


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _as_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _as_int(value: Any, *, default: int) -> int:
    parsed = _as_float(value)
    return int(parsed) if parsed is not None else default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
