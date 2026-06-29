# Generated at 2026-04-08 22:33:28 +08:00 (Asia/Shanghai)
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.condition_aware_dataset_generation.pipeline import ConditionAwareDatasetPipeline
from src.condition_aware_dataset_generation.teacher_generation import (
    TeacherGenerator,
    combine_geometry_constraints,
    evaluate_geometry_sizing,
)
from src.condition_aware_dataset_generation.utils import load_json, read_jsonl
from src.tasks.domains.geometry_util import volume_to_edge_length

from tests.condition_aware_dataset_generation.test_ingestion_and_sampling import build_config


def _build_step_config(step_geometry_root: Path, output_root: Path) -> dict:
    config = build_config(step_geometry_root, output_root, patterns=["*.step"])
    config["condition_sampling"]["default_conditions_per_geometry"] = 1
    config["condition_sampling"]["budgets"] = [240]
    config["smoke"]["smoke_target_num_elements"] = 240
    config["teacher"]["initial_mesh_element_volume"] = 0.08
    config["teacher"]["max_adaptive_steps"] = 1
    config["teacher"]["reference_refinement_levels"] = 1
    config["teacher"]["initial_target_num_elements"] = 60
    config["teacher"]["initial_target_num_surface_faces"] = 160
    config["teacher"]["initial_max_nodes"] = 240
    config["teacher"]["initial_max_dofs"] = 320
    config["teacher"]["initial_max_budget_fraction"] = 0.95
    config["teacher"]["budget_calibration_tolerance"] = 0.3
    config["teacher"]["budget_calibration_timeout_seconds"] = 20.0
    config["teacher"]["minimum_viable_budget"] = 120
    config["teacher"]["desired_budget"] = 240
    config["teacher"]["hard_max_budget"] = 360
    config["teacher"]["reject_if_initial_mesh_too_dense"] = False
    config["teacher"]["adaptive_refinement_local_refine_for_complex_3d_enable"] = True
    config["teacher"]["adaptive_refinement_local_refine_complexity_threshold"] = 1.85
    return config


def test_geometry_feature_detection_on_step(step_geometry_root: Path, case_root: Path):
    config = _build_step_config(step_geometry_root, case_root / "step_preprocess")
    pipeline = ConditionAwareDatasetPipeline(config)

    ingest_summary = pipeline.ingest_geometries()
    assert ingest_summary["num_geometries"] == 1
    preprocess_summary = pipeline.preprocess_geometries()
    assert preprocess_summary["num_success"] == 1

    preprocess_record = pipeline._load_preprocess_records()[0]
    features = preprocess_record.geometry_features
    assert preprocess_record.geometry_feature_metadata_path is not None
    assert Path(preprocess_record.geometry_feature_metadata_path).exists()
    assert any(record["surface_type"] == "cylinder" for record in features["surface_patches"])
    assert any(record["curve_type"] == "circle" for record in features["curves"])
    assert features["feature_anchors"]["hole_curve_loops"]

    cached_summary = pipeline.preprocess_geometries()
    assert cached_summary["num_success"] == 1


def test_surface_sizing_constraints_shrink_near_hole(step_geometry_root: Path, case_root: Path):
    config = _build_step_config(step_geometry_root, case_root / "step_sizing")
    pipeline = ConditionAwareDatasetPipeline(config)
    pipeline.ingest_geometries()
    pipeline.preprocess_geometries()
    preprocess_record = pipeline._load_preprocess_records()[0]

    base_size = float(volume_to_edge_length(config["teacher"]["initial_mesh_element_volume"], dim=3))
    constraints = combine_geometry_constraints(preprocess_record.geometry_features, base_size, config["teacher"])
    hole = constraints["holes"][0]
    near = np.asarray(hole["curve_anchor_points"][0], dtype=float)
    far = np.asarray([0.05, 0.05, 0.5], dtype=float)
    sizes = evaluate_geometry_sizing(
        points=np.vstack([near, far]),
        geometry_features=preprocess_record.geometry_features,
        constraint_summary=constraints,
        base_size=base_size,
        config=config["teacher"],
    )
    assert sizes[0] < sizes[1]
    assert hole["min_segments"] >= config["teacher"]["min_circle_segments"]
    assert hole["target_size"] <= config["teacher"]["hole_edge_length_ratio"] * hole["radius"] + 1.0e-12


def test_complex_step_prefers_local_refine_for_adaptive_step(step_geometry_root: Path, case_root: Path):
    config = _build_step_config(step_geometry_root, case_root / "step_local_refine_pref")
    pipeline = ConditionAwareDatasetPipeline(config)
    pipeline.ingest_geometries()
    pipeline.preprocess_geometries()

    preprocess_record = pipeline._load_preprocess_records()[0]
    teacher = TeacherGenerator(config["teacher"], config["smoke"])

    assert teacher._prefer_local_refine_for_complex_3d_step(preprocess_record) is True


def test_meshing_quality_and_retry_on_step(step_geometry_root: Path, case_root: Path):
    config = _build_step_config(step_geometry_root, case_root / "step_teacher")
    config["condition_sampling"]["budgets"] = [90]
    config["teacher"]["initial_mesh_element_volume"] = 3.5
    config["teacher"]["hole_edge_length_ratio"] = 0.7
    config["teacher"]["min_circle_segments"] = 8
    config["teacher"]["max_circle_fit_error"] = 0.03
    config["teacher"]["max_geometry_retry"] = 2
    config["teacher"]["initial_target_num_elements"] = 120
    config["teacher"]["initial_target_num_surface_faces"] = 240
    config["teacher"]["initial_max_nodes"] = 360
    config["teacher"]["initial_max_dofs"] = 420

    pipeline = ConditionAwareDatasetPipeline(config)
    pipeline.ingest_geometries()
    pipeline.preprocess_geometries()
    pipeline.sample_conditions()

    geometry_record = pipeline._load_geometry_records()[0]
    preprocess_record = pipeline._load_preprocess_records()[0]
    condition_record = pipeline._load_condition_records()[0]
    teacher = TeacherGenerator(config["teacher"], config["smoke"])
    teacher_record, sample_records, failure = teacher.generate(
        geometry_record=geometry_record,
        preprocess_record=preprocess_record,
        condition_record=condition_record,
        layout=pipeline.layout,
        overwrite=True,
    )
    assert failure is None
    assert teacher_record.status in {"success", "success_budget_closed", "success_near_desired_budget", "success_partial_under_budget", "budget_closure_failed"}
    assert teacher_record.initial_mesh_diagnostics["topology_preserved"]
    assert (not teacher_record.initial_mesh_diagnostics["initial_is_too_dense"]) or teacher_record.initial_mesh_diagnostics.get("used_preprocess_coarse_fallback", False)
    if teacher_record.surface_quality_metrics.get("hole_sampling"):
        assert teacher_record.surface_quality_metrics["hole_sampling"].get("min_segments", 0) >= 0
    assert teacher_record.geometry_retry_history
    assert sample_records
    assert all(record.status != "failed" for record in sample_records)


def test_bad_initial_mesh_is_recorded_on_failure(step_geometry_root: Path, case_root: Path):
    config = _build_step_config(step_geometry_root, case_root / "step_failure")
    config["teacher"]["initial_target_num_elements"] = 4
    config["teacher"]["initial_target_num_surface_faces"] = 8
    config["teacher"]["initial_max_nodes"] = 16
    config["teacher"]["initial_max_dofs"] = 24
    config["teacher"]["initial_max_budget_fraction"] = 0.1
    config["teacher"]["reject_if_initial_mesh_too_dense"] = True

    pipeline = ConditionAwareDatasetPipeline(config)
    pipeline.ingest_geometries()
    pipeline.preprocess_geometries()
    pipeline.sample_conditions()

    teacher_summary = pipeline.generate_teacher_targets()
    assert teacher_summary["num_failures"] >= 1
    sample_records = pipeline._load_sample_records()
    assert sample_records
    assert any(record.status == "failed" for record in sample_records)
    assert any(record.failure_category == "reject_bad_initial_mesh" for record in sample_records)


def test_full_pipeline_step_smoke(step_geometry_root: Path, case_root: Path):
    config = _build_step_config(step_geometry_root, case_root / "step_smoke")
    pipeline = ConditionAwareDatasetPipeline(config)

    summary = pipeline.run_full_pipeline()
    assert summary["manifest"]["num_samples"] > 0
    assert summary["manifest"]["num_successful_samples"] > 0
    sample_manifest = read_jsonl(pipeline.layout.manifest_path("sample_manifest"))
    assert sample_manifest
    success_rows = [row for row in sample_manifest if row["status"] != "failed"]
    assert success_rows
    assert all("geometry_feature_metadata_path" in row["geometry_artifact_paths"] for row in success_rows)
    assert all("surface_quality_metrics" in row["teacher_metadata"] for row in success_rows)
    split_manifest = load_json(pipeline.layout.split_manifest_path)
    assert split_manifest["geometry_to_split"]


