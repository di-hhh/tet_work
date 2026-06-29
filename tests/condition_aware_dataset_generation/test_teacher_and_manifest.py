from __future__ import annotations

from pathlib import Path

from src.condition_aware_dataset_generation.pipeline import ConditionAwareDatasetPipeline
from src.condition_aware_dataset_generation.records import ConditionRecord
from src.condition_aware_dataset_generation.teacher_generation import TeacherGenerator
from src.condition_aware_dataset_generation.utils import load_json, read_jsonl

from tests.condition_aware_dataset_generation.test_ingestion_and_sampling import build_config


def test_teacher_generation_and_manifest(geometry_root: Path, case_root: Path):
    config = build_config(geometry_root, case_root / "teacher_output")
    pipeline = ConditionAwareDatasetPipeline(config)
    pipeline.ingest_geometries()
    pipeline.preprocess_geometries()
    pipeline.sample_conditions()

    teacher_summary = pipeline.generate_teacher_targets()
    assert teacher_summary["num_samples"] > 0

    sample_records = pipeline._load_sample_records()
    success_records = [record for record in sample_records if record.status != "failed"]
    assert success_records
    assert all(Path(record.initial_mesh_path).exists() for record in success_records)
    assert all(Path(record.final_target_mesh_path).exists() for record in success_records)
    assert any(record.optional_intermediate_mesh_paths for record in success_records)
    assert all("surface_quality_metrics" in record.teacher_metadata for record in success_records)
    assert all("geometry_feature_metadata_path" in record.geometry_artifact_paths for record in success_records)

    manifest_summary = pipeline.build_dataset_manifest()
    assert manifest_summary["num_samples"] == len(sample_records)
    manifest_rows = read_jsonl(pipeline.layout.manifest_path("sample_manifest"))
    assert len(manifest_rows) == len(sample_records)
    split_manifest = load_json(pipeline.layout.split_manifest_path)
    assert split_manifest["geometry_to_split"]

    split_by_geometry = {}
    for row in manifest_rows:
        split_by_geometry.setdefault(row["geometry_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in split_by_geometry.values())


def test_teacher_generation_failure_is_captured(geometry_root: Path, case_root: Path):
    config = build_config(geometry_root, case_root / "teacher_failure_output")
    pipeline = ConditionAwareDatasetPipeline(config)
    pipeline.ingest_geometries()
    pipeline.preprocess_geometries()

    geometry_record = pipeline._load_geometry_records()[0]
    preprocess_record = pipeline._load_preprocess_records()[0]
    broken_condition = ConditionRecord(
        condition_id="broken_condition",
        geometry_id=geometry_record.geometry_id,
        pde_family="unknown_family",
        condition_index=0,
        condition_spec={"pde_family": "unknown_family"},
        budget_or_tolerance_spec={"budgets": [10]},
        source_name=geometry_record.source_name,
    )
    teacher = TeacherGenerator(config["teacher"])
    teacher_record, sample_records, failure = teacher.generate(
        geometry_record=geometry_record,
        preprocess_record=preprocess_record,
        condition_record=broken_condition,
        layout=pipeline.layout,
        overwrite=True,
    )
    assert teacher_record is not None
    assert teacher_record.status == "failed"
    assert sample_records[0].status == "failed"
    assert failure is not None


def test_budget_calibration_records_diagnostics(geometry_root: Path, case_root: Path):
    config = build_config(geometry_root, case_root / 'budget_calibration_output')
    config['condition_sampling']['default_conditions_per_geometry'] = 1
    config['condition_sampling']['pde_families'] = ['scalar_elliptic']
    config['condition_sampling']['budgets'] = [18]
    config['teacher']['max_adaptive_steps'] = 1

    pipeline = ConditionAwareDatasetPipeline(config)
    pipeline.ingest_geometries()
    pipeline.preprocess_geometries()
    pipeline.sample_conditions()
    pipeline.generate_teacher_targets()

    sample_records = [record for record in pipeline._load_sample_records() if record.status != 'failed']
    assert sample_records
    diagnostics = sample_records[0].teacher_metadata['budget_diagnostics']
    initial_diagnostics = sample_records[0].teacher_metadata['initial_mesh_diagnostics']
    assert diagnostics['target_budget'] == 18
    assert 'actual_budget' in diagnostics
    assert 'budget_ratio' in diagnostics
    assert 'lambda' in diagnostics
    assert 'calibration_iters' in diagnostics
    assert 'calibration_converged' in diagnostics
    assert 'initial_budget_fraction' in initial_diagnostics


def test_budget_status_classification_is_layered():
    teacher = TeacherGenerator(
        {
            'budget_calibration_tolerance': 0.1,
            'allow_success_partial_under_budget': True,
            'minimum_viable_budget': 35,
            'desired_budget': 100,
            'hard_max_budget': 125,
        }
    )
    tiers = teacher._budget_tiers(100)

    assert teacher._classify_budget_status(actual_budget=100, budget_tiers=tiers) == 'success_budget_closed'
    assert teacher._classify_budget_status(actual_budget=82, budget_tiers=tiers) == 'success_near_desired_budget'
    assert teacher._classify_budget_status(actual_budget=40, budget_tiers=tiers) == 'success_partial_under_budget'
    assert teacher._classify_budget_status(actual_budget=20, budget_tiers=tiers, growth_stalled=True) == 'fail_budget_growth_stalled'
    assert teacher._classify_budget_status(actual_budget=130, budget_tiers=tiers) == 'fail_budget_hard_cap_exceeded'


def test_high_budget_initial_caps_scale_with_budget():
    teacher = TeacherGenerator(
        {
            'initial_target_num_elements': 2400,
            'initial_target_num_surface_faces': 4800,
            'initial_max_nodes': 6000,
            'initial_max_dofs': 9000,
            'initial_max_budget_fraction': 0.25,
            'high_budget_threshold': 50_000,
            'initial_absolute_caps_scale_with_budget': True,
        }
    )

    assert teacher._effective_initial_element_cap(12_000) == 2400
    assert teacher._effective_initial_element_cap(150_000) == 37_500
    assert teacher._effective_initial_surface_face_cap(150_000) == 75_000
    assert teacher._effective_initial_node_cap(150_000) == 93_750
    assert teacher._effective_initial_dof_cap(150_000) == 140_625


def test_high_budget_growth_controls_expand_for_large_shortfall():
    teacher = TeacherGenerator(
        {
            'budget_growth_max_steps': 5,
            'budget_growth_max_steps_cap': 18,
            'budget_growth_batch_refine_fraction': 0.16,
            'budget_growth_batch_refine_fraction_max': 0.32,
            'budget_growth_dynamic_step_enable': True,
            'high_budget_threshold': 50_000,
        }
    )

    low_budget_steps, low_budget_fraction = teacher._effective_budget_growth_controls(current_elements=4_500, desired_budget=12_000)
    high_budget_steps, high_budget_fraction = teacher._effective_budget_growth_controls(current_elements=24_000, desired_budget=150_000)

    assert low_budget_steps >= 5
    assert low_budget_fraction >= 0.16
    assert high_budget_steps > low_budget_steps
    assert high_budget_steps >= 8
    assert high_budget_fraction >= 0.22
    assert high_budget_fraction <= 0.32


def test_budget_growth_loop_expands_coarse_seed(geometry_root: Path, case_root: Path):
    config = build_config(geometry_root, case_root / 'budget_growth_output')
    config['condition_sampling']['default_conditions_per_geometry'] = 1
    config['condition_sampling']['pde_families'] = ['scalar_elliptic']
    config['condition_sampling']['budgets'] = [80]
    config['teacher']['max_adaptive_steps'] = 0
    config['smoke']['smoke_max_refinement_steps'] = 0
    config['teacher']['enable_budget_calibration'] = False
    config['teacher']['minimum_viable_budget'] = 30
    config['teacher']['desired_budget'] = 80
    config['teacher']['hard_max_budget'] = 120
    config['teacher']['budget_growth_enable'] = True
    config['teacher']['budget_growth_max_steps'] = 3
    config['teacher']['budget_growth_batch_refine_fraction'] = 0.35
    config['teacher']['budget_growth_timeout_seconds'] = 15.0

    pipeline = ConditionAwareDatasetPipeline(config)
    pipeline.ingest_geometries()
    pipeline.preprocess_geometries()
    pipeline.sample_conditions()
    pipeline.generate_teacher_targets()

    sample_records = [record for record in pipeline._load_sample_records() if record.status != 'failed']
    assert sample_records
    metadata = sample_records[0].teacher_metadata
    growth = metadata['budget_growth_diagnostics']
    initial_elements = metadata['initial_mesh_diagnostics']['initial_num_elements']
    achieved_elements = metadata['achieved_num_elements']
    assert growth['history']
    assert achieved_elements > initial_elements
    assert achieved_elements >= 30
    assert metadata['budget_status'] in {'success_budget_closed', 'success_near_desired_budget', 'success_partial_under_budget'}
    assert 'final_allocation_diagnostics' in metadata


def test_preprocess_coarse_mode_uses_coarse_mesh_as_initial(geometry_root: Path, case_root: Path):
    config = build_config(geometry_root, case_root / 'preprocess_coarse_initial_output')
    config['condition_sampling']['default_conditions_per_geometry'] = 1
    config['condition_sampling']['pde_families'] = ['scalar_elliptic']
    config['condition_sampling']['budgets'] = [80]
    config['teacher']['initial_mesh_generation_mode'] = 'preprocess_coarse'
    config['teacher']['reject_if_initial_mesh_too_dense'] = False
    config['teacher']['enable_budget_calibration'] = False
    config['smoke']['smoke_max_refinement_steps'] = 0
    config['teacher']['max_adaptive_steps'] = 0

    pipeline = ConditionAwareDatasetPipeline(config)
    pipeline.ingest_geometries()
    pipeline.preprocess_geometries()
    pipeline.sample_conditions()
    pipeline.generate_teacher_targets()

    sample_records = [record for record in pipeline._load_sample_records() if record.status != 'failed']
    assert sample_records
    metadata = sample_records[0].teacher_metadata
    initial = metadata['initial_mesh_diagnostics']
    assert initial['initial_mesh_generation_mode'] == 'preprocess_coarse'
    assert initial['initial_mesh_source'] == 'preprocess_coarse_seed'
    assert initial.get('used_preprocess_coarse_as_initial', False) is True
