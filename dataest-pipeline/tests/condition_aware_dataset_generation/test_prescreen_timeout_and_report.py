# 生成时间：2026-04-09 21:12:00 +08:00（北京时间）
from __future__ import annotations

from pathlib import Path

from src.condition_aware_dataset_generation.pipeline import ConditionAwareDatasetPipeline
from src.condition_aware_dataset_generation.utils import load_json

from tests.condition_aware_dataset_generation.test_ingestion_and_sampling import build_config


def test_prescreen_emits_accept_and_reject_labels(step_geometry_root: Path, case_root: Path):
    config = build_config(step_geometry_root, case_root / 'prescreen_labels', patterns=['*.step'])
    config['condition_sampling']['default_conditions_per_geometry'] = 2
    config['condition_sampling']['pde_families'] = ['scalar_elliptic', 'linear_elasticity']
    config['condition_sampling']['budgets'] = [30]
    config['prescreen']['enable_prescreen'] = True
    config['prescreen']['prescreen_min_hotspot_concentration'] = 0.1
    config['prescreen']['prescreen_min_allocation_gain'] = 0.9
    config['smoke']['elasticity_smoke_enable'] = False

    pipeline = ConditionAwareDatasetPipeline(config)
    pipeline.ingest_geometries()
    pipeline.preprocess_geometries()
    pipeline.sample_conditions()

    summary = pipeline.prescreen_conditions()
    prescreen_records = pipeline._load_prescreen_records()

    assert summary['num_conditions'] == 2
    assert any(record.label == 'accept_for_smoke' for record in prescreen_records)
    assert any(record.label == 'reject_too_expensive' for record in prescreen_records)
    assert all(record.started_at for record in prescreen_records)
    assert all(record.finished_at for record in prescreen_records)


def test_internal_teacher_timeout_is_recorded(step_geometry_root: Path, case_root: Path):
    config = build_config(step_geometry_root, case_root / 'teacher_timeout', patterns=['*.step'])
    config['condition_sampling']['default_conditions_per_geometry'] = 1
    config['condition_sampling']['pde_families'] = ['scalar_elliptic']
    config['condition_sampling']['budgets'] = [30]
    config['teacher']['initial_mesh_element_volume'] = 0.02
    config['smoke']['smoke_max_runtime_seconds_per_sample'] = 0.05
    config['smoke']['smoke_max_runtime_seconds_per_stage']['surface_meshing'] = 0.05
    config['smoke']['adaptive_stage_timeouts_enable'] = False

    pipeline = ConditionAwareDatasetPipeline(config)
    pipeline.ingest_geometries()
    pipeline.preprocess_geometries()
    pipeline.sample_conditions()

    summary = pipeline.generate_teacher_targets()
    sample_records = pipeline._load_sample_records()

    assert summary['num_failures'] >= 1
    assert sample_records
    assert any(record.failure_category and 'timeout' in record.failure_category for record in sample_records)
    assert any(record.stage_where_stopped in {'surface_meshing', 'teacher_runtime'} for record in sample_records if record.failure_category)
    teacher_records = pipeline._load_teacher_records()
    assert teacher_records
    runtime_observation = teacher_records[0].solver_metadata.get('runtime_observation', {})
    assert runtime_observation.get('parent_observed_elapsed_seconds') is not None
    assert 'worker_reported_elapsed_seconds' in runtime_observation
    assert sample_records[0].elapsed_seconds == runtime_observation['parent_observed_elapsed_seconds']


def test_adaptive_teacher_timeout_scales_for_complex_step(step_geometry_root: Path, case_root: Path):
    config = build_config(step_geometry_root, case_root / 'adaptive_teacher_timeout', patterns=['*.step'])
    config['condition_sampling']['default_conditions_per_geometry'] = 2

    pipeline = ConditionAwareDatasetPipeline(config)
    pipeline.ingest_geometries()
    pipeline.preprocess_geometries()
    pipeline.sample_conditions()

    preprocess_record = pipeline._load_preprocess_records()[0]
    base_stage_timeouts = pipeline._stage_timeouts()
    scalar_condition = next(record for record in pipeline._load_condition_records() if record.pde_family == 'scalar_elliptic')
    elasticity_condition = next(record for record in pipeline._load_condition_records() if record.pde_family == 'linear_elasticity')

    scalar_limits = pipeline._teacher_runtime_limits(preprocess_record, scalar_condition)
    elasticity_limits = pipeline._teacher_runtime_limits(preprocess_record, elasticity_condition)

    assert scalar_limits['adaptive_stage_timeouts_enable'] is True
    assert scalar_limits['complexity_multiplier'] > 1.0
    assert scalar_limits['stage_timeout_seconds']['adaptive_refinement'] > base_stage_timeouts['adaptive_refinement']
    assert scalar_limits['sample_timeout_seconds'] >= config['smoke']['smoke_max_runtime_seconds_per_sample']
    assert elasticity_limits['stage_timeout_seconds']['adaptive_refinement'] >= scalar_limits['stage_timeout_seconds']['adaptive_refinement']


def test_smoke_report_contains_contrast_and_difference_metrics(geometry_root: Path, case_root: Path):
    config = build_config(geometry_root, case_root / 'report_metrics')
    config['condition_sampling']['default_conditions_per_geometry'] = 2
    config['condition_sampling']['pde_families'] = ['scalar_elliptic']
    config['condition_sampling']['budgets'] = [18]
    config['teacher']['max_adaptive_steps'] = 1

    pipeline = ConditionAwareDatasetPipeline(config)
    summary = pipeline.run_full_pipeline()
    report = load_json(pipeline.layout.report_path('smoke_report'))

    assert summary['manifest']['num_samples'] > 0
    assert report['geometry_reports']
    first_geometry = report['geometry_reports'][0]
    assert first_geometry['sample_metrics']
    sample_metrics = first_geometry['sample_metrics'][0]
    assert 'hotspot_size_ratio' in sample_metrics
    assert 'final_hotspot_size_ratio' in sample_metrics
    assert 'allocation_gain' in sample_metrics
    assert 'final_allocation_gain' in sample_metrics
    assert 'budget_status' in sample_metrics
    if first_geometry['pairwise_condition_metrics']:
        first_pair = next(iter(first_geometry['pairwise_condition_metrics'].values()))
        assert 'probe_wise_sizing_pearson' in first_pair
        assert 'final_sizing_field_pearson' in first_pair
        assert 'region_wise_density_difference' in first_pair
        assert 'finest_20_region_jaccard' in first_pair


def test_smoke_report_contains_stage_separability_metrics(geometry_root: Path, case_root: Path):
    config = build_config(geometry_root, case_root / 'report_stage_metrics')
    config['condition_sampling']['default_conditions_per_geometry'] = 2
    config['condition_sampling']['pde_families'] = ['scalar_elliptic']
    config['condition_sampling']['budgets'] = [18]
    config['teacher']['max_adaptive_steps'] = 1

    pipeline = ConditionAwareDatasetPipeline(config)
    pipeline.run_full_pipeline()
    report = load_json(pipeline.layout.report_path('smoke_report'))

    first_geometry = report['scalar_smoke']['geometry_reports'][0]
    stage_metrics = first_geometry['field_stage_pairwise_metrics']
    assert set(stage_metrics) == {'s_pde_raw', 'h_pde_only', 'h_after_geometry_fusion', 'h_after_budget_calibration'}
    if stage_metrics['h_after_budget_calibration']:
        first_pair = next(iter(stage_metrics['h_after_budget_calibration'].values()))
        assert 'pearson' in first_pair
        assert 'spearman' in first_pair
        assert 'relative_l2_difference' in first_pair
        assert 'finest_20_region_jaccard' in first_pair
        assert 'hotspot_region_jaccard' in first_pair
    assert 'collapse_stage' in first_geometry['collapse_diagnostics']


def test_elasticity_gate_does_not_block_scalar_smoke(geometry_root: Path, case_root: Path):
    config = build_config(geometry_root, case_root / 'elasticity_gate_non_blocking')
    config['condition_sampling']['default_conditions_per_geometry'] = 2
    config['condition_sampling']['pde_families'] = ['scalar_elliptic', 'linear_elasticity']
    config['condition_sampling']['budgets'] = [30]
    config['prescreen']['enable_prescreen'] = True
    config['prescreen']['prescreen_min_hotspot_concentration'] = 0.05
    config['prescreen']['prescreen_min_allocation_gain'] = 0.8
    config['smoke']['elasticity_smoke_enable'] = True
    config['smoke']['elasticity_smoke_max_samples'] = 0
    config['smoke']['elasticity_smoke_strict_cost_gate'] = True

    pipeline = ConditionAwareDatasetPipeline(config)
    summary = pipeline.run_full_pipeline()
    sample_records = pipeline._load_sample_records()

    assert summary['manifest']['num_samples'] > 0
    assert any(record.pde_family == 'scalar_elliptic' and record.status != 'failed' for record in sample_records)
    assert not any(record.pde_family == 'linear_elasticity' and record.status != 'failed' for record in sample_records)
    prescreen_records = pipeline._load_prescreen_records()
    assert any(record.pde_family == 'linear_elasticity' and record.label == 'reject_too_expensive' for record in prescreen_records)


def test_elasticity_cheap_smoke_uses_coarse_reference(geometry_root: Path, case_root: Path):
    config = build_config(geometry_root, case_root / 'elasticity_cheap_smoke')
    config['condition_sampling']['default_conditions_per_geometry'] = 1
    config['condition_sampling']['pde_families'] = ['linear_elasticity']
    config['condition_sampling']['budgets'] = [35]
    config['prescreen']['enable_prescreen'] = False
    config['teacher']['max_adaptive_steps'] = 0
    config['teacher']['enable_budget_calibration'] = False
    config['teacher']['minimum_viable_budget'] = 18
    config['teacher']['desired_budget'] = 35
    config['teacher']['hard_max_budget'] = 60
    config['smoke']['elasticity_smoke_enable'] = True
    config['smoke']['elasticity_smoke_mode'] = 'cheap_reference'
    config['smoke']['elasticity_smoke_reference_level'] = 0
    config['smoke']['smoke_max_dofs'] = 60000
    config['smoke']['smoke_max_matrix_nnz'] = 4000000

    pipeline = ConditionAwareDatasetPipeline(config)
    summary = pipeline.run_full_pipeline()
    sample_records = pipeline._load_sample_records()

    assert summary['manifest']['num_samples'] > 0
    assert any(record.pde_family == 'linear_elasticity' and record.status != 'failed' for record in sample_records)
    report = load_json(pipeline.layout.report_path('smoke_report'))
    assert report['elasticity_smoke']['summary']['num_samples'] >= 1



