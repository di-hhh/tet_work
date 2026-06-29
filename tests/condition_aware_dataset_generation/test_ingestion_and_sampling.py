from __future__ import annotations

from pathlib import Path

from src.condition_aware_dataset_generation.condition_sampling import ConditionSampler
from src.condition_aware_dataset_generation.pipeline import ConditionAwareDatasetPipeline
from src.condition_aware_dataset_generation.records import GeometryPreprocessRecord


def build_config(geometry_root: Path, output_root: Path, patterns: list[str] | None = None) -> dict:
    return {
        'output_root': str(output_root),
        'random_seed': 5,
        'workers': 1,
        'overwrite': True,
        'geometry_source': {
            'name': 'local_directory',
            'source_name': 'unit_local',
            'root': str(geometry_root),
            'patterns': patterns or ['*.json'],
            'recursive': False,
        },
        'preprocessing': {
            'coarse_element_volume': 0.08,
            'min_extent': 1.0e-6,
            'sharp_dihedral_deg': 40.0,
            'save_geometry_feature_metadata': True,
        },
        'condition_sampling': {
            'default_conditions_per_geometry': 4,
            'pde_families': ['scalar_elliptic', 'linear_elasticity'],
            'budgets': [20, 35],
        },
        'prescreen': {
            'enable_prescreen': False,
            'prescreen_max_elements': 4000,
            'prescreen_max_runtime_seconds': 10,
            'prescreen_probe_count': 128,
            'prescreen_hotspot_quantile': 0.9,
            'prescreen_condition_overlap_threshold': 0.65,
            'prescreen_min_allocation_gain': 1.02,
        },
        'teacher': {
            'initial_mesh_element_volume': 0.12,
            'reference_refinement_levels': 1,
            'max_adaptive_steps': 2,
            'refine_theta': 0.6,
            'store_trajectory': True,
            'allow_budget_shortfall': True,
            'enable_geometry_fidelity_constraints': True,
            'min_circle_segments': 16,
            'hole_edge_length_ratio': 0.28,
            'hole_radial_refinement_layers': 3,
            'hole_radial_growth_rate': 1.4,
            'curvature_refinement_strength': 1.5,
            'feature_size_refinement_strength': 1.5,
            'surface_first_meshing': True,
            'max_circle_fit_error': 0.05,
            'max_boundary_deviation': 0.03,
            'max_normal_deviation': 25.0,
            'max_geometry_retry': 2,
            'initial_target_num_elements': 20,
            'initial_target_num_surface_faces': 32,
            'initial_max_nodes': 64,
            'initial_max_dofs': 64,
            'initial_max_runtime_seconds': 10.0,
            'initial_max_budget_fraction': 0.7,
            'reject_if_initial_mesh_too_dense': True,
            'enable_budget_calibration': True,
            'budget_calibration_max_iters': 4,
            'budget_calibration_tolerance': 0.2,
            'budget_calibration_timeout_seconds': 10.0,
            'field_stage_debug_dump': True,
            'enable_low_importance_inflation': True,
            'condition_difference_preservation_enable': True,
        },
        'smoke': {
            'smoke_target_num_elements': 35,
            'smoke_max_runtime_seconds_per_sample': 180,
            'smoke_max_runtime_seconds_per_stage': {
                'geometry_preprocessing': 15,
                'prescreen_solve': 10,
                'surface_meshing': 60,
                'volume_meshing': 60,
                'pde_solve': 45,
                'reference_solve': 45,
                'adaptive_refinement': 30,
            },
            'adaptive_stage_timeouts_enable': True,
            'adaptive_stage_timeout_reference_elements': 2500,
            'adaptive_stage_timeout_min_multiplier': 1.0,
            'adaptive_stage_timeout_max_multiplier': 4.0,
            'adaptive_stage_timeout_3d_bonus': 0.65,
            'adaptive_stage_timeout_elasticity_bonus': 0.15,
            'adaptive_stage_timeout_boundary_patch_baseline': 8,
            'adaptive_stage_timeout_boundary_patch_weight': 0.035,
            'adaptive_stage_timeout_sharp_edge_baseline': 10,
            'adaptive_stage_timeout_sharp_edge_weight': 0.015,
            'adaptive_stage_timeout_hole_weight': 0.20,
            'adaptive_stage_timeout_stage_bias': {
                'surface_meshing': 0.75,
                'volume_meshing': 0.9,
                'budget_calibration': 0.6,
                'pde_solve': 0.65,
                'reference_solve': 0.75,
                'adaptive_refinement': 1.35,
            },
            'adaptive_stage_timeout_sample_scale': 2.25,
            'adaptive_stage_timeout_sample_slack_seconds': 20.0,
            'adaptive_stage_timeout_max_sample_seconds': 300.0,
            'smoke_max_refinement_steps': 2,
            'smoke_max_retries': 2,
            'smoke_max_dofs': 60000,
            'smoke_max_matrix_nnz': 4000000,
            'smoke_max_budget_overrun_ratio': 1.5,
            'contrast_mode': 'hybrid',
            'contrast_gamma': 1.8,
            'hotspot_quantile': 0.9,
            'medium_quantile': 0.7,
            'low_importance_size_boost': 1.35,
            'target_hotspot_size_ratio': 0.8,
            'enable_global_budget_rescaling': True,
            'elasticity_smoke_enable': True,
            'elasticity_smoke_max_samples': 1,
            'skip_expensive_elasticity': True,
        },
        'split': {'seed': 5, 'ratios': {'train': 0.5, 'val': 0.25, 'test': 0.25}},
    }


def test_ingestion_preprocess_and_condition_sampling(geometry_root: Path, case_root: Path):
    config = build_config(geometry_root, case_root / 'output')
    pipeline = ConditionAwareDatasetPipeline(config)

    ingest_summary = pipeline.ingest_geometries()
    assert ingest_summary['num_geometries'] == 2
    assert ingest_summary['num_failures'] == 1

    preprocess_summary = pipeline.preprocess_geometries()
    assert preprocess_summary['num_success'] == 2
    preprocess_records = pipeline._load_preprocess_records()
    assert all(isinstance(record, GeometryPreprocessRecord) for record in preprocess_records)
    assert all(record.validation['is_meshable'] for record in preprocess_records)
    assert all(Path(record.geometry_feature_metadata_path).exists() for record in preprocess_records if record.geometry_feature_metadata_path)

    sample_summary = pipeline.sample_conditions()
    assert sample_summary['num_conditions'] == 8
    condition_records = pipeline._load_condition_records()
    assert len(condition_records) == 8
    assert {record.pde_family for record in condition_records} == {'scalar_elliptic', 'linear_elasticity'}

    sampler = ConditionSampler(config['condition_sampling'])
    geometry_record = pipeline._load_geometry_records()[0]
    preprocess_record = preprocess_records[0]
    left = [record.to_dict() for record in sampler.sample_for_geometry(geometry_record, preprocess_record, seed=17)]
    right = [record.to_dict() for record in sampler.sample_for_geometry(geometry_record, preprocess_record, seed=17)]
    assert left == right




