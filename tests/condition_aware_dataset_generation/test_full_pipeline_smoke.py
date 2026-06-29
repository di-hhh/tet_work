from __future__ import annotations

from src.condition_aware_dataset_generation.pipeline import ConditionAwareDatasetPipeline
from src.condition_aware_dataset_generation.utils import load_json, read_jsonl

from tests.condition_aware_dataset_generation.test_ingestion_and_sampling import build_config


def test_full_pipeline_smoke(geometry_root: Path, case_root):
    config = build_config(geometry_root, case_root / "smoke_output")
    config["condition_sampling"]["default_conditions_per_geometry"] = 2
    config["teacher"]["max_adaptive_steps"] = 1
    pipeline = ConditionAwareDatasetPipeline(config)

    summary = pipeline.run_full_pipeline()
    assert summary["manifest"]["num_samples"] > 0
    assert (pipeline.layout.output_root / "geometries").exists()
    assert (pipeline.layout.output_root / "teachers").exists()
    assert (pipeline.layout.output_root / "manifests" / "sample_manifest.jsonl").exists()

    sample_manifest = read_jsonl(pipeline.layout.manifest_path("sample_manifest"))
    assert sample_manifest
    smoke_report = load_json(pipeline.layout.report_path('smoke_report'))
    assert len(smoke_report['scalar_smoke']['geometry_reports']) >= 2
    split_manifest = load_json(pipeline.layout.split_manifest_path)
    assert "geometry_to_split" in split_manifest

