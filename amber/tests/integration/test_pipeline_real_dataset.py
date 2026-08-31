from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.algorithm.dataloader import get_datasets
from src.algorithm.loss.amber_loss import AmberLoss
from src.algorithm.prediction_transform.no_transform import NoTransform
from src.tasks.pipeline_dataset_audit import audit_pipeline_dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
FIXED_SAMPLE_IDS = [
    "sample_7000_008db8c153",  # scalar train
    "sample_7000_01f22c2d04",  # elasticity train
    "sample_7000_430dd5be4e",  # scalar val
    "sample_7000_001446d1c7",  # elasticity val
    "sample_7000_193d319437",  # scalar test
    "sample_7000_01ad06036c",  # elasticity test
    "sample_7000_00358c9316",  # quality FAIL control
]


@pytest.mark.real_dataset
def test_fixed_real_samples_cover_splits_families_weights_features_and_loss():
    root_value = os.environ.get("AMBER_REAL_DATASET_ROOT")
    if not root_value:
        pytest.skip("Set AMBER_REAL_DATASET_ROOT to run the fixed real-data integration test")
    root = Path(root_value).resolve()
    dataest_root = root.parents[2]
    geometry_root = dataest_root / "data" / "mold"

    cfg = _compose(root=root, geometry_root=geometry_root, run_name="amber_pipeline_weighted")
    audit = audit_pipeline_dataset(cfg.task)
    audit.raise_for_errors()
    assert audit.split_counts == {"train": 2, "val": 2, "test": 2}
    assert audit.counts["quality_filtered"] == 1
    datasets = get_datasets(cfg.algorithm, cfg.task)

    transform = NoTransform(
        OmegaConf.create({"predict_residual": False, "inverse_transform_in_loss": False})
    )
    criterion = AmberLoss(
        label_transform=transform,
        loss_type="mse",
        weighted_imitation_config=OmegaConf.to_container(cfg.algorithm.weighted_imitation, resolve=True),
    )
    observed_families = set()
    for split in ("train", "val", "test"):
        assert len(datasets[split]) == 2
        for data in datasets[split].data:
            graph = data.observation
            assert graph.x.shape[0] == graph.y.shape[0]
            assert graph.imitation_weights.shape == graph.y.shape
            assert bool(graph.imitation_weights_loaded.max())
            assert not bool(graph.imitation_weights_fallback.max())
            observed_families.add(data.source_data.imitation_weight_cache["pde_family"])
            loss, _ = criterion(predictions=graph.y.clone(), labels=graph.y, graph_batch=graph)
            assert torch.isfinite(loss)
    assert observed_families == {"scalar_elliptic", "linear_elasticity"}

    stage_cfg = _compose(
        root=root,
        geometry_root=geometry_root,
        run_name="amber_pipeline_physics_correction_stage_field",
    )
    stage_data = get_datasets(stage_cfg.algorithm, stage_cfg.task)["train"][0]
    stage_graph = stage_data.observation
    assert stage_data.source_data.imitation_weight_cache["physics_feature_source"] == "stage_field_fusion"
    assert bool(stage_graph.physics_feature_stage_field_loaded.max())
    assert stage_graph.physics_feature.shape[0] == stage_graph.y.shape[0]


def _compose(*, root: Path, geometry_root: Path, run_name: str):
    sample_ids = ",".join(FIXED_SAMPLE_IDS)
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(
            config_name="training_config",
            overrides=[
                f"+_runs/amber={run_name}",
                f"task.pipeline_output_root={root.as_posix()}",
                f"task.geometry_source_root={geometry_root.as_posix()}",
                f"task.sample_id_filter=[{sample_ids}]",
                "task.required_splits=[train,val,test]",
                "algorithm.sizing_field_interpolation_type=element_weighted_sum",
                "algorithm.initial_mesh_handling=exclude",
                "task.features.edge.edge_curvature=False",
            ],
        )
