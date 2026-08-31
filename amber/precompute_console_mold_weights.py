import argparse
from pathlib import Path

DATASET_EXTENSIONS = {
    "console": "nas",
    "mold": "vtk",
}


def _iter_sample_bases(dataset_name: str, dataset_modes):
    dataset_root = Path("data") / dataset_name
    for dataset_mode in dataset_modes:
        mode_dir = dataset_root / dataset_mode
        if not mode_dir.exists():
            continue
        for geometry_path in sorted(mode_dir.glob("*.step")):
            yield geometry_path.with_suffix("")


def _build_source_data(*, dataset_name: str, sample_base: Path):
    from src.algorithm.dataloader.source_data import SourceData
    from src.mesh_util.load_mesh import load_expert_mesh
    from src.tasks.domains.mesh_wrapper import MeshWrapper

    expert_mesh = MeshWrapper(load_expert_mesh(expert_mesh_path=str(sample_base), extension=DATASET_EXTENSIONS[dataset_name]))
    feature_provider = None
    if dataset_name == "mold":
        from src.tasks.features.inlet_feature_provider import InletFeatureProvider

        inlet_file = sample_base.parent / f"{sample_base.name}_features.txt"
        feature_provider = InletFeatureProvider(inlet_file=inlet_file, observation_features=[])
    return SourceData(
        expert_mesh=expert_mesh,
        initial_mesh=expert_mesh,
        feature_provider=feature_provider,
        dataset_name=dataset_name,
        data_point_path=str(sample_base),
    )


def main():
    from src.algorithm.util.console_mold_reference import ensure_console_mold_reference_cache

    parser = argparse.ArgumentParser(description="Precompute weighted imitation reference caches for console and mold.")
    parser.add_argument("--datasets", nargs="+", default=["console", "mold"], choices=["console", "mold"])
    parser.add_argument("--modes", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--cache-dir", default="data/weighted_imitation")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    weighted_imitation_config = {
        "enabled": True,
        "datasets": args.datasets,
        "weight_source_mode": "console_mold_reference",
        "reference_physics_type": "linear_elasticity",
        "importance_metric": "strain_energy_density",
        "cache_dir": args.cache_dir,
        "auto_prepare": True,
        "overwrite_cache": args.overwrite,
        "young_modulus": 1.0,
        "poisson_ratio": 0.3,
        "fallback_to_ones": False,
    }  # [CodeX] 预处理脚本默认使用三维线弹性 + 应变能密度，与当前训练侧默认配置保持一致。

    prepared = 0
    for dataset_name in args.datasets:
        for sample_base in _iter_sample_bases(dataset_name=dataset_name, dataset_modes=args.modes):
            source_data = _build_source_data(dataset_name=dataset_name, sample_base=sample_base)
            ensure_console_mold_reference_cache(
                source_data=source_data,
                weighted_imitation_config=weighted_imitation_config,
                overwrite=args.overwrite,
            )
            prepared += 1
            print(f"Prepared weighted imitation cache for {dataset_name}:{sample_base.parent.name}/{sample_base.name}")

    print(f"Prepared {prepared} cache files.")


if __name__ == "__main__":
    main()
