from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from lightning import Trainer
from omegaconf import OmegaConf

from src.experiment_artifacts import (
    ExperimentProtocolError,
    initialize_evaluation_context,
    preflight_experiment_protocol,
)
from src.initialization import initialize
from src.initialization.init_config import load_omega_conf_resolvers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one completed formal fit without running or resuming training"
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--mode",
        choices=("normal", "condition_shuffle"),
        default="normal",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_root = Path(args.run_root).resolve()
    config_path = run_root / "resolved_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    load_omega_conf_resolvers()
    config = OmegaConf.load(config_path)
    config.logger.wandb.enabled = False
    config.algorithm.plotting.sample_idxs = []
    preflight = preflight_experiment_protocol(config)

    artifact_root = (
        run_root
        if args.mode == "normal"
        else run_root / "diagnostics" / "condition_shuffle"
    )
    marker_path = artifact_root / "evaluation_complete.json"
    failure_path = artifact_root / "evaluation_failure.json"
    prediction_manifest = artifact_root / "test_predictions" / "prediction_manifest.csv"
    if not args.overwrite and (marker_path.exists() or prediction_manifest.exists()):
        raise FileExistsError(
            f"Evaluation artifacts already exist under {artifact_root}; use --overwrite to rerun"
        )
    if args.overwrite:
        marker_path.unlink(missing_ok=True)
        failure_path.unlink(missing_ok=True)

    started_counter = time.perf_counter()
    started_at = _beijing_now()
    try:
        initialization_return = initialize(config=config)
        metadata_overrides: dict[str, object] = {"evaluation_mode": args.mode}
        if args.mode == "condition_shuffle":
            if preflight["analysis_id"] not in {"M2", "M4"}:
                raise ExperimentProtocolError(
                    "condition_shuffle is preregistered only for M2 and M4"
                )
            mapping = apply_same_geometry_condition_shuffle(
                initialization_return.datasets["test"]
            )
            artifact_root.mkdir(parents=True, exist_ok=True)
            (artifact_root / "condition_shuffle_mapping.json").write_text(
                json.dumps(mapping, indent=2, sort_keys=True, ensure_ascii=False),
                encoding="utf-8",
            )
            metadata_overrides.update(
                {
                    "run_id": f"{preflight['protocol_id']}:{preflight['analysis_id']}-SHUFFLED:seed{int(config.seed)}",
                    "analysis_id": f"{preflight['analysis_id']}-SHUFFLED",
                    "method_role": "diagnostic",
                    "evaluation_prediction_variants": ["final"],
                }
            )

        evaluation_metadata = initialize_evaluation_context(
            config=config,
            run_root=run_root,
            artifact_root=artifact_root,
            initialization_return=initialization_return,
            preflight=preflight,
            metadata_overrides=metadata_overrides,
        )
        algorithm = initialization_return.algorithm
        if args.mode == "condition_shuffle":
            algorithm.local_evaluation_prediction_variants = ["final"]

        trainer_config = config.trainer
        if trainer_config.get("matmul_precision") is not None:
            torch.set_float32_matmul_precision(str(trainer_config.matmul_precision))
        trainer = Trainer(
            logger=False,
            callbacks=[],
            default_root_dir=str(artifact_root),
            accelerator=trainer_config.accelerator,
            devices=trainer_config.devices,
            precision=trainer_config.precision,
            enable_checkpointing=False,
            enable_progress_bar=True,
            enable_model_summary=False,
        )
        trainer.test(
            algorithm,
            dataloaders=initialization_return.dataloaders.get("test"),
            ckpt_path=str(run_root / "checkpoints" / "last.ckpt"),
        )

        expected_samples = len(initialization_return.datasets["test"])
        actual_samples = _csv_row_count(prediction_manifest)
        if actual_samples != expected_samples:
            raise RuntimeError(
                f"Prediction manifest has {actual_samples} rows; expected {expected_samples}"
            )
        if args.mode == "normal" and "expert_only" in preflight["evaluation_prediction_variants"]:
            expert_manifest = artifact_root / "test_predictions" / "expert_only_prediction_manifest.csv"
            expert_samples = _csv_row_count(expert_manifest)
            if expert_samples != expected_samples:
                raise RuntimeError(
                    f"Expert-only manifest has {expert_samples} rows; expected {expected_samples}"
                )

        payload = {
            "status": "complete",
            "mode": args.mode,
            "started_at_beijing": started_at,
            "finished_at_beijing": _beijing_now(),
            "evaluation_wall_time_seconds": float(time.perf_counter() - started_counter),
            "num_test_samples": expected_samples,
            "checkpoint": evaluation_metadata["evaluation_checkpoint"],
            "checkpoint_sha256": evaluation_metadata["evaluation_checkpoint_sha256"],
            "prediction_manifest": str(prediction_manifest),
        }
        marker_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        failure_path.unlink(missing_ok=True)
    except Exception as exc:
        artifact_root.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "mode": args.mode,
                    "started_at_beijing": started_at,
                    "finished_at_beijing": _beijing_now(),
                    "failure": f"{type(exc).__name__}: {exc}",
                },
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        raise
    return 0


def apply_same_geometry_condition_shuffle(dataset) -> list[dict[str, object]]:
    groups: dict[str, list] = defaultdict(list)
    for data in dataset.data:
        cache = data.source_data.imitation_weight_cache or {}
        groups[str(cache.get("geometry_id"))].append(data)

    mapping: list[dict[str, object]] = []
    for geometry_id, group in sorted(groups.items()):
        ordered = sorted(
            group,
            key=lambda data: str(data.source_data.imitation_weight_cache.get("sample_id")),
        )
        if len(ordered) < 2:
            raise ExperimentProtocolError(
                f"Geometry '{geometry_id}' has fewer than two conditions and cannot be shuffled"
            )
        assigned = ordered[1:] + ordered[:1]
        assigned_cache_snapshots = [
            dict(source.source_data.imitation_weight_cache or {}) for source in assigned
        ]
        for destination, source_cache in zip(ordered, assigned_cache_snapshots):
            destination_cache = destination.source_data.imitation_weight_cache
            destination_cache["stage_field_path"] = source_cache.get("stage_field_path")
            for key in list(destination_cache):
                if key.startswith("_pipeline_stage_field"):
                    destination_cache.pop(key, None)
            destination._observation = None
            destination.__dict__.pop("_physics_feature_bundle_cache_codex", None)
            mapping.append(
                {
                    "geometry_id": geometry_id,
                    "destination_sample_id": destination_cache.get("sample_id"),
                    "destination_condition_id": destination_cache.get("condition_id"),
                    "source_sample_id": source_cache.get("sample_id"),
                    "source_condition_id": source_cache.get("condition_id"),
                    "source_pde_family": source_cache.get("pde_family"),
                }
            )
    return mapping


def _csv_row_count(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _beijing_now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
