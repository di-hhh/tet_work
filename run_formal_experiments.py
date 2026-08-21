from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from hydra import compose, initialize_config_dir

from src.formal_experiment_plan import (
    PIPELINE_REPO_ROOT,
    FormalPlanError,
    FormalRunSpec,
    dependency_checkpoint,
    iter_run_specs,
    load_formal_plan,
    resolve_amber_path,
    validate_frozen_dataset,
    validate_repository_freeze,
)
from src.initialization.init_config import load_omega_conf_resolvers
from src.tasks.pipeline_dataset_audit import AMBER_REPO_ROOT, audit_pipeline_dataset
from src.tasks.pipeline_dataset_fingerprint import verify_dataset_fingerprint


DEFAULT_PLAN = AMBER_REPO_ROOT / "config" / "formal_experiment_plan.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen formal experiment protocol")
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    train = subparsers.add_parser("train")
    train.add_argument("--dry-run", action="store_true")
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--dry-run", action="store_true")
    evaluate.add_argument("--overwrite", action="store_true")
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--output-dir")
    subparsers.add_parser("status")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = load_formal_plan(args.plan)
    if args.command == "status":
        print(json.dumps(protocol_status(plan), indent=2, ensure_ascii=False))
        return 0

    validate_formal_prerequisites(plan)
    if args.command == "validate":
        print(json.dumps({"status": "valid", "protocol_id": plan["protocol_id"]}, indent=2))
        return 0
    if args.command == "train":
        train_all(plan, dry_run=args.dry_run)
        return 0
    if args.command == "analyze":
        analyze_all(plan, output_dir=args.output_dir)
        return 0
    evaluate_all(plan, dry_run=args.dry_run, overwrite=args.overwrite)
    return 0


def validate_formal_prerequisites(plan: dict) -> None:
    expected_environment = str(plan["execution"]["conda_environment"])
    actual_environment = os.environ.get("CONDA_DEFAULT_ENV") or Path(sys.prefix).name
    if actual_environment != expected_environment:
        raise FormalPlanError(
            f"Formal protocol requires conda environment '{expected_environment}', got '{actual_environment}'"
        )
    frozen = validate_frozen_dataset(plan)
    validate_repository_freeze(plan)
    validate_composed_configs(plan)
    validate_live_dataset(plan, frozen=frozen)


def validate_composed_configs(plan: dict) -> None:
    load_omega_conf_resolvers()
    pipeline_root = resolve_amber_path(plan["dataset"]["pipeline_output_root"])
    geometry_root = resolve_amber_path(plan["dataset"]["geometry_source_root"])
    expected_epochs = int(plan["execution"]["training_budget_epochs"])
    expected_steps = int(plan["execution"]["inference_steps"])
    with initialize_config_dir(version_base=None, config_dir=str(AMBER_REPO_ROOT / "config")):
        for spec in iter_run_specs(plan):
            config = compose(
                config_name="training_config",
                overrides=[
                    f"+_runs/amber={spec.run_config}",
                    f"task.pipeline_output_root={pipeline_root.as_posix()}",
                    f"task.geometry_source_root={geometry_root.as_posix()}",
                    f"seed={spec.seed}",
                ],
            )
            identity = (
                str(config.experiment_protocol.method_id),
                str(config.experiment_protocol.analysis_id),
                str(config.experiment_protocol.method_role),
            )
            expected_identity = (spec.method_id, spec.analysis_id, spec.role)
            if identity != expected_identity:
                raise FormalPlanError(
                    f"Config identity mismatch for {spec.analysis_id}: expected={expected_identity}, actual={identity}"
                )
            if int(config.trainer.max_epochs) != expected_epochs:
                raise FormalPlanError(f"Training epochs changed for {spec.analysis_id}")
            if int(config.algorithm.inference_steps) != expected_steps:
                raise FormalPlanError(f"Inference steps changed for {spec.analysis_id}")
            if bool(config.experiment_protocol.run_test_after_fit):
                raise FormalPlanError(f"{spec.analysis_id} would test immediately after fit")
            if spec.analysis_id in {"M2", "M4"} and str(
                config.task.stage_field.projection_target
            ) != "learner_mesh":
                raise FormalPlanError(f"{spec.analysis_id} does not use direct learner-mesh projection")


def validate_live_dataset(plan: dict, *, frozen: dict) -> None:
    """Re-hash the current consumed dataset and verify frozen test composition."""

    pipeline_root = resolve_amber_path(plan["dataset"]["pipeline_output_root"])
    geometry_root = resolve_amber_path(plan["dataset"]["geometry_source_root"])
    with initialize_config_dir(version_base=None, config_dir=str(AMBER_REPO_ROOT / "config")):
        config = compose(
            config_name="training_config",
            overrides=[
                "+_runs/amber=amber_pipeline_baseline",
                f"task.pipeline_output_root={pipeline_root.as_posix()}",
                f"task.geometry_source_root={geometry_root.as_posix()}",
            ],
        )
    audit = audit_pipeline_dataset(config.task)
    audit.raise_for_errors()
    verify_dataset_fingerprint(
        frozen_payload=frozen,
        audit_result=audit,
        task_config=config.task,
    )

    test_records = audit.records_by_split.get("test", [])
    sample_counts = Counter(str(row.get("pde_family")) for row in test_records)
    geometry_sets: dict[str, set[str]] = defaultdict(set)
    for row in test_records:
        geometry_sets[str(row.get("pde_family"))].add(str(row.get("geometry_id")))
    geometry_counts = {family: len(values) for family, values in geometry_sets.items()}
    expected_samples = {
        str(key): int(value)
        for key, value in plan["dataset"]["expected_test_samples_by_pde"].items()
    }
    expected_geometries = {
        str(key): int(value)
        for key, value in plan["dataset"]["expected_test_geometries_by_pde"].items()
    }
    if dict(sample_counts) != expected_samples:
        raise FormalPlanError(
            f"Frozen per-PDE test sample counts changed: expected={expected_samples}, actual={dict(sample_counts)}"
        )
    if geometry_counts != expected_geometries:
        raise FormalPlanError(
            "Frozen per-PDE test geometry counts changed: "
            f"expected={expected_geometries}, actual={geometry_counts}"
        )


def train_all(plan: dict, *, dry_run: bool) -> None:
    for spec in iter_run_specs(plan):
        if _fit_complete(spec):
            print(f"SKIP complete fit: {spec.analysis_id} seed={spec.seed}")
            continue
        if spec.run_root.exists() and any(spec.run_root.iterdir()):
            raise FormalPlanError(
                f"Refusing to overwrite incomplete formal run: {spec.run_root}"
            )
        dependency = dependency_checkpoint(plan, spec)
        if dependency is not None and not dependency.exists() and not dry_run:
            raise FormalPlanError(f"Initialization checkpoint is missing: {dependency}")
        command = _training_command(plan=plan, spec=spec, dependency=dependency)
        _execute(command, cwd=AMBER_REPO_ROOT, dry_run=dry_run)


def evaluate_all(plan: dict, *, dry_run: bool, overwrite: bool) -> None:
    specs = list(iter_run_specs(plan))
    incomplete = [str(spec.run_root) for spec in specs if not _fit_complete(spec)]
    if incomplete and not dry_run:
        raise FormalPlanError(
            "No test may start until every preregistered fit is complete; "
            f"incomplete={incomplete[:10]}"
        )

    for spec in specs:
        marker = spec.run_root / "evaluation_complete.json"
        if marker.exists() and not overwrite:
            print(f"SKIP complete evaluation: {spec.analysis_id} seed={spec.seed}")
            continue
        command = [sys.executable, str(AMBER_REPO_ROOT / "evaluate_formal_checkpoint.py"), "--run-root", str(spec.run_root)]
        if overwrite:
            command.append("--overwrite")
        _execute(command, cwd=AMBER_REPO_ROOT, dry_run=dry_run)

    diagnostic_methods = set(plan.get("diagnostics", {}).get("same_geometry_condition_shuffle", []))
    for spec in specs:
        if spec.analysis_id not in diagnostic_methods:
            continue
        marker = spec.run_root / "diagnostics" / "condition_shuffle" / "evaluation_complete.json"
        if marker.exists() and not overwrite:
            print(f"SKIP complete shuffle diagnostic: {spec.analysis_id} seed={spec.seed}")
            continue
        command = [
            sys.executable,
            str(AMBER_REPO_ROOT / "evaluate_formal_checkpoint.py"),
            "--run-root",
            str(spec.run_root),
            "--mode",
            "condition_shuffle",
        ]
        if overwrite:
            command.append("--overwrite")
        _execute(command, cwd=AMBER_REPO_ROOT, dry_run=dry_run)

    reference_dir = resolve_amber_path(plan["execution"]["formal_root"]) / "static_references"
    reference_metadata = reference_dir / "reference_manifest_metadata.json"
    if overwrite or not reference_metadata.exists():
        _execute(
            [
                sys.executable,
                str(AMBER_REPO_ROOT / "build_formal_reference_manifests.py"),
                "--pipeline-root",
                str(resolve_amber_path(plan["dataset"]["pipeline_output_root"])),
                "--geometry-source-root",
                str(resolve_amber_path(plan["dataset"]["geometry_source_root"])),
                "--output-dir",
                str(reference_dir),
                "--protocol-id",
                str(plan["protocol_id"]),
                "--fingerprint-path",
                str(resolve_amber_path(plan["dataset"]["fingerprint_path"])),
            ],
            cwd=AMBER_REPO_ROOT,
            dry_run=dry_run,
        )

    pde_jobs = _pde_jobs(plan, specs=specs, reference_dir=reference_dir)
    for manifest, output_csv, aggregate_json in pde_jobs:
        if output_csv.exists() and aggregate_json.exists() and not overwrite:
            print(f"SKIP complete PDE evaluation: {output_csv}")
            continue
        _execute(
            [
                sys.executable,
                str(PIPELINE_REPO_ROOT / "evaluate_pipeline_pde.py"),
                "evaluate",
                "--root",
                str(resolve_amber_path(plan["dataset"]["pipeline_output_root"])),
                "--geometry-source-root",
                str(resolve_amber_path(plan["dataset"]["geometry_source_root"])),
                "--predictions",
                str(manifest),
                "--output-csv",
                str(output_csv),
                "--aggregate-json",
                str(aggregate_json),
                "--allow-failures",
            ],
            cwd=PIPELINE_REPO_ROOT,
            dry_run=dry_run,
        )


def protocol_status(plan: dict) -> dict:
    rows = []
    for spec in iter_run_specs(plan):
        rows.append(
            {
                "analysis_id": spec.analysis_id,
                "seed": spec.seed,
                "fit_complete": _fit_complete(spec),
                "test_complete": (spec.run_root / "evaluation_complete.json").exists(),
                "pde_complete": (spec.run_root / "pde_aggregate.json").exists(),
                "run_root": str(spec.run_root),
            }
        )
    return {"protocol_id": plan["protocol_id"], "runs": rows}


def analyze_all(plan: dict, *, output_dir: str | None) -> None:
    specs = list(iter_run_specs(plan))
    missing = [str(spec.run_root / "pde_metrics.csv") for spec in specs if not (spec.run_root / "pde_metrics.csv").exists()]
    if missing:
        raise FormalPlanError(f"Formal analysis requires every PDE evaluation: missing={missing[:10]}")
    command = [
        sys.executable,
        str(AMBER_REPO_ROOT / "analyze_formal_experiments.py"),
        "--plan",
        str(plan["_plan_path"]),
    ]
    if output_dir:
        command.extend(["--output-dir", str(Path(output_dir).resolve())])
    _execute(command, cwd=AMBER_REPO_ROOT, dry_run=False)


def _training_command(
    *,
    plan: dict,
    spec: FormalRunSpec,
    dependency: Path | None,
) -> list[str]:
    command = [
        sys.executable,
        str(AMBER_REPO_ROOT / "main.py"),
        f"+_runs/amber={spec.run_config}",
        f"task.pipeline_output_root={resolve_amber_path(plan['dataset']['pipeline_output_root']).as_posix()}",
        f"task.geometry_source_root={resolve_amber_path(plan['dataset']['geometry_source_root']).as_posix()}",
        f"seed={spec.seed}",
        f"hydra.run.dir={spec.run_root.as_posix()}",
        "logger.wandb.enabled=false",
    ]
    if dependency is not None:
        command.append(f"algorithm.init_from_weighted_baseline_checkpoint={dependency.as_posix()}")
    return command


def _pde_jobs(
    plan: dict,
    *,
    specs: list[FormalRunSpec],
    reference_dir: Path,
) -> list[tuple[Path, Path, Path]]:
    jobs: list[tuple[Path, Path, Path]] = []
    for spec in specs:
        jobs.append(
            (
                spec.run_root / "test_predictions" / "prediction_manifest.csv",
                spec.run_root / "pde_metrics.csv",
                spec.run_root / "pde_aggregate.json",
            )
        )
        if spec.analysis_id == "M4":
            jobs.append(
                (
                    spec.run_root / "test_predictions" / "expert_only_prediction_manifest.csv",
                    spec.run_root / "expert_only_pde_metrics.csv",
                    spec.run_root / "expert_only_pde_aggregate.json",
                )
            )
        if spec.analysis_id in set(plan.get("diagnostics", {}).get("same_geometry_condition_shuffle", [])):
            diagnostic = spec.run_root / "diagnostics" / "condition_shuffle"
            jobs.append(
                (
                    diagnostic / "test_predictions" / "prediction_manifest.csv",
                    diagnostic / "pde_metrics.csv",
                    diagnostic / "pde_aggregate.json",
                )
            )
    for reference_id in ("r0_initial", "r1_teacher"):
        jobs.append(
            (
                reference_dir / f"{reference_id}_prediction_manifest.csv",
                reference_dir / f"{reference_id}_pde_metrics.csv",
                reference_dir / f"{reference_id}_pde_aggregate.json",
            )
        )
    return jobs


def _fit_complete(spec: FormalRunSpec) -> bool:
    summary_path = spec.run_root / "training_summary.json"
    checkpoint_path = spec.run_root / "checkpoints" / "last.ckpt"
    if not summary_path.exists() or not checkpoint_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return summary.get("status") == "complete"


def _execute(command: list[str], *, cwd: Path, dry_run: bool) -> None:
    print(subprocess.list2cmdline(command))
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
