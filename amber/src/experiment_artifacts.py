from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import lightning
import meshio
import skfem
import torch
import torch_geometric
from lightning.pytorch.callbacks import Callback
from omegaconf import DictConfig, ListConfig, OmegaConf

from src.tasks.pipeline_dataset_audit import AMBER_REPO_ROOT, DatasetPathResolver
from src.tasks.pipeline_dataset_fingerprint import verify_dataset_fingerprint


TET_WORK_ROOT = AMBER_REPO_ROOT.parent
PIPELINE_REPO_ROOT = TET_WORK_ROOT / "dataest-pipeline"


class ExperimentProtocolError(ValueError):
    pass


class LocalMetricsCallback(Callback):
    """Persist epoch metrics independently of W&B."""

    def __init__(self, run_root: str | Path):
        super().__init__()
        self.run_root = Path(run_root)
        self.rows: dict[str, dict[int, dict[str, Any]]] = {"train": {}, "val": {}}

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        if trainer.current_epoch > 0:
            self._capture(trainer, "train", epoch=int(trainer.current_epoch) - 1)

    def on_validation_end(self, trainer, pl_module) -> None:
        if not trainer.sanity_checking:
            self._capture(trainer, "val", epoch=int(trainer.current_epoch))

    def on_fit_end(self, trainer, pl_module) -> None:
        completed = int(trainer.fit_loop.epoch_progress.current.completed)
        last_epoch = max(completed - 1, 0)
        self._capture(trainer, "train", epoch=last_epoch)
        self._capture(trainer, "val", epoch=last_epoch)

    def _capture(self, trainer, split: str, *, epoch: int) -> None:
        prefix = f"metrics.{split}"
        row: dict[str, Any] = {"epoch": epoch}
        for key, value in trainer.callback_metrics.items():
            key_text = str(key)
            if not key_text.startswith(prefix):
                continue
            scalar = _to_scalar(value)
            if scalar is not None:
                row[key_text] = scalar
        if len(row) == 1:
            return
        self.rows[split][epoch] = row
        _write_csv_rows(
            self.run_root / f"{split}_metrics.csv",
            [self.rows[split][key] for key in sorted(self.rows[split])],
        )


class FormalLastCheckpointCallback(Callback):
    """Guarantee that last.ckpt represents the state at fit completion."""

    def __init__(self, run_root: str | Path):
        super().__init__()
        self.checkpoint_path = Path(run_root) / "checkpoints" / "last.ckpt"

    def on_fit_end(self, trainer, pl_module) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(self.checkpoint_path))


class TrainingRuntimeCallback(Callback):
    """Persist wall-clock training cost without changing optimization."""

    def __init__(self, run_root: str | Path):
        super().__init__()
        self.run_root = Path(run_root)
        self.started_counter: float | None = None
        self.started_at: str | None = None

    def on_fit_start(self, trainer, pl_module) -> None:
        self.started_counter = time.perf_counter()
        self.started_at = _beijing_now()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._write(trainer=trainer, status="running")

    def on_fit_end(self, trainer, pl_module) -> None:
        self._write(trainer=trainer, status="complete")

    def on_exception(self, trainer, pl_module, exception: BaseException) -> None:
        self._write(
            trainer=trainer,
            status="failed",
            failure=f"{type(exception).__name__}: {exception}",
        )

    def _write(self, *, trainer, status: str, failure: str | None = None) -> None:
        elapsed = None
        if self.started_counter is not None:
            elapsed = float(time.perf_counter() - self.started_counter)
        peak_gpu_memory = None
        if torch.cuda.is_available():
            peak_gpu_memory = int(torch.cuda.max_memory_allocated())
        payload = {
            "status": status,
            "started_at_beijing": self.started_at,
            "finished_at_beijing": _beijing_now() if status != "running" else None,
            "training_wall_time_seconds": elapsed,
            "training_updates": int(getattr(trainer, "global_step", 0)),
            "completed_epochs": int(trainer.fit_loop.epoch_progress.current.completed),
            "peak_gpu_memory_bytes": peak_gpu_memory,
            "failure": failure,
        }
        _write_json(self.run_root / "training_summary.json", payload)


def preflight_experiment_protocol(config: DictConfig) -> dict[str, Any]:
    protocol = OmegaConf.to_container(config.get("experiment_protocol", {}), resolve=True) or {}
    method_id = str(protocol.get("method_id", ""))
    if method_id not in {"M0", "M1", "M2", "M3", "M4"}:
        raise ExperimentProtocolError(f"Unknown or missing method_id '{method_id}'")
    if str(protocol.get("evaluation_checkpoint")) != "last.ckpt":
        raise ExperimentProtocolError("Formal evaluation_checkpoint must be exactly 'last.ckpt'")

    analysis_id = str(protocol.get("analysis_id", method_id))
    method_role = str(protocol.get("method_role", "core"))
    allowed_identity = {
        "M0": {("M0", "core")},
        "M1": {("M1", "core"), ("M1-FT", "auxiliary")},
        "M2": {("M2", "core")},
        "M3": {("M3", "oracle")},
        "M4": {("M4", "core")},
    }[method_id]
    if (analysis_id, method_role) not in allowed_identity:
        raise ExperimentProtocolError(
            f"Invalid experiment identity for {method_id}: analysis_id={analysis_id}, role={method_role}"
        )
    if method_id == "M3" and not bool(protocol.get("oracle_only", False)):
        raise ExperimentProtocolError("M3 must be explicitly marked oracle_only=true")
    if bool(protocol.get("formal_run", False)) and bool(protocol.get("run_test_after_fit", False)):
        raise ExperimentProtocolError("Formal fits must defer test evaluation until every preregistered fit is complete")

    evaluation_variants = [str(value) for value in protocol.get("evaluation_prediction_variants", ["final"])]
    if not evaluation_variants or any(value not in {"final", "expert_only"} for value in evaluation_variants):
        raise ExperimentProtocolError(f"Unsupported evaluation_prediction_variants={evaluation_variants}")
    if method_id == "M4" and set(evaluation_variants) != {"final", "expert_only"}:
        raise ExperimentProtocolError("M4 must evaluate both final and expert_only from the same checkpoint")
    if method_id != "M4" and evaluation_variants != ["final"]:
        raise ExperimentProtocolError(f"{method_id} must use evaluation_prediction_variants=[final]")

    budget_close_relative_tolerance = float(protocol.get("budget_close_relative_tolerance", 0.18))
    budget_valid_min_ratio = float(protocol.get("budget_valid_min_ratio", 0.8))
    budget_valid_max_ratio = float(protocol.get("budget_valid_max_ratio", 11000.0 / 7000.0))
    if not 0.0 <= budget_close_relative_tolerance < 1.0:
        raise ExperimentProtocolError("budget_close_relative_tolerance must be in [0, 1)")
    if not 0.0 < budget_valid_min_ratio <= 1.0 <= budget_valid_max_ratio:
        raise ExperimentProtocolError("Invalid formal budget-valid ratio interval")

    algorithm = config.algorithm
    task = config.task
    weighted_enabled = bool((algorithm.get("weighted_imitation") or {}).get("enabled", False))
    correction_enabled = bool(algorithm.get("enable_physics_correction_branch", False))
    feature_source = str(task.get("physics_feature_source"))
    gate_max = float(algorithm.get("gate_max", 1.0))
    expected = {
        "M0": (False, False, "pipeline_indicator"),
        "M1": (True, False, "pipeline_indicator"),
        "M2": (True, True, "stage_field_fusion"),
        "M3": (True, True, "pipeline_indicator"),
        "M4": (True, True, "stage_field_fusion"),
    }[method_id]
    actual = (weighted_enabled, correction_enabled, feature_source)
    if actual != expected:
        raise ExperimentProtocolError(
            f"{method_id} information flow mismatch: expected={expected}, actual={actual}"
        )
    if method_id == "M2":
        aux_values = {
            "gate_max": gate_max,
            "lambda_expert_aux": float(algorithm.get("lambda_expert_aux", 0.0)),
            "lambda_corr_aux": float(algorithm.get("lambda_corr_aux", 0.0)),
            "lambda_corr_reg": float(algorithm.get("lambda_corr_reg", 0.0)),
        }
        if any(value != 0.0 for value in aux_values.values()):
            raise ExperimentProtocolError(f"M2 final=expert contract requires all zeros: {aux_values}")
    if method_id in {"M2", "M4"}:
        projection_target = str((task.get("stage_field") or {}).get("projection_target", ""))
        if projection_target != "learner_mesh":
            raise ExperimentProtocolError(
                f"{method_id} requires direct probe projection to learner_mesh, got '{projection_target}'"
            )

    if analysis_id == "M1-FT":
        if str(protocol.get("initialization_mode")) != "weights_only_fresh_optimizer":
            raise ExperimentProtocolError("M1-FT must declare weights_only_fresh_optimizer initialization")
        if config.trainer.get("ckpt_path") not in {None, ""}:
            raise ExperimentProtocolError("M1-FT must not resume optimizer/scheduler state through trainer.ckpt_path")

    expected_budget = int(protocol.get("training_budget_epochs", config.trainer.max_epochs))
    if int(config.trainer.max_epochs) != expected_budget:
        raise ExperimentProtocolError(
            f"Training budget mismatch: trainer.max_epochs={config.trainer.max_epochs}, "
            f"protocol={expected_budget}"
        )

    init_report = _validate_seed_matched_initialization(config, protocol)
    fingerprint = _load_frozen_dataset_fingerprint(config, required=bool(protocol.get("formal_run", False)))
    code_versions = collect_code_versions()
    if bool(protocol.get("require_clean_repositories", False)):
        dirty = [name for name, info in code_versions.items() if info.get("dirty")]
        if dirty:
            raise ExperimentProtocolError(f"Formal run requires clean repositories; dirty={dirty}")

    return {
        "method_id": method_id,
        "analysis_id": analysis_id,
        "method_role": method_role,
        "oracle_only": bool(protocol.get("oracle_only", False)),
        "protocol_id": str(protocol.get("protocol_id", "")),
        "formal_run": bool(protocol.get("formal_run", False)),
        "run_test_after_fit": bool(protocol.get("run_test_after_fit", False)),
        "evaluation_prediction_variants": evaluation_variants,
        "budget_close_relative_tolerance": budget_close_relative_tolerance,
        "budget_valid_min_ratio": budget_valid_min_ratio,
        "budget_valid_max_ratio": budget_valid_max_ratio,
        "weighted_enabled": weighted_enabled,
        "weighted_source": str((algorithm.get("weighted_imitation") or {}).get("weight_source_mode")),
        "weighted_mode": str((algorithm.get("weighted_imitation") or {}).get("weight_mode")),
        "correction_enabled": correction_enabled,
        "physics_feature_source": feature_source,
        "stage_projection_target": str((task.get("stage_field") or {}).get("projection_target", "learner_mesh")),
        "gate_max": gate_max,
        "checkpoint_init_path": init_report.get("checkpoint_path"),
        "checkpoint_init_validated": init_report.get("validated", False),
        "inference_steps": int(algorithm.inference_steps),
        "training_budget_epochs": expected_budget,
        "evaluation_checkpoint": "last.ckpt",
        "dataset_fingerprint_sha256": fingerprint.get("dataset_fingerprint_sha256"),
        "code_versions": code_versions,
    }


def initialize_run_artifacts(
    *,
    config: DictConfig,
    run_root: str | Path,
    initialization_return,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    root = Path(run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=config, f=str(root / "resolved_config.yaml"), resolve=True)

    audit_result = getattr(initialization_return.datasets.get("train"), "pipeline_audit_result", None)
    if audit_result is None:
        raise ExperimentProtocolError("Pipeline audit result is unavailable after dataset initialization")
    frozen = _load_frozen_dataset_fingerprint(
        config,
        required=bool(preflight.get("formal_run", False)),
    )
    if bool(preflight.get("formal_run", False)):
        _validate_formal_dataset_artifacts(
            config=config,
            audit_result=audit_result,
            frozen=frozen,
        )
    split_sizes = {split: len(dataset) for split, dataset in initialization_return.datasets.items()}
    information_flow = {**preflight, "retained_split_sizes": split_sizes}
    _write_json(root / "information_flow.json", information_flow)
    print(json.dumps({"experiment_information_flow": information_flow}, indent=2, ensure_ascii=False))

    environment = collect_environment_versions()
    code_payload = {
        "repositories": preflight["code_versions"],
        "environment": environment,
    }
    _write_json(root / "code_version.json", code_payload)

    dataset_fingerprint = frozen.get("dataset_fingerprint_sha256")
    manifest_sha256 = _named_file_hash(frozen, "pipeline_output/manifests/sample_manifest.jsonl")
    dataset_payload = {
        "dataset_fingerprint_sha256": dataset_fingerprint,
        "frozen_fingerprint_path": _frozen_fingerprint_path(config).as_posix(),
        "manifest_sha256": manifest_sha256,
        "quality_report_sha256": _named_file_hash(frozen, "pipeline_output/reports/smoke_report.json"),
        "split_counts": audit_result.split_counts,
        "geometry_ids_by_split": audit_result.geometry_ids_by_split,
        "filter_counts": audit_result.counts,
        "filter_config": {
            key: _plain_config_value(config.task.get(key))
            for key in (
                "allowed_statuses",
                "budget_filter",
                "pde_family_filter",
                "quality_filter",
                "require_indicator",
                "physics_weight_source",
                "physics_feature_source",
            )
        },
    }
    _write_json(root / "dataset_version.json", dataset_payload)
    run_id = f"{preflight['protocol_id']}:{preflight['analysis_id']}:seed{int(config.seed)}"
    initialization_report = getattr(initialization_return.algorithm, "_weighted_baseline_init_report", None)
    run_metadata = {
        "run_id": run_id,
        "experiment_id": f"{config.exp_name}-v{config._version}-seed{config.seed}",
        "protocol_id": preflight["protocol_id"],
        "method_id": preflight["method_id"],
        "analysis_id": preflight["analysis_id"],
        "method_role": preflight["method_role"],
        "oracle_only": preflight["oracle_only"],
        "seed": int(config.seed),
        "evaluation_checkpoint": "checkpoints/last.ckpt",
        "evaluation_prediction_variants": preflight["evaluation_prediction_variants"],
        "budget_close_relative_tolerance": preflight["budget_close_relative_tolerance"],
        "budget_valid_min_ratio": preflight["budget_valid_min_ratio"],
        "budget_valid_max_ratio": preflight["budget_valid_max_ratio"],
        "dataset_fingerprint_sha256": dataset_fingerprint,
        "manifest_sha256": manifest_sha256,
        "amber_code_commit": preflight["code_versions"]["amber"]["commit"],
        "pipeline_code_commit": preflight["code_versions"]["dataest-pipeline"]["commit"],
        "initialization_mode": (OmegaConf.to_container(config.experiment_protocol, resolve=True) or {}).get(
            "initialization_mode",
            "from_scratch" if not preflight["checkpoint_init_path"] else "weights_only_fresh_optimizer",
        ),
        "initialization_checkpoint": preflight["checkpoint_init_path"],
        "initialization_report": initialization_report,
        "launch_command": subprocess.list2cmdline(sys.argv),
        "created_at_beijing": _beijing_now(),
    }
    _write_json(root / "run_metadata.json", run_metadata)

    algorithm = initialization_return.algorithm
    algorithm.local_artifact_root = str(root)
    algorithm.local_run_metadata = run_metadata
    algorithm.local_evaluation_prediction_variants = preflight["evaluation_prediction_variants"]
    return information_flow


def initialize_evaluation_context(
    *,
    config: DictConfig,
    run_root: str | Path,
    artifact_root: str | Path,
    initialization_return,
    preflight: dict[str, Any],
    metadata_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a completed fit and attach read-only test artifact metadata."""

    root = Path(run_root).resolve()
    output_root = Path(artifact_root).resolve()
    metadata_path = root / "run_metadata.json"
    training_summary_path = root / "training_summary.json"
    checkpoint_path = root / "checkpoints" / "last.ckpt"
    for required_path in (metadata_path, training_summary_path, checkpoint_path):
        if not required_path.exists():
            raise ExperimentProtocolError(f"Formal evaluation prerequisite is missing: {required_path}")

    training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
    if training_summary.get("status") != "complete":
        raise ExperimentProtocolError(
            f"Formal evaluation requires a complete fit, got status={training_summary.get('status')}"
        )

    run_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "method_id": preflight["method_id"],
        "analysis_id": preflight["analysis_id"],
        "seed": int(config.seed),
        "dataset_fingerprint_sha256": preflight["dataset_fingerprint_sha256"],
        "amber_code_commit": preflight["code_versions"]["amber"]["commit"],
        "pipeline_code_commit": preflight["code_versions"]["dataest-pipeline"]["commit"],
    }
    mismatches = {
        key: {"fit": run_metadata.get(key), "evaluation": value}
        for key, value in expected.items()
        if run_metadata.get(key) != value
    }
    if mismatches:
        raise ExperimentProtocolError(f"Fit/evaluation identity mismatch: {mismatches}")

    audit_result = getattr(initialization_return.datasets.get("train"), "pipeline_audit_result", None)
    if audit_result is None:
        raise ExperimentProtocolError("Pipeline audit result is unavailable for formal evaluation")
    is_formal_run = bool(preflight.get("formal_run", False))
    frozen = _load_frozen_dataset_fingerprint(config, required=is_formal_run)
    if is_formal_run:
        _validate_formal_dataset_artifacts(config=config, audit_result=audit_result, frozen=frozen)

    evaluation_metadata = dict(run_metadata)
    evaluation_metadata.update(metadata_overrides or {})
    evaluation_metadata["evaluation_checkpoint"] = str(checkpoint_path)
    evaluation_metadata["evaluation_checkpoint_sha256"] = _sha256_file(checkpoint_path)
    evaluation_metadata["evaluation_started_at_beijing"] = _beijing_now()
    evaluation_metadata["artifact_root"] = str(output_root)

    output_root.mkdir(parents=True, exist_ok=True)
    algorithm = initialization_return.algorithm
    algorithm.local_artifact_root = str(output_root)
    algorithm.local_run_metadata = evaluation_metadata
    algorithm.local_evaluation_prediction_variants = preflight["evaluation_prediction_variants"]
    return evaluation_metadata


def _validate_formal_dataset_artifacts(
    *,
    config: DictConfig,
    audit_result,
    frozen: dict[str, Any],
) -> None:
    """Require Gate B/D artifacts before a formal fit can start."""
    verify_dataset_fingerprint(
        frozen_payload=frozen,
        audit_result=audit_result,
        task_config=config.task,
    )
    required_split_values = config.task.get("evaluation_reference_required_splits", ["test"])
    if not isinstance(required_split_values, (list, tuple, ListConfig)):
        required_split_values = [required_split_values]
    required_splits = {str(value) for value in required_split_values}
    missing_references: list[str] = []
    reference_paths: list[Path] = []
    for split in sorted(required_splits):
        for record in audit_result.records_by_split.get(split, []):
            resolved = dict(record.get("_resolved_paths", {}) or {})
            reference_value = resolved.get("optional_evaluation_reference_path")
            metadata_value = resolved.get("evaluation_reference_metadata_path")
            if not reference_value or not metadata_value:
                missing_references.append(str(record.get("sample_id")))
                continue
            reference_paths.extend([Path(reference_value).resolve(), Path(metadata_value).resolve()])
    if missing_references:
        raise ExperimentProtocolError(
            "Formal run requires validated strong references for every required evaluation sample; "
            f"missing={missing_references[:10]}"
        )

    expected_counts = frozen.get("split_counts")
    if expected_counts != audit_result.split_counts:
        raise ExperimentProtocolError(
            "Frozen dataset split counts do not match the audited run view: "
            f"frozen={expected_counts}, audited={audit_result.split_counts}"
        )
    frozen_paths = {str(entry.get("path")) for entry in frozen.get("files", [])}
    required_catalog_paths = {
        "pipeline_output/manifests/evaluation_reference_manifest.jsonl"
    }
    output_root = Path(audit_result.pipeline_output_root).resolve()
    for path in reference_paths:
        try:
            required_catalog_paths.add(f"pipeline_output/{path.relative_to(output_root).as_posix()}")
        except ValueError as exc:
            raise ExperimentProtocolError(
                f"Strong evaluation reference is outside pipeline_output_root: {path}"
            ) from exc
    missing_from_fingerprint = sorted(required_catalog_paths.difference(frozen_paths))
    if missing_from_fingerprint:
        raise ExperimentProtocolError(
            "Frozen dataset fingerprint predates or omits strong evaluation references: "
            f"missing={missing_from_fingerprint[:10]}"
        )


def collect_code_versions() -> dict[str, dict[str, Any]]:
    return {
        "amber": _git_version(AMBER_REPO_ROOT),
        "dataest-pipeline": _git_version(PIPELINE_REPO_ROOT),
    }


def collect_environment_versions() -> dict[str, Any]:
    import gmsh
    import sklearn
    import torch_scatter

    scatter_source = torch.tensor([1.0, 2.0, 3.0])
    scatter_index = torch.tensor([0, 1, 0])
    scatter_value = torch_scatter.scatter_add(scatter_source, scatter_index).tolist()
    cuda_devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": cuda_devices,
        "lightning": lightning.__version__,
        "torch_geometric": torch_geometric.__version__,
        "torch_scatter": torch_scatter.__version__,
        "torch_scatter_operation": scatter_value,
        "meshio": meshio.__version__,
        "scikit_fem": skfem.__version__,
        "scikit_learn": sklearn.__version__,
        "gmsh": getattr(gmsh, "__version__", "import-ok"),
    }


def _validate_seed_matched_initialization(
    config: DictConfig, protocol: dict[str, Any]
) -> dict[str, Any]:
    required = bool(protocol.get("require_seed_matched_m1_checkpoint", False))
    raw_path = config.algorithm.get("init_from_weighted_baseline_checkpoint")
    if not required:
        return {"validated": raw_path in {None, ""}, "checkpoint_path": raw_path}
    if raw_path in {None, ""}:
        raise ExperimentProtocolError(
            f"{protocol.get('method_id')} requires the same-seed M1 last.ckpt initialization"
        )
    if str(raw_path).lower() == "auto":
        raw_path = _find_unique_m1_checkpoint(config, protocol)
    checkpoint_path = Path(str(raw_path))
    if not checkpoint_path.is_absolute():
        checkpoint_path = AMBER_REPO_ROOT / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.exists() or checkpoint_path.name != "last.ckpt":
        raise ExperimentProtocolError(f"Required M1 last.ckpt not found: {checkpoint_path}")
    m1_config_path = checkpoint_path.parent.parent / "resolved_config.yaml"
    if not m1_config_path.exists():
        raise ExperimentProtocolError(f"M1 resolved_config.yaml not found beside checkpoint: {m1_config_path}")
    m1_config = OmegaConf.load(m1_config_path)
    if str(m1_config.experiment_protocol.method_id) != "M1":
        raise ExperimentProtocolError(f"Initialization checkpoint is not from M1: {checkpoint_path}")
    source_analysis_id = str(m1_config.experiment_protocol.get("analysis_id", "M1"))
    if source_analysis_id != "M1":
        raise ExperimentProtocolError(
            f"Initialization checkpoint must be the base M1 run, got analysis_id={source_analysis_id}: "
            f"{checkpoint_path}"
        )
    if int(m1_config.seed) != int(config.seed):
        raise ExperimentProtocolError(
            f"Seed mismatch: current={config.seed}, M1 checkpoint seed={m1_config.seed}"
        )
    if int(m1_config.trainer.max_epochs) != int(config.trainer.max_epochs):
        raise ExperimentProtocolError(
            "M1 initialization and current method do not share the same training budget"
        )
    current_fingerprint = _load_frozen_dataset_fingerprint(config, required=False).get(
        "dataset_fingerprint_sha256"
    )
    source_metadata_path = checkpoint_path.parent.parent / "run_metadata.json"
    if current_fingerprint and source_metadata_path.exists():
        source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
        if source_metadata.get("dataset_fingerprint_sha256") != current_fingerprint:
            raise ExperimentProtocolError(
                "M1 initialization and current method do not share the same dataset fingerprint"
            )
    config.algorithm.init_from_weighted_baseline_checkpoint = str(checkpoint_path)
    return {"validated": True, "checkpoint_path": str(checkpoint_path)}


def _find_unique_m1_checkpoint(config: DictConfig, protocol: dict[str, Any]) -> str:
    root_value = protocol.get("m1_checkpoint_search_root", "output/hydra/training")
    search_root = Path(str(root_value))
    if not search_root.is_absolute():
        search_root = AMBER_REPO_ROOT / search_root
    search_root = search_root.resolve()
    expected_fingerprint = _load_frozen_dataset_fingerprint(config, required=False).get(
        "dataset_fingerprint_sha256"
    )
    candidates: list[Path] = []
    if search_root.exists():
        for metadata_path in search_root.rglob("run_metadata.json"):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if str(metadata.get("method_id")) != "M1" or int(metadata.get("seed", -1)) != int(config.seed):
                continue
            if str(metadata.get("analysis_id", "M1")) != "M1":
                continue
            if expected_fingerprint and metadata.get("dataset_fingerprint_sha256") != expected_fingerprint:
                continue
            checkpoint = metadata_path.parent / "checkpoints" / "last.ckpt"
            if checkpoint.exists():
                candidates.append(checkpoint.resolve())
    if len(candidates) != 1:
        raise ExperimentProtocolError(
            f"Automatic same-seed M1 lookup under '{search_root}' found {len(candidates)} candidates; "
            f"expected exactly one. candidates={[str(path) for path in candidates[:5]]}"
        )
    return str(candidates[0])


def _load_frozen_dataset_fingerprint(config: DictConfig, *, required: bool) -> dict[str, Any]:
    path = _frozen_fingerprint_path(config)
    if not path.exists():
        if required:
            raise ExperimentProtocolError(
                f"Frozen dataset fingerprint is required before formal training: {path}"
            )
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _frozen_fingerprint_path(config: DictConfig) -> Path:
    resolver = DatasetPathResolver(OmegaConf.to_container(config.task, resolve=True))
    value = str(config.task.get("dataset_fingerprint_path", "manifests/dataset_fingerprint.json"))
    path = Path(value)
    if not path.is_absolute():
        path = resolver.pipeline_output_root / path
    return path.resolve()


def _git_version(root: Path) -> dict[str, Any]:
    commit = _git(root, "rev-parse", "HEAD")
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    status = _git(root, "status", "--porcelain")
    return {
        "root": str(root),
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "status_porcelain": status.splitlines(),
    }


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _named_file_hash(payload: dict[str, Any], display_path: str) -> str | None:
    for entry in payload.get("files", []):
        if entry.get("path") == display_path:
            return str(entry.get("sha256"))
    return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def _beijing_now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}, key=lambda key: (key != "epoch", key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _to_scalar(value: Any) -> float | int | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu().item())
    if isinstance(value, (int, float)):
        return value
    return None


def _plain_config_value(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value
