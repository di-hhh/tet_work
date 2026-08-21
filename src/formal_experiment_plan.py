from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from omegaconf import OmegaConf

from src.tasks.pipeline_dataset_audit import AMBER_REPO_ROOT


PIPELINE_REPO_ROOT = AMBER_REPO_ROOT.parent / "dataest-pipeline"


class FormalPlanError(ValueError):
    pass


@dataclass(frozen=True)
class FormalRunSpec:
    analysis_id: str
    method_id: str
    role: str
    run_config: str
    seed: int
    run_root: Path
    initialize_from: str | None = None
    oracle_only: bool = False


def load_formal_plan(path: str | Path) -> dict[str, Any]:
    plan_path = Path(path).resolve()
    payload = OmegaConf.to_container(OmegaConf.load(plan_path), resolve=True)
    if not isinstance(payload, dict):
        raise FormalPlanError(f"Formal plan must be a mapping: {plan_path}")
    validate_plan_schema(payload)
    payload["_plan_path"] = str(plan_path)
    return payload


def validate_plan_schema(plan: dict[str, Any]) -> None:
    if int(plan.get("schema_version", -1)) != 1:
        raise FormalPlanError("Unsupported formal plan schema_version")
    if not plan.get("protocol_id"):
        raise FormalPlanError("protocol_id is required")
    constraints = plan.get("hard_constraints", {})
    if constraints.get("modify_model_architecture") is not False:
        raise FormalPlanError("Formal plan must forbid model architecture changes")
    if constraints.get("modify_loss") is not False:
        raise FormalPlanError("Formal plan must forbid loss changes")
    if constraints.get("stage_projection_target") != "learner_mesh":
        raise FormalPlanError("Formal plan must project stage probes directly to learner_mesh")
    if constraints.get("test_after_fit") is not False:
        raise FormalPlanError("Formal plan must defer all tests")

    methods = plan.get("methods", [])
    identities = [(str(row.get("analysis_id")), int(seed)) for row in methods for seed in row.get("seeds", [])]
    if not identities or len(identities) != len(set(identities)):
        raise FormalPlanError("Formal method/seed identities must be non-empty and unique")
    by_analysis = {str(row.get("analysis_id")): row for row in methods}
    required_core = {"M0", "M1", "M2", "M4"}
    if not required_core.issubset(by_analysis):
        raise FormalPlanError(f"Missing core methods: {sorted(required_core.difference(by_analysis))}")
    for method in required_core:
        if list(by_analysis[method].get("seeds", [])) != [0, 1, 2]:
            raise FormalPlanError(f"{method} must use seeds [0, 1, 2]")
    if by_analysis.get("M3", {}).get("role") != "oracle" or not by_analysis.get("M3", {}).get("oracle_only"):
        raise FormalPlanError("M3 must be an explicitly oracle-only diagnostic")
    if by_analysis.get("M1-FT", {}).get("method_id") != "M1":
        raise FormalPlanError("M1-FT must remain method_id M1")
    for row in methods:
        dependency = row.get("initialize_from")
        if dependency and str(dependency) not in by_analysis:
            raise FormalPlanError(f"Unknown initialization dependency '{dependency}'")

    planned = {
        (str(row.get("left")), str(row.get("right")))
        for row in plan.get("statistics", {}).get("planned_contrasts", [])
    }
    expected = {("M1", "M0"), ("M1-FT", "M1"), ("M2", "M1-FT"), ("M4", "M2")}
    if planned != expected:
        raise FormalPlanError(f"Planned contrasts changed: expected={expected}, actual={planned}")
    if plan.get("dataset", {}).get("expected_test_samples_by_pde") != {
        "scalar_elliptic": 20,
        "linear_elasticity": 22,
    }:
        raise FormalPlanError("Frozen per-PDE test sample counts changed")
    if plan.get("dataset", {}).get("expected_test_geometries_by_pde") != {
        "scalar_elliptic": 10,
        "linear_elasticity": 11,
    }:
        raise FormalPlanError("Frozen per-PDE test geometry counts changed")


def iter_run_specs(plan: dict[str, Any]) -> Iterable[FormalRunSpec]:
    formal_root = resolve_amber_path(plan["execution"]["formal_root"])
    for method in plan["methods"]:
        for seed in method["seeds"]:
            yield FormalRunSpec(
                analysis_id=str(method["analysis_id"]),
                method_id=str(method["method_id"]),
                role=str(method["role"]),
                run_config=str(method["run_config"]),
                seed=int(seed),
                run_root=formal_root / str(method["analysis_id"]) / f"seed_{int(seed):02d}",
                initialize_from=(
                    str(method["initialize_from"]) if method.get("initialize_from") else None
                ),
                oracle_only=bool(method.get("oracle_only", False)),
            )


def dependency_checkpoint(plan: dict[str, Any], spec: FormalRunSpec) -> Path | None:
    if spec.initialize_from is None:
        return None
    matches = [
        candidate
        for candidate in iter_run_specs(plan)
        if candidate.analysis_id == spec.initialize_from and candidate.seed == spec.seed
    ]
    if len(matches) != 1:
        raise FormalPlanError(
            f"Expected one {spec.initialize_from} seed {spec.seed} dependency, got {len(matches)}"
        )
    return matches[0].run_root / "checkpoints" / "last.ckpt"


def validate_frozen_dataset(plan: dict[str, Any]) -> dict[str, Any]:
    fingerprint_path = resolve_amber_path(plan["dataset"]["fingerprint_path"])
    if not fingerprint_path.exists():
        raise FormalPlanError(f"Frozen fingerprint does not exist: {fingerprint_path}")
    payload = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    expected = str(plan["dataset"]["fingerprint_sha256"])
    actual = str(payload.get("dataset_fingerprint_sha256"))
    if actual != expected:
        raise FormalPlanError(f"Frozen dataset fingerprint mismatch: expected={expected}, actual={actual}")
    expected_splits = {key: int(value) for key, value in plan["dataset"]["expected_split_samples"].items()}
    if payload.get("split_counts") != expected_splits:
        raise FormalPlanError(
            f"Frozen split counts changed: expected={expected_splits}, actual={payload.get('split_counts')}"
        )
    return payload


def validate_repository_freeze(plan: dict[str, Any]) -> None:
    required_tag = str(plan["required_local_tag"])
    for name, root in (("amber", AMBER_REPO_ROOT), ("dataest-pipeline", PIPELINE_REPO_ROOT)):
        status = _git(root, "status", "--porcelain")
        if status:
            raise FormalPlanError(f"{name} repository is dirty")
        tags = set(_git(root, "tag", "--points-at", "HEAD").splitlines())
        if required_tag not in tags:
            raise FormalPlanError(f"{name} HEAD is not tagged '{required_tag}'")


def resolve_amber_path(value: str | Path) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = AMBER_REPO_ROOT / path
    return path.resolve()


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()
