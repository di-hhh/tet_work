from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from omegaconf import OmegaConf

from src.tasks.pipeline_dataset_audit import audit_pipeline_dataset
from src.tasks.pipeline_dataset_fingerprint import (
    build_dataset_fingerprint,
    verify_dataset_fingerprint,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_TASK_CONFIG = REPO_ROOT / "config" / "task" / "pipeline_condition_aware.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit a pipeline dataset using the Amber task protocol")
    parser.add_argument("--root", required=True, help="Pipeline output root")
    parser.add_argument("--geometry-source-root", required=True, help="Independent geometry-source anchor")
    parser.add_argument("--config", default=str(DEFAULT_TASK_CONFIG), help="Task or resolved run YAML")
    parser.add_argument("--json-output", help="Optional audit JSON path")
    parser.add_argument("--csv-output", help="Optional retained-sample CSV path")
    parser.add_argument("--freeze-fingerprint", help="Write a new frozen dataset fingerprint JSON")
    parser.add_argument("--verify-fingerprint", help="Verify an existing frozen fingerprint JSON")
    parser.add_argument("--allow-errors", action="store_true", help="Write diagnostics and exit zero despite audit issues")
    return parser


def load_task_config(path_value: str, *, root: str, geometry_source_root: str):
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    payload = OmegaConf.load(path.resolve())
    if "task" in payload:
        payload = payload.task
    payload.pipeline_output_root = root
    payload.geometry_source_root = geometry_source_root
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_task_config(args.config, root=args.root, geometry_source_root=args.geometry_source_root)
    result = audit_pipeline_dataset(config)
    payload = result.to_dict(include_records=False)
    console_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"consumed_paths", "geometry_ids_by_split"}
    }
    console_payload["consumed_path_count"] = len(result.consumed_paths)
    print(json.dumps(console_payload, indent=2, ensure_ascii=False))

    if args.json_output:
        _write_json(Path(args.json_output), payload)
    if args.csv_output:
        _write_retained_csv(Path(args.csv_output), result.records_by_split)
    if args.freeze_fingerprint:
        frozen = build_dataset_fingerprint(audit_result=result, task_config=config)
        _write_json(Path(args.freeze_fingerprint), frozen)
    if args.verify_fingerprint:
        frozen = json.loads(Path(args.verify_fingerprint).read_text(encoding="utf-8"))
        verify_dataset_fingerprint(frozen_payload=frozen, audit_result=result, task_config=config)
    if result.issues and not args.allow_errors:
        result.raise_for_errors()
    return 0


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def _write_retained_csv(path: Path, records_by_split: dict[str, list[dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_id", "geometry_id", "condition_id", "split", "pde_family", "budget", "status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split in ("train", "val", "test"):
            for record in records_by_split.get(split, []):
                writer.writerow({key: record.get(key) for key in fieldnames})


if __name__ == "__main__":
    raise SystemExit(main())
