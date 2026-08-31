from __future__ import annotations

import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir

from src.experiment_artifacts import collect_code_versions
from src.formal_reference_manifests import build_static_reference_manifests
from src.initialization.init_config import load_omega_conf_resolvers
from src.tasks.pipeline_dataset_audit import audit_pipeline_dataset


REPO_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build frozen R0-Initial and R1-Teacher manifests")
    parser.add_argument("--pipeline-root", required=True)
    parser.add_argument("--geometry-source-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protocol-id", default="formal-mold7000-v3")
    parser.add_argument(
        "--fingerprint-path",
        help="Frozen fingerprint path (defaults to task.dataset_fingerprint_path under --pipeline-root)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_omega_conf_resolvers()
    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "config")):
        config = compose(
            config_name="training_config",
            overrides=[
                "+_runs/amber=amber_pipeline_baseline",
                f"task.pipeline_output_root={Path(args.pipeline_root).resolve().as_posix()}",
                f"task.geometry_source_root={Path(args.geometry_source_root).resolve().as_posix()}",
            ],
        )
    audit = audit_pipeline_dataset(config.task)
    audit.raise_for_errors()
    fingerprint_path = (
        Path(args.fingerprint_path).resolve()
        if args.fingerprint_path
        else Path(args.pipeline_root).resolve() / str(config.task.dataset_fingerprint_path)
    )
    fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    manifest_sha256 = next(
        (
            row.get("sha256")
            for row in fingerprint.get("files", [])
            if row.get("path") == "pipeline_output/manifests/sample_manifest.jsonl"
        ),
        None,
    )
    result = build_static_reference_manifests(
        audit_result=audit,
        output_dir=args.output_dir,
        protocol_id=args.protocol_id,
        dataset_fingerprint_sha256=str(fingerprint["dataset_fingerprint_sha256"]),
        manifest_sha256=manifest_sha256,
        code_versions=collect_code_versions(),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
