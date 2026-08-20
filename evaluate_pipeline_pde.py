from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.condition_aware_dataset_generation.evaluation import (
    build_evaluation_references,
    evaluate_prediction_manifest,
)
from src.condition_aware_dataset_generation.evaluation.pde_evaluator import aggregate_pde_rows
from src.condition_aware_dataset_generation.utils import dump_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build strong references or evaluate predicted tetra meshes")
    subparsers = parser.add_subparsers(dest="command", required=True)

    reference = subparsers.add_parser("build-references")
    _add_roots(reference)
    reference.add_argument("--sample-id", action="append", dest="sample_ids")
    reference.add_argument("--no-reuse-scalar", action="store_true")
    reference.add_argument("--max-dofs", type=int)
    reference.add_argument("--max-matrix-nnz", type=int)

    evaluate = subparsers.add_parser("evaluate")
    _add_roots(evaluate)
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--prediction-root")
    evaluate.add_argument("--output-csv", required=True)
    evaluate.add_argument("--aggregate-json", required=True)
    evaluate.add_argument("--max-dofs", type=int)
    evaluate.add_argument("--max-matrix-nnz", type=int)
    evaluate.add_argument("--allow-failures", action="store_true")

    aggregate = subparsers.add_parser("aggregate", help="Recompute aggregate JSON from local per-sample CSV")
    aggregate.add_argument("--input-csv", required=True)
    aggregate.add_argument("--aggregate-json", required=True)
    return parser


def _add_roots(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", required=True, help="Pipeline output root")
    parser.add_argument("--geometry-source-root", required=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build-references":
        result = build_evaluation_references(
            pipeline_output_root=args.root,
            geometry_source_root=args.geometry_source_root,
            sample_ids=set(args.sample_ids) if args.sample_ids else None,
            uniform_refinement_level=1,
            reuse_audited_scalar_reference=not args.no_reuse_scalar,
            max_dofs=args.max_dofs,
            max_matrix_nnz=args.max_matrix_nnz,
        )
    elif args.command == "evaluate":
        result = evaluate_prediction_manifest(
            pipeline_output_root=args.root,
            geometry_source_root=args.geometry_source_root,
            prediction_manifest_path=args.predictions,
            prediction_root=args.prediction_root,
            output_csv_path=args.output_csv,
            aggregate_json_path=args.aggregate_json,
            max_dofs=args.max_dofs,
            max_matrix_nnz=args.max_matrix_nnz,
            fail_on_any_error=not args.allow_failures,
        )["aggregate"]
    else:
        with Path(args.input_csv).open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        result = aggregate_pde_rows(rows)
        dump_json(Path(args.aggregate_json), result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
