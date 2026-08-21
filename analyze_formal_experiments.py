from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.formal_experiment_analysis import analyze_formal_protocol
from src.formal_experiment_plan import load_formal_plan, resolve_amber_path


REPO_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run frozen geometry-level formal statistics")
    parser.add_argument("--plan", default=str(REPO_ROOT / "config" / "formal_experiment_plan.yaml"))
    parser.add_argument("--output-dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = load_formal_plan(args.plan)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else resolve_amber_path(plan["execution"]["formal_root"]) / "analysis"
    )
    result = analyze_formal_protocol(plan=plan, output_dir=output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
