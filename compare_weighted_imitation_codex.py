from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from src.algorithm.util.weighted_imitation_diagnostics_codex import (
    build_weights_from_normalized_importance_codex,
    compute_distribution_stats_codex,
    normalize_importance_codex,
)


def build_mode_overrides_codex(dataset: str) -> List[Dict[str, object]]:
    # [CodeX] 统一定义最小对比实验候选，保证 baseline / linear / power / binary / stage2 命令可复现。
    base_run = f"amber_{dataset}"
    weighted_run = f"amber_{dataset}_weighted"
    return [
        {
            "name": "baseline",
            "run": base_run,
            "cache_mode": "ones",
            "overrides": [
                "algorithm.weighted_imitation.enabled=False",
                "algorithm.weighted_imitation.metric_use_physics_weights=True",
            ],
        },
        {
            "name": "linear",
            "run": weighted_run,
            "overrides": [
                "algorithm.weighted_imitation.weight_mode=linear",
            ],
        },
        {
            "name": "power",
            "run": weighted_run,
            "overrides": [
                "algorithm.weighted_imitation.weight_mode=power",
                "algorithm.weighted_imitation.gamma=3.0",
            ],
        },
        {
            "name": "binary_topk",
            "run": weighted_run,
            "overrides": [
                "algorithm.weighted_imitation.weight_mode=binary_topk",
                "algorithm.weighted_imitation.lambda_high=8.0",
                "algorithm.weighted_imitation.topk_percent=0.2",
            ],
        },
        {
            "name": "ternary_quantile",
            "run": weighted_run,
            "overrides": [
                "algorithm.weighted_imitation.weight_mode=ternary_quantile",
                "algorithm.weighted_imitation.lambda_mid=2.0",
                "algorithm.weighted_imitation.lambda_high=8.0",
                "algorithm.weighted_imitation.ternary_low_quantile=0.5",
                "algorithm.weighted_imitation.ternary_high_quantile=0.8",
            ],
        },
    ]


def append_stage2_overrides_codex(commands: List[Dict[str, object]], *, ckpt_path: str) -> List[Dict[str, object]]:
    # [CodeX] 若提供 checkpoint，则自动补两阶段微调候选，便于直接比较“先学全局、再压高重要区”的效果。
    stage2_candidates = [
        {
            "name": "baseline_stage2",
            "run": commands[0]["run"],
            "overrides": [
                f"trainer.ckpt_path={ckpt_path}",
                "algorithm.weighted_imitation.stage2_enable=True",
                "algorithm.weighted_imitation.stage2_weight_mode=binary_topk",
                "algorithm.weighted_imitation.stage2_high_importance_only=True",
            ],
        },
        {
            "name": "power_stage2",
            "run": commands[2]["run"],
            "overrides": commands[2]["overrides"]
            + [
                f"trainer.ckpt_path={ckpt_path}",
                "algorithm.weighted_imitation.stage2_enable=True",
                "algorithm.weighted_imitation.stage2_weight_mode=binary_topk",
                "algorithm.weighted_imitation.stage2_high_importance_only=True",
                "algorithm.weighted_imitation.stage2_lambda_high=8.0",
            ],
        },
    ]
    return commands + stage2_candidates


def analyze_cache_weight_modes_codex(*, dataset: str, cache_dir: Path, max_samples: int) -> Dict[str, Dict[str, float]]:
    # [CodeX] 在无完整训练依赖时，直接基于缓存重要性比较不同权重模式的集中度，先回答“当前权重是不是太平”。
    sample_paths = sorted((cache_dir / dataset).rglob("*.npz"))
    if max_samples > 0:
        sample_paths = sample_paths[:max_samples]
    if not sample_paths:
        raise FileNotFoundError(f"No cache files found under '{cache_dir / dataset}'.")

    mode_stats = {mode_spec["name"]: [] for mode_spec in build_mode_overrides_codex(dataset)}
    stale_cache_count = 0
    for sample_path in sample_paths:
        with np.load(sample_path, allow_pickle=True) as data:
            raw_importance = np.asarray(data["vertex_importance"], dtype=np.float64)
            if "reference_physics_type" not in data.files:
                stale_cache_count += 1
            normalized = normalize_importance_codex(raw_importance, config={})
            for mode_spec in build_mode_overrides_codex(dataset):
                if mode_spec.get("cache_mode") == "ones":
                    weights = np.ones_like(normalized, dtype=np.float32)
                else:
                    mode_config = _mode_config_from_overrides_codex(mode_spec["overrides"])
                    weights = build_weights_from_normalized_importance_codex(normalized, config=mode_config)
                stats = compute_distribution_stats_codex(weights, prefix="")
                mode_stats[mode_spec["name"]].append(stats)

    aggregated = {mode_name: _aggregate_stats_codex(stats_list) for mode_name, stats_list in mode_stats.items()}
    aggregated["_meta"] = {
        "num_samples": float(len(sample_paths)),
        "stale_cache_count": float(stale_cache_count),
    }
    return aggregated


def run_commands_codex(*, dataset: str, commands: Iterable[Dict[str, object]], python_executable: str, execute: bool) -> List[List[str]]:
    emitted_commands = []
    for command_spec in commands:
        command = [
            python_executable,
            "main.py",
            f"+_runs/amber={command_spec['run']}",
            *command_spec["overrides"],
        ]
        emitted_commands.append(command)
        print(" ".join(command))
        if execute:
            subprocess.run(command, check=True)
    return emitted_commands


def _mode_config_from_overrides_codex(overrides: Iterable[str]) -> Dict[str, object]:
    config = {
        "weight_mode": "linear",
        "beta": 1.0,
        "gamma": 2.0,
        "lambda_high": 4.0,
        "lambda_mid": 2.0,
        "topk_percent": 0.2,
        "ternary_low_quantile": 0.5,
        "ternary_high_quantile": 0.8,
        "clip_min": 1.0,
        "clip_max": 10.0,
        "epsilon": 1.0e-8,
    }
    for override in overrides:
        if "=" not in override:
            continue
        key, raw_value = override.split("=", 1)
        if not key.startswith("algorithm.weighted_imitation."):
            continue
        short_key = key.split(".")[-1]
        config[short_key] = _parse_override_value_codex(raw_value)
    return config


def _parse_override_value_codex(raw_value: str):
    lowered = raw_value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in raw_value or "e" in lowered:
            return float(raw_value)
        return int(raw_value)
    except ValueError:
        return raw_value


def _aggregate_stats_codex(stats_list: List[Dict[str, float]]) -> Dict[str, float]:
    keys = stats_list[0].keys()
    return {key: float(np.mean([entry[key] for entry in stats_list])) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare physics-weighted imitation modes for Console/Mold.")
    parser.add_argument("--dataset", choices=["console", "mold"], required=True)
    parser.add_argument("--cache-dir", default="data/weighted_imitation")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--stage2-ckpt", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    commands = build_mode_overrides_codex(args.dataset)
    if args.stage2_ckpt:
        commands = append_stage2_overrides_codex(commands, ckpt_path=args.stage2_ckpt)

    cache_stats = analyze_cache_weight_modes_codex(
        dataset=args.dataset,
        cache_dir=Path(args.cache_dir),
        max_samples=args.max_samples,
    )
    emitted_commands = run_commands_codex(
        dataset=args.dataset,
        commands=commands,
        python_executable=args.python_executable,
        execute=args.execute,
    )

    result = {
        "dataset": args.dataset,
        "cache_mode_stats": cache_stats,
        "commands": emitted_commands,
    }
    print(json.dumps(result, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
