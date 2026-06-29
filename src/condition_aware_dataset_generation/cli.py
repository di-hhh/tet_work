from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from src.condition_aware_dataset_generation.pipeline import ConditionAwareDatasetPipeline


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / 'config' / 'condition_aware_dataset_generation' / 'default.yaml'
SUPPORTED_COMMANDS = (
    'ingest_geometries',
    'preprocess_geometries',
    'sample_conditions',
    'prescreen_conditions',
    'generate_teacher_targets',
    'build_dataset_manifest',
    'build_smoke_report',
    'run_full_pipeline',
)


def load_pipeline_config(config_path: str | None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    config = OmegaConf.load(path)
    return OmegaConf.to_container(config, resolve=True)


def build_argument_parser(prog: str = 'condition_aware_dataset_generation') -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog)
    subparsers = parser.add_subparsers(dest='command', required=True)

    for command in SUPPORTED_COMMANDS:
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument('--config', type=str, default=str(DEFAULT_CONFIG_PATH), help='Path to the pipeline YAML config')
        command_parser.add_argument('--output-root', type=str, default=None, help='Override output_root from config')
        command_parser.add_argument('--workers', type=int, default=None, help='Override the number of workers')
        command_parser.add_argument('--overwrite', action='store_true', help='Overwrite cached stage outputs')
        command_parser.add_argument('--limit-geometries', type=int, default=None, help='Optional cap on ingested geometries')

    return parser


def _apply_config_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.output_root is not None:
        config['output_root'] = args.output_root
    if args.workers is not None:
        config['workers'] = args.workers
    if args.limit_geometries is not None:
        config['limit_geometries'] = args.limit_geometries
    if args.overwrite:
        config['overwrite'] = True
    return config


def dispatch_command(command: str, pipeline: ConditionAwareDatasetPipeline) -> dict:
    if command == 'ingest_geometries':
        return pipeline.ingest_geometries()
    if command == 'preprocess_geometries':
        return pipeline.preprocess_geometries()
    if command == 'sample_conditions':
        return pipeline.sample_conditions()
    if command == 'prescreen_conditions':
        return pipeline.prescreen_conditions()
    if command == 'generate_teacher_targets':
        return pipeline.generate_teacher_targets()
    if command == 'build_dataset_manifest':
        return pipeline.build_dataset_manifest()
    if command == 'build_smoke_report':
        return pipeline.build_smoke_report()
    if command == 'run_full_pipeline':
        return pipeline.run_full_pipeline()
    raise ValueError(f'Unsupported command: {command}')


def run_cli(argv: list[str] | None = None, prog: str = 'condition_aware_dataset_generation') -> dict:
    parser = build_argument_parser(prog=prog)
    args = parser.parse_args(argv)
    config = _apply_config_overrides(load_pipeline_config(args.config), args)
    pipeline = ConditionAwareDatasetPipeline(config)
    return dispatch_command(args.command, pipeline)


def run_entrypoint(entrypoint: str, argv: list[str] | None = None) -> dict:
    if entrypoint not in SUPPORTED_COMMANDS:
        raise ValueError(f'Unsupported entrypoint: {entrypoint}')
    return run_cli([entrypoint, *(argv or [])], prog=entrypoint)
