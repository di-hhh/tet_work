from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from src.condition_aware_dataset_generation.records import FailureRecord, GeometryRecord


class GeometrySourceBackend(ABC):
    def __init__(self, source_config: dict):
        self.source_config = source_config
        self.root = Path(source_config["root"]).resolve()
        self.source_name = source_config.get("source_name", source_config["name"])

    @abstractmethod
    def ingest(self) -> tuple[list[GeometryRecord], list[FailureRecord]]:
        raise NotImplementedError

    def _scan_patterns(self, patterns: Iterable[str], recursive: bool) -> list[Path]:
        paths: list[Path] = []
        for pattern in patterns:
            iterator = self.root.rglob(pattern) if recursive else self.root.glob(pattern)
            paths.extend(iterator)
        return sorted({path.resolve() for path in paths if path.is_file()})


def build_geometry_source(source_config: dict) -> GeometrySourceBackend:
    source_name = source_config["name"]
    if source_name == "local_directory":
        from .local_directory_source import LocalDirectoryGeometrySource

        return LocalDirectoryGeometrySource(source_config)
    if source_name == "abc_dataset":
        from .abc_source import ABCDatasetGeometrySource

        return ABCDatasetGeometrySource(source_config)
    raise ValueError(f"Unknown geometry source backend: {source_name}")
