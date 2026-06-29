from __future__ import annotations

from src.condition_aware_dataset_generation.geometry_sources.base import GeometrySourceBackend
from src.condition_aware_dataset_generation.records import FailureRecord, GeometryRecord
from src.condition_aware_dataset_generation.utils import stable_identifier


class LocalDirectoryGeometrySource(GeometrySourceBackend):
    DEFAULT_PATTERNS = ["*.step", "*.stp", "*.brep", "*.iges", "*.igs", "*.json"]
    SUPPORTED_SUFFIXES = {".step", ".stp", ".brep", ".iges", ".igs", ".json"}

    def ingest(self) -> tuple[list[GeometryRecord], list[FailureRecord]]:
        patterns = self.source_config.get("patterns", self.DEFAULT_PATTERNS)
        recursive = bool(self.source_config.get("recursive", True))
        records: list[GeometryRecord] = []
        failures: list[FailureRecord] = []

        for path in self._scan_patterns(patterns=patterns, recursive=recursive):
            relative_source_path = str(path.relative_to(self.root))
            if path.suffix.lower() not in self.SUPPORTED_SUFFIXES:
                failures.append(
                    FailureRecord(
                        stage="ingest",
                        item_id=relative_source_path,
                        source_path=str(path),
                        reason=f"Unsupported geometry suffix: {path.suffix.lower()}",
                    )
                )
                continue

            if path.stat().st_size == 0:
                failures.append(
                    FailureRecord(
                        stage="ingest",
                        item_id=relative_source_path,
                        source_path=str(path),
                        reason="Geometry file is empty",
                    )
                )
                continue

            geometry_id = stable_identifier(
                prefix=path.stem,
                text=f"{self.source_name}::{relative_source_path}",
            )
            records.append(
                GeometryRecord(
                    geometry_id=geometry_id,
                    source_name=self.source_name,
                    source_path=str(path),
                    relative_source_path=relative_source_path,
                    metadata={
                        "suffix": path.suffix.lower(),
                        "file_size": path.stat().st_size,
                        "root": str(self.root),
                    },
                )
            )

        return records, failures
