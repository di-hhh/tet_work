from __future__ import annotations

from src.condition_aware_dataset_generation.geometry_sources.local_directory_source import LocalDirectoryGeometrySource
from src.condition_aware_dataset_generation.records import FailureRecord


class ABCDatasetGeometrySource(LocalDirectoryGeometrySource):
    DEFAULT_PATTERNS = ["*.step", "*.stp", "*.7z"]

    def ingest(self):
        records, failures = super().ingest()
        archive_paths = self._scan_patterns(patterns=["*.7z"], recursive=True)
        for archive_path in archive_paths:
            relative_source_path = str(archive_path.relative_to(self.root))
            failures.append(
                FailureRecord(
                    stage="ingest",
                    item_id=relative_source_path,
                    source_path=str(archive_path),
                    reason="ABC archive detected but not extracted; extract STEP files first for preprocessing",
                    details={"archive_path": str(archive_path)},
                )
            )
        for record in records:
            record.metadata["dataset_family"] = "ABC-Dataset"
        return records, failures
