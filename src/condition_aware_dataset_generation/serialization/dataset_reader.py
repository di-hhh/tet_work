from __future__ import annotations

from pathlib import Path

from src.condition_aware_dataset_generation.utils import read_jsonl


class ConditionAwareSampleDataset:
    def __init__(self, output_root: str | Path, manifest_name: str = "sample_manifest"):
        self.output_root = Path(output_root).resolve()
        manifest_path = self.output_root / "manifests" / f"{manifest_name}.jsonl"
        self._records = read_jsonl(manifest_path)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict:
        return self._records[index]

    @property
    def records(self) -> list[dict]:
        return list(self._records)
