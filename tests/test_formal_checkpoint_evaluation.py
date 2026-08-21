import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from evaluate_formal_checkpoint import apply_same_geometry_condition_shuffle
from src.experiment_artifacts import TrainingRuntimeCallback


class FormalCheckpointEvaluationTests(unittest.TestCase):
    def test_training_runtime_callback_records_wall_time_updates_and_epochs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SimpleNamespace(
                global_step=7,
                fit_loop=SimpleNamespace(
                    epoch_progress=SimpleNamespace(current=SimpleNamespace(completed=2))
                ),
            )
            callback = TrainingRuntimeCallback(tmpdir)
            callback.on_fit_start(trainer, None)
            callback.on_fit_end(trainer, None)
            payload = json.loads(
                (Path(tmpdir) / "training_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["status"], "complete")
            self.assertEqual(payload["training_updates"], 7)
            self.assertEqual(payload["completed_epochs"], 2)
            self.assertGreaterEqual(payload["training_wall_time_seconds"], 0.0)
            self.assertIn("+08:00", payload["finished_at_beijing"])

    def test_condition_shuffle_is_a_within_geometry_derangement(self):
        data = [
            _data("g0", "s0", "c0", "field0.npz"),
            _data("g0", "s1", "c1", "field1.npz"),
            _data("g0", "s2", "c2", "field2.npz"),
            _data("g1", "s3", "c3", "field3.npz"),
            _data("g1", "s4", "c4", "field4.npz"),
        ]
        dataset = SimpleNamespace(data=data)

        mapping = apply_same_geometry_condition_shuffle(dataset)

        self.assertEqual(len(mapping), len(data))
        by_sample = {row["destination_sample_id"]: row for row in mapping}
        for item in data:
            cache = item.source_data.imitation_weight_cache
            row = by_sample[cache["sample_id"]]
            self.assertNotEqual(row["destination_sample_id"], row["source_sample_id"])
            self.assertEqual(row["geometry_id"], cache["geometry_id"])
            self.assertIsNone(item._observation)
        self.assertEqual(data[0].source_data.imitation_weight_cache["stage_field_path"], "field1.npz")
        self.assertEqual(data[2].source_data.imitation_weight_cache["stage_field_path"], "field0.npz")
        self.assertEqual(data[3].source_data.imitation_weight_cache["stage_field_path"], "field4.npz")


def _data(geometry_id: str, sample_id: str, condition_id: str, stage_field_path: str):
    cache = {
        "geometry_id": geometry_id,
        "sample_id": sample_id,
        "condition_id": condition_id,
        "pde_family": "scalar_elliptic",
        "stage_field_path": stage_field_path,
        "_pipeline_stage_field_fusion_probe_field": ("stale", "cache"),
    }
    return SimpleNamespace(
        source_data=SimpleNamespace(imitation_weight_cache=cache),
        _observation=object(),
        _physics_feature_bundle_cache_codex={"stale": True},
    )


if __name__ == "__main__":
    unittest.main()
