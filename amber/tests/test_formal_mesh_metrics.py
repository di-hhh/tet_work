import unittest
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

from src.algorithm.util.weighted_imitation_diagnostics_codex import compute_size_error_metrics_codex
from src.mesh_util.mesh_metrics import MeshMetrics


class FormalMeshMetricTests(unittest.TestCase):
    def test_legacy_l2_named_sizing_metrics_have_explicit_mse_aliases(self):
        metrics = compute_size_error_metrics_codex(
            predictions=np.array([1.0, 3.0]),
            labels=np.array([2.0, 1.0]),
            weights=np.array([1.0, 2.0]),
            importance=np.array([0.0, 1.0]),
            bucket_count=2,
        )
        self.assertEqual(metrics["weighted_size_mse"], metrics["weighted_size_l2"])
        self.assertEqual(metrics["topk_high_importance_mse"], metrics["topk_high_importance_l2"])
        self.assertEqual(metrics["bucket_low_size_mse"], metrics["bucket_low_size_l2"])
        self.assertEqual(metrics["bucket_high_size_mse"], metrics["bucket_high_size_l2"])

    def test_tetra_quality_reports_median_degenerate_and_inverted_counts(self):
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        evaluated = SimpleNamespace(
            vertex_positions=points,
            element_indices=np.array(
                [
                    [0, 1, 2, 3],
                    [0, 2, 1, 3],
                    [0, 0, 1, 2],
                ],
                dtype=np.int64,
            ),
        )
        metrics = MeshMetrics(
            metric_config=OmegaConf.create({}),
            reference_mesh=evaluated,
            evaluated_mesh=evaluated,
            fem_problem=None,
        ).tetra_quality_metrics()

        self.assertEqual(metrics["tetra_inverted_count"], 1)
        self.assertEqual(metrics["tetra_degenerate_count"], 1)
        self.assertEqual(metrics["tetra_invalid_count"], 2)
        self.assertEqual(metrics["tetra_quality_q05"], metrics["tetra_quality_p05"])
        self.assertTrue(np.isfinite(metrics["tetra_quality_median"]))


if __name__ == "__main__":
    unittest.main()
