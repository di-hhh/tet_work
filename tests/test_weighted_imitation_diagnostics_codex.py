import unittest

import numpy as np

from src.algorithm.util.weighted_imitation_diagnostics_codex import (
    build_weights_from_normalized_importance_codex,
    compute_size_error_metrics_codex,
    should_enable_stage2_codex,
)


class WeightedImitationDiagnosticsTestsCodex(unittest.TestCase):
    def test_weighted_size_l2_matches_manual_result(self):
        predictions = np.array([1.0, 3.0], dtype=np.float64)
        labels = np.array([0.0, 1.0], dtype=np.float64)
        weights = np.array([1.0, 3.0], dtype=np.float64)
        importance = np.array([0.1, 0.9], dtype=np.float64)
        metrics = compute_size_error_metrics_codex(predictions, labels, weights, importance, epsilon=1.0e-8, topk_percent=0.5, bucket_count=2)
        expected = (1.0 * 1.0 + 3.0 * 4.0) / 4.0
        self.assertAlmostEqual(metrics["weighted_size_l2"], expected, places=6)

    def test_topk_high_importance_l2_matches_manual_result(self):
        predictions = np.array([1.0, 3.0, 4.0, 2.0], dtype=np.float64)
        labels = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        weights = np.ones(4, dtype=np.float64)
        importance = np.array([0.1, 0.2, 0.9, 0.8], dtype=np.float64)
        metrics = compute_size_error_metrics_codex(predictions, labels, weights, importance, epsilon=1.0e-8, topk_percent=0.5, bucket_count=2)
        expected = np.mean(np.array([(4.0 - 1.0) ** 2, (2.0 - 1.0) ** 2], dtype=np.float64))
        self.assertAlmostEqual(metrics["topk_high_importance_l2"], expected, places=6)

    def test_bucketed_error_tracks_low_and_high_buckets(self):
        predictions = np.array([0.0, 0.0, 2.0, 4.0], dtype=np.float64)
        labels = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        weights = np.ones(4, dtype=np.float64)
        importance = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
        metrics = compute_size_error_metrics_codex(predictions, labels, weights, importance, epsilon=1.0e-8, topk_percent=0.5, bucket_count=2)
        self.assertAlmostEqual(metrics["bucket_low_size_l2"], 0.0, places=6)
        self.assertAlmostEqual(metrics["bucket_high_size_l2"], 10.0, places=6)
        self.assertTrue(np.isnan(metrics["bucket_high_low_ratio"]) or metrics["bucket_high_low_ratio"] > 1.0)

    def test_weight_modes_follow_expected_values(self):
        importance = np.array([0.0, 0.2, 0.7, 1.0], dtype=np.float64)

        linear = build_weights_from_normalized_importance_codex(importance, {"weight_mode": "linear", "beta": 1.0, "clip_min": 1.0, "clip_max": 10.0})
        power = build_weights_from_normalized_importance_codex(importance, {"weight_mode": "power", "beta": 1.0, "gamma": 2.0, "clip_min": 1.0, "clip_max": 10.0})
        binary = build_weights_from_normalized_importance_codex(importance, {"weight_mode": "binary_topk", "lambda_high": 5.0, "topk_percent": 0.25, "clip_min": 1.0, "clip_max": 10.0})
        ternary = build_weights_from_normalized_importance_codex(
            importance,
            {
                "weight_mode": "ternary_quantile",
                "lambda_mid": 2.0,
                "lambda_high": 4.0,
                "ternary_low_quantile": 0.25,
                "ternary_high_quantile": 0.75,
                "clip_min": 1.0,
                "clip_max": 10.0,
            },
        )

        self.assertTrue(np.all(linear > 0))
        self.assertTrue(np.all(power > 0))
        self.assertTrue(np.all(binary > 0))
        self.assertTrue(np.all(ternary > 0))
        np.testing.assert_allclose(linear, np.array([1.0, 1.2, 1.7, 2.0], dtype=np.float32), atol=1.0e-6)
        np.testing.assert_allclose(power, np.array([1.0, 1.04, 1.49, 2.0], dtype=np.float32), atol=1.0e-6)
        np.testing.assert_allclose(binary, np.array([1.0, 1.0, 1.0, 5.0], dtype=np.float32), atol=1.0e-6)
        np.testing.assert_allclose(ternary, np.array([1.0, 2.0, 2.0, 4.0], dtype=np.float32), atol=1.0e-6)

    def test_stage2_schedule_switches_as_expected(self):
        self.assertFalse(
            should_enable_stage2_codex(
                current_epoch=10,
                max_epochs=100,
                stage2_enable=False,
                stage2_epochs=20,
                resumed_from_checkpoint=False,
                stage2_resume_mode=True,
            )
        )
        self.assertFalse(
            should_enable_stage2_codex(
                current_epoch=70,
                max_epochs=100,
                stage2_enable=True,
                stage2_epochs=20,
                resumed_from_checkpoint=False,
                stage2_resume_mode=True,
            )
        )
        self.assertTrue(
            should_enable_stage2_codex(
                current_epoch=80,
                max_epochs=100,
                stage2_enable=True,
                stage2_epochs=20,
                resumed_from_checkpoint=False,
                stage2_resume_mode=True,
            )
        )
        self.assertTrue(
            should_enable_stage2_codex(
                current_epoch=0,
                max_epochs=100,
                stage2_enable=True,
                stage2_epochs=20,
                resumed_from_checkpoint=True,
                stage2_resume_mode=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
