import unittest

import numpy as np

from src.formal_experiment_analysis import (
    _apply_holm_correction,
    _normalize_sample_row,
    aggregate_geometry_rows,
    aggregate_method_seed_rows,
    compute_paired_geometry_contrast,
    paired_geometry_bootstrap_ci,
)


class FormalExperimentAnalysisTests(unittest.TestCase):
    def test_real_slash_prefixed_last_step_metrics_are_normalized(self):
        row = _normalize_sample_row(
            {
                "last/projected_l2_error_symmetric": "0.25",
                "last/tetra_quality_q05": "0.75",
                "last/gate_mean": "0.125",
                "mesh_generation_success": "True",
                "solver_success": "True",
            }
        )

        self.assertEqual(row["projected_l2_error_symmetric"], 0.25)
        self.assertEqual(row["tetra_quality_q05"], 0.75)
        self.assertEqual(row["gate_mean"], 0.125)

    def test_geometry_aggregation_does_not_treat_conditions_as_independent(self):
        rows = [
            _sample("M0", 0, "g0", "c0", 1.0, True),
            _sample("M0", 0, "g0", "c1", 3.0, True),
            _sample("M0", 0, "g1", "c0", 7.0, True),
            _sample("M0", 0, "g1", "c1", None, False),
        ]

        geometry = aggregate_geometry_rows(rows)
        scalar = {
            row["geometry_id"]: row
            for row in geometry
            if row["pde_family"] == "scalar_elliptic"
        }

        self.assertEqual(len(scalar), 2)
        self.assertEqual(scalar["g0"]["solution_l2_relative"], 2.0)
        self.assertEqual(scalar["g0"]["num_conditions"], 2)
        self.assertIsNone(scalar["g1"]["solution_l2_relative"])
        self.assertEqual(scalar["g1"]["joint_success_rate"], 0.5)
        self.assertFalse(scalar["g1"]["joint_success_all"])

    def test_paired_contrast_requires_every_common_seed_for_a_geometry(self):
        geometry_rows = []
        for seed in (0, 1):
            geometry_rows.extend(
                [
                    _geometry("M0", seed, "g0", 1.0),
                    _geometry("M1", seed, "g0", 0.8),
                    _geometry("M0", seed, "g1", 2.0),
                    _geometry("M1", seed, "g1", 1.5 if seed == 0 else None),
                ]
            )

        result = compute_paired_geometry_contrast(
            geometry_rows=geometry_rows,
            left="M1",
            right="M0",
            claim="weighted_loss",
            pde_family="scalar_elliptic",
            metric="solution_l2_relative",
            bootstrap_iterations=500,
            bootstrap_seed=17,
            confidence_level=0.95,
        )

        self.assertEqual(result["candidate_geometries"], 2)
        self.assertEqual(result["paired_complete_geometries"], 1)
        self.assertEqual(result["missing_seed_geometry_pairs"], 1)
        self.assertAlmostEqual(result["mean_paired_difference"], -0.2)
        self.assertEqual(result["geometry_win_rate"], 1.0)

    def test_method_seed_summary_reports_mean_and_sample_std(self):
        rows = [
            {
                "analysis_id": "M0",
                "seed": 0,
                "pde_family": "scalar_elliptic",
                "geometry_joint_success_rate": 1.0,
                "geometry_mesh_success_rate": 1.0,
                "mean_solution_l2_relative": 0.2,
            },
            {
                "analysis_id": "M0",
                "seed": 1,
                "pde_family": "scalar_elliptic",
                "geometry_joint_success_rate": 0.5,
                "geometry_mesh_success_rate": 1.0,
                "mean_solution_l2_relative": 0.4,
            },
        ]

        summary = aggregate_method_seed_rows(rows)[0]
        self.assertEqual(summary["num_seeds_total"], 2)
        self.assertEqual(summary["seeds"], "0,1")
        self.assertAlmostEqual(summary["seed_mean_mean_solution_l2_relative"], 0.3)
        self.assertAlmostEqual(
            summary["seed_std_mean_solution_l2_relative"],
            np.sqrt(0.02),
        )

    def test_bootstrap_and_holm_are_deterministic(self):
        differences = np.array([-0.3, -0.1, 0.2, -0.4])
        first = paired_geometry_bootstrap_ci(
            differences,
            iterations=1000,
            seed=123,
            confidence_level=0.95,
        )
        second = paired_geometry_bootstrap_ci(
            differences,
            iterations=1000,
            seed=123,
            confidence_level=0.95,
        )
        self.assertEqual(first, second)

        rows = [
            {"wilcoxon_p_raw": 0.01, "wilcoxon_p_holm": None},
            {"wilcoxon_p_raw": 0.04, "wilcoxon_p_holm": None},
            {"wilcoxon_p_raw": 0.03, "wilcoxon_p_holm": None},
        ]
        _apply_holm_correction(rows)
        self.assertAlmostEqual(rows[0]["wilcoxon_p_holm"], 0.03)
        self.assertAlmostEqual(rows[2]["wilcoxon_p_holm"], 0.06)
        self.assertAlmostEqual(rows[1]["wilcoxon_p_holm"], 0.06)


def _sample(
    analysis_id: str,
    seed: int,
    geometry_id: str,
    condition_id: str,
    error: float | None,
    success: bool,
) -> dict:
    return {
        "analysis_id": analysis_id,
        "seed": seed,
        "geometry_id": geometry_id,
        "condition_id": condition_id,
        "pde_family": "scalar_elliptic",
        "mesh_generation_success": success,
        "solver_success": success,
        "joint_success": success,
        "solution_l2_relative": error,
        "qoi_absolute_error": error,
        "qoi_relative_error": error,
        "projected_l2_error_symmetric": 0.1,
        "physics_weighted_projected_l2_error": 0.1,
        "weighted_size_mse": 0.1,
        "topk_high_importance_mse": 0.1,
        "bucket_low_size_mse": 0.1,
        "bucket_high_size_mse": 0.1,
        "predicted_elements": 7000,
        "predicted_vertices": 1500,
        "budget_ratio": 1.0,
        "absolute_budget_deviation": 0.0,
        "absolute_budget_relative_deviation": 0.0,
    }


def _geometry(analysis_id: str, seed: int, geometry_id: str, error: float | None) -> dict:
    return {
        "analysis_id": analysis_id,
        "seed": seed,
        "pde_family": "scalar_elliptic",
        "geometry_id": geometry_id,
        "solution_l2_relative": error,
        "joint_success_rate": 1.0 if error is not None else 0.0,
    }


if __name__ == "__main__":
    unittest.main()
