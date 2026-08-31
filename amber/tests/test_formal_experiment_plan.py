import unittest
from pathlib import Path

from src.formal_experiment_plan import (
    dependency_checkpoint,
    iter_run_specs,
    load_formal_plan,
    validate_frozen_dataset,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class FormalExperimentPlanTests(unittest.TestCase):
    def test_frozen_plan_has_expected_runs_dependencies_and_data(self):
        plan = load_formal_plan(REPO_ROOT / "config" / "formal_experiment_plan.yaml")
        specs = list(iter_run_specs(plan))

        self.assertEqual(len(specs), 16)
        self.assertEqual(
            [(spec.analysis_id, spec.seed) for spec in specs[:6]],
            [("M0", 0), ("M0", 1), ("M0", 2), ("M1", 0), ("M1", 1), ("M1", 2)],
        )
        m1_ft_seed1 = next(
            spec for spec in specs if spec.analysis_id == "M1-FT" and spec.seed == 1
        )
        dependency = dependency_checkpoint(plan, m1_ft_seed1)
        self.assertEqual(
            dependency,
            next(spec for spec in specs if spec.analysis_id == "M1" and spec.seed == 1).run_root
            / "checkpoints"
            / "last.ckpt",
        )
        m3 = [spec for spec in specs if spec.analysis_id == "M3"]
        self.assertEqual([(spec.seed, spec.role, spec.oracle_only) for spec in m3], [(0, "oracle", True)])
        frozen = validate_frozen_dataset(plan)
        self.assertEqual(frozen["split_counts"], {"train": 182, "val": 34, "test": 42})


if __name__ == "__main__":
    unittest.main()
