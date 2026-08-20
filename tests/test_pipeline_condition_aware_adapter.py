import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import meshio
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.algorithm.dataloader import get_datasets
from src.experiment_artifacts import (
    ExperimentProtocolError,
    _validate_formal_dataset_artifacts,
    preflight_experiment_protocol,
)
from src.tasks.pipeline_condition_aware_dataset_preparator import _select_cells
from src.tasks.pipeline_dataset_audit import PipelineDatasetAuditError, audit_pipeline_dataset
from src.tasks.pipeline_dataset_fingerprint import (
    DatasetFingerprintMismatch,
    build_dataset_fingerprint,
    verify_dataset_fingerprint,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"


class PipelineConditionAwareAdapterTests(unittest.TestCase):
    def test_formal_dataset_contract_requires_references_and_fingerprint_catalog(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = OmegaConf.create(
                {"task": {"evaluation_reference_required_splits": ["test"]}}
            )
            record = {"sample_id": "sample_test", "_resolved_paths": {}}
            audit = SimpleNamespace(
                records_by_split={"test": [record]},
                split_counts={"train": 1, "val": 1, "test": 1},
                pipeline_output_root=str(root),
            )
            frozen = {"split_counts": audit.split_counts, "files": []}
            with self.assertRaisesRegex(ExperimentProtocolError, "strong references"):
                _validate_formal_dataset_artifacts(
                    config=config,
                    audit_result=audit,
                    frozen=frozen,
                )

            reference = root / "evaluation_references" / "v1" / "sample_test" / "reference_solution.npz"
            metadata = reference.with_name("metadata.json")
            record["_resolved_paths"] = {
                "optional_evaluation_reference_path": str(reference),
                "evaluation_reference_metadata_path": str(metadata),
            }
            frozen["files"] = [
                {"path": "pipeline_output/manifests/evaluation_reference_manifest.jsonl"},
                {
                    "path": "pipeline_output/evaluation_references/v1/sample_test/reference_solution.npz"
                },
                {"path": "pipeline_output/evaluation_references/v1/sample_test/metadata.json"},
            ]
            _validate_formal_dataset_artifacts(
                config=config,
                audit_result=audit,
                frozen=frozen,
            )

    def test_m0_to_m4_configs_compose_and_preserve_comparison_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            configs = {
                method: _compose_pipeline_config(root=root, run_name=run_name, overrides=[])
                for method, run_name in {
                    "M0": "amber_pipeline_baseline",
                    "M1": "amber_pipeline_weighted",
                    "M2": "amber_pipeline_feature_only",
                    "M3": "amber_pipeline_physics_correction_codex",
                    "M4": "amber_pipeline_physics_correction_stage_field",
                }.items()
            }
            self.assertEqual(
                {
                    method: str(config.experiment_protocol.method_id)
                    for method, config in configs.items()
                },
                {method: method for method in configs},
            )
            for key in ("architecture", "dataloader", "optimizer", "inference_steps"):
                self.assertEqual(configs["M0"].algorithm.get(key), configs["M1"].algorithm.get(key))
            self.assertEqual(configs["M0"].trainer.max_epochs, configs["M1"].trainer.max_epochs)
            self.assertFalse(configs["M0"].algorithm.weighted_imitation.enabled)
            self.assertTrue(configs["M1"].algorithm.weighted_imitation.enabled)
            self.assertEqual(float(configs["M2"].algorithm.gate_max), 0.0)
            self.assertEqual(float(configs["M2"].algorithm.lambda_expert_aux), 0.0)
            self.assertEqual(float(configs["M2"].algorithm.lambda_corr_aux), 0.0)
            self.assertEqual(float(configs["M2"].algorithm.lambda_corr_reg), 0.0)
            self.assertEqual(configs["M3"].task.physics_feature_source, "pipeline_indicator")
            self.assertEqual(configs["M4"].task.physics_feature_source, "stage_field_fusion")
            configs["M0"].experiment_protocol.formal_run = False
            configs["M0"].experiment_protocol.require_clean_repositories = False
            self.assertEqual(preflight_experiment_protocol(configs["M0"])["method_id"], "M0")
            configs["M2"].experiment_protocol.formal_run = False
            configs["M2"].experiment_protocol.require_clean_repositories = False
            with self.assertRaisesRegex(ExperimentProtocolError, "Automatic same-seed M1 lookup"):
                preflight_experiment_protocol(configs["M2"])

    def test_relative_fixture_can_be_copied_and_loaded_from_a_new_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            original = _write_pipeline_fixture(Path(tmpdir) / "original")
            moved = Path(tmpdir) / "moved"
            shutil.copytree(original, moved)
            cfg = _compose_pipeline_config(
                root=moved,
                run_name="amber_pipeline_weighted",
                overrides=[
                    "task.required_splits=[train,val]",
                    "algorithm.sizing_field_interpolation_type=element_weighted_sum",
                    "algorithm.initial_mesh_handling=exclude",
                    "task.features.edge.edge_curvature=False",
                ],
            )
            datasets = get_datasets(cfg.algorithm, cfg.task)
            self.assertEqual((len(datasets["train"]), len(datasets["val"])), (1, 1))

    def test_legacy_absolute_paths_use_only_explicit_relocation_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_root = _write_pipeline_fixture(Path(tmpdir) / "old")
            _rewrite_fixture_manifest_as_absolute(old_root)
            new_root = Path(tmpdir) / "new"
            shutil.copytree(old_root, new_root)
            hidden_old_root = Path(tmpdir) / "old_hidden"
            old_root.rename(hidden_old_root)

            cfg = _compose_pipeline_config(
                root=new_root,
                run_name="amber_pipeline_weighted",
                overrides=[
                    "task.required_splits=[train,val]",
                    "algorithm.sizing_field_interpolation_type=element_weighted_sum",
                    "algorithm.initial_mesh_handling=exclude",
                    "task.features.edge.edge_curvature=False",
                ],
            )
            cfg.task.path_relocation.enabled = True
            cfg.task.path_relocation.old_root = str(old_root)
            cfg.task.path_relocation.new_root = str(new_root)
            datasets = get_datasets(cfg.algorithm, cfg.task)
            self.assertEqual((len(datasets["train"]), len(datasets["val"])), (1, 1))

    def test_structural_path_errors_are_aggregated_and_never_silently_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _write_pipeline_fixture(Path(tmpdir))
            (root / "meshes" / "mesh_with_extra_triangle.vtk").unlink()
            cfg = _compose_pipeline_config(
                root=root,
                run_name="amber_pipeline_weighted",
                overrides=[
                    "task.required_splits=[train]",
                    "task.over_limit_policy=skip",
                    "algorithm.sizing_field_interpolation_type=element_weighted_sum",
                    "algorithm.initial_mesh_handling=exclude",
                    "task.features.edge.edge_curvature=False",
                ],
            )
            with self.assertRaises(PipelineDatasetAuditError) as caught:
                get_datasets(cfg.algorithm, cfg.task)
            message = str(caught.exception)
            self.assertIn("path_not_found", message)
            self.assertIn("original_records=4", message)
            self.assertIn("First 3 issue(s)", message)

    def test_fingerprint_hashes_consumed_content_and_filter_view(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _write_pipeline_fixture(Path(tmpdir))
            cfg = _compose_pipeline_config(
                root=root,
                run_name="amber_pipeline_weighted",
                overrides=[
                    "task.required_splits=[train,val]",
                    "algorithm.sizing_field_interpolation_type=element_weighted_sum",
                    "algorithm.initial_mesh_handling=exclude",
                    "task.features.edge.edge_curvature=False",
                ],
            )
            audit = audit_pipeline_dataset(cfg.task)
            frozen = build_dataset_fingerprint(audit_result=audit, task_config=cfg.task)

            # The frozen dataset identity is shared by M1/M3/M4.  Selecting an
            # already-frozen indicator or stage field is method information,
            # not a different retained dataset.
            stage_view = cfg.task.copy()
            stage_view.physics_feature_source = "stage_field_fusion"
            stage_audit = audit_pipeline_dataset(stage_view)
            stage_fingerprint = build_dataset_fingerprint(
                audit_result=stage_audit,
                task_config=stage_view,
            )
            self.assertEqual(
                frozen["dataset_fingerprint_sha256"],
                stage_fingerprint["dataset_fingerprint_sha256"],
            )

            changed_view = cfg.task.copy()
            changed_view.pde_family_filter = ["scalar_elliptic"]
            changed_audit = audit_pipeline_dataset(changed_view)
            changed_view_fingerprint = build_dataset_fingerprint(
                audit_result=changed_audit,
                task_config=changed_view,
            )
            self.assertNotEqual(
                frozen["dataset_fingerprint_sha256"],
                changed_view_fingerprint["dataset_fingerprint_sha256"],
            )

            np.save(root / "meshes" / "error_indicator.npy", np.array([0.5], dtype=np.float32))
            changed_content_audit = audit_pipeline_dataset(cfg.task)
            changed_content = build_dataset_fingerprint(
                audit_result=changed_content_audit,
                task_config=cfg.task,
            )
            self.assertNotEqual(
                frozen["dataset_fingerprint_sha256"],
                changed_content["dataset_fingerprint_sha256"],
            )
            with self.assertRaises(DatasetFingerprintMismatch):
                verify_dataset_fingerprint(
                    frozen_payload=frozen,
                    audit_result=changed_content_audit,
                    task_config=cfg.task,
                )

    def test_formal_preflight_accepts_only_same_seed_m1_last_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cfg = _compose_pipeline_config(
                root=root,
                run_name="amber_pipeline_feature_only",
                overrides=[],
            )
            cfg.experiment_protocol.formal_run = False
            cfg.experiment_protocol.require_clean_repositories = False
            m1_root = root / "m1_seed0"
            checkpoint = m1_root / "checkpoints" / "last.ckpt"
            checkpoint.parent.mkdir(parents=True)
            torch.save({"state_dict": {}}, checkpoint)
            OmegaConf.save(
                OmegaConf.create(
                    {
                        "seed": int(cfg.seed),
                        "trainer": {"max_epochs": int(cfg.trainer.max_epochs)},
                        "experiment_protocol": {"method_id": "M1"},
                    }
                ),
                m1_root / "resolved_config.yaml",
            )
            cfg.algorithm.init_from_weighted_baseline_checkpoint = str(checkpoint)
            result = preflight_experiment_protocol(cfg)
            self.assertTrue(result["checkpoint_init_validated"])
            self.assertEqual(Path(result["checkpoint_init_path"]), checkpoint.resolve())

            cfg.seed = int(cfg.seed) + 1
            with self.assertRaisesRegex(ExperimentProtocolError, "Seed mismatch"):
                preflight_experiment_protocol(cfg)

    def test_pipeline_adapter_loads_tetra_records_and_connects_indicator_to_physics_feature(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _write_pipeline_fixture(Path(tmpdir))
            cfg = _compose_pipeline_config(
                root=root,
                run_name="amber_pipeline_physics_correction_codex",
                overrides=[
                    "task.required_splits=[train,val]",
                    "task.empty_split_policy=fail",
                    "algorithm.sizing_field_interpolation_type=element_weighted_sum",
                    "algorithm.initial_mesh_handling=exclude",
                    "task.features.edge.edge_curvature=False",
                ],
            )

            datasets = get_datasets(cfg.algorithm, cfg.task)

            self.assertEqual(len(datasets["train"]), 1)
            self.assertEqual(len(datasets["val"]), 1)
            self.assertEqual(len(datasets["test"]), 0)

            data = datasets["train"][0]
            self.assertEqual(data.source_data.dataset_name, "pipeline_condition_aware")
            self.assertEqual(data.mesh.num_elements, 1)
            self.assertEqual(data.source_data.expert_mesh.num_elements, 1)
            self.assertEqual(data.source_data.imitation_weight_cache["weight_source_mode"], "pipeline_indicator")
            self.assertEqual(data.source_data.imitation_weight_cache["condition_spec"]["pde_family"], "scalar_elliptic")

            bundle = data._imitation_weight_bundle
            np.testing.assert_allclose(bundle["raw_importance"], np.array([0.25], dtype=np.float32))
            self.assertTrue(bundle["loaded"])
            self.assertFalse(bundle["fallback"])
            self.assertEqual(bundle["weights"].shape, (1,))

            graph = data._get_observation_graph()
            self.assertTrue(hasattr(graph, "physics_feature"))
            self.assertEqual(tuple(graph.physics_feature.shape), (1, 1))
            self.assertEqual(float(graph.physics_feature_available.max()), 1.0)
            self.assertEqual(float(graph.physics_feature_pipeline_indicator_loaded.max()), 1.0)
            self.assertEqual(float(graph.physics_feature_stage_field_loaded.max()), 0.0)
            self.assertEqual(graph.x.shape[0], 1)

    def test_stage_field_fusion_can_drive_physics_feature_without_changing_loss_weight_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _write_pipeline_fixture(Path(tmpdir))
            cfg = _compose_pipeline_config(
                root=root,
                run_name="amber_pipeline_physics_correction_stage_field",
                overrides=[
                    "task.required_splits=[train]",
                    "task.empty_split_policy=fail",
                    "algorithm.sizing_field_interpolation_type=interpolated_vertex",
                    "algorithm.initial_mesh_handling=exclude",
                    "task.features.edge.edge_curvature=False",
                ],
            )

            datasets = get_datasets(cfg.algorithm, cfg.task)
            data = datasets["train"][0]

            self.assertEqual(data.source_data.imitation_weight_cache["weight_source_mode"], "pipeline_indicator")
            self.assertEqual(data.source_data.imitation_weight_cache["physics_feature_source"], "stage_field_fusion")
            np.testing.assert_allclose(data._imitation_weight_bundle["raw_importance"], np.full(4, 0.25, dtype=np.float32))

            graph = data._get_observation_graph()
            self.assertEqual(tuple(graph.physics_feature.shape), (4, 1))
            self.assertEqual(float(graph.physics_feature_available.max()), 1.0)
            self.assertEqual(float(graph.physics_feature_stage_field_loaded.max()), 1.0)
            self.assertEqual(float(graph.physics_feature_pipeline_indicator_loaded.max()), 0.0)
            self.assertGreater(float(graph.physics_feature.max()), 0.0)

    def test_quality_filter_excludes_fail_global_overrefine_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _write_pipeline_fixture(Path(tmpdir))
            filtered_cfg = _compose_pipeline_config(
                root=root,
                run_name="amber_pipeline_weighted",
                overrides=[
                    "task.required_splits=[train]",
                    "task.empty_split_policy=fail",
                    "algorithm.sizing_field_interpolation_type=element_weighted_sum",
                    "algorithm.initial_mesh_handling=exclude",
                    "task.features.edge.edge_curvature=False",
                ],
            )
            unfiltered_cfg = _compose_pipeline_config(
                root=root,
                run_name="amber_pipeline_weighted",
                overrides=[
                    "task.quality_filter.enabled=false",
                    "task.required_splits=[train]",
                    "task.empty_split_policy=fail",
                    "algorithm.sizing_field_interpolation_type=element_weighted_sum",
                    "algorithm.initial_mesh_handling=exclude",
                    "task.features.edge.edge_curvature=False",
                ],
            )

            filtered = get_datasets(filtered_cfg.algorithm, filtered_cfg.task)
            unfiltered = get_datasets(unfiltered_cfg.algorithm, unfiltered_cfg.task)

            self.assertEqual(len(filtered["train"]), 1)
            self.assertEqual(len(unfiltered["train"]), 2)

    def test_allowed_statuses_and_split_manifest_are_yaml_configurable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _write_pipeline_fixture(Path(tmpdir))
            cfg = _compose_pipeline_config(
                root=root,
                run_name="amber_pipeline_weighted",
                overrides=[
                    "task.split_source=split_manifest",
                    "task.allowed_statuses=[success_partial_under_budget]",
                    "task.required_splits=[test]",
                    "task.empty_split_policy=fail",
                    "algorithm.sizing_field_interpolation_type=element_weighted_sum",
                    "algorithm.initial_mesh_handling=exclude",
                    "task.features.edge.edge_curvature=False",
                ],
            )

            datasets = get_datasets(cfg.algorithm, cfg.task)

            self.assertEqual(len(datasets["train"]), 0)
            self.assertEqual(len(datasets["val"]), 0)
            self.assertEqual(len(datasets["test"]), 1)

    def test_default_cell_policy_filters_tetra_and_fails_if_empty(self):
        points = _tet_points()
        mixed_mesh = meshio.Mesh(
            points=points,
            cells=[
                ("triangle", np.array([[0, 1, 2]], dtype=np.int32)),
                ("tetra", np.array([[0, 1, 2, 3]], dtype=np.int32)),
            ],
        )
        selected = _select_cells(
            mixed_mesh,
            cell_type="tetra",
            policy="filter_tetra_then_fail_if_empty",
            path=Path("mixed.vtk"),
        )
        np.testing.assert_array_equal(selected, np.array([[0, 1, 2, 3]], dtype=np.int32))

        surface_only_mesh = meshio.Mesh(points=points, cells=[("triangle", np.array([[0, 1, 2]], dtype=np.int32))])
        with self.assertRaisesRegex(ValueError, "does not contain 'tetra' cells"):
            _select_cells(
                surface_only_mesh,
                cell_type="tetra",
                policy="filter_tetra_then_fail_if_empty",
                path=Path("surface.vtk"),
            )


def _compose_pipeline_config(*, root: Path, run_name: str, overrides: list[str]):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(
            config_name="training_config",
            overrides=[
                f"+_runs/amber={run_name}",
                f"task.pipeline_output_root={root.as_posix()}",
                f"task.geometry_source_root={root.as_posix()}",
                *overrides,
            ],
        )


def _write_pipeline_fixture(root: Path) -> Path:
    manifests_dir = root / "manifests"
    manifests_dir.mkdir(parents=True)
    mesh_dir = root / "meshes"
    mesh_dir.mkdir()
    geometry_dir = root / "geometries" / "g_train"
    geometry_dir.mkdir(parents=True)
    source_path = geometry_dir / "source.step"
    source_path.write_text("", encoding="utf-8")

    mesh_path = mesh_dir / "mesh_with_extra_triangle.vtk"
    _write_mixed_tet_mesh(mesh_path)
    indicator_path = mesh_dir / "error_indicator.npy"
    np.save(indicator_path, np.array([0.25], dtype=np.float32))
    stage_field_path = mesh_dir / "stage_fields.npz"
    _write_stage_fields(stage_field_path)

    records = [
        _record(root, mesh_path, source_path, indicator_path, stage_field_path, "sample_train", "g_train", "c_train", "train", "success_budget_closed"),
        _record(root, mesh_path, source_path, indicator_path, stage_field_path, "sample_val", "g_val", "c_val", "val", "success_near_desired_budget"),
        _record(root, mesh_path, source_path, indicator_path, stage_field_path, "sample_test", "g_test", "c_test", "test", "success_partial_under_budget"),
        _record(root, mesh_path, source_path, indicator_path, stage_field_path, "sample_fail", "g_fail", "c_fail", "train", "success_budget_closed"),
    ]
    with (manifests_dir / "sample_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    split_manifest = {
        "geometry_to_split": {
            "g_train": "train",
            "g_val": "val",
            "g_test": "test",
        }
    }
    (manifests_dir / "split_manifest.json").write_text(json.dumps(split_manifest), encoding="utf-8")
    _write_smoke_report(root)
    return root


def _rewrite_fixture_manifest_as_absolute(root: Path) -> None:
    manifest_path = root / "manifests" / "sample_manifest.jsonl"
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    for record in records:
        record["initial_mesh_path"] = str((root / record["initial_mesh_path"]).resolve())
        record["final_target_mesh_path"] = str((root / record["final_target_mesh_path"]).resolve())
        record["optional_error_indicator_path"] = str(
            (root / record["optional_error_indicator_path"]).resolve()
        )
        record["optional_stage_field_path"] = str(
            (root / record["optional_stage_field_path"]).resolve()
        )
        artifacts = record["geometry_artifact_paths"]
        artifacts["coarse_mesh_path"] = str((root / artifacts["coarse_mesh_path"]).resolve())
        artifacts["source_path"] = str((root / artifacts["source_path"]).resolve())
    manifest_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _record(
    root: Path,
    mesh_path: Path,
    source_path: Path,
    indicator_path: Path,
    stage_field_path: Path,
    sample_id: str,
    geometry_id: str,
    condition_id: str,
    split: str,
    status: str,
) -> dict:
    mesh_rel = mesh_path.relative_to(root).as_posix()
    source_rel = source_path.relative_to(root).as_posix()
    indicator_rel = indicator_path.relative_to(root).as_posix()
    stage_field_rel = stage_field_path.relative_to(root).as_posix()
    return {
        "sample_id": sample_id,
        "geometry_id": geometry_id,
        "condition_id": condition_id,
        "pde_family": "scalar_elliptic",
        "budget": 10,
        "condition_spec": {"pde_family": "scalar_elliptic", "budget_or_tolerance_spec": {"budgets": [10]}},
        "geometry_artifact_paths": {
            "coarse_mesh_path": mesh_rel,
            "source_path": source_rel,
        },
        "initial_mesh_path": mesh_rel,
        "final_target_mesh_path": mesh_rel,
        "optional_error_indicator_path": indicator_rel,
        "optional_stage_field_path": stage_field_rel,
        "split": split,
        "status": status,
    }


def _write_mixed_tet_mesh(path: Path) -> None:
    mesh = meshio.Mesh(
        points=_tet_points(),
        cells=[
            ("tetra", np.array([[0, 1, 2, 3]], dtype=np.int32)),
            ("triangle", np.array([[0, 1, 2]], dtype=np.int32)),
        ],
    )
    meshio.write(path, mesh)


def _write_stage_fields(path: Path) -> None:
    points = _tet_points()
    np.savez(
        path,
        probe_points=points,
        s_pde_raw=np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        h_pde_only=np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32),
        h_after_geometry_fusion=np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32),
        h_after_budget_calibration=np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32),
    )


def _write_smoke_report(root: Path) -> None:
    reports_dir = root / "reports"
    reports_dir.mkdir()
    report = {
        "scalar_smoke": {
            "geometry_reports": [
                {"geometry_id": "g_train", "verdict": "PASS_STRONG", "sample_metrics": [{"sample_id": "sample_train"}]},
                {"geometry_id": "g_val", "verdict": "PASS_WEAK", "sample_metrics": [{"sample_id": "sample_val"}]},
                {"geometry_id": "g_test", "verdict": "PASS_WEAK", "sample_metrics": [{"sample_id": "sample_test"}]},
                {
                    "geometry_id": "g_fail",
                    "verdict": "FAIL_GLOBAL_OVERREFINE",
                    "sample_metrics": [{"sample_id": "sample_fail"}],
                },
            ]
        },
        "elasticity_smoke": {"geometry_reports": []},
        "geometry_reports": [],
        "summary": {},
    }
    (reports_dir / "smoke_report.json").write_text(json.dumps(report), encoding="utf-8")


def _tet_points() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


if __name__ == "__main__":
    unittest.main()
