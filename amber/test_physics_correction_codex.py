import os
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch_geometric.data import Data

from src.algorithm.dataloader.amber_data import AmberData
from src.algorithm.architecture.supervised_mpn import SupervisedMPN
from src.algorithm.core.mesh_generation_algorithm import MeshGenerationAlgorithm
from src.algorithm.loss.amber_loss import AmberLoss
from src.algorithm.normalizer.dummy_running_normalizer import DummyRunningNormalizer
from src.algorithm.prediction_transform.no_transform import NoTransform
from src.mesh_util.mesh_metrics import MeshMetrics


def _make_graph(*, node_features: int) -> Data:
    # [CodeX] 构造一个最小可运行图，用于验证 physics correction 头与 gate 头的输出形状。
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]],
        dtype=torch.long,
    )
    edge_attr = torch.ones((edge_index.shape[1], 1), dtype=torch.float32)
    graph = Data(
        x=torch.randn((4, node_features), dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    graph.mask_output = torch.ones(graph.num_nodes, dtype=torch.bool)
    return graph


def _mpn_config(*, enable_branch: bool, fallback: str = "gate_zero"):
    # [CodeX] 使用最小 MPN 配置触发双头 + gate 路径，避免把测试耦合到完整 Hydra 训练配置。
    return OmegaConf.create(
        {
            "name": "mpn",
            "latent_dimension": 8,
            "create_graph_copy": True,
            "assert_graph_shapes": False,
            "edge_dropout": 0.0,
            "enable_physics_correction_branch": enable_branch,
            "gate_activation": "sigmoid",
            "gate_max": 1.0,
            "inference_missing_physics_fallback": fallback,
            "stack": {
                "layer_norm": None,
                "num_steps": 2,
                "residual_connections": "inner",
                "mlp": {
                    "activation_function": "relu",
                    "num_layers": 1,
                    "add_output_layer": False,
                    "regularization": {
                        "dropout": 0.0,
                        "spectral_norm": False,
                        "latent_normalization": None,
                    },
                },
            },
            "decoder": {
                "activation_function": "relu",
                "num_layers": 1,
                "regularization": {
                    "dropout": 0.0,
                    "spectral_norm": False,
                    "latent_normalization": None,
                },
            },
        }
    )


class PhysicsCorrectionTests(unittest.TestCase):
    def _no_transform(self):
        return NoTransform(OmegaConf.create({"predict_residual": False, "inverse_transform_in_loss": False}))

    def test_branch_enabled_returns_expert_physics_and_gate(self):
        graph = _make_graph(node_features=4)
        graph.physics_feature_available = torch.ones((graph.num_nodes, 1), dtype=torch.float32)
        model = SupervisedMPN(architecture_config=_mpn_config(enable_branch=True), example_graph=graph)

        outputs = model(graph, correction_warmup_factor=1.0)

        self.assertIsInstance(outputs, dict)
        self.assertEqual(outputs["expert_output"].shape, (graph.num_nodes, 1))
        self.assertEqual(outputs["physics_output"].shape, (graph.num_nodes, 1))
        self.assertEqual(outputs["gate"].shape, (graph.num_nodes, 1))

    def test_gate_zero_fallback_for_missing_physics(self):
        graph = _make_graph(node_features=4)
        graph.physics_feature_available = torch.zeros((graph.num_nodes, 1), dtype=torch.float32)
        model = SupervisedMPN(
            architecture_config=_mpn_config(enable_branch=True, fallback="gate_zero"),
            example_graph=graph,
        )

        outputs = model(graph, correction_warmup_factor=1.0)

        self.assertTrue(torch.allclose(outputs["gate"], torch.zeros_like(outputs["gate"])))

    def test_branch_disabled_keeps_tensor_output_path(self):
        graph = _make_graph(node_features=3)
        model = SupervisedMPN(architecture_config=_mpn_config(enable_branch=False), example_graph=graph)

        outputs = model(graph)

        self.assertIsInstance(outputs, torch.Tensor)
        self.assertEqual(outputs.shape, (graph.num_nodes, 1))

    def test_prediction_bundle_degenerates_to_expert_when_gate_is_zero(self):
        fake_algorithm = SimpleNamespace(
            normalizer=DummyRunningNormalizer(),
            prediction_transform=self._no_transform(),
        )
        fake_algorithm._denormalize_correction_residual_codex = types.MethodType(
            MeshGenerationAlgorithm._denormalize_correction_residual_codex,
            fake_algorithm,
        )
        batch = Data(current_sizing_field=torch.zeros(4, dtype=torch.float32))
        prediction_bundle = MeshGenerationAlgorithm._build_prediction_bundle_codex(
            fake_algorithm,
            batch=batch,
            model_outputs={
                "expert_output": torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32),
                "physics_output": torch.tensor([[9.0], [8.0], [7.0], [6.0]], dtype=torch.float32),
                "gate": torch.zeros((4, 1), dtype=torch.float32),
                "gate_logits": torch.zeros((4, 1), dtype=torch.float32),
                "physics_feature_available": torch.zeros((4, 1), dtype=torch.float32),
            },
            is_train=True,
            flatten=True,
        )

        self.assertTrue(torch.allclose(prediction_bundle["final"], prediction_bundle["expert"]))

    def test_loss_matches_original_when_branch_is_disabled(self):
        labels = torch.tensor([1.0, 2.0], dtype=torch.float32)
        predictions = torch.tensor([0.0, 4.0], dtype=torch.float32)
        graph = Data(
            current_sizing_field=torch.zeros_like(labels),
            imitation_weights=torch.ones_like(labels),
            imitation_normalized_importance=torch.tensor([0.1, 0.9], dtype=torch.float32),
        )

        reference_loss = AmberLoss(
            label_transform=self._no_transform(),
            loss_type="mse",
            weighted_imitation_config={
                "enabled": True,
                "epsilon": 1.0e-8,
                "fallback_to_ones": True,
                "lambda_expert_aux": 0.0,
                "lambda_corr_reg": 0.0,
            },
        )
        correction_loss = AmberLoss(
            label_transform=self._no_transform(),
            loss_type="mse",
            weighted_imitation_config={
                "enabled": True,
                "epsilon": 1.0e-8,
                "fallback_to_ones": True,
                "lambda_expert_aux": 0.25,
                "lambda_corr_reg": 1.0e-3,
            },
        )

        reference_value, _ = reference_loss(predictions=predictions, labels=labels, graph_batch=graph)
        correction_value, _ = correction_loss(predictions=predictions, labels=labels, graph_batch=graph)
        self.assertAlmostEqual(reference_value.item(), correction_value.item(), places=6)

    def test_loss_bundle_reports_aux_and_correction_terms(self):
        labels = torch.tensor([1.0, 2.0], dtype=torch.float32)
        graph = Data(
            current_sizing_field=torch.zeros_like(labels),
            imitation_weights=torch.ones_like(labels),
            imitation_normalized_importance=torch.tensor([0.1, 0.9], dtype=torch.float32),
        )
        correction_loss = AmberLoss(
            label_transform=self._no_transform(),
            loss_type="mse",
            weighted_imitation_config={
                "enabled": True,
                "epsilon": 1.0e-8,
                "fallback_to_ones": True,
                "lambda_expert_aux": 0.25,
                "lambda_corr_aux": 0.5,
                "lambda_corr_reg": 1.0e-3,
            },
        )
        prediction_bundle = {
            "final": torch.tensor([0.0, 4.0], dtype=torch.float32),
            "expert": torch.tensor([0.5, 3.5], dtype=torch.float32),
            "semantic_expert_delta": torch.tensor([0.5, 3.5], dtype=torch.float32),
            "semantic_phys_delta": torch.tensor([0.3, -0.3], dtype=torch.float32),
            "applied_correction": torch.tensor([0.1, -0.2], dtype=torch.float32),
            "gate": torch.tensor([0.5, 0.5], dtype=torch.float32),
        }

        loss_value, _ = correction_loss(predictions=prediction_bundle, labels=labels, graph_batch=graph)

        self.assertGreater(loss_value.item(), 0.0)
        self.assertGreater(correction_loss.last_loss_metrics["loss_expert_aux"], 0.0)
        self.assertGreater(correction_loss.last_loss_metrics["loss_corr_aux"], 0.0)
        self.assertGreater(correction_loss.last_loss_metrics["loss_corr_reg"], 0.0)

    def test_prediction_diagnostics_include_full_branch_statistics(self):
        # [CodeX] 验证 physics correction 诊断只保留关键指标，去掉低价值的 gate 范围和 delta_phys 细分统计。
        labels = torch.tensor([1.0, 2.0], dtype=torch.float32)
        graph = Data(
            current_sizing_field=torch.zeros_like(labels),
            imitation_weights=torch.tensor([1.0, 2.0], dtype=torch.float32),
            imitation_normalized_importance=torch.tensor([0.1, 0.9], dtype=torch.float32),
        )
        loss_module = AmberLoss(
            label_transform=self._no_transform(),
            loss_type="mse",
            weighted_imitation_config={
                "enabled": True,
                "epsilon": 1.0e-8,
                "fallback_to_ones": True,
            },
        )
        prediction_bundle = {
            "final": torch.tensor([0.8, 2.2], dtype=torch.float32),
            "expert": torch.tensor([0.9, 2.4], dtype=torch.float32),
            "semantic_phys_delta": torch.tensor([0.3, -0.3], dtype=torch.float32),
            "applied_correction": torch.tensor([0.1, -0.2], dtype=torch.float32),
            "gate": torch.tensor([0.2, 0.8], dtype=torch.float32),
        }

        diagnostics = loss_module.get_prediction_diagnostics(
            predictions=prediction_bundle,
            labels=labels,
            graph_batch=graph,
        )

        self.assertIn("gate_mean", diagnostics)
        self.assertIn("gate_std", diagnostics)
        self.assertIn("gate_min", diagnostics)
        self.assertIn("gate_max", diagnostics)
        self.assertIn("gate_high_importance_mean", diagnostics)
        self.assertIn("delta_phys_abs_mean", diagnostics)
        self.assertIn("delta_phys_high_importance_abs_mean", diagnostics)
        self.assertIn("applied_correction_abs_mean", diagnostics)
        self.assertIn("expert_prior_weighted_size_l2", diagnostics)
        self.assertIn("final_prediction_topk_high_importance_l2", diagnostics)
        self.assertNotIn("final_prediction_bucket_high_size_l2", diagnostics)

    def test_correction_warmup_starts_non_zero_on_first_epoch(self):
        fake_algorithm = SimpleNamespace(
            config={"correction_warmup_epochs": 3},
            current_epoch=0,
        )

        warmup_factor = MeshGenerationAlgorithm._get_correction_warmup_factor_codex(
            fake_algorithm,
            is_train=True,
        )

        self.assertAlmostEqual(warmup_factor, 1.0 / 3.0, places=6)

    def test_checkpoint_adaptation_copies_old_columns_and_zeros_new_column(self):
        current_value = torch.ones((3, 5), dtype=torch.float32)
        checkpoint_value = torch.arange(12, dtype=torch.float32).reshape(3, 4)

        adapted = MeshGenerationAlgorithm._adapt_checkpoint_tensor_codex(
            current_value=current_value,
            checkpoint_value=checkpoint_value,
        )

        self.assertIsNotNone(adapted)
        self.assertTrue(torch.allclose(adapted[:, :4], checkpoint_value))
        self.assertTrue(torch.allclose(adapted[:, 4], torch.zeros(3, dtype=torch.float32)))

    def test_log_plots_skips_when_experiment_logger_is_missing(self):
        # [CodeX] 验证关闭 WandB 后绘图日志会安全跳过，而不是访问空 experiment 导致崩溃。
        fake_algorithm = SimpleNamespace(
            logger=None,
            current_epoch=3,
        )
        fake_algorithm._get_experiment_logger_codex = types.MethodType(
            MeshGenerationAlgorithm._get_experiment_logger_codex,
            fake_algorithm,
        )

        MeshGenerationAlgorithm._log_plots(fake_algorithm, {"demo": object()})

    def test_on_test_end_skips_external_logging_when_experiment_logger_is_missing(self):
        # [CodeX] 验证 test end 在无外部 logger 时仍可完成清理与后续流程调用。
        log_constant_calls = []
        fake_algorithm = SimpleNamespace(
            logger=None,
            current_epoch=2,
            test_step_outputs=[{"mesh_l2": 1.0}, {"mesh_l2": 3.0}],
            trainer=SimpleNamespace(test_dataloaders=[]),
        )
        fake_algorithm._get_experiment_logger_codex = types.MethodType(
            MeshGenerationAlgorithm._get_experiment_logger_codex,
            fake_algorithm,
        )
        fake_algorithm._log_constant_plots = lambda dataloader, prefix: log_constant_calls.append((dataloader, prefix))

        MeshGenerationAlgorithm.on_test_end(fake_algorithm)

        self.assertEqual(fake_algorithm.test_step_outputs, [])
        self.assertEqual(log_constant_calls, [([], "test")])

    def test_local_test_artifacts_are_reconstructible_without_wandb(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_algorithm = SimpleNamespace(
                local_artifact_root=tmpdir,
                test_sample_rows=[
                    {
                        "sample_id": "sample_a",
                        "mesh_generation_success": True,
                        "last_cur_elements": 8,
                    }
                ],
                test_prediction_rows=[
                    {
                        "sample_id": "sample_a",
                        "prediction_mesh_path": "test_predictions/sample_a.vtk",
                        "mesh_generation_success": True,
                    }
                ],
            )
            MeshGenerationAlgorithm._write_local_test_artifacts(
                fake_algorithm,
                {"metrics.test_last_cur_elements": 8.0},
            )

            root = Path(tmpdir)
            self.assertTrue((root / "per_sample_metrics.csv").exists())
            self.assertTrue((root / "test_predictions" / "prediction_manifest.csv").exists())
            aggregate = (root / "aggregate_metrics.json").read_text(encoding="utf-8")
            self.assertIn('"checkpoint": "checkpoints/last.ckpt"', aggregate)
            self.assertEqual(fake_algorithm.test_sample_rows, [])
            self.assertEqual(fake_algorithm.test_prediction_rows, [])

    def test_regular_tetra_mean_ratio_quality_is_one(self):
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, np.sqrt(3.0) / 2.0, 0.0],
                [0.5, np.sqrt(3.0) / 6.0, np.sqrt(2.0 / 3.0)],
            ]
        )
        fake_metrics = SimpleNamespace(
            evaluated_mesh=SimpleNamespace(
                vertex_positions=points,
                element_indices=np.array([[0, 1, 2, 3]], dtype=np.int64),
            )
        )
        metrics = MeshMetrics.tetra_quality_metrics(fake_metrics)
        self.assertAlmostEqual(metrics["tetra_quality_mean"], 1.0, places=12)
        self.assertEqual(metrics["tetra_degenerate_fraction"], 0.0)

    def test_console_run_config_composes_physics_correction_preset(self):
        # [CodeX] 验证新增的 Console run 入口能正确展开到 physics correction 预设，便于直接启动实验。
        config_dir = os.path.join(os.path.dirname(__file__), "config")
        with initialize_config_dir(version_base=None, config_dir=os.path.abspath(config_dir)):
            cfg = compose(
                config_name="training_config",
                overrides=["+_runs/amber=amber_console_physics_correction_codex"],
            )

        self.assertEqual(cfg.task.name, "console")
        self.assertTrue(cfg.algorithm.enable_physics_correction_branch)
        self.assertTrue(cfg.algorithm.weighted_imitation.enabled)
        self.assertEqual(cfg.exp_name, "amber_console_physics_correction_codex")

    def test_mold_run_config_composes_physics_correction_preset(self):
        # [CodeX] 验证新增的 Mold run 入口能正确展开到 physics correction 预设，避免手工覆盖一长串参数。
        config_dir = os.path.join(os.path.dirname(__file__), "config")
        with initialize_config_dir(version_base=None, config_dir=os.path.abspath(config_dir)):
            cfg = compose(
                config_name="training_config",
                overrides=["+_runs/amber=amber_mold_physics_correction_codex"],
            )

        self.assertEqual(cfg.task.name, "mold")
        self.assertTrue(cfg.algorithm.enable_physics_correction_branch)
        self.assertTrue(cfg.algorithm.weighted_imitation.enabled)
        self.assertEqual(cfg.exp_name, "amber_mold_physics_correction_codex")

    def test_physics_feature_uses_mesh_specific_bundle_in_hierarchical_graph(self):
        # [CodeX] 验证层级图中的 initial mesh 会查询自己的 importance bundle，而不是误用当前 mesh 的长度。
        current_mesh = SimpleNamespace(num_vertices=4, num_elements=4)
        initial_mesh = SimpleNamespace(num_vertices=6, num_elements=6)
        bundles = {
            id(current_mesh): {
                "raw_importance": np.arange(4, dtype=np.float32),
                "normalized_importance": np.linspace(0.0, 1.0, 4, dtype=np.float32),
                "loaded": True,
            },
            id(initial_mesh): {
                "raw_importance": np.arange(6, dtype=np.float32),
                "normalized_importance": np.linspace(0.0, 1.0, 6, dtype=np.float32),
                "loaded": True,
            },
        }
        fake_data = SimpleNamespace(
            physics_correction_config={"physics_feature_mode": "normalized_importance"},
            node_type="vertex",
        )
        fake_data._get_imitation_weight_bundle_for_mesh_codex = types.MethodType(
            lambda self, mesh: bundles[id(mesh)],
            fake_data,
        )
        fake_data._select_physics_feature_values_from_bundle_codex = types.MethodType(
            AmberData._select_physics_feature_values_from_bundle_codex,
            fake_data,
        )
        fake_data._reproject_physics_feature_values_codex = types.MethodType(
            lambda self, **kwargs: (None, False),
            fake_data,
        )

        feature_values, feature_available = AmberData._get_physics_feature_values(
            fake_data,
            mesh=initial_mesh,
            expected_size=6,
        )

        self.assertTrue(feature_available)
        self.assertEqual(feature_values.shape[0], 6)
        self.assertAlmostEqual(float(feature_values[-1]), 1.0, places=6)

    def test_physics_feature_length_mismatch_falls_back_to_zero_feature(self):
        # [CodeX] 验证多步推理中若 importance 长度仍不匹配，也会回退为零特征而不是直接抛异常。
        mismatched_mesh = SimpleNamespace(num_vertices=6, num_elements=6)
        fake_data = SimpleNamespace(
            physics_correction_config={"physics_feature_mode": "normalized_importance"},
            node_type="vertex",
        )
        fake_data._get_imitation_weight_bundle_for_mesh_codex = types.MethodType(
            lambda self, mesh: {
                "raw_importance": np.arange(4, dtype=np.float32),
                "normalized_importance": np.linspace(0.0, 1.0, 4, dtype=np.float32),
                "loaded": True,
            },
            fake_data,
        )
        fake_data._select_physics_feature_values_from_bundle_codex = types.MethodType(
            AmberData._select_physics_feature_values_from_bundle_codex,
            fake_data,
        )
        fake_data._reproject_physics_feature_values_codex = types.MethodType(
            lambda self, **kwargs: (None, False),
            fake_data,
        )

        feature_values, feature_available = AmberData._get_physics_feature_values(
            fake_data,
            mesh=mismatched_mesh,
            expected_size=6,
        )

        self.assertFalse(feature_available)
        self.assertEqual(feature_values.shape[0], 6)
        self.assertTrue(np.allclose(feature_values, np.zeros(6, dtype=np.float32)))

    def test_physics_feature_length_mismatch_reprojects_before_zero_feature(self):
        # [CodeX] 验证长度不匹配时会先尝试重投影补救，只有补救失败时才回退为零特征。
        mismatched_mesh = SimpleNamespace(num_vertices=6, num_elements=6)
        fake_data = SimpleNamespace(
            physics_correction_config={"physics_feature_mode": "normalized_importance"},
            node_type="vertex",
        )
        fake_data._get_imitation_weight_bundle_for_mesh_codex = types.MethodType(
            lambda self, mesh: {
                "raw_importance": np.arange(4, dtype=np.float32),
                "normalized_importance": np.linspace(0.0, 1.0, 4, dtype=np.float32),
                "loaded": True,
            },
            fake_data,
        )
        fake_data._select_physics_feature_values_from_bundle_codex = types.MethodType(
            AmberData._select_physics_feature_values_from_bundle_codex,
            fake_data,
        )
        fake_data._reproject_physics_feature_values_codex = types.MethodType(
            lambda self, **kwargs: (np.linspace(0.0, 1.0, 6, dtype=np.float32), True),
            fake_data,
        )

        feature_values, feature_available = AmberData._get_physics_feature_values(
            fake_data,
            mesh=mismatched_mesh,
            expected_size=6,
        )

        self.assertTrue(feature_available)
        self.assertEqual(feature_values.shape[0], 6)
        self.assertAlmostEqual(float(feature_values[-1]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
