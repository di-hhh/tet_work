import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data

from src.algorithm.dataloader.amber_data import AmberData
from src.algorithm.dataloader.source_data import SourceData
from src.algorithm.loss.amber_loss import AmberLoss
from src.algorithm.prediction_transform.no_transform import NoTransform
from src.tasks.domains.extended_mesh_tet1 import ExtendedMeshTet1
from src.tasks.domains.mesh_wrapper import MeshWrapper


def _make_simple_tet_mesh() -> MeshWrapper:
    vertex_positions = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    element_indices = np.array([[0], [1], [2], [3]], dtype=np.int32)
    return MeshWrapper(ExtendedMeshTet1(vertex_positions, element_indices))


class DummyAmberData(AmberData):
    def _mesh_to_graph(self, mesh: MeshWrapper) -> Data:
        graph = Data(x=torch.zeros((mesh.num_vertices, 1), dtype=torch.float32))
        graph.current_sizing_field = torch.ones(mesh.num_vertices, dtype=torch.float32)
        return graph


class WeightedImitationTests(unittest.TestCase):
    def _no_transform(self):
        return NoTransform(OmegaConf.create({"predict_residual": False, "inverse_transform_in_loss": False}))

    def test_all_one_weights_match_original_loss(self):
        labels = torch.tensor([1.0, 2.0], dtype=torch.float32)
        predictions = torch.tensor([0.0, 4.0], dtype=torch.float32)
        graph = Data(current_sizing_field=torch.zeros_like(labels), imitation_weights=torch.ones_like(labels))

        original_loss = AmberLoss(label_transform=self._no_transform(), loss_type="mse", weighted_imitation_config={"enabled": False})
        weighted_loss = AmberLoss(
            label_transform=self._no_transform(),
            loss_type="mse",
            weighted_imitation_config={"enabled": True, "epsilon": 1.0e-8, "fallback_to_ones": True},
        )

        original_value, _ = original_loss(predictions=predictions, labels=labels, graph_batch=graph)
        weighted_value, _ = weighted_loss(predictions=predictions, labels=labels, graph_batch=graph)
        self.assertAlmostEqual(original_value.item(), weighted_value.item(), places=6)

    def test_manual_weights_match_weighted_average(self):
        labels = torch.tensor([1.0, 2.0], dtype=torch.float32)
        predictions = torch.tensor([0.0, 4.0], dtype=torch.float32)
        weights = torch.tensor([1.0, 3.0], dtype=torch.float32)
        graph = Data(current_sizing_field=torch.zeros_like(labels), imitation_weights=weights)

        weighted_loss = AmberLoss(
            label_transform=self._no_transform(),
            loss_type="mse",
            weighted_imitation_config={"enabled": True, "epsilon": 1.0e-8, "fallback_to_ones": True},
        )
        weighted_value, _ = weighted_loss(predictions=predictions, labels=labels, graph_batch=graph)

        element_loss = torch.tensor([1.0, 4.0], dtype=torch.float32)
        expected = torch.sum(weights * element_loss) / torch.sum(weights)
        self.assertAlmostEqual(weighted_value.item(), expected.item(), places=6)

    def test_missing_weights_fallback_to_ones(self):
        labels = torch.tensor([1.0, 2.0], dtype=torch.float32)
        predictions = torch.tensor([0.0, 4.0], dtype=torch.float32)
        graph = Data(current_sizing_field=torch.zeros_like(labels))

        weighted_loss = AmberLoss(
            label_transform=self._no_transform(),
            loss_type="mse",
            weighted_imitation_config={"enabled": True, "epsilon": 1.0e-8, "fallback_to_ones": True},
        )
        weighted_value, _ = weighted_loss(predictions=predictions, labels=labels, graph_batch=graph)

        expected = torch.mean(torch.tensor([1.0, 4.0], dtype=torch.float32))
        self.assertAlmostEqual(weighted_value.item(), expected.item(), places=6)

    def test_console_cache_projects_weights_to_vertex_labels(self):
        expert_mesh = _make_simple_tet_mesh()
        queried_mesh = _make_simple_tet_mesh()

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "console" / "train" / "001.npz"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                vertex_importance=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                element_importance=np.array([2.5], dtype=np.float32),
            )

            data = DummyAmberData(
                mesh=queried_mesh,
                source_data=SourceData(
                    expert_mesh=expert_mesh,
                    initial_mesh=queried_mesh,
                    dataset_name="console",
                    data_point_path=str(Path("data") / "console" / "train" / "001"),
                ),
                node_feature_names=[],
                node_type="vertex",
                sizing_field_interpolation_type="interpolated_vertex",
                weighted_imitation_config={
                    "enabled": True,
                    "datasets": ["console", "mold"],
                    "weight_source_mode": "console_mold_reference",
                    "cache_dir": tmpdir,
                    "beta": 1.0,
                    "epsilon": 1.0e-8,
                    "weight_clip_min": 1.0,
                    "weight_clip_max": 10.0,
                    "fallback_to_ones": True,
                },
            )

            observation = data.observation
            self.assertEqual(observation.imitation_weights.shape, observation.y.shape)
            self.assertEqual(observation.imitation_weights.dtype, torch.float32)
            self.assertEqual(observation.imitation_weights.device.type, "cpu")
            self.assertEqual(float(observation.imitation_weights_loaded.item()), 1.0)
            self.assertEqual(float(observation.imitation_weights_fallback.item()), 0.0)
            self.assertTrue(torch.all(observation.imitation_weights > 0))

    def test_missing_console_cache_falls_back_to_ones(self):
        expert_mesh = _make_simple_tet_mesh()
        queried_mesh = _make_simple_tet_mesh()

        with tempfile.TemporaryDirectory() as tmpdir:
            data = DummyAmberData(
                mesh=queried_mesh,
                source_data=SourceData(
                    expert_mesh=expert_mesh,
                    initial_mesh=queried_mesh,
                    dataset_name="console",
                    data_point_path=str(Path("data") / "console" / "train" / "001"),
                ),
                node_feature_names=[],
                node_type="vertex",
                sizing_field_interpolation_type="interpolated_vertex",
                weighted_imitation_config={
                    "enabled": True,
                    "datasets": ["console", "mold"],
                    "weight_source_mode": "console_mold_reference",
                    "cache_dir": tmpdir,
                    "beta": 1.0,
                    "epsilon": 1.0e-8,
                    "weight_clip_min": 1.0,
                    "weight_clip_max": 10.0,
                    "fallback_to_ones": True,
                },
            )

            observation = data.observation
            self.assertTrue(torch.allclose(observation.imitation_weights, torch.ones_like(observation.imitation_weights)))
            self.assertEqual(float(observation.imitation_weights_loaded.item()), 0.0)
            self.assertEqual(float(observation.imitation_weights_fallback.item()), 1.0)


if __name__ == "__main__":
    unittest.main()
