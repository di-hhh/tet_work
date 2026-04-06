import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.algorithm.util.console_mold_reference import (
    ensure_console_mold_reference_cache,
    get_console_mold_reference_fields,
)
from src.algorithm.util.linear_elasticity_reference_codex import (
    assemble_linear_elasticity_system_codex,
    solve_linear_elasticity_reference_codex,
)


def _boundary_faces_and_tets(element_indices: np.ndarray):
    face_to_tet = {}
    tetra_faces = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    for tet_idx, tet in enumerate(element_indices):
        for local_face in tetra_faces:
            face = tuple(sorted(int(tet[node_idx]) for node_idx in local_face))
            face_to_tet.setdefault(face, []).append(tet_idx)
    boundary_faces = []
    boundary_tets = []
    for face, tet_ids in face_to_tet.items():
        if len(tet_ids) == 1:
            boundary_faces.append(face)
            boundary_tets.append(tet_ids[0])
    return np.asarray(boundary_faces, dtype=np.int64), np.asarray(boundary_tets, dtype=np.int64)


class _FakeBoundaryMesh:
    # [CodeX] 最小边界网格桩对象：只提供 Console/Mold 参考缓存生成所需的 boundary_facets/facets/f2t 接口。
    def __init__(self, element_indices: np.ndarray):
        boundary_faces, boundary_tets = _boundary_faces_and_tets(element_indices)
        self.facets = boundary_faces.T
        self.f2t = np.vstack([boundary_tets, -np.ones_like(boundary_tets)])
        self._boundary_facet_ids = np.arange(boundary_faces.shape[0], dtype=np.int64)

    def boundary_facets(self) -> np.ndarray:
        return self._boundary_facet_ids


class _FakeMeshWrapper:
    # [CodeX] 最小专家网格桩对象：复用真实参考求解器，但避免依赖完整 AMBER 网格类与训练环境。
    def __init__(self, vertex_positions: np.ndarray, element_indices: np.ndarray):
        self.vertex_positions = np.asarray(vertex_positions, dtype=np.float64)
        self.element_indices = np.asarray(element_indices, dtype=np.int64)
        self.num_vertices = int(self.vertex_positions.shape[0])
        self.num_elements = int(self.element_indices.shape[0])
        self.mesh = _FakeBoundaryMesh(self.element_indices)

    def dim(self) -> int:
        return 3


@dataclass
class _FakeFeatureProvider:
    inlet_position: Optional[np.ndarray] = None


@dataclass
class _FakeSourceData:
    expert_mesh: _FakeMeshWrapper
    initial_mesh: _FakeMeshWrapper
    dataset_name: str
    data_point_path: str
    feature_provider: Optional[_FakeFeatureProvider] = None
    imitation_weight_cache: Optional[dict] = None


def _make_box_tet_mesh() -> _FakeMeshWrapper:
    vertex_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.6],
            [2.0, 0.0, 0.6],
            [2.0, 1.0, 0.6],
            [0.0, 1.0, 0.6],
        ],
        dtype=np.float64,
    )
    element_indices = np.array(
        [
            [0, 1, 2, 6],
            [0, 2, 3, 6],
            [0, 3, 7, 6],
            [0, 7, 4, 6],
            [0, 4, 5, 6],
            [0, 5, 1, 6],
        ],
        dtype=np.int64,
    )
    return _FakeMeshWrapper(vertex_positions, element_indices)


class LinearElasticityReferenceTestsCodex(unittest.TestCase):
    def test_small_tetra_solver_produces_finite_displacement_and_nonnegative_energy(self):
        vertex_positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        element_indices = np.array([[0, 1, 2, 3]], dtype=np.int64)

        stiffness_bundle = assemble_linear_elasticity_system_codex(
            vertex_positions=vertex_positions,
            element_indices=element_indices,
            young_modulus=1.0,
            poisson_ratio=0.3,
        )
        self.assertEqual(stiffness_bundle["stiffness"].shape, (12, 12))

        solution_bundle = solve_linear_elasticity_reference_codex(
            vertex_positions=vertex_positions,
            element_indices=element_indices,
            fixed_nodes=np.array([0, 1, 2], dtype=np.int64),
            load_faces=np.array([[1, 2, 3]], dtype=np.int64),
            load_vectors=np.array([[0.0, -1.0, 0.0]], dtype=np.float64),
            body_force=None,
            young_modulus=1.0,
            poisson_ratio=0.3,
            importance_metric="strain_energy_density",
            regularization_epsilon=1.0e-10,
        )

        self.assertTrue(np.all(np.isfinite(solution_bundle["displacement"])))
        self.assertTrue(np.all(solution_bundle["element_importance"] >= 0.0))
        self.assertGreater(float(np.max(solution_bundle["element_importance"])), 0.0)

    def test_console_reference_weights_are_positive_and_cached(self):
        mesh = _make_box_tet_mesh()
        source_data = _FakeSourceData(
            expert_mesh=mesh,
            initial_mesh=mesh,
            dataset_name="console",
            data_point_path=str(Path("data") / "console" / "train" / "sample_console"),
            feature_provider=_FakeFeatureProvider(),
            imitation_weight_cache={},
        )
        config = {
            "datasets": ["console", "mold"],
            "reference_physics_type": "linear_elasticity",
            "importance_metric": "strain_energy_density",
            "young_modulus": 1.0,
            "poisson_ratio": 0.3,
            "console_support_quantile": 0.10,
            "console_load_quantile": 0.90,
            "console_min_region_faces": 1,
            "console_load_type": "surface_traction",
            "console_surface_traction_magnitude": 1.0,
            "epsilon": 1.0e-8,
            "fallback_to_ones": False,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config["cache_dir"] = tmpdir
            cache_path = ensure_console_mold_reference_cache(
                source_data=source_data,
                weighted_imitation_config=config,
                overwrite=True,
            )
            fields = get_console_mold_reference_fields(
                source_data=source_data,
                weighted_imitation_config=config,
            )
            self.assertTrue(cache_path.exists())

        self.assertEqual(fields["vertex_importance"].shape[0], mesh.num_vertices)
        self.assertTrue(np.all(fields["vertex_importance"] >= 0.0))
        self.assertGreater(float(np.std(fields["vertex_importance"])), 0.0)
        self.assertEqual(str(fields["reference_physics_type"][0]), "linear_elasticity")

    def test_mold_reference_weights_are_positive_and_cached(self):
        mesh = _make_box_tet_mesh()
        source_data = _FakeSourceData(
            expert_mesh=mesh,
            initial_mesh=mesh,
            dataset_name="mold",
            data_point_path=str(Path("data") / "mold" / "train" / "sample_mold"),
            feature_provider=_FakeFeatureProvider(inlet_position=np.array([0.0, 0.2, 0.2], dtype=np.float64)),
            imitation_weight_cache={},
        )
        config = {
            "datasets": ["console", "mold"],
            "reference_physics_type": "linear_elasticity",
            "importance_metric": "strain_energy_density",
            "young_modulus": 1.0,
            "poisson_ratio": 0.3,
            "mold_inlet_radius_scale": 0.35,
            "mold_far_region_quantile": 0.80,
            "mold_min_region_faces": 1,
            "mold_pressure_magnitude": 1.0,
            "epsilon": 1.0e-8,
            "fallback_to_ones": False,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config["cache_dir"] = tmpdir
            cache_path = ensure_console_mold_reference_cache(
                source_data=source_data,
                weighted_imitation_config=config,
                overwrite=True,
            )
            fields = get_console_mold_reference_fields(
                source_data=source_data,
                weighted_imitation_config=config,
            )
            self.assertTrue(cache_path.exists())

        self.assertEqual(fields["vertex_importance"].shape[0], mesh.num_vertices)
        self.assertTrue(np.all(fields["vertex_importance"] >= 0.0))
        self.assertGreater(float(np.linalg.norm(fields["load_vectors"])), 0.0)
        self.assertEqual(str(fields["importance_metric"][0]), "strain_energy_density")


if __name__ == "__main__":
    unittest.main()
