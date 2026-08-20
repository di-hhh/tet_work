from __future__ import annotations

from functools import cached_property
from typing import Dict, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from src.algorithm.dataloader.source_data import SourceData
from src.algorithm.util.fem_imitation_weights import get_imitation_weight_bundle
from src.algorithm.util.weighted_imitation_diagnostics_codex import compute_size_error_metrics_codex
from src.helpers.custom_types import MetricDict
from src.helpers.qol import prefix_keys
from src.mesh_util.sizing_field_util import get_sizing_field
from src.tasks.domains.mesh_wrapper import MeshWrapper
from src.tasks.features.fem.fem_problem import FEMProblem


class MeshMetrics:
    """
    Utility class that computes similarity metrics between two meshes. Contains different functions that define
    (relative) quality and similarity metrics between a reference mesh and an evaluated mesh.
    """

    def __init__(
        self,
        metric_config: DictConfig,
        reference_mesh: MeshWrapper,
        evaluated_mesh: MeshWrapper,
        fem_problem: Optional[FEMProblem],
        source_data: Optional[SourceData] = None,
        weighted_imitation_config: Optional[Dict] = None,
    ):
        self.metric_config = metric_config
        self.reference_mesh = reference_mesh
        self.evaluated_mesh = evaluated_mesh
        self.fem_problem = fem_problem
        self.source_data = source_data if source_data is not None else getattr(reference_mesh, "source_data", None)
        self.weighted_imitation_config = (
            weighted_imitation_config if weighted_imitation_config is not None else getattr(reference_mesh, "weighted_imitation_config", {})
        ) or {}

    def __call__(self) -> MetricDict:
        metrics = self.get_similarity_metrics()

        if self.fem_problem is not None:
            metrics |= self.get_fem_metrics()
        return metrics

    def get_similarity_metrics(self) -> MetricDict:
        """
        Compute similarity metrics between two meshes.

        The following metrics are computed on element midpoints and/or mesh vertices:
        - The (symmetric) Chamfer distance between the evaluated mesh and the reference mesh, as well as its
        exponentiated and density-aware variants
        - The (symmetric) Earth Mover's Distance (EMD) between the evaluated mesh and the reference mesh-

        Additionally, the following metrics are computed:
        - The element size difference between the evaluated mesh and the reference mesh. These differences are
            calculated as the absolute, squared, mean and maximum differences in element size.
            We calculate them both in the original and in the log space, and either for the original mesh only or
            symmetrized between the two meshes.

        Finally, we take the difference in the number of elements between the two meshes.

        Returns: A dictionary of similarity metrics as key-value pairs

        """
        computed_metrics = {}

        pointcloud_metrics = self.metric_config.pointcloud
        pointcloud_distances = self.pointcloud_distances(pointcloud_metrics)
        computed_metrics = computed_metrics | pointcloud_distances

        if self.metric_config.projected_l2_error:
            computed_metrics["projected_l2_error"] = self.projected_l2_error()
            computed_metrics["projected_l2_error_reverse"] = self.projected_l2_error(reverse=True)
            computed_metrics["projected_l2_error_symmetric"] = (
                computed_metrics["projected_l2_error"] + computed_metrics["projected_l2_error_reverse"]
            ) / 2

        if self.metric_config.get("physics_weighted_projected_l2_error", False):
            computed_metrics["physics_weighted_projected_l2_error"] = self.physics_weighted_projected_l2_error()
        if any(
            self.metric_config.get(metric_name, False)
            for metric_name in ["weighted_size_l2", "topk_high_importance_l2", "bucketed_error"]
        ):
            computed_metrics |= self._importance_weighted_projected_metrics()

        if self.metric_config.element_delta:
            computed_metrics["element_delta"] = self.evaluated_mesh.num_elements - self.reference_mesh.num_elements

        if self.metric_config.get("tetra_quality", False) and self.evaluated_mesh.mesh.dim() == 3:
            computed_metrics |= self.tetra_quality_metrics()

        return computed_metrics

    def tetra_quality_metrics(self) -> MetricDict:
        points = np.asarray(self.evaluated_mesh.vertex_positions, dtype=np.float64)
        tetra = np.asarray(self.evaluated_mesh.element_indices, dtype=np.int64)
        vertices = points[tetra]
        signed_six_volume = np.einsum(
            "ij,ij->i",
            np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]),
            vertices[:, 3] - vertices[:, 0],
        )
        volumes = np.abs(signed_six_volume) / 6.0
        edge_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        squared_edge_sum = sum(
            np.sum((vertices[:, left] - vertices[:, right]) ** 2, axis=1)
            for left, right in edge_pairs
        )
        valid = (volumes > 1.0e-15) & np.isfinite(volumes) & (squared_edge_sum > 0.0)
        quality = np.zeros(len(tetra), dtype=np.float64)
        quality[valid] = 12.0 * np.power(3.0 * volumes[valid], 2.0 / 3.0) / squared_edge_sum[valid]
        return {
            "tetra_quality_mean": float(np.mean(quality)) if quality.size else float("nan"),
            "tetra_quality_min": float(np.min(quality)) if quality.size else float("nan"),
            "tetra_quality_p05": float(np.quantile(quality, 0.05)) if quality.size else float("nan"),
            "tetra_degenerate_fraction": float(np.mean(~valid)) if quality.size else float("nan"),
        }

    def get_fem_metrics(self) -> MetricDict:
        """
        Calculate fem-specific mesh quality metrics, i.e., metrics that evaluate the mesh w.r.t. an underlying PDE.
        Only available if we actually have such a PDE.
        Returns:

        """
        assert self.fem_problem is not None
        fem_metrics = self.fem_problem.get_quality_metrics(self.evaluated_mesh)
        return prefix_keys(fem_metrics, "fem", separator="_")

    def projected_l2_error(self, reverse: bool = False) -> float:
        """
        Computes the relative L2 norm error between the sizing fields of the adaptive
        and reference meshes after projecting the reference field onto the adaptive mesh.

        Args:
            reverse (bool): If True, swaps the roles of the adaptive and reference meshes.

        Returns:
            float: Relative L2 error metric quantifying the difference between the two meshes.
        """
        from src.algorithm.util.amber_util import interpolate_vertex_field

        # Compute vertex-based sizing fields
        sizing_1 = get_sizing_field(self.reference_mesh, mesh_node_type="vertex")
        sizing_2 = get_sizing_field(self.evaluated_mesh, mesh_node_type="vertex")

        # Reverse case: swap reference and adaptive mesh roles
        if reverse:
            sizing_1, sizing_2 = sizing_2, sizing_1
            mesh_1, mesh_2 = self.evaluated_mesh, self.reference_mesh
        else:
            mesh_1, mesh_2 = self.reference_mesh, self.evaluated_mesh

        # Project the sizing field from the vertices of mesh_1 onto those of mesh_2
        sizing_1_projected = interpolate_vertex_field(mesh_1, mesh_2, sizing_1)

        # Compute L2 norm of the difference
        l2_diff = np.linalg.norm(sizing_2 - sizing_1_projected, ord=2)
        l2_ref = np.linalg.norm(sizing_1_projected, ord=2)

        # Normalize by reference field norm with small epsilon to prevent division by zero
        return l2_diff / (l2_ref + 1e-10)

    def physics_weighted_projected_l2_error(self) -> float:
        from src.algorithm.util.amber_util import interpolate_vertex_field

        if self.source_data is None:
            return float("nan")

        sizing_reference = get_sizing_field(self.reference_mesh, mesh_node_type="vertex")
        sizing_evaluated = get_sizing_field(self.evaluated_mesh, mesh_node_type="vertex")
        sizing_reference_projected = interpolate_vertex_field(self.reference_mesh, self.evaluated_mesh, sizing_reference)
        weight_bundle = get_imitation_weight_bundle(
            queried_mesh=self.evaluated_mesh,
            source_data=self.source_data,
            sizing_field_interpolation_type="interpolated_vertex",
            node_type="vertex",
            weighted_imitation_config=self.weighted_imitation_config,
        )  # [CodeX] 评估阶段复用训练侧同一套投影权重，保证 physics-weighted 指标与加权模仿目标一致。
        weights = np.asarray(weight_bundle["weights"], dtype=np.float64)
        differences = sizing_evaluated - sizing_reference_projected
        epsilon = float(self.weighted_imitation_config.get("epsilon", 1.0e-10))

        weighted_diff = np.sqrt(np.sum(weights * differences**2) / (np.sum(weights) + epsilon))
        weighted_reference = np.sqrt(np.sum(weights * sizing_reference_projected**2) / (np.sum(weights) + epsilon))
        return weighted_diff / (weighted_reference + epsilon)

    def _importance_weighted_projected_metrics(self) -> MetricDict:
        from src.algorithm.util.amber_util import interpolate_vertex_field

        if self.source_data is None:
            return {
                "weighted_size_l2": float("nan"),
                "topk_high_importance_l2": float("nan"),
                "bucket_low_size_l2": float("nan"),
                "bucket_high_size_l2": float("nan"),
                "bucket_high_low_ratio": float("nan"),
            }

        sizing_reference = get_sizing_field(self.reference_mesh, mesh_node_type="vertex")
        sizing_evaluated = get_sizing_field(self.evaluated_mesh, mesh_node_type="vertex")
        sizing_reference_projected = interpolate_vertex_field(self.reference_mesh, self.evaluated_mesh, sizing_reference)
        weight_bundle = get_imitation_weight_bundle(
            queried_mesh=self.evaluated_mesh,
            source_data=self.source_data,
            sizing_field_interpolation_type="interpolated_vertex",
            node_type="vertex",
            weighted_imitation_config=self.weighted_imitation_config,
        )
        importance = np.asarray(weight_bundle.get("normalized_importance", weight_bundle["weights"]), dtype=np.float64)
        metrics = compute_size_error_metrics_codex(
            sizing_evaluated,
            sizing_reference_projected,
            np.asarray(weight_bundle["weights"], dtype=np.float64),
            importance,
            epsilon=float(self.weighted_imitation_config.get("epsilon", 1.0e-8)),
            topk_percent=float(self.weighted_imitation_config.get("topk_percent", 0.2)),
            bucket_count=int(self.weighted_imitation_config.get("bucket_count", 5)),
        )
        return metrics

    def pointcloud_distances(self, all_point_metrics: Dict | DictConfig) -> MetricDict:
        """
        Compute pointcloud distances between the reference and evaluated meshes. This includes the Chamfer distance,
        the density-aware Chamfer distance, the exponentiated Chamfer distance, and the Earth Mover's Distance (EMD).
        Args:
            all_point_metrics: Dictionary of names of metrics to include. Has structure
                midpoints: [{cd, dcd, ecd, emd}],
                vertices: [{cd, dcd, ecd, emd}]
            to include the Chamfer distance, density-aware Chamfer distance, exponentiated Chamfer distance, and
            Earth Mover's Distance (EMD) for element midpoints and/or mesh vertices.

        Returns:

        """
        pointcloud_distances = {}
        for scope, point_metrics in all_point_metrics.items():
            if "cd" in point_metrics:  # chamfer distance
                pointcloud_distances[f"cd_{scope}"] = self._chamfer_distance(scope=scope, distance_type="vanilla")
            if "dcd" in point_metrics:  # density-aware chamfer distance
                pointcloud_distances[f"dcd_{scope}"] = self._chamfer_distance(scope=scope, distance_type="density_aware")
            if "ecd" in point_metrics:  # exponentiated chamfer distance
                pointcloud_distances[f"ecd_{scope}"] = self._chamfer_distance(scope=scope, distance_type="exponentiated")
        return pointcloud_distances

    def _chamfer_distance(self, scope: str = "midpoint", distance_type: str = "vanilla") -> float:
        if scope == "midpoint":
            distances1, indices1, distances2, indices2 = self._midpoint_distances_and_indices
        elif scope == "vertex":
            distances1, indices1, distances2, indices2 = self._vertex_distances_and_indices
        else:
            raise ValueError(f"Unknown scope '{scope}'")
        if distance_type == "vanilla":
            distance = 0.5 * (np.mean(distances1) + np.mean(distances2))
        elif distance_type == "density_aware" or distance_type == "exponentiated":
            exp_distances1 = np.exp(-distances1)
            exp_distances2 = np.exp(-distances2)
            if distance_type == "exponentiated":
                distance = 0.5 * (np.mean(1 - exp_distances1) + np.mean(1 - exp_distances2))
            elif distance_type == "density_aware":
                weighted_distances1 = exp_distances1 / np.bincount(indices1).astype(np.float32)[indices1]
                weighted_distances2 = exp_distances2 / np.bincount(indices2).astype(np.float32)[indices2]
                distance = 0.5 * (np.mean(1 - weighted_distances1) + np.mean(1 - weighted_distances2))
            else:
                raise ValueError(f"Unknown distance type '{distance_type}'")
        else:
            raise ValueError(f"Unknown distance type '{distance_type}'")
        return distance

    @cached_property
    def _midpoint_distances_and_indices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        distances1, indices1 = self.reference_mesh.midpoint_tree.query(self.evaluated_mesh.element_midpoints, k=1)
        distances2, indices2 = self.evaluated_mesh.midpoint_tree.query(self.reference_mesh.element_midpoints, k=1)
        return distances1, indices1, distances2, indices2

    @cached_property
    def _vertex_distances_and_indices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        distances1, indices1 = self.reference_mesh.vertex_tree.query(self.evaluated_mesh.vertex_positions, k=1)
        distances2, indices2 = self.evaluated_mesh.vertex_tree.query(self.reference_mesh.vertex_positions, k=1)
        return distances1, indices1, distances2, indices2
