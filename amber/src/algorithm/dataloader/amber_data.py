from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import List, Literal

import numpy as np
import torch
from torch_geometric.data.data import Data

from src.algorithm.dataloader.mesh_generation_data import MeshGenerationData
from src.mesh_util.sizing_field_util import get_sizing_field
from src.tasks.domains.mesh_wrapper import MeshWrapper


@dataclass
class AmberData(MeshGenerationData):
    edge_feature_names: List[str] = None
    weighted_imitation_config: dict = None
    physics_correction_config: dict = None
    add_self_edges: bool = True
    initial_mesh_handling: Literal["exclude", "topology_only", "full"] = "exclude"
    refinement_depth: int = 0
    sampled_count: int = 0

    def __post_init__(self):
        super().__post_init__()
        if self.initial_mesh_handling not in {"exclude", "topology_only", "full"}:
            raise ValueError(
                f"Invalid initial_mesh_handling: {self.initial_mesh_handling}. "
                "Must be one of 'exclude', 'topology_only', or 'full'."
            )

    @classmethod
    def from_reference(cls, reference: "AmberData", new_mesh: MeshWrapper) -> "AmberData":
        return cls(
            mesh=new_mesh,
            source_data=reference.source_data,
            node_type=reference.node_type,
            sizing_field_interpolation_type=reference.sizing_field_interpolation_type,
            weighted_imitation_config=reference.weighted_imitation_config,
            physics_correction_config=reference.physics_correction_config,
            node_feature_names=reference.node_feature_names,
            edge_feature_names=reference.edge_feature_names,
            add_self_edges=reference.add_self_edges,
            initial_mesh_handling=reference.initial_mesh_handling,
            refinement_depth=reference.refinement_depth + 1,
            sampled_count=reference.sampled_count,
        )

    def increment_sampled_count(self):
        self.sampled_count += 1

    @property
    def observation(self) -> Data:
        if self._observation is None:
            graph = self._get_observation_graph()
            graph.y = torch.tensor(self._labels, dtype=torch.float32)
            graph.imitation_weights = torch.tensor(self._imitation_weights, dtype=torch.float32)
            # [CodeX] 将当前输出节点对应的物理重要性和权重显式挂到图对象上，继续复用现有 weighted imitation loss 与诊断逻辑。
            graph.imitation_raw_importance = torch.tensor(
                self._imitation_weight_bundle["raw_importance"],
                dtype=torch.float32,
            )
            graph.imitation_normalized_importance = torch.tensor(
                self._imitation_weight_bundle["normalized_importance"],
                dtype=torch.float32,
            )
            graph.imitation_weights_loaded = torch.tensor(
                [float(self._imitation_weights_loaded)],
                dtype=torch.float32,
            )
            graph.imitation_weights_fallback = torch.tensor(
                [float(self._imitation_weights_fallback)],
                dtype=torch.float32,
            )
            for key, value in self._imitation_weight_bundle.get("diagnostic_scalars", {}).items():
                setattr(graph, key, torch.tensor([float(value)], dtype=torch.float32))
            self._observation = graph
        return self._observation

    @cached_property
    def _imitation_weight_bundle(self):
        return self._get_imitation_weight_bundle_for_mesh_codex(mesh=self.mesh)

    @cached_property
    def _imitation_weights(self):
        return self._imitation_weight_bundle["weights"]

    @cached_property
    def _imitation_weights_loaded(self):
        return self._imitation_weight_bundle["loaded"]

    @cached_property
    def _imitation_weights_fallback(self):
        return self._imitation_weight_bundle["fallback"]

    @observation.setter
    def observation(self, value) -> None:
        self._observation = value

    @property
    def graph_size(self) -> int:
        return self.observation.num_nodes + self.observation.num_edges

    def to(self, device) -> "AmberData":
        self.observation = self.observation.to(device)
        return self

    def _get_observation_graph(self) -> Data:
        graph = self._mesh_to_graph(self.mesh)
        if self.initial_mesh_handling in ["full", "topology_only"]:
            graph = self._extend_to_hierarchical_graph(graph)
        return graph

    def _extend_to_hierarchical_graph(self, graph: Data) -> Data:
        if self.refinement_depth == 0:
            graph.x = torch.cat([graph.x, torch.zeros(len(graph.x))[:, None]], dim=1)
            graph.mask_output = torch.ones(len(graph.x)).bool()
        else:
            initial_graph = self._mesh_to_graph(self.source_data.initial_mesh)
            from src.mesh_util.transforms.mesh_to_graph import get_inter_graph_edges

            inter_edge_attr, inter_edge_index = get_inter_graph_edges(
                src_mesh=self.mesh,
                dest_mesh=self.source_data.initial_mesh,
                node_type=self.node_type,
                edge_feature_names=self.edge_feature_names,
            )
            graph.edge_index = torch.cat(
                [graph.edge_index, initial_graph.edge_index + len(graph.x), inter_edge_index],
                dim=1,
            )
            graph.edge_attr = torch.cat([graph.edge_attr, initial_graph.edge_attr, inter_edge_attr], dim=0)

            graph.mask_output = torch.cat(
                [torch.ones(len(graph.x)), torch.zeros(len(initial_graph.x))],
                dim=0,
            ).bool()
            graph.x = torch.cat([graph.x, torch.zeros(len(graph.x))[:, None]], dim=1)

            if hasattr(graph, "physics_feature_available") and hasattr(initial_graph, "physics_feature_available"):
                graph.physics_feature_available = torch.cat(
                    [graph.physics_feature_available, initial_graph.physics_feature_available],
                    dim=0,
                )
            if hasattr(graph, "physics_feature") and hasattr(initial_graph, "physics_feature"):
                graph.physics_feature = torch.cat(
                    [graph.physics_feature, initial_graph.physics_feature],
                    dim=0,
                )
            for attr_name in [
                "physics_feature_stage_field_loaded",
                "physics_feature_pipeline_indicator_loaded",
            ]:
                if hasattr(graph, attr_name) and hasattr(initial_graph, attr_name):
                    setattr(
                        graph,
                        attr_name,
                        torch.cat([getattr(graph, attr_name), getattr(initial_graph, attr_name)], dim=0),
                    )

            if self.initial_mesh_handling == "topology_only":
                initial_graph.x = torch.zeros_like(initial_graph.x)

            initial_graph.x = torch.cat([initial_graph.x, torch.ones(len(initial_graph.x))[:, None]], dim=1)
            graph.x = torch.cat([graph.x, initial_graph.x], dim=0)
        return graph

    def _mesh_to_graph(self, mesh: MeshWrapper) -> Data:
        from src.mesh_util.transforms.mesh_to_graph import mesh_to_graph

        graph = mesh_to_graph(
            wrapped_mesh=mesh,
            node_feature_names=self.node_feature_names,
            node_type=self.node_type,
            edge_feature_names=self.edge_feature_names,
            feature_provider=self.feature_provider,
            add_self_edges=self.add_self_edges,
        )
        graph = self._add_physics_feature(mesh=mesh, graph=graph)
        graph = self._add_current_sizing_field(mesh=mesh, graph=graph)
        return graph

    def _add_current_sizing_field(self, mesh: MeshWrapper, graph: Data) -> Data:
        sizing_field = get_sizing_field(mesh, mesh_node_type=self.node_type)
        graph.current_sizing_field = torch.Tensor(sizing_field).float()
        return graph

    def _add_physics_feature(self, mesh: MeshWrapper, graph: Data) -> Data:
        config = self.physics_correction_config or {}
        if not config.get("enable_physics_correction_branch", False):
            return graph

        num_nodes = graph.x.shape[0]
        availability = torch.zeros((num_nodes, 1), dtype=torch.float32)
        physics_feature = torch.zeros((num_nodes, 1), dtype=torch.float32)
        feature_available = False
        if self._should_append_physics_feature():
            # [CodeX] Reuse the existing importance feature slot for legacy Console/Mold and pipeline samples.
            feature_values, feature_available = self._get_physics_feature_values(
                mesh=mesh,
                expected_size=num_nodes,
            )
            physics_feature = torch.tensor(feature_values, dtype=torch.float32).reshape(num_nodes, 1)
            graph.x = torch.cat([graph.x, physics_feature], dim=1)
            availability = torch.full((num_nodes, 1), float(feature_available), dtype=torch.float32)

        # [CodeX] 记录每个节点的物理特征可用性，供 gate_zero / disable_branch 在缺特征时安全回退。
        graph.physics_feature_available = availability
        graph.physics_feature = physics_feature
        feature_source = self._get_physics_feature_source()
        graph.physics_feature_stage_field_loaded = torch.full(
            (num_nodes, 1),
            float(feature_available and feature_source in {"stage_field", "stage_field_fusion"}),
            dtype=torch.float32,
        )
        graph.physics_feature_pipeline_indicator_loaded = torch.full(
            (num_nodes, 1),
            float(feature_available and feature_source == "pipeline_indicator"),
            dtype=torch.float32,
        )
        return graph

    def _should_append_physics_feature(self) -> bool:
        dataset_name = getattr(self.source_data, "dataset_name", None)
        if dataset_name in {"console", "mold"}:
            return True
        return self._get_physics_feature_source() in {"pipeline_indicator", "stage_field", "stage_field_fusion"}

    def _get_physics_feature_source(self) -> str:
        cache = getattr(self.source_data, "imitation_weight_cache", None) or {}
        if cache.get("physics_feature_source"):
            return str(cache.get("physics_feature_source"))
        if cache.get("weight_source_mode"):
            return str(cache.get("weight_source_mode"))
        weighted_config = self.weighted_imitation_config or {}
        return str(weighted_config.get("weight_source_mode", "console_mold_reference"))

    def _get_imitation_weight_bundle_for_mesh_codex(self, mesh: MeshWrapper) -> dict:
        cache = getattr(self, "_imitation_weight_bundle_cache_codex", None)
        if cache is None:
            cache = {}
            self._imitation_weight_bundle_cache_codex = cache

        cache_key = id(mesh)
        if cache_key not in cache:
            cache[cache_key] = self._get_uncached_imitation_weight_bundle_for_mesh_codex(mesh=mesh)
        return cache[cache_key]

    def _get_uncached_imitation_weight_bundle_for_mesh_codex(self, mesh: MeshWrapper) -> dict:
        from src.algorithm.util.fem_imitation_weights import get_imitation_weight_bundle

        # [CodeX] 统一复用现有 reference physics -> queried mesh 投影链路，但显式绕过 AmberData 的本地缓存。
        return get_imitation_weight_bundle(
            queried_mesh=mesh,
            source_data=self.source_data,
            sizing_field_interpolation_type=self.sizing_field_interpolation_type,
            node_type=self.node_type,
            weighted_imitation_config=self.weighted_imitation_config,
        )

    def _get_physics_feature_bundle_for_mesh_codex(self, mesh: MeshWrapper) -> dict:
        cache = getattr(self, "_physics_feature_bundle_cache_codex", None)
        if cache is None:
            cache = {}
            self._physics_feature_bundle_cache_codex = cache

        feature_source = self._get_physics_feature_source()
        cache_key = (id(mesh), feature_source)
        if cache_key not in cache:
            cache[cache_key] = self._get_uncached_physics_feature_bundle_for_mesh_codex(
                mesh=mesh,
                feature_source=feature_source,
            )
        return cache[cache_key]

    def _get_uncached_physics_feature_bundle_for_mesh_codex(self, mesh: MeshWrapper, feature_source: str) -> dict:
        from src.algorithm.util.fem_imitation_weights import get_imitation_weight_bundle

        feature_config = dict(self.weighted_imitation_config or {})
        feature_config["weight_source_mode"] = feature_source
        return get_imitation_weight_bundle(
            queried_mesh=mesh,
            source_data=self.source_data,
            sizing_field_interpolation_type=self.sizing_field_interpolation_type,
            node_type=self.node_type,
            weighted_imitation_config=feature_config,
        )

    def _select_physics_feature_values_from_bundle_codex(self, *, bundle: dict, feature_mode: str) -> tuple[np.ndarray, bool]:
        raw_importance = np.asarray(bundle["raw_importance"], dtype=np.float32)
        normalized_importance = np.asarray(bundle["normalized_importance"], dtype=np.float32)
        feature_available = bool(bundle.get("loaded", False))

        if feature_mode in {"normalized_importance", "importance", "node_importance"}:
            feature_values = normalized_importance
        elif feature_mode == "raw_importance":
            feature_values = raw_importance
        elif feature_mode == "log_raw_importance":
            feature_values = np.log1p(np.maximum(raw_importance, 0.0)).astype(np.float32)
        elif feature_mode == "clipped_raw_importance":
            clip_value = float(np.quantile(raw_importance, 0.95)) if raw_importance.size > 0 else 0.0
            feature_values = np.clip(raw_importance, a_min=0.0, a_max=clip_value).astype(np.float32)
        else:
            raise ValueError(f"Unsupported physics_feature_mode '{feature_mode}'")
        return feature_values.astype(np.float32), feature_available

    def _reproject_physics_feature_values_codex(
        self,
        *,
        mesh: MeshWrapper,
        feature_mode: str,
        expected_size: int,
    ) -> tuple[np.ndarray | None, bool]:
        try:
            reprojected_bundle = self._get_uncached_physics_feature_bundle_for_mesh_codex(
                mesh=mesh,
                feature_source=self._get_physics_feature_source(),
            )
        except Exception:
            return None, False

        cache = getattr(self, "_physics_feature_bundle_cache_codex", None)
        if cache is not None:
            # [CodeX] 若重投影成功，则刷新当前 mesh 的本地 bundle 缓存，避免同一轮重复触发 mismatch 补救。
            cache[(id(mesh), self._get_physics_feature_source())] = reprojected_bundle

        reprojected_values, reprojected_available = self._select_physics_feature_values_from_bundle_codex(
            bundle=reprojected_bundle,
            feature_mode=feature_mode,
        )
        if not reprojected_available:
            reprojected_values = np.zeros_like(reprojected_values, dtype=np.float32)
        if reprojected_values.shape[0] != expected_size:
            return None, False
        return reprojected_values.astype(np.float32), reprojected_available

    def _get_physics_feature_values(self, mesh: MeshWrapper, expected_size: int | None = None) -> tuple[np.ndarray, bool]:
        config = self.physics_correction_config or {}
        feature_mode = str(config.get("physics_feature_mode", "normalized_importance"))
        if hasattr(self, "_get_physics_feature_bundle_for_mesh_codex"):
            bundle = self._get_physics_feature_bundle_for_mesh_codex(mesh=mesh)
        else:
            bundle = self._get_imitation_weight_bundle_for_mesh_codex(mesh=mesh)
        feature_values, feature_available = self._select_physics_feature_values_from_bundle_codex(
            bundle=bundle,
            feature_mode=feature_mode,
        )

        if not feature_available:
            # [CodeX] 若当前 mesh 没有可用 physics cache，则先退化到零特征；只有长度 mismatch 时再尝试显式重投影补救。
            feature_values = np.zeros_like(feature_values, dtype=np.float32)

        if expected_size is None:
            expected_size = _graph_node_count_from_mesh(mesh=mesh, node_type=self.node_type)
        if feature_values.shape[0] != expected_size:
            # [CodeX] 长度不匹配时先基于 reference fields 对当前 mesh 再做一次显式重投影；只有仍失败时才退化为零特征。
            reprojected_values, reprojected_available = self._reproject_physics_feature_values_codex(
                mesh=mesh,
                feature_mode=feature_mode,
                expected_size=int(expected_size),
            )
            if reprojected_values is not None:
                return reprojected_values.astype(np.float32), reprojected_available
            feature_values = np.zeros(int(expected_size), dtype=np.float32)
            feature_available = False
        return feature_values.astype(np.float32), feature_available


def _graph_node_count_from_mesh(*, mesh: MeshWrapper, node_type: str) -> int:
    # [CodeX] 统一根据 node_type 计算图节点数，避免顶点图和单元图切换时出现静默错位。
    if node_type == "element":
        return mesh.num_elements
    return mesh.num_vertices
