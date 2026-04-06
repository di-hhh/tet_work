from dataclasses import dataclass
from functools import cached_property
from typing import List, Literal

import torch
from torch_geometric.data.data import Data

from src.algorithm.dataloader.mesh_generation_data import MeshGenerationData
from src.mesh_util.sizing_field_util import get_sizing_field
from src.tasks.domains.mesh_wrapper import MeshWrapper


@dataclass
class AmberData(MeshGenerationData):
    edge_feature_names: List[str] = None
    weighted_imitation_config: dict = None
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
            graph.imitation_weights = torch.tensor(
                self._imitation_weights,
                dtype=torch.float32,
            )  # [CodeX] 将投影后的最终训练权重绑定到监督节点上，保证损失聚合直接使用同一组节点。
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
        from src.algorithm.util.fem_imitation_weights import get_imitation_weight_bundle

        return get_imitation_weight_bundle(
            queried_mesh=self.mesh,
            source_data=self.source_data,
            sizing_field_interpolation_type=self.sizing_field_interpolation_type,
            node_type=self.node_type,
            weighted_imitation_config=self.weighted_imitation_config,
        )

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
        graph = self._add_current_sizing_field(mesh=mesh, graph=graph)
        return graph

    def _add_current_sizing_field(self, mesh: MeshWrapper, graph: Data) -> Data:
        sizing_field = get_sizing_field(mesh, mesh_node_type=self.node_type)
        graph.current_sizing_field = torch.Tensor(sizing_field).float()
        return graph
