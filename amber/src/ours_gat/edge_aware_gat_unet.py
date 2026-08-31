"""
多尺度 U-Net 式 Edge-Aware GAT 网络 (EdgeAwareGATUNet)
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from src.ours_gat.edge_aware_gat_block import EdgeAwareGATBlock


_AMBER_PROTECTED_ATTRS = ("y", "mask_output", "current_sizing_field")


def _scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: Optional[int] = None) -> torch.Tensor:
    if dim_size is None:
        dim_size = index.max().item() + 1

    if src.dim() == 1:
        out = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
        count = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
        out.scatter_add_(0, index, src)
        count.scatter_add_(0, index, torch.ones_like(src))
        return out / count.clamp(min=1)

    out = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)
    count = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    idx_expanded = index.unsqueeze(-1).expand_as(src)
    out.scatter_add_(dim, idx_expanded, src)
    count.scatter_add_(0, index, torch.ones(index.size(0), dtype=src.dtype, device=src.device))
    count = count.clamp(min=1)
    for _ in range(src.dim() - 1):
        count = count.unsqueeze(-1)
    return out / count


def _graclus(edge_index: torch.Tensor, num_nodes: int, max_rounds: int = 5) -> torch.Tensor:
    device = edge_index.device
    src, dst = edge_index[0], edge_index[1]

    no_self = src != dst
    src, dst = src[no_self], dst[no_self]

    node_ids = torch.arange(num_nodes, device=device)
    cluster = node_ids.clone()
    matched = torch.zeros(num_nodes, dtype=torch.bool, device=device)

    for _ in range(max_rounds):
        if matched.all():
            break

        src_ok = ~matched[src]
        dst_ok = ~matched[dst]
        valid = src_ok & dst_ok
        if not valid.any():
            break
        v_src, v_dst = src[valid], dst[valid]

        perm = torch.randperm(v_src.size(0), device=device)
        v_src, v_dst = v_src[perm], v_dst[perm]

        proposal = node_ids.clone()
        proposal[v_src] = v_dst

        mutual = (proposal[proposal] == node_ids) & (~matched)
        is_match = mutual & (node_ids < proposal)

        matched_nodes = is_match.nonzero(as_tuple=True)[0]
        if matched_nodes.numel() == 0:
            break

        partners = proposal[matched_nodes]
        cluster[partners] = matched_nodes
        matched[matched_nodes] = True
        matched[partners] = True

    _, cluster = torch.unique(cluster, return_inverse=True)
    return cluster


def _hash_edges(src: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    return src * num_nodes + dst


def _lookup_edge_indices(query_hash: torch.Tensor, ref_hash: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    sorted_ref, sort_perm = ref_hash.sort()
    insert_pos = torch.searchsorted(sorted_ref, query_hash)
    insert_pos_clamped = insert_pos.clamp(max=sorted_ref.size(0) - 1)
    found = sorted_ref[insert_pos_clamped] == query_hash
    indices = sort_perm[insert_pos_clamped]
    indices[~found] = 0
    return indices, found


class EdgeAwareGATUNet(nn.Module):
    def __init__(
        self,
        latent_dimension: int,
        stack_config: Optional[Dict[str, Any]] = None,
        unet_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        if unet_config is not None:
            config = unet_config
        elif stack_config is not None:
            config = stack_config
        else:
            config = {}

        self._latent_dimension = latent_dimension
        self._num_levels = config.get("num_levels", 3)
        self._steps_per_level = config.get("steps_per_level", 3)

        block_config = config.get("stack_config", {})

        self._encoder_blocks = nn.ModuleList()
        for _ in range(self._num_levels):
            level_blocks = nn.ModuleList(
                [
                    EdgeAwareGATBlock(
                        stack_config=block_config,
                        latent_dimension=latent_dimension,
                    )
                    for _ in range(self._steps_per_level)
                ]
            )
            self._encoder_blocks.append(level_blocks)

        self._decoder_blocks = nn.ModuleList()
        for _ in range(self._num_levels - 1):
            level_blocks = nn.ModuleList(
                [
                    EdgeAwareGATBlock(
                        stack_config=block_config,
                        latent_dimension=latent_dimension,
                    )
                    for _ in range(self._steps_per_level)
                ]
            )
            self._decoder_blocks.append(level_blocks)

        self._skip_node_projections = nn.ModuleList(
            [nn.Linear(2 * latent_dimension, latent_dimension) for _ in range(self._num_levels - 1)]
        )
        self._skip_edge_projections = nn.ModuleList(
            [nn.Linear(2 * latent_dimension, latent_dimension) for _ in range(self._num_levels - 1)]
        )

    @property
    def latent_dimension(self) -> int:
        return self._latent_dimension

    def _build_graph_hierarchy(
        self, graph: Union[Data, Batch]
    ) -> Tuple[List[Union[Data, Batch]], List[torch.Tensor]]:
        graphs = [graph]
        clusters = []

        current = graph
        for _ in range(self._num_levels - 1):
            cluster = _graclus(current.edge_index, num_nodes=current.x.size(0))
            coarse = self._manual_pool(current, cluster)
            graphs.append(coarse)
            clusters.append(cluster)
            current = coarse

        return graphs, clusters

    @staticmethod
    def _manual_pool(graph: Union[Data, Batch], cluster: torch.Tensor) -> Union[Data, Batch]:
        num_coarse_nodes = cluster.max().item() + 1

        coarse_x = _scatter_mean(graph.x, cluster, dim=0, dim_size=num_coarse_nodes)

        edge_index = graph.edge_index
        coarse_src = cluster[edge_index[0]]
        coarse_dst = cluster[edge_index[1]]

        not_self_loop = coarse_src != coarse_dst
        coarse_src = coarse_src[not_self_loop]
        coarse_dst = coarse_dst[not_self_loop]

        if graph.edge_attr is not None:
            filtered_edge_attr = graph.edge_attr[not_self_loop]
        else:
            filtered_edge_attr = None

        edge_hash = coarse_src * num_coarse_nodes + coarse_dst
        unique_hash, inverse_indices = torch.unique(edge_hash, return_inverse=True)

        new_src = unique_hash // num_coarse_nodes
        new_dst = unique_hash % num_coarse_nodes
        coarse_edge_index = torch.stack([new_src, new_dst], dim=0)

        if filtered_edge_attr is not None:
            coarse_edge_attr = _scatter_mean(
                filtered_edge_attr,
                inverse_indices,
                dim=0,
                dim_size=unique_hash.size(0),
            )
        else:
            coarse_edge_attr = None

        coarse = Data(x=coarse_x, edge_index=coarse_edge_index, edge_attr=coarse_edge_attr)

        if hasattr(graph, "batch") and graph.batch is not None:
            coarse_batch = _scatter_mean(
                graph.batch.float(),
                cluster,
                dim=0,
                dim_size=num_coarse_nodes,
            ).long()
            coarse.batch = coarse_batch

        # codex: preserve AMBER-specific attributes that should survive the U-Net hierarchy.
        for attr_name in _AMBER_PROTECTED_ATTRS:
            if hasattr(graph, attr_name):
                setattr(coarse, attr_name, getattr(graph, attr_name))

        return coarse

    def forward(self, graph: Batch) -> None:
        graphs, clusters = self._build_graph_hierarchy(graph)

        encoder_node_features = []
        encoder_edge_features = []

        for level in range(self._num_levels):
            for block in self._encoder_blocks[level]:
                block(graphs[level])

            encoder_node_features.append(graphs[level].x.clone())
            encoder_edge_features.append(graphs[level].edge_attr.clone())

            if level < self._num_levels - 1:
                cluster = clusters[level]
                coarse_graph = graphs[level + 1]

                coarse_graph.x = _scatter_mean(
                    graphs[level].x,
                    cluster,
                    dim=0,
                    dim_size=coarse_graph.x.size(0),
                )

                fine_edge_index = graphs[level].edge_index
                fine_edge_attr = graphs[level].edge_attr
                coarse_src = cluster[fine_edge_index[0]]
                coarse_dst = cluster[fine_edge_index[1]]

                not_self_loop = coarse_src != coarse_dst
                if not_self_loop.any():
                    c_src = coarse_src[not_self_loop]
                    c_dst = coarse_dst[not_self_loop]
                    c_attr = fine_edge_attr[not_self_loop]
                    num_coarse_nodes = coarse_graph.x.size(0)
                    edge_hash = _hash_edges(c_src, c_dst, num_coarse_nodes)

                    coarse_ei = coarse_graph.edge_index
                    coarse_hash = _hash_edges(coarse_ei[0], coarse_ei[1], num_coarse_nodes)
                    coarse_edge_idx, found = _lookup_edge_indices(edge_hash, coarse_hash)
                    if found.any():
                        coarse_graph.edge_attr = _scatter_mean(
                            c_attr[found],
                            coarse_edge_idx[found],
                            dim=0,
                            dim_size=coarse_graph.edge_attr.size(0),
                        )

        for level in range(self._num_levels - 2, -1, -1):
            coarse_x = graphs[level + 1].x
            cluster = clusters[level]
            unpooled_x = coarse_x[cluster]

            skip_x = torch.cat([encoder_node_features[level], unpooled_x], dim=-1)
            graphs[level].x = self._skip_node_projections[level](skip_x)

            coarse_edge_attr = graphs[level + 1].edge_attr
            fine_edge_index = graphs[level].edge_index
            encoder_edge = encoder_edge_features[level]

            coarse_src = cluster[fine_edge_index[0]]
            coarse_dst = cluster[fine_edge_index[1]]
            num_coarse_nodes = graphs[level + 1].x.size(0)

            unpooled_edge = encoder_edge.clone()
            not_self_loop = coarse_src != coarse_dst

            if not_self_loop.any() and coarse_edge_attr is not None:
                c_src = coarse_src[not_self_loop]
                c_dst = coarse_dst[not_self_loop]
                edge_hash = _hash_edges(c_src, c_dst, num_coarse_nodes)

                coarse_ei = graphs[level + 1].edge_index
                coarse_hash = _hash_edges(coarse_ei[0], coarse_ei[1], num_coarse_nodes)
                coarse_edge_indices, found = _lookup_edge_indices(edge_hash, coarse_hash)

                nsl_indices = not_self_loop.nonzero(as_tuple=True)[0]
                unpooled_edge[nsl_indices[found]] = coarse_edge_attr[coarse_edge_indices[found]]

            skip_edge = torch.cat([encoder_edge, unpooled_edge], dim=-1)
            graphs[level].edge_attr = self._skip_edge_projections[level](skip_edge)

            for block in self._decoder_blocks[level]:
                block(graphs[level])

        graph.x = graphs[0].x
        graph.edge_attr = graphs[0].edge_attr

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  latent_dimension={self._latent_dimension},\n"
            f"  num_levels={self._num_levels},\n"
            f"  steps_per_level={self._steps_per_level},\n"
            f"  encoder_blocks={len(self._encoder_blocks)},\n"
            f"  decoder_blocks={len(self._decoder_blocks)}\n"
            f")"
        )
