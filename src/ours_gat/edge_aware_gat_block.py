from typing import Any, Dict, Optional

import torch.nn as nn
from torch_geometric.data.batch import Batch
from torch_geometric.data.data import Data

from src.mpn.common.mpn_util import noop
from src.ours_gat.edge_aware_gat_modules import EdgeAwareGATEdgeModule, EdgeAwareGATNodeModule


class EdgeAwareGATBlock(nn.Module):
    """
    对齐 AMBER 的 MessagePassingBlock：
      - 先 edge_module，再 node_module
      - residual_connections: None | "inner" | "outer"
      - layer_norm: None | "inner" | "outer"
    forward 顺序与 AMBER 一致 :contentReference[oaicite:5]{index=5}
    """
    def __init__(
        self,
        stack_config: Dict[str, Any],
        latent_dimension: int,
    ):
        super().__init__()
        self._latent_dimension = latent_dimension   # 隐式维度

        # 残差连接和归一化配置读取（同AMBER）
        # 残差连接：把更新前的特征加回来，防止网络太深训练崩掉、信息丢失。
        # LayerNorm（层归一化）就是：对每个节点（或每条边）的特征向量做“标准化”，让训练更稳定、数值不爆炸。
        # “把每个向量的大小/分布拉回到比较正常的范围”。
        residual_connections: Optional[str] = stack_config.get("residual_connections")
        residual_connections = residual_connections.lower() if residual_connections is not None else None
        layer_norm: Optional[str] = stack_config.get("layer_norm")
        layer_norm = layer_norm.lower() if layer_norm is not None else None
        self.use_layer_norm = layer_norm in ["outer", "inner"]

        # 维度对齐 AMBER：edge_in=3d, node_in=2d
        edge_in_features = 3 * latent_dimension
        node_in_features = 2 * latent_dimension

        self.edge_module = EdgeAwareGATEdgeModule(
            in_features=edge_in_features,
            latent_dimension=latent_dimension,
            stack_config=stack_config,
        )
        self.node_module = EdgeAwareGATNodeModule(
            d = latent_dimension,
            stack_config=stack_config,
        )

        self.reset_parameters()

        # 层归一化层初始化
        if self.use_layer_norm:
            self._node_layer_norms = nn.LayerNorm(normalized_shape=latent_dimension)
            self._edge_layer_norms = nn.LayerNorm(normalized_shape=latent_dimension)
        else:
            self._node_layer_norms = None
            self._edge_layer_norms = None

        # 保存旧graph，用于残差连接
        self._old_graph: Dict[str, Any] = {}

        # 初始化残差连接和层归一化所需的函数
        self._initialize_maybes()

        # residual switches（对齐 AMBER）
        if residual_connections == "outer":
            self.maybe_store_old_graph = self._store_old_graph
            self.maybe_outer_residual = self._add_graph_residuals
        elif residual_connections == "inner":
            self.maybe_store_old_graph = self._store_old_graph
            self.maybe_inner_node_residual = self._add_node_residual
            self.maybe_inner_edge_residual = self._add_edge_residual

        # layer_norm switches（对齐 AMBER）
        if layer_norm == "outer":
            self.maybe_outer_layer_norm = self._graph_layer_norm
        elif layer_norm == "inner":
            self.maybe_inner_node_layer_norm = self._node_layer_norm
            self.maybe_inner_edge_layer_norm = self._edge_layer_norm

    def reset_parameters(self):
        """
        This resets all the parameters for all modules
        """
        for item in [self.node_module, self.edge_module]:
            if hasattr(item, "reset_parameters"):
                # hasattr函数，检查item 是否有 reset_parameters 这个方法
                item.reset_parameters()

    def _initialize_maybes(self):
        self.maybe_store_old_graph = noop

        self.maybe_outer_residual = noop
        self.maybe_inner_node_residual = noop
        self.maybe_inner_edge_residual = noop

        self.maybe_outer_layer_norm = noop
        self.maybe_inner_node_layer_norm = noop
        self.maybe_inner_edge_layer_norm = noop

    def forward(self, graph: Data):
        """
        block（edge-MLP + Node-GAT）前向传播
        """
        self.maybe_store_old_graph(graph=graph)     # 保存old_graph

        # edge first (inplace)
        self.edge_module(graph)
        # 当residual_connections == "inner"，非noop
        self.maybe_inner_edge_residual(graph=graph) # 内部残差，在每个子模块更新之后立刻加残差。
        # 当layer_norm == "inner"，非noop
        self.maybe_inner_edge_layer_norm(graph=graph)

        # then node (inplace, uses UPDATED edge_attr)
        self.node_module(graph)
        self.maybe_inner_node_residual(graph=graph)
        self.maybe_inner_node_layer_norm(graph=graph)

        # 当residual_connections == "outer"，非noop
        self.maybe_outer_residual(graph=graph) # 外部残差，整层 Block 跑完（边+点都更新完）再统一加回去。
        # 当layer_norm == "outer"，非noop
        self.maybe_outer_layer_norm(graph=graph) # 外部归一化，整层 Block 结束后再对两个量一起做归一化：

        return graph

    # --- residual/layernorm helpers（对齐 AMBER） ---
    def _store_old_graph(self, graph: Batch):
        # 保存old_graph（点特征、边特征）
        self._old_graph["x"] = graph.x
        self._old_graph["edge_attr"] = graph.edge_attr

    def _add_graph_residuals(self, graph: Batch):
        # 残差连接
        self._add_node_residual(graph)
        self._add_edge_residual(graph)

    def _add_node_residual(self, graph: Batch):
        graph.__setattr__("x", graph.x + self._old_graph["x"])

    def _add_edge_residual(self, graph: Batch):
        graph.__setattr__("edge_attr", graph.edge_attr + self._old_graph["edge_attr"])

    def _graph_layer_norm(self, graph: Batch) -> None:
        # 层归一化
        self._node_layer_norm(graph)
        self._edge_layer_norm(graph)

    def _node_layer_norm(self, graph: Batch) -> None:
        graph.__setattr__("x", self._node_layer_norms(graph.x))

    def _edge_layer_norm(self, graph: Batch) -> None:
        graph.__setattr__("edge_attr", self._edge_layer_norms(graph.edge_attr))