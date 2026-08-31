# src/ours_gat/edge_aware_gat_block.py
# ---------------------------------
# Block：对齐你原 AMBER-style 框架（residual/layernorm 都保留）
# 但在 strict_paper=True 时，严格按论文：只做 node 的 Edge-aware GAT 更新（e_ij来自pos）
# ---------------------------------

from typing import Any, Dict, Optional  # 导入类型标注

import torch.nn as nn  # 导入神经网络模块
from torch_geometric.data.batch import Batch  # 导入 Batch 类型
from torch_geometric.data.data import Data  # 导入 Data 类型（与 Batch 兼容）

from src.mpn.common.mpn_util import noop  # 导入 noop（空操作函数）
from src.ours_gat.edge_aware_gat_modules import EdgeAwareGATEdgeModule, EdgeAwareGATNodeModule  # 导入模块（已改为严格论文版）


class EdgeAwareGATBlock(nn.Module):
    """
    Block 结构：
      - （可选）edge update（AMBER 风格，不是论文必需）
      - node update（严格论文 Edge-aware GAT：用 pos 构造 e_ij，并按 Eq.(3) 计算注意力）
      - residual_connections / layer_norm：保持你原工程逻辑不变
    """
    def __init__(self, stack_config: Dict[str, Any], latent_dimension: int):  # 初始化
        super().__init__()  # 调用父类初始化
        self._latent_dimension = latent_dimension  # 保存隐维度 d

        # 读取 residual_connections 配置（outer/inner/None）
        residual_connections: Optional[str] = stack_config.get("residual_connections")  # 取出配置
        residual_connections = residual_connections.lower() if residual_connections is not None else None  # 统一小写

        # 读取 layer_norm 配置（outer/inner/None）
        layer_norm: Optional[str] = stack_config.get("layer_norm")  # 取出配置
        layer_norm = layer_norm.lower() if layer_norm is not None else None  # 统一小写
        self.use_layer_norm = layer_norm in ["outer", "inner"]  # 是否启用 layer norm

        # strict_paper=True：严格按论文，不进行 edge_module 的学习更新
        self.strict_paper = bool(stack_config.get("strict_paper", True))  # 默认 True（严格论文）

        # edge_module（仅当 strict_paper=False 时才会真正用）
        edge_in_features = 3 * latent_dimension  # AMBER 风格 edge update 需要 3d 输入
        self.edge_module = EdgeAwareGATEdgeModule(  # 构建 edge module
            in_features=edge_in_features,  # 输入维度 3d
            latent_dimension=latent_dimension,  # 隐维度 d
            stack_config=stack_config,  # 传入配置
        )

        # node_module（严格论文版：会用 graph.pos 构造 e_ij）
        self.node_module = EdgeAwareGATNodeModule(  # 构建 node module
            d=latent_dimension,  # 节点隐维度 d
            stack_config=stack_config,  # 传入配置（num_heads、dropout等）
        )

        self.reset_parameters()  # 统一 reset

        # LayerNorm（按你的原逻辑）
        if self.use_layer_norm:  # 若启用 LN
            self._node_layer_norms = nn.LayerNorm(normalized_shape=latent_dimension)  # 节点 LN
            self._edge_layer_norms = nn.LayerNorm(normalized_shape=latent_dimension)  # 边 LN
        else:
            self._node_layer_norms = None  # 不启用则为 None
            self._edge_layer_norms = None  # 不启用则为 None

        self._old_graph: Dict[str, Any] = {}  # 缓存旧图特征（用于残差）

        self._initialize_maybes()  # 初始化“可能执行/可能noop”的函数指针

        # residual switches（对齐你原工程）
        if residual_connections == "outer":  # 外部残差
            self.maybe_store_old_graph = self._store_old_graph  # 存旧图
            self.maybe_outer_residual = self._add_graph_residuals  # block 结束后加残差
        elif residual_connections == "inner":  # 内部残差
            self.maybe_store_old_graph = self._store_old_graph  # 存旧图
            self.maybe_inner_node_residual = self._add_node_residual  # node 后加残差
            self.maybe_inner_edge_residual = self._add_edge_residual  # edge 后加残差

        # layer_norm switches（对齐你原工程）
        if layer_norm == "outer":  # 外部 LN
            self.maybe_outer_layer_norm = self._graph_layer_norm  # block 结束后 LN
        elif layer_norm == "inner":  # 内部 LN
            self.maybe_inner_node_layer_norm = self._node_layer_norm  # node 后 LN
            self.maybe_inner_edge_layer_norm = self._edge_layer_norm  # edge 后 LN

    def reset_parameters(self) -> None:  # reset 所有子模块参数
        for item in [self.node_module, self.edge_module]:  # 遍历 node/edge 模块
            if hasattr(item, "reset_parameters"):  # 如果模块实现了 reset_parameters
                item.reset_parameters()  # 调用 reset

    def _initialize_maybes(self) -> None:  # 初始化所有 maybe_* 为 noop
        self.maybe_store_old_graph = noop  # 默认不存旧图
        self.maybe_outer_residual = noop  # 默认无外部残差
        self.maybe_inner_node_residual = noop  # 默认无内部 node 残差
        self.maybe_inner_edge_residual = noop  # 默认无内部 edge 残差
        self.maybe_outer_layer_norm = noop  # 默认无外部 LN
        self.maybe_inner_node_layer_norm = noop  # 默认无内部 node LN
        self.maybe_inner_edge_layer_norm = noop  # 默认无内部 edge LN

    def forward(self, graph: Data):  # 前向传播
        self.maybe_store_old_graph(graph=graph)  # 根据 residual 配置决定是否存旧图

        # ---------- Edge update（严格论文默认跳过）----------
        if not self.strict_paper:  # 若不严格论文（想保留 AMBER 风格）
            self.edge_module(graph)  # 先更新 edge_attr（AMBER 风格）
            self.maybe_inner_edge_residual(graph=graph)  # 若 inner residual 则 edge 后加残差
            self.maybe_inner_edge_layer_norm(graph=graph)  # 若 inner LN 则 edge 后做 LN

        # ---------- Node update（严格论文 Edge-aware GAT）----------
        self.node_module(graph)  # 使用 pos 构造 e_ij，并按 Eq.(3) 更新 graph.x，同时写入 graph.p_scalar
        self.maybe_inner_node_residual(graph=graph)  # 若 inner residual 则 node 后加残差
        self.maybe_inner_node_layer_norm(graph=graph)  # 若 inner LN 则 node 后 LN

        # ---------- Outer residual / LN（对齐你原工程）----------
        self.maybe_outer_residual(graph=graph)  # 若 outer residual 则整个 block 结束后加残差
        self.maybe_outer_layer_norm(graph=graph)  # 若 outer LN 则整个 block 结束后 LN

        return graph  # 返回 graph（虽然你也可原地使用）

    # ----- residual / layernorm helper（对齐你原工程）-----
    def _store_old_graph(self, graph: Batch) -> None:  # 存旧特征
        self._old_graph["x"] = graph.x  # 保存旧 node 特征
        self._old_graph["edge_attr"] = graph.edge_attr  # 保存旧 edge 特征

    def _add_graph_residuals(self, graph: Batch) -> None:  # 对整图加残差
        self._add_node_residual(graph)  # 加 node 残差
        self._add_edge_residual(graph)  # 加 edge 残差

    def _add_node_residual(self, graph: Batch) -> None:  # node 残差
        graph.__setattr__("x", graph.x + self._old_graph["x"])  # x = x + x_old

    def _add_edge_residual(self, graph: Batch) -> None:  # edge 残差
        graph.__setattr__("edge_attr", graph.edge_attr + self._old_graph["edge_attr"])  # e = e + e_old

    def _graph_layer_norm(self, graph: Batch) -> None:  # 整图 LN
        self._node_layer_norm(graph)  # node LN
        self._edge_layer_norm(graph)  # edge LN

    def _node_layer_norm(self, graph: Batch) -> None:  # node LN
        graph.__setattr__("x", self._node_layer_norms(graph.x))  # 对 graph.x 做 LN

    def _edge_layer_norm(self, graph: Batch) -> None:  # edge LN
        graph.__setattr__("edge_attr", self._edge_layer_norms(graph.edge_attr))  # 对 graph.edge_attr 做 LN
