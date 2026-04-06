import abc
from typing import Callable, Dict, Optional, Type, Any


from torch import nn
from torch_geometric.data import Batch

from src.mpn.common.mpn_util import get_create_copy, noop, unpack_features
from src.mpn.graph_assertions import MessagePassingGraphAssertions
from src.mpn.input_embedding import MessagePassingInputEmbedding
from src.ours_gat.edge_aware_gat_stack import EdgeAwareGatStack
from src.ours_gat.edge_aware_gat_unet import EdgeAwareGATUNet


class EdgeAwareGatBase(nn.Module, abc.ABC):
    """
    Graph Neural Network (GNN) Base module processes the graph observations of the environment.
    It uses a stack of GNN Blocks. Each block defines a single GNN pass.
    图神经网络（GNN）基础模块负责处理环境的图观测数据。
    它使用一个由多个 GNN 块组成的堆栈，其中每个块定义了一次独立的图神经网络传递。
    """

    def __init__(
        self,
        *,
        in_node_features: int,
        in_edge_features: int,
        latent_dimension: int,
        stack_config: Dict,
        embedding_config: Dict,
        output_type: str = "features",
        edge_dropout: float = 0.0,
        create_graph_copy: bool = True,
        assert_graph_shapes: bool = True,
        device: Optional = None,
        node_type: str = "node",
    ):
        super().__init__()
        self._latent_dimension = latent_dimension   # 嵌入维度（隐空间维度）

        self._node_type = node_type

        self.maybe_assertions: MessagePassingGraphAssertions
        if assert_graph_shapes:
            self.maybe_assertions = self._get_assertions()(in_node_features=in_node_features, in_edge_features=in_edge_features)
        else:
            self.maybe_assertions = noop

        if edge_dropout > 0.0:
            from src.mpn.common.edge_dropout import EdgeDropout
            self.maybe_edge_dropout = EdgeDropout(dropout_prob=edge_dropout)
        else:
            self.maybe_edge_dropout = noop

        self.maybe_create_copy: Callable = get_create_copy(create_graph_copy=create_graph_copy) # 创建图副本

        self.maybe_transform_output = self._get_transform_output(output_type=output_type) # 输出转换层

        # 定义输入嵌入层（初始节点特征维度、初始边特征维度、隐空间维度、嵌入配置）
        self.input_embeddings = MessagePassingInputEmbedding(
            in_node_features=in_node_features, # 3维
            in_edge_features=in_edge_features, # 2维
            latent_dimension=latent_dimension, # 隐空间维度=64
            embedding_config=embedding_config,
            device=device,
        )

        # 根据 stack_config 中的 stack_type 字段选择 Stack 或 UNet
        stack_type = stack_config.get("stack_type", "default") if stack_config else "default"
        if stack_type == "unet":
            self.edge_aware_gat_stack = EdgeAwareGATUNet(
                latent_dimension=latent_dimension,
                stack_config=stack_config,
            )
        else:
            self.edge_aware_gat_stack = EdgeAwareGatStack(
                stack_config=stack_config, latent_dimension=latent_dimension
            )

    def _get_assertions(self) -> Type[MessagePassingGraphAssertions]:
        return MessagePassingGraphAssertions

    # [Claude Code] 修复：移除 _get_input_embeddings 和 _get_message_passing_stack 两个死方法。
    # 这两个静态方法从 MessagePassingBase 复制而来，但 __init__ 中并未通过它们创建子模块，
    # 直接硬编码调用了 MessagePassingInputEmbedding 和 EdgeAwareGatStack，导致方法定义毫无意义。

    def transform_to_features(self, graph: Batch) -> Batch:
        return unpack_features(graph, agent_node_type=self._node_type)

    def _get_transform_output(self, output_type: str):
        """
        返回一个将网络输出转换为指定输出类型的函数。
        参数：
            output_type：可选 "features" 或 "graph"。
        返回：
            如果 output_type 为 "features"，则返回将网络输出转换为对应格式的函数；
            如果 output_type 为 "graph"，则返回一个不执行转换（或直接返回原图）的函数。
        """
        if output_type == "features":
            return self.transform_to_features
        elif output_type == "graph":
            return noop # 无操作函数，返回null
        else:
            raise ValueError(f"Unknown output_type '{output_type}'")

    def forward(self, graph: Batch) -> Batch:
        """
        对给定的输入执行消息传递/图神经网络的前向传播。
        参数：
            graph：PyTorch Geometric 的 Batch 对象。
                表示一个（批量的）图。
        返回：
            根据类初始化时的配置，返回一个修改后的图或（节点特征，边特征）元组。
            所有节点和边特征均经过嵌入处理及多轮消息传递。
        """
        self.maybe_assertions(graph)                # 输入graph, observations
        graph = self.maybe_create_copy(graph)
        self.maybe_edge_dropout(graph)
        self.input_embeddings(graph)                # 输入嵌入层
        self.edge_aware_gat_stack(graph)           # Edge-Aware GAT Layer * nums
        return self.maybe_transform_output(graph)   # 输出转换层