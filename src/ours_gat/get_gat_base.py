from typing import Any, Dict, Optional, Tuple, Union

from torch_geometric.data.data import Data
from src.ours_gat.edge_aware_gat_base import EdgeAwareGatBase


def get_gat_from_graph(
    *,
    example_graph: Data,
    latent_dimension: int,
    base_config: Dict[str, Any],
    node_name: str = "node",
    device: Optional = None,
) -> EdgeAwareGatBase:
    """
    Build and return a Message Passing Base specified in the config from the provided example graph.
    构建并返回基于配置文件中指定的消息传递基础架构（该架构根据提供的示例图生成）。
    """
    # example_graph：示例图，用于推断节点和边的输入特征维度
    in_node_features = example_graph.x.shape[1]
    in_edge_features = example_graph.edge_attr.shape[1]

    return get_gat(
        in_node_features=in_node_features, # 节点特征维度
        in_edge_features=in_edge_features, # 边特征维度
        latent_dimension=latent_dimension, # 潜在维度
        base_config=base_config,           # 配置
        node_name=node_name,
        device=device,
    )

def get_gat(
    *,
    in_node_features: Union[int, Dict[str, int]],
    in_edge_features: Union[int, Dict[Tuple[str, str, str], int]],
    latent_dimension: int,
    base_config: Dict[str, Any],
    node_name: str = "node",
    device: Optional = None,
) -> EdgeAwareGatBase:
    """
    Build and return a Message Passing Base specified in the config.
    构建并返回配置中指定的消息传递基础架构。
    """
    assert type(in_node_features) == type(in_edge_features), (
        f"May either provide feature dimensions as int or Dict, " f"but not both. " f"Given '{in_node_features}', '{in_edge_features}'"
    )

    create_graph_copy = base_config.get("create_graph_copy")
    assert_graph_shapes = base_config.get("assert_graph_shapes")
    stack_config = base_config.get("stack")
    embedding_config = base_config.get("embedding")
    edge_dropout = base_config.get("edge_dropout")
    # [Claude Code] 修复：移除从未使用的 patch_config 死代码；
    # 将 device 放入 params 直接传给构造函数，与 get_mpn_from_graph 保持一致。

    params = dict(
        in_node_features=in_node_features,
        in_edge_features=in_edge_features,
        latent_dimension=latent_dimension,
        stack_config=stack_config,             # 堆栈配置
        embedding_config=embedding_config,     # 嵌入配置
        edge_dropout=edge_dropout,             # 边丢弃
        create_graph_copy=create_graph_copy,   # 创建图副本
        assert_graph_shapes=assert_graph_shapes,
        device=device,
    )

    base = EdgeAwareGatBase(**params, node_type=node_name)
    return base
