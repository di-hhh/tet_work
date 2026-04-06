import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch_geometric.data import Batch, Data

from src.algorithm.architecture.mlp import MLP

from src.ours_gat.get_gat_base import get_gat_from_graph


class EdgeAwareGat(nn.Module):
    def __init__(self, architecture_config: DictConfig, example_graph: Data):
        """

        初始化 Edge-Aware GAT架构。
        这是一个带有边特征感知的 GAT 网络

        参数：
            example_graph：示例图，用于推断节点和边的输入特征维度
            architecture_config：策略网络和价值网络的配置

        """
        super(EdgeAwareGat, self).__init__()

        self._node_type = "node"
        latent_dimension = architecture_config.latent_dimension # 潜在维度=64
        self.gat = get_gat_from_graph(
            example_graph=example_graph,
            latent_dimension=latent_dimension,
            node_name=self._node_type,
            base_config=architecture_config,
        ) # Edge-Aware GAT Layer

        # 解码器 mlp
        mlp_config = architecture_config.decoder
        self.decoder_mlp = MLP(
            in_features=latent_dimension,
            mlp_config=mlp_config,
            latent_dimension=latent_dimension,
        )
        self.readout = nn.Linear(latent_dimension, 1) # 线性层

    def forward(self, observations: Batch, **kwargs) -> torch.Tensor:
        """
        Args:
            observations: (Batch of) observation graph(s)
        Returns:
            A scalar value for each node in the batch of graphs

        参数：
            observations：（批量）观测图
        返回：
            图批次中每个节点的标量值
        """
        node_features, _, _ = self.gat(observations)       # MPN（20层message passing step）处理输入的图数据 observations
        node_features = node_features.get(self._node_type) # 特征提取，hvL（最终层输出）

        #  `mask_output` 属性，它是一个形状为 `(num_nodes,)` 的布尔张量，其中属于学习网格的节点为 `True`，属于初始网格的节点为 `False`
        #  该属性用于在 GNN 前向传播时屏蔽初始网格节点的输出，确保预测仅针对当前物理网格（学习网格）进行，而非初始网格。
        if hasattr(observations, "mask_output"):                     # 如果 mask_output 存在，则使用它作为索引从完整的 node_features 张量中提取一个子集。
            node_features = node_features[observations.mask_output]  # 仅保留属于学习网格的节点特征（topology_only模式下，初始节点特征子集中元素均为0）

        decoded_node_features = self.decoder_mlp(node_features)   # 将MPN最终层输出的节点特征hvL输入到 self.decoder_mlp（多层感知机）进行解码
        outputs = self.readout(decoded_node_features)             # 通过 self.readout 线性层生成最终输出，每个节点对应一个标量值

        return outputs  # 返回一个 torch.Tensor，包含批次中每个节点的一个标量预测值，xj
