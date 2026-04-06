from typing import Any, Dict, Optional

import torch.nn as nn
from torch_geometric.data.batch import Batch

from src.mpn.message_passing_block import MessagePassingBlock
from src.ours_gat.edge_aware_gat_block import EdgeAwareGATBlock


class EdgeAwareGatStack(nn.Module):
    def __init__(
        self,
        latent_dimension: int,
        stack_config: Dict[str, Any],   # stack的配置
    ):
        """
        定义一个新的神经网络模块类，继承自PyTorch的nn.Module
        EdgeAwareGATStack是边感知GAT堆栈的意思
        这个类将多个EdgeAwareGATLayer层堆叠在一起
        """

        super().__init__()
        self._num_steps: int = stack_config.get("num_steps")

        self._num_step_repeats: int = stack_config.get("num_step_repeats", 1)
        self._residual_connections: Optional[str] = stack_config.get("residual_connections")
        self._latent_dimension: int = latent_dimension

        # 创建层堆栈
        self.blocks = nn.ModuleList([
            EdgeAwareGATBlock(
                stack_config=stack_config,
                latent_dimension=self._latent_dimension,    # 64
            )
            for _ in range(self._num_steps)
        ])

        # 3. nn.ModuleList 的作用：
        #    - 将多个层打包成一个列表
        #    - PyTorch能识别并管理这些层的参数
        #    - 可以像普通列表一样遍历：for layer in self.layers:

        # 这里所有层都是相同的配置。
        # 有些高级设计会让不同层有不同的参数（如逐渐减少维度）。

    @property
    def num_steps(self) -> int:
        """
        How many steps this stack is composed of.
        """
        return self._num_steps

    @property
    def latent_dimension(self) -> int:
        """
        Dimensionality of the features that are handled in this stack
        Returns:
        """
        return self._latent_dimension

    def forward(self, graph: Batch) -> None:
        """
        前向传播，原地计算
        """
        # [Claude Code] 修复：原实现未使用 _num_step_repeats，每个 block 始终只执行一次。
        # 对齐 MessagePassingStack 的行为，支持同一 block 重复执行（权重共享）。
        for blk in self.blocks:
            for _ in range(self._num_step_repeats):
                graph = blk(graph)

    def __repr__(self):
        # [Claude Code] 修复：原实现引用 self._message_passing_steps（不存在），改为 self.blocks
        if self.blocks:
            return (
                f"{self.__class__.__name__}(\n"
                f" num_message_passing_steps={self.num_steps},\n"
                f" num_step_repeats={self._num_step_repeats},\n"
                f" first_step={self.blocks[0]}\n"
            )
        else:
            return f"{self.__class__.__name__}(\n" f" num_message_passing_steps={self.num_steps}\n"

