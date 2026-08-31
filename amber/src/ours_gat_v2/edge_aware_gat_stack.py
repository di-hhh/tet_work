# src/ours_gat/edge_aware_gat_stack.py
# -----------------------------
# Stack：把多个 EdgeAwareGATBlock 堆叠起来
# 注意：forward 仍保持“原地更新 graph”，与你原工程一致
# -----------------------------

from typing import Any, Dict, Optional  # 导入类型标注

import torch.nn as nn  # 导入 PyTorch 神经网络模块
from torch_geometric.data.batch import Batch  # 导入 PyG 的 Batch

from src.ours_gat.edge_aware_gat_block import EdgeAwareGATBlock  # 导入我们改好的 Block


class EdgeAwareGatStack(nn.Module):  # 定义堆叠模块
    def __init__(self, latent_dimension: int, stack_config: Dict[str, Any]):  # 构造函数
        super().__init__()  # 调用父类构造
        self._num_steps: int = int(stack_config.get("num_steps"))  # 堆叠层数（步数）
        self._num_step_repeats: int = int(stack_config.get("num_step_repeats", 1))  # 每步重复次数（若你要用）
        self._residual_connections: Optional[str] = stack_config.get("residual_connections")  # 记录残差配置
        self._latent_dimension: int = latent_dimension  # 记录隐维度 d

        # 创建 block 列表：每个 block 内部实现一次 Edge-aware GAT 更新
        self.blocks = nn.ModuleList([  # ModuleList 让 PyTorch 能注册参数
            EdgeAwareGATBlock(  # 创建一个 block
                stack_config=stack_config,  # 传入配置（包含 strict_paper 等）
                latent_dimension=self._latent_dimension,  # 隐维度 d
            )
            for _ in range(self._num_steps)  # 重复 num_steps 次
        ])

    @property
    def num_steps(self) -> int:  # 返回堆叠层数
        return self._num_steps  # 直接返回

    @property
    def latent_dimension(self) -> int:  # 返回隐维度
        return self._latent_dimension  # 直接返回

    def forward(self, graph: Batch) -> Batch:  # 前向传播
        # 逐层执行 block
        for blk in self.blocks:  # 遍历每一层
            graph = blk(graph)  # 执行一次 block（内部会原地更新 graph.x，并可能更新 graph.p_scalar）
        return graph  # 返回最终 graph（便于链式调用/调试）

    def __repr__(self) -> str:  # 打印结构用
        return (  # 返回字符串
            f"{self.__class__.__name__}(\n"  # 打印类名
            f" num_steps={self._num_steps},\n"  # 打印层数
            f" num_step_repeats={self._num_step_repeats},\n"  # 打印重复次数
            f" latent_dimension={self._latent_dimension},\n"  # 打印隐维度
            f" residual_connections={self._residual_connections}\n"  # 打印残差配置
            f")"  # 结束括号
        )
