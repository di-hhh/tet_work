# src/ours_gat/edge_aware_gat_modules.py
# -----------------------------
# 严格按论文的 Edge-aware GAT：
#   Eq.(1) d_ij = ||x_i - x_j||
#   Eq.(2) u_ij = (x_i - x_j) / ||x_i - x_j||
#   Eq.(3) alpha_ij = softmax( LeakyReLU( a^T [W h_i || W h_j || e_ij] ) )
#   Eq.(4) P_i = sum_j alpha_ij * d_ij   (保存到 graph.p_scalar)
# 注意：这里的 x_i/x_j 指的是坐标，h_i/h_j 指的是节点特征向量（graph.x）
# -----------------------------

from typing import Dict, Any, Optional  # 导入类型标注工具（字典/任意类型/可选类型）

import torch  # 导入 PyTorch 主库
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入常用函数（激活/Dropout等）
from torch_geometric.data.batch import Batch  # 导入 PyG 的 Batch 数据结构
from torch_geometric.utils import softmax  # 导入 PyG 的按节点归一化 softmax
from torch_scatter import scatter_add  # 导入 scatter_add 用于按 dst 聚合消息

from src.mpn.common.latent_mlp import LatentMLP  # 复用你工程中的 LatentMLP（如果还要 edge update）


class EdgeAwareGATEdgeModule(nn.Module):
    """
    （可选）AMBER 风格 edge update：
        e_new = MLP([h_src, h_dst, e])
    注意：论文 Edge-aware GAT 不要求更新 edge_attr；严格模式可以不用这个模块。
    """
    def __init__(self, *, in_features: int, latent_dimension: int, stack_config: Dict):  # 初始化
        super().__init__()  # 调用父类初始化
        mlp_config = stack_config.get("mlp")  # 读取 MLP 配置
        self._mlp = LatentMLP(in_features=in_features, latent_dimension=latent_dimension, config=mlp_config)  # 构建 MLP

    def forward(self, graph: Batch) -> None:  # 前向传播（原地更新）
        src, dst = graph.edge_index  # 取出边两端索引（src=j, dst=i）
        x_src = graph.x[src]  # 取出源节点特征 [E, d]
        x_dst = graph.x[dst]  # 取出目标节点特征 [E, d]
        e = graph.edge_attr  # 取出边特征 [E, d]
        aggregated = torch.cat([x_src, x_dst, e], dim=1)  # 拼接得到 [E, 3d]
        graph.__setattr__("edge_attr", self._mlp(aggregated))  # 用 MLP 更新 edge_attr 为 [E, d]


class EdgeAwareGATNodeModule(nn.Module):
    """
    严格按论文 Eq.(1)(2)(3)(4) 的节点更新模块（Edge-aware GAT）。

    关键点：
    1) e_ij 必须来自几何：e_ij = [d_ij, u_ij]，由 graph.pos 计算（不是学习出来的 edge_attr）。
    2) 注意力 logits = a^T [W h_i || W h_j || e_ij]，其中 i=dst, j=src（与你的 edge_index 约定一致）。
    3) 节点更新 h_i' = sum_j alpha_ij * (W h_j) （标准 GAT 聚合）
    4) 同时计算 P_i = sum_j alpha_ij * d_ij（论文 Eq.4），写入 graph.p_scalar

    输入：
      graph.x: [N, d]  节点隐向量 h
      graph.pos: [N, 3] 节点坐标（用于 d_ij, u_ij）
      graph.edge_index: [2, E]，src=j, dst=i

    输出（原地）：
      graph.x: [N, d] 更新后的节点向量
      graph.p_scalar: [N, 1] 每个节点的距离传播量 P_i（可选用于后续）
    """

    def __init__(self, d: int, stack_config: Dict[str, Any]):  # 构造函数
        super().__init__()  # 调用父类构造
        self.d = d  # 保存节点隐维度 d

        # -------- 多头设置（与标准 GAT 一致）--------
        self.num_heads = int(stack_config.get("num_heads", 4))  # 读取注意力头数 H（默认4）
        if d % self.num_heads != 0:  # 检查 d 是否能被 H 整除
            raise ValueError(f"d={d} must be divisible by num_heads={self.num_heads}")  # 不整除则报错
        self.d_head = d // self.num_heads  # 每个 head 的维度 Dh

        # -------- Dropout 设置 --------
        self.feat_drop = nn.Dropout(float(stack_config.get("feat_drop", 0.0)))  # 节点特征 dropout
        self.attn_drop = nn.Dropout(float(stack_config.get("attn_drop", 0.0)))  # 注意力权重 dropout

        # -------- 论文 Eq.(3) 里的 W：对节点特征做线性变换 --------
        self.fc_node = nn.Linear(d, d, bias=bool(stack_config.get("use_bias", True)))  # W: R^d -> R^d

        # -------- 论文 e_ij = [d_ij, u_ij]，这里 e_dim = 1 + 3 = 4 --------
        self.e_dim = 4  # e_ij 的维度（严格按论文：距离1维 + 方向3维）
        # 将 e_ij 投影到 head 维度，便于拼接进注意力输入（不改变“e_ij来自几何”的事实）
        self.fc_edge_geom = nn.Linear(self.e_dim, self.num_heads * self.d_head, bias=True)  # 几何边特征投影层

        # -------- 论文 Eq.(3) 里的 a^T：注意力打分向量（每个 head 一个）--------
        # 注意力输入维度： [W h_i || W h_j || e_ij_projected] => Dh + Dh + Dh = 3Dh
        self.attn_fc = nn.Linear(3 * self.d_head, 1, bias=True)  # 输出每条边每个 head 的一个标量 logit

        # -------- 输出投影（可选，但建议保留：混合 heads 后再线性）--------
        self.out_proj = nn.Linear(d, d, bias=True)  # 将 concat/merge 后的结果再投影回 d

        # -------- 可选的自环（论文未强调；默认关闭以保持严格最小改动）--------
        self.self_loop = bool(stack_config.get("self_loop", False))  # 是否加自环残差
        self.self_node_transform = bool(stack_config.get("self_node_transform", False)) and self.self_loop  # 自环是否过线性
        self.fc_self = nn.Linear(d, d, bias=True) if self.self_node_transform else None  # 自环线性层（可选）

        # -------- 初始化参数 --------
        self.reset_parameters()  # 调用初始化

    def reset_parameters(self) -> None:  # 参数初始化函数
        gain = nn.init.calculate_gain("relu")  # 计算 xavier 的增益系数
        nn.init.xavier_normal_(self.fc_node.weight, gain=gain)  # 初始化节点线性层权重
        if self.fc_node.bias is not None:  # 若存在偏置
            nn.init.zeros_(self.fc_node.bias)  # 偏置置零

        nn.init.xavier_normal_(self.fc_edge_geom.weight, gain=gain)  # 初始化几何边投影权重
        nn.init.zeros_(self.fc_edge_geom.bias)  # 初始化 bias 为 0（存在则置零）

        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)  # 初始化注意力层权重
        nn.init.zeros_(self.attn_fc.bias)  # 初始化注意力 bias 为 0

        nn.init.xavier_normal_(self.out_proj.weight, gain=gain)  # 初始化输出投影权重
        nn.init.zeros_(self.out_proj.bias)  # 初始化输出投影 bias 为 0

        if self.fc_self is not None:  # 若启用自环线性
            nn.init.xavier_normal_(self.fc_self.weight, gain=gain)  # 初始化自环线性权重
            nn.init.zeros_(self.fc_self.bias)  # 初始化自环 bias 为 0

    def forward(self, graph: Batch) -> None:  # 前向传播（原地更新 graph.x）
        x = graph.x  # 取出节点特征 h: [N, d]
        pos = graph.pos  # 取出节点坐标: [N, 3]（严格按论文必须有）
        src, dst = graph.edge_index  # 取出边索引（src=j, dst=i）
        N = x.size(0)  # 节点数 N
        E = src.size(0)  # 边数 E

        # -------- 1) 论文 Eq.(1)(2)：从 pos 计算 d_ij 与 u_ij --------
        # 论文中 u_ij = (x_i - x_j)/||x_i-x_j||，这里 i=dst, j=src（与你的约定一致）
        vec_ij = pos[dst] - pos[src]  # 计算向量 (x_i - x_j): [E, 3]
        d_ij = torch.norm(vec_ij, dim=-1, keepdim=True)  # 计算距离 ||x_i-x_j||: [E, 1]
        eps = 1e-12  # 防止除零的小量
        u_ij = vec_ij / (d_ij + eps)  # 单位方向向量: [E, 3]
        e_ij = torch.cat([d_ij, u_ij], dim=-1)  # 拼接得到 e_ij=[d_ij,u_ij]: [E, 4]

        # -------- 2) 对节点特征做 Dropout，然后计算 z = W h（论文 Eq.(3) 的 Wh）--------
        x_in = x  # 保留输入，用于可选自环残差
        x = self.feat_drop(x)  # 节点特征 dropout
        z = self.fc_node(x).view(N, self.num_heads, self.d_head)  # z=Wh: [N, H, Dh]

        # -------- 3) 将 e_ij 投影到 head 维度（仍然是几何特征，只是维度对齐）--------
        e_proj = self.fc_edge_geom(e_ij).view(E, self.num_heads, self.d_head)  # [E, H, Dh]

        # -------- 4) 构造注意力输入 [z_i || z_j || e_ij] 并计算 logits（论文 Eq.(3)）--------
        z_i = z[dst]  # 目标节点 i 的 z: [E, H, Dh]
        z_j = z[src]  # 源节点 j 的 z: [E, H, Dh]
        att_in = torch.cat([z_i, z_j, e_proj], dim=-1)  # 拼接得到 [E, H, 3Dh]
        logits = self.attn_fc(att_in).squeeze(-1)  # 线性得到 logits: [E, H]
        logits = F.leaky_relu(logits, negative_slope=0.2)  # 论文用 LeakyReLU（Eq.3）

        # -------- 5) alpha = softmax over incoming edges of each dst（按 i 的入边归一化）--------
        alpha = softmax(logits, dst)  # 对每个 dst 节点的入边做 softmax: [E, H]
        alpha = self.attn_drop(alpha)  # 注意力 dropout: [E, H]

        # -------- 6) 节点更新：h_i' = sum_j alpha_ij * z_j（标准 GAT 聚合）--------
        msg = z_j * alpha.unsqueeze(-1)  # 边上消息 alpha_ij * z_j: [E, H, Dh]
        h = scatter_add(msg, dst, dim=0, dim_size=N)  # 聚合到节点 i: [N, H, Dh]
        h = h.reshape(N, self.d)  # 合并 heads: [N, d]
        h = self.out_proj(h)  # 输出投影: [N, d]

        # -------- 7) 可选自环残差（严格论文可关闭；默认 stack_config.self_loop=False）--------
        if self.self_loop:  # 若启用自环
            if self.fc_self is not None:  # 若自环需要线性变换
                h = h + self.fc_self(x_in)  # 加上变换后的输入
            else:
                h = h + x_in  # 直接残差相加

        # -------- 8) 论文 Eq.(4)：P_i = sum_j alpha_ij * d_ij，并写入 graph.p_scalar --------
        # 这里 alpha 是 [E,H]，d_ij 是 [E,1]，先按 head 计算再对 head 求均值 -> [E,1]
        p_edge = (alpha * d_ij).mean(dim=1, keepdim=True)  # 每条边对 P 的贡献: [E, 1]
        p_node = scatter_add(p_edge, dst, dim=0, dim_size=N)  # 聚合到节点 i: [N, 1]
        graph.__setattr__("p_scalar", p_node)  # 将 P_i 保存到图对象（后续可用）

        # -------- 9) 写回节点特征 --------
        graph.__setattr__("x", h)  # 原地更新 graph.x
