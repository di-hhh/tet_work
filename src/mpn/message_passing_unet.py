"""
多尺度 U-Net 式消息传递网络 (Message Passing U-Net)

============ 背景知识 ============

【什么是 U-Net？】
U-Net 最初是图像分割领域的经典架构，核心思想是"先缩小再放大"：
  - 编码器（Encoder）：逐层降低分辨率，提取越来越抽象的高级特征
  - 解码器（Decoder）：逐层恢复分辨率，把高级特征映射回原始尺寸
  - Skip Connection：编码器每层的特征直接"跳接"到解码器对应层，
                      防止细节信息在压缩过程中丢失

【为什么要用在图上？】
原来的 MessagePassingStack 在一张固定的图上反复跑消息传递，
信息传播的速度受限于图的直径（两个最远节点之间的最短路径长度）。
比如一个 1000 节点的网格，信息从一端传到另一端可能需要 30+ 层。

U-Net 的做法：把图"粗化"成更小的图（节点数减半），
在小图上信息传播更快（等效于在原图上一步跨越多个节点），
然后再"还原"回原始大小。这样用更少的层数就能获得全局感受野。

【本文件的结构示意】

    G_0 (原始图, 比如1000个节点)
     │ ──encoder──> 保存特征 h_0
     │ pool（粗化，节点数减半）
    G_1 (约500个节点)
     │ ──encoder──> 保存特征 h_1
     │ pool
    G_2 (约250个节点)
     │ ──encoder──> 保存特征 h_2
     │ pool
    G_3 (约125个节点) ──encoder──> h_3 (最粗层/瓶颈层)
     │
     │ 开始解码，从粗到细：
     │
    G_2: unpool(h_3) + skip(h_2) ──decoder──> 更新后的 G_2 特征
     │
    G_1: unpool(G_2特征) + skip(h_1) ──decoder──> 更新后的 G_1 特征
     │
    G_0: unpool(G_1特征) + skip(h_0) ──decoder──> 最终输出

============ 文件组织 ============

1. 工具函数（文件顶部）：
   - _scatter_mean:  按分组求平均（替代 torch_scatter 库）
   - _graclus:       图粗化算法（替代 torch_cluster 库）
   - _hash_edges:    把边编码成整数，用于快速查找
   - _lookup_edge_indices: 在边集合中快速查找特定边

2. MessagePassingUNet 类：
   - __init__:                构建编码器/解码器的网络层
   - _build_graph_hierarchy:  构建多层粗化图
   - _manual_pool:            实现图的粗化操作
   - forward:                 前向传播（编码→解码→输出）
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from src.mpn.message_passing_block import MessagePassingBlock


# AMBER 中一些特殊属性，在图粗化时不能被平均（比如布尔标签）
_AMBER_PROTECTED_ATTRS = ("y", "mask_output", "current_sizing_field")


# =============================================================================
# 工具函数 1: _scatter_mean
# =============================================================================
#
# 功能：按分组对张量求平均。
#
# 举个例子理解：
#   假设有 5 个学生的成绩：src = [90, 80, 70, 85, 95]
#   他们分属 3 个小组：     index = [0,  1,  0,  2,  1]
#   （学生0属于组0，学生1属于组1，学生2属于组0，...）
#
#   _scatter_mean(src, index, dim_size=3) 的结果：
#     组0 = (90 + 70) / 2 = 80
#     组1 = (80 + 95) / 2 = 87.5
#     组2 = 85 / 1         = 85
#     返回 [80, 87.5, 85]
#
# 在本文件中的用途：
#   - 图粗化时，多个细节点合并成一个粗节点，对它们的特征求平均
#   - 多条细边映射到同一条粗边时，对边特征求平均
#
def _scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: Optional[int] = None) -> torch.Tensor:
    """
    按 index 分组，对 src 求每组的平均值。纯 PyTorch 实现。

    Args:
        src:      源数据张量。
                  - 1D 情况: 形状 [N]，比如每个节点的某个标量属性
                  - 2D 情况: 形状 [N, D]，比如每个节点的 D 维特征向量
        index:    分组索引，形状 [N]。index[i] = g 表示第 i 个元素属于第 g 组。
        dim:      沿哪个维度做聚合（通常是 0，即按行/节点聚合）。
        dim_size: 输出的组数。如果不指定，自动取 index 的最大值 + 1。

    Returns:
        形状 [dim_size, ...] 的张量，每行是对应组的平均特征。
    """
    if dim_size is None:
        dim_size = index.max().item() + 1

    # ---- 1D 张量的情况（比如 batch 向量） ----
    if src.dim() == 1:
        # 创建全零的输出（每个组一个位置）
        out = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
        # 创建计数器，记录每个组有多少个元素
        count = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
        # scatter_add_: 把 src[i] 累加到 out[index[i]]
        out.scatter_add_(0, index, src)
        # 同时统计每个组的元素个数
        count.scatter_add_(0, index, torch.ones_like(src))
        # 求平均（clamp(min=1) 防止除以 0）
        return out / count.clamp(min=1)

    # ---- 多维张量的情况（比如 [N, 64] 的节点特征） ----
    else:
        out = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)
        count = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
        # index 是 [N]，src 是 [N, D]，需要把 index 扩展成 [N, D] 才能用 scatter_add_
        idx_expanded = index.unsqueeze(-1).expand_as(src)  # [N] -> [N, 1] -> [N, D]
        out.scatter_add_(dim, idx_expanded, src)
        count.scatter_add_(0, index, torch.ones(index.size(0), dtype=src.dtype, device=src.device))
        count = count.clamp(min=1)
        # 把 count 从 [dim_size] 扩展到 [dim_size, 1, 1, ...] 以便广播除法
        for _ in range(src.dim() - 1):
            count = count.unsqueeze(-1)
        return out / count


# =============================================================================
# 工具函数 2: _graclus (图粗化算法)
# =============================================================================
#
# 功能：把一张图的节点数大约减半。
#
# 原理（用社交网络举例）：
#   假设有 6 个人（节点）互相认识（边）。
#   Graclus 算法会把他们两两配对：
#
#   第1轮：
#     节点 A 提议和 B 配对，节点 B 也提议和 A 配对 → 配对成功！合并为超级节点 {A,B}
#     节点 C 提议和 D 配对，节点 D 提议和 E 配对 → C 和 D 不互相提议，配对失败
#     节点 E 提议和 F 配对，节点 F 也提议和 E 配对 → 配对成功！合并为 {E,F}
#
#   第2轮（只对未配对的 C、D 继续）：
#     节点 C 提议和 D 配对，节点 D 也提议和 C 配对 → 配对成功！合并为 {C,D}
#
#   最终：6 个节点 → 3 个超级节点 {A,B}, {C,D}, {E,F}，节点数减半。
#
# 返回值 cluster：
#   cluster[i] = j 表示原始节点 i 被分到了第 j 个超级节点。
#   比如 cluster = [0, 0, 1, 1, 2, 2] 表示：
#     节点 0,1 → 超级节点 0
#     节点 2,3 → 超级节点 1
#     节点 4,5 → 超级节点 2
#
def _graclus(edge_index: torch.Tensor, num_nodes: int, max_rounds: int = 5) -> torch.Tensor:
    """
    GPU 友好的多轮并行 Graclus 聚类算法。

    Args:
        edge_index: 图的边列表，形状 [2, E]。
                    edge_index[0] 是源节点，edge_index[1] 是目标节点。
        num_nodes:  图中的节点总数。
        max_rounds: 最多跑几轮匹配（轮数越多，配对率越高，但耗时也越多）。

    Returns:
        cluster: 形状 [num_nodes] 的张量，cluster[i] 是节点 i 所属的粗节点编号。
                 编号从 0 连续递增到 (粗节点数 - 1)。
    """
    device = edge_index.device
    src, dst = edge_index[0], edge_index[1]

    # 去掉自环边（自己连自己的边没有意义）
    no_self = src != dst
    src, dst = src[no_self], dst[no_self]

    # 初始化：每个节点自成一个 cluster，谁都没配对
    node_ids = torch.arange(num_nodes, device=device)  # [0, 1, 2, ..., N-1]
    cluster = node_ids.clone()  # 初始时 cluster[i] = i，每个节点是自己的 cluster
    matched = torch.zeros(num_nodes, dtype=torch.bool, device=device)  # 标记谁已经配对了

    for _ in range(max_rounds):
        # 所有节点都配对了，提前结束
        if matched.all():
            break

        # 只看那些两端都还没配对的边
        src_ok = ~matched[src]   # 源节点未配对？
        dst_ok = ~matched[dst]   # 目标节点未配对？
        valid = src_ok & dst_ok  # 两端都未配对的边
        if not valid.any():
            break  # 没有可用的边了（剩余未配对节点互相没连边）
        v_src, v_dst = src[valid], dst[valid]

        # 随机打乱边的顺序，这样每个节点随机选一个邻居来"提议"
        perm = torch.randperm(v_src.size(0), device=device)
        v_src, v_dst = v_src[perm], v_dst[perm]

        # proposal[i] = j 表示节点 i "提议"和节点 j 配对
        # 默认提议自己（即不提议任何人），只有有边的节点才会提议邻居
        proposal = node_ids.clone()
        # 对于打乱后的边 (v_src[k], v_dst[k])，节点 v_src[k] 提议和 v_dst[k] 配对
        # 如果一个节点有多条边，只保留最后一条（因为是覆盖写入）
        proposal[v_src] = v_dst

        # 互相提议才算配对成功：
        #   节点 i 提议了 j (proposal[i] == j)
        #   且节点 j 也提议了 i (proposal[j] == i)
        #   即 proposal[proposal[i]] == i
        mutual = (proposal[proposal] == node_ids) & (~matched)
        # 为了避免重复计数（i 和 j 互相提议，只算一次），只保留编号较小的那个
        is_match = mutual & (node_ids < proposal)

        matched_nodes = is_match.nonzero(as_tuple=True)[0]  # 配对成功的较小编号节点
        if matched_nodes.numel() == 0:
            break  # 这一轮没有新的配对，结束

        # 执行合并：把 partner 节点的 cluster 指向 matched_node
        # 比如节点 3 和节点 7 配对，cluster[7] = 3
        partners = proposal[matched_nodes]
        cluster[partners] = matched_nodes
        matched[matched_nodes] = True
        matched[partners] = True

    # 最后把 cluster ID 压缩成连续的 [0, 1, 2, ...]
    # 比如 cluster = [0, 0, 2, 2, 5, 5] → [0, 0, 1, 1, 2, 2]
    _, cluster = torch.unique(cluster, return_inverse=True)
    return cluster


# =============================================================================
# 工具函数 3: _hash_edges (边编码)
# =============================================================================
#
# 把 (源节点, 目标节点) 对编码成一个唯一的整数，方便后续快速比较和查找。
#
# 原理：类似于把二维坐标 (x, y) 编码成一维索引 x * width + y。
# 比如 num_nodes=100，边 (3, 7) → 3 * 100 + 7 = 307。
# 只要 src 和 dst 都 < num_nodes，每条边的 hash 就是唯一的。
#
def _hash_edges(src: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """把 (src, dst) 节点对编码为唯一整数。"""
    return src * num_nodes + dst


# =============================================================================
# 工具函数 4: _lookup_edge_indices (边查找)
# =============================================================================
#
# 问题场景：
#   我手里有一批边的 hash 值 (query_hash)，想知道它们各自对应
#   另一个边集合 (ref_hash) 中的第几条边。
#
# 类比：
#   你有一份名单（ref_hash = [307, 512, 809, ...]），
#   你想查 "307 在名单中排第几？" → 答案是第 0 个。
#   "512 排第几？" → 第 1 个。
#
# 实现方法：先排序，然后用二分查找（searchsorted），复杂度 O(N log N)。
# 比 Python dict 快得多，而且全程在 GPU 上运行。
#
# 安全处理：如果查找的值不在名单中（理论上不应该发生，但防御性编程），
#           返回 found=False，调用方可以跳过这些边。
#
def _lookup_edge_indices(query_hash: torch.Tensor, ref_hash: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    在 ref_hash 中查找 query_hash 中每个值的位置。

    Args:
        query_hash: 要查找的 hash 值，形状 [Q]。
        ref_hash:   参考边集的 hash 值，形状 [R]。

    Returns:
        indices: 形状 [Q]，indices[i] 是 query_hash[i] 在 ref_hash 中的位置。
                 如果没找到，indices[i] = 0（安全的占位值，但会被 found mask 过滤掉）。
        found:   形状 [Q] 的布尔张量，found[i] = True 表示 query_hash[i] 确实在 ref_hash 中找到了。
    """
    # 第1步：对 ref_hash 排序（searchsorted 要求输入有序）
    sorted_ref, sort_perm = ref_hash.sort()
    # sort_perm 记录了排序后的位置 → 原始位置的映射
    # 比如 ref_hash = [809, 307, 512]，排序后 sorted_ref = [307, 512, 809]
    # sort_perm = [1, 2, 0]，意思是排序后第0个来自原始第1个

    # 第2步：二分查找每个 query 在 sorted_ref 中的插入位置
    insert_pos = torch.searchsorted(sorted_ref, query_hash)
    # clamp 防止越界（如果 query 比所有 ref 都大，insert_pos 会等于 len）
    insert_pos_clamped = insert_pos.clamp(max=sorted_ref.size(0) - 1)

    # 第3步：验证是否真的找到了（searchsorted 只给插入位置，不保证匹配）
    found = sorted_ref[insert_pos_clamped] == query_hash

    # 第4步：把排序后的位置转回原始位置
    indices = sort_perm[insert_pos_clamped]
    indices[~found] = 0  # 没找到的设为 0（不会被使用，因为 found=False）
    return indices, found


# =============================================================================
# 主类: MessagePassingUNet
# =============================================================================

class MessagePassingUNet(nn.Module):
    """
    多尺度 U-Net 式消息传递网络。

    【用法】
    直接替换原来的 MessagePassingStack，输入输出接口完全一致：
      - 输入：PyG 的 Batch/Data 对象（包含 graph.x 节点特征和 graph.edge_attr 边特征）
      - 输出：原地修改 graph.x 和 graph.edge_attr

    【相比 MessagePassingStack 的优势】
    1. 多尺度感受野：通过图粗化，底层节点的信息可以快速传播到远处
    2. Skip Connection：细层的局部细节不会在粗化过程中丢失
    3. 参数效率：在粗图上做消息传递，节点/边更少，计算更快
    """

    def __init__(
        self,
        latent_dimension: int,
        stack_config: Optional[Dict[str, Any]] = None,
        unet_config: Optional[Dict[str, Any]] = None,
    ):
        """
        构造 U-Net 消息传递网络。

        Args:
            latent_dimension: 隐空间维度（所有节点特征和边特征的维度）。
                              比如 latent_dimension=64 表示每个节点/边用 64 维向量表示。

            stack_config: 兼容 MessagePassingStack 的参数名。
                         如果同时提供了 unet_config，则 stack_config 被忽略。

            unet_config: U-Net 专用配置字典，包含：
                num_levels (int):      图的层级数，默认 4。
                    即 G_0(原始) → G_1(粗) → G_2(更粗) → G_3(最粗)。
                    层级越多，最粗层节点越少，全局感受野越大，但计算开销也越大。

                steps_per_level (int): 每层做几轮消息传递，默认 3。
                    每一轮消息传递 = 一次 MessagePassingBlock。
                    轮数越多，每层的局部信息提取越充分。

                stack_config (dict):   传给每个 MessagePassingBlock 的配置
                    （包含 residual_connections, layer_norm, mlp 等子配置）
        """
        super().__init__()

        # 统一处理两种参数名，保持接口兼容
        if unet_config is not None:
            config = unet_config
        elif stack_config is not None:
            config = stack_config
        else:
            config = {}

        self._latent_dimension = latent_dimension
        self._num_levels = config.get("num_levels", 4)        # 图层级数
        self._steps_per_level = config.get("steps_per_level", 3)  # 每层消息传递轮数

        # 传给每个 MessagePassingBlock 的配置
        block_config = config.get("stack_config", {})

        # ======================== 编码器 ========================
        # 每一层（从细到粗）都有若干个 MessagePassingBlock。
        # 结构：encoder_blocks[level][step]
        # 比如 num_levels=4, steps_per_level=3，共有 4 层 × 3 个 block = 12 个编码器 block。
        self._encoder_blocks = nn.ModuleList()
        for _ in range(self._num_levels):
            level_blocks = nn.ModuleList([
                MessagePassingBlock(
                    stack_config=block_config,
                    latent_dimension=latent_dimension
                )
                for _ in range(self._steps_per_level)
            ])
            self._encoder_blocks.append(level_blocks)

        # ======================== 解码器 ========================
        # 除了最粗层（bottleneck），每层都有解码器 block。
        # 所以解码器层数 = num_levels - 1。
        self._decoder_blocks = nn.ModuleList()
        for _ in range(self._num_levels - 1):
            level_blocks = nn.ModuleList([
                MessagePassingBlock(
                    stack_config=block_config,
                    latent_dimension=latent_dimension
                )
                for _ in range(self._steps_per_level)
            ])
            self._decoder_blocks.append(level_blocks)

        # ======================== Skip Connection 投影层 ========================
        #
        # 解码器每一层需要融合两种信息：
        #   (a) 编码器保存的细层特征（保留了局部细节）
        #   (b) 从粗层 unpool 回来的特征（包含全局/远程信息）
        #
        # 融合方式：把 (a) 和 (b) 拼接起来 [a; b]，维度变成 2×latent_dimension，
        # 然后用一个线性层投影回 latent_dimension。
        #
        # 节点特征的投影层：
        self._skip_node_projections = nn.ModuleList([
            nn.Linear(2 * latent_dimension, latent_dimension)
            for _ in range(self._num_levels - 1)
        ])

        # 边特征的投影层（和节点一样的道理）：
        self._skip_edge_projections = nn.ModuleList([
            nn.Linear(2 * latent_dimension, latent_dimension)
            for _ in range(self._num_levels - 1)
        ])

    @property
    def latent_dimension(self) -> int:
        return self._latent_dimension

    # =========================================================================
    # _build_graph_hierarchy: 构建多层粗化图
    # =========================================================================
    #
    # 把原始图逐层粗化，得到一系列从细到粗的图：
    #   G_0 (原始, ~1000节点) → G_1 (~500) → G_2 (~250) → G_3 (~125)
    #
    # 同时记录每层的 cluster 映射关系（谁和谁合并了），
    # 解码器需要用这个映射来做 unpool（把粗层特征还原到细层）。
    #
    def _build_graph_hierarchy(
        self, graph: Union[Data, Batch]
    ) -> Tuple[List[Union[Data, Batch]], List[torch.Tensor]]:
        """
        构建多层粗化图层次结构。

        Args:
            graph: 输入的原始图（最细层）。

        Returns:
            graphs:   [G_0, G_1, ..., G_L] 从细到粗的图列表。
                      G_0 是原始图，G_L 是最粗的图。
            clusters: [c_0, c_1, ..., c_{L-1}] 每层的节点映射关系。
                      c_l[i] = j 表示第 l 层的节点 i 在第 l+1 层变成了节点 j。
        """
        graphs = [graph]    # 第 0 层就是原始图
        clusters = []

        current = graph
        for _ in range(self._num_levels - 1):
            # 用 Graclus 算法把当前图的节点两两配对
            cluster = _graclus(current.edge_index, num_nodes=current.x.size(0))

            # 根据配对结果，构建更粗的图
            coarse = self._manual_pool(current, cluster)

            graphs.append(coarse)
            clusters.append(cluster)
            current = coarse  # 下一轮在更粗的图上继续粗化

        return graphs, clusters

    # =========================================================================
    # _manual_pool: 图粗化的具体实现
    # =========================================================================
    #
    # 给定一张图和 cluster 映射，生成一张更粗的图。
    #
    # 需要处理三件事：
    #   1. 节点特征：同一个 cluster 内的多个节点特征取平均
    #   2. 边的拓扑：原来连接不同 cluster 的边保留，同一 cluster 内的边删除（自环）
    #   3. 边特征：映射到同一条粗边的多条细边，特征取平均
    #
    # 举例：
    #   细图有 6 个节点 [A, B, C, D, E, F]，cluster = [0, 0, 1, 1, 2, 2]
    #   即 {A,B}→粗节点0, {C,D}→粗节点1, {E,F}→粗节点2
    #
    #   细边 A→C 的两端分别属于粗节点 0 和 1 → 保留为粗边 0→1
    #   细边 A→B 的两端都属于粗节点 0      → 变成自环，删除
    #   细边 A→C 和 B→D 都映射到粗边 0→1  → 特征取平均
    #
    @staticmethod
    def _manual_pool(graph: Union[Data, Batch], cluster: torch.Tensor) -> Union[Data, Batch]:
        """
        根据 cluster 映射，将细图粗化为粗图。

        Args:
            graph:   细层图对象，包含 x (节点特征), edge_index (边), edge_attr (边特征)。
            cluster: 形状 [N_fine]，cluster[i] 是细节点 i 对应的粗节点编号。

        Returns:
            粗化后的图对象。
        """
        num_coarse_nodes = cluster.max().item() + 1

        # ---- 步骤 1：节点特征粗化 ----
        # 同一个 cluster 内的所有细节点特征取平均，作为粗节点的特征。
        # 比如细节点 0 的特征是 [1, 2, 3]，细节点 1 的特征是 [3, 4, 5]，
        # 它们同属 cluster 0，则粗节点 0 的特征 = [2, 3, 4]（逐元素平均）。
        coarse_x = _scatter_mean(graph.x, cluster, dim=0, dim_size=num_coarse_nodes)

        # ---- 步骤 2：边的重映射 ----
        edge_index = graph.edge_index

        # 把边的两端从细节点编号映射到粗节点编号
        # 比如细边 (3, 5)，cluster[3]=1, cluster[5]=2 → 粗边 (1, 2)
        coarse_src = cluster[edge_index[0]]
        coarse_dst = cluster[edge_index[1]]

        # 去掉自环：如果边的两端被合并到了同一个 cluster，这条边就没意义了
        not_self_loop = coarse_src != coarse_dst
        coarse_src = coarse_src[not_self_loop]
        coarse_dst = coarse_dst[not_self_loop]

        # 同步过滤边特征
        if graph.edge_attr is not None:
            filtered_edge_attr = graph.edge_attr[not_self_loop]
        else:
            filtered_edge_attr = None

        # ---- 步骤 3：边的去重 ----
        # 多条细边可能映射到同一条粗边（比如细边 A→C 和 B→D 都映射到 粗0→粗1）。
        # 用 hash 编码来识别和去重。
        edge_hash = coarse_src * num_coarse_nodes + coarse_dst
        unique_hash, inverse_indices = torch.unique(edge_hash, return_inverse=True)
        # inverse_indices[i] 表示第 i 条过滤后的边对应 unique_hash 中的第几条

        # 从 unique hash 还原粗边的 (src, dst)
        new_src = unique_hash // num_coarse_nodes
        new_dst = unique_hash % num_coarse_nodes
        coarse_edge_index = torch.stack([new_src, new_dst], dim=0)

        # ---- 步骤 4：边特征去重（平均） ----
        # 映射到同一条粗边的多条细边，特征取平均
        if filtered_edge_attr is not None:
            coarse_edge_attr = _scatter_mean(
                filtered_edge_attr, inverse_indices, dim=0,
                dim_size=unique_hash.size(0)
            )
        else:
            coarse_edge_attr = None

        # ---- 步骤 5：构建粗图对象 ----
        coarse = Data(x=coarse_x, edge_index=coarse_edge_index, edge_attr=coarse_edge_attr)

        # 传播 batch 信息（如果是多图的 Batch）
        # batch 向量标记每个节点属于 batch 中的第几张图
        if hasattr(graph, "batch") and graph.batch is not None:
            coarse_batch = _scatter_mean(
                graph.batch.float(), cluster, dim=0, dim_size=num_coarse_nodes
            ).long()
            coarse.batch = coarse_batch

        return coarse

    # =========================================================================
    # forward: 前向传播（核心！）
    # =========================================================================
    #
    # 整体流程：
    #
    #   1. 构建图层次：G_0 → G_1 → G_2 → G_3（粗化）
    #
    #   2. 编码器（从细到粗）：
    #      在每一层跑消息传递，提取特征，然后 pool 到下一层
    #      同时保存每层的特征（给 skip connection 用）
    #
    #   3. 解码器（从粗到细）：
    #      从最粗层开始，把特征 unpool 回细层，
    #      和编码器保存的特征拼接（skip connection），
    #      再跑消息传递进一步融合
    #
    #   4. 最终把最细层的特征写回原始图
    #
    def forward(self, graph: Batch) -> None:
        """
        前向传播，原地修改 graph 的节点特征 (graph.x) 和边特征 (graph.edge_attr)。

        Args:
            graph: 输入图（PyG 的 Batch 对象，可能包含多张图）。
        """

        # ==================== 第1步：构建多尺度图层次 ====================
        # graphs  = [G_0(原始), G_1(粗), G_2(更粗), G_3(最粗)]
        # clusters = [c_0, c_1, c_2]  每层的节点映射
        graphs, clusters = self._build_graph_hierarchy(graph)

        # ==================== 第2步：编码器（从细到粗） ====================
        # 目标：在每一层图上跑消息传递，让节点"了解"周围的信息，
        #       然后把信息 pool 到更粗的层级。
        #
        # 同时保存每层编码后的特征，给解码器做 skip connection 用。

        encoder_node_features = []  # 保存每层编码器输出的节点特征
        encoder_edge_features = []  # 保存每层编码器输出的边特征

        for level in range(self._num_levels):
            # ---- 在当前层跑若干轮消息传递 ----
            # 每轮消息传递：每个节点收集邻居信息 → 更新自己的特征
            # 多轮之后，每个节点就"知道"了更远范围内的信息
            for block in self._encoder_blocks[level]:
                block(graphs[level])  # 原地修改 graphs[level].x 和 .edge_attr

            # ---- 保存这一层的特征（用于 skip connection）----
            # 必须 clone，因为后续 pool 操作会覆盖这些值
            encoder_node_features.append(graphs[level].x.clone())
            encoder_edge_features.append(graphs[level].edge_attr.clone())

            # ---- Pool: 把特征传到下一层（更粗的图）----
            # 最粗层没有下一层，不需要 pool
            if level < self._num_levels - 1:
                cluster = clusters[level]
                coarse_graph = graphs[level + 1]

                # 节点特征 pool：同一 cluster 内的细节点特征取平均 → 粗节点特征
                coarse_graph.x = _scatter_mean(
                    graphs[level].x, cluster, dim=0,
                    dim_size=coarse_graph.x.size(0)
                )

                # 边特征 pool：细边映射到粗边，同一粗边的多条细边特征取平均
                # （这里的逻辑和 _manual_pool 类似，但用的是编码后的最新特征）
                fine_edge_index = graphs[level].edge_index
                fine_edge_attr = graphs[level].edge_attr
                coarse_src = cluster[fine_edge_index[0]]
                coarse_dst = cluster[fine_edge_index[1]]

                # 只保留跨 cluster 的边（非自环）
                not_self_loop = coarse_src != coarse_dst
                if not_self_loop.any():
                    c_src = coarse_src[not_self_loop]
                    c_dst = coarse_dst[not_self_loop]
                    c_attr = fine_edge_attr[not_self_loop]
                    num_coarse_nodes = coarse_graph.x.size(0)
                    edge_hash = _hash_edges(c_src, c_dst, num_coarse_nodes)

                    # 找到每条细边对应粗图中的哪条边
                    coarse_ei = coarse_graph.edge_index
                    coarse_hash = _hash_edges(coarse_ei[0], coarse_ei[1], num_coarse_nodes)
                    coarse_edge_idx, found = _lookup_edge_indices(edge_hash, coarse_hash)
                    # 只用确实找到对应关系的边来更新粗层边特征
                    if found.any():
                        coarse_graph.edge_attr = _scatter_mean(
                            c_attr[found], coarse_edge_idx[found], dim=0,
                            dim_size=coarse_graph.edge_attr.size(0)
                        )

        # ==================== 第3步：解码器（从粗到细） ====================
        # 从倒数第二层开始，逐层往细的方向还原。
        #
        # 每一层的操作：
        #   (a) Unpool: 把粗层特征广播回细层（粗节点的特征复制给它包含的所有细节点）
        #   (b) Skip Connection: 把 unpool 特征和编码器保存的特征拼接，投影回原维度
        #   (c) 跑解码器消息传递，进一步融合信息

        for level in range(self._num_levels - 2, -1, -1):  # 从 L-2 到 0
            # ---- (a) Unpool 节点特征 ----
            # 把粗层节点的特征"广播"回细层：
            # 如果 cluster[i] = j，则细节点 i 拿到粗节点 j 的特征
            coarse_x = graphs[level + 1].x    # 粗层当前的节点特征
            cluster = clusters[level]          # 细→粗的映射
            unpooled_x = coarse_x[cluster]     # 广播：[N_fine, D]，每个细节点取对应粗节点特征

            # ---- (b) Skip Connection 节点特征 ----
            # 拼接：[编码器特征(局部细节) | unpool特征(全局信息)]，维度变成 2D
            # 然后用线性层投影回 D
            skip_x = torch.cat([encoder_node_features[level], unpooled_x], dim=-1)
            graphs[level].x = self._skip_node_projections[level](skip_x)

            # ---- (b) Skip Connection 边特征 ----
            # 边特征的 unpool 比节点复杂，因为：
            #   - 细边和粗边不是简单的一对一关系
            #   - 两端在同一 cluster 内的细边，在粗图中不存在（已被删除为自环）
            #
            # 策略：
            #   - 对于跨 cluster 的细边：从粗图中找到对应的粗边，取其特征
            #   - 对于同 cluster 内的细边（自环边）：用编码器保存的原始特征填充
            #     （比用零向量好，至少包含有意义的局部信息）
            coarse_edge_attr = graphs[level + 1].edge_attr
            fine_edge_index = graphs[level].edge_index
            encoder_edge = encoder_edge_features[level]

            # 计算细边两端对应的粗节点编号
            coarse_src = cluster[fine_edge_index[0]]
            coarse_dst = cluster[fine_edge_index[1]]
            num_coarse_nodes = graphs[level + 1].x.size(0)

            # 默认用编码器边特征填充（对于自环边和查找失败的边都用这个）
            unpooled_edge = encoder_edge.clone()
            not_self_loop = coarse_src != coarse_dst  # 跨 cluster 的边

            if not_self_loop.any() and coarse_edge_attr is not None:
                c_src = coarse_src[not_self_loop]
                c_dst = coarse_dst[not_self_loop]
                edge_hash = _hash_edges(c_src, c_dst, num_coarse_nodes)

                # 在粗图的边集中查找对应边
                coarse_ei = graphs[level + 1].edge_index
                coarse_hash = _hash_edges(coarse_ei[0], coarse_ei[1], num_coarse_nodes)
                coarse_edge_indices, found = _lookup_edge_indices(edge_hash, coarse_hash)

                # 只对确实找到对应粗边的细边，写入粗层边特征
                nsl_indices = not_self_loop.nonzero(as_tuple=True)[0]  # 跨cluster细边的原始位置
                unpooled_edge[nsl_indices[found]] = coarse_edge_attr[coarse_edge_indices[found]]

            # 拼接 + 投影，和节点一样
            skip_edge = torch.cat([encoder_edge, unpooled_edge], dim=-1)
            graphs[level].edge_attr = self._skip_edge_projections[level](skip_edge)

            # ---- (c) 解码器消息传递 ----
            for block in self._decoder_blocks[level]:
                block(graphs[level])

        # ==================== 第4步：写回结果 ====================
        # graphs[0] 本质上就是原始 graph 的引用，但消息传递过程中
        # .x 和 .edge_attr 可能被替换成新的张量对象，
        # 所以这里显式写回，确保调用方拿到最终结果。
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
