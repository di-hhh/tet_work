# Edge-Aware GAT 模型架构说明

> 基于当前代码（`d=64, H=4, Dh=16, num_steps=20`）的完整架构文档。

---

## 整体流水线

```
原始图 (节点特征 [N,3], 边特征 [E,2])
        ↓
  Input Embedding（线性投影）
        ↓
  × 20 个 GAT Block
  ┌──────────────────────┐
  │  EdgeModule          │
  │  NodeModule (GAT)    │
  │  残差 + LayerNorm    │
  └──────────────────────┘
        ↓
  Decoder MLP + Linear(d→1)
        ↓
  每个节点输出一个 sizing field 标量
```

---

## 第一阶段：Input Embedding

原始节点特征 3 维（坐标 x,y + FEM 值），边特征 2 维（边长、角度），通过线性层投影到 d=64：

```
节点: [N, 3]  →  Linear(3, 64)  →  [N, 64]
边:   [E, 2]  →  Linear(2, 64)  →  [E, 64]
```

此后所有特征在 64 维空间中流动。

---

## 第二阶段：GAT Block（× 20）

每个 Block 由两个子模块顺序执行：**先更新边，再更新节点**。

### 子模块一：Edge Module（边更新）

```
输入: x_src [E,64], x_dst [E,64], e [E,64]
  ↓ concat
[E, 192]  →  MLP(2层, LeakyReLU)  →  [E, 64]
  ↓
e_new [E, 64]
```

每条边将两端节点特征与自身特征拼接，过 2 层 MLP，得到新边特征。
使边特征实时感知所连节点的当前状态。

---

### 子模块二：Node Module（节点更新）

#### 步骤 1：多头特征投影

```
x [N, 64]  →  Dropout(0.1)  →  fc: Linear(64→64)  →  reshape  →  z [N, 4, 16]
```

将 64 维特征切分为 4 个头，每头 16 维（d_head = d / num_heads = 64 / 4 = 16）。

---

#### 步骤 2：边感知注意力打分（每头独立）

对每条边 (j→i)，在每个头 h 上独立计算注意力分数：

```
e [E, 64]  →  fc_edge_for_att_calc: Linear(64→64)  →  reshape  →  e_att [E, 4, 16]

att_in = cat([z_src, z_dst, e_att], dim=-1)       # [E, 4, 48]

logit_{h,ij} = attn_vec[h] · att_in[:, h, :]     # [E, 4]
logit        = LeakyReLU(logit, slope=0.2)
```

`attn_vec` 是形状 `[4, 48]` 的可学习参数矩阵，每行对应一个头的独立打分向量。
注意力依赖边的几何特征，使网格中边长、角度不同的邻居获得差异化权重。

---

#### 步骤 3：Softmax 归一化

```
对每个目标节点 i 的所有入边，在每个头上独立做 softmax：
alpha [E, 4]  ←  softmax(logit, index=dst)
alpha          =  AttnDropout(alpha, p=0.1)
```

---

#### 步骤 4：双路加权聚合

```
路径 A（节点聚合）:
  msg_node = z_src * alpha              # [E, 4, 16]，注意力加权邻居特征
  h_node   = scatter_add(msg_node, dst) # [N, 4, 16]

路径 B（边聚合）:
  ez       = fc_edge(e).reshape(-1, 4, 16)  # [E, 4, 16]，边特征变换
  msg_edge = ez * alpha                      # 与节点聚合共享注意力权重
  h_edge   = scatter_add(msg_edge, dst)      # [N, 4, 16]

h = h_node + h_edge                          # [N, 4, 16]
```

同时聚合邻居节点和邻居边，且两路共用同一套注意力权重，统一衡量"这条连接重不重要"。

---

#### 步骤 5：多头合并 → out_proj → 残差 → FFN

```
h [N, 4, 16]  →  reshape  →  [N, 64]

→  out_proj: Linear(64→64) + LayerNorm  →  h [N, 64]

→  + x_in [N, 64]          （self_loop 残差，使用 Dropout 前的原始 x）

→  + FFN(LayerNorm(h))
      FFN: Linear(64→128) → SiLU → Linear(128→64)
      （Pre-LN 残差风格）

→  graph.x = h [N, 64]
```

`out_proj` 负责混合各头输出。self_loop 残差保留原始节点信息。FFN 补充深度非线性变换能力。

---

### Block 层面的残差和 LayerNorm（inner 模式）

Block 结束后，对边和节点各自施加残差与归一化：

```
edge_attr = LayerNorm(edge_attr_new + edge_attr_old)
x         = LayerNorm(x_new + x_old)
```

节点特征共经历两次残差：Node Module 内部的 self_loop 残差，以及 Block 外部的 inner 残差。

---

## 第三阶段：Decoder

20 个 Block 后，每个节点的 64 维特征已融合 20 跳范围内所有邻居的几何与 FEM 信息：

```
node_features [N, 64]
  →  mask_output（过滤初始网格节点，仅保留当前物理网格节点）
  →  Decoder MLP（2层, LeakyReLU）     [N, 64]
  →  Linear(64 → 1)
  →  outputs [N, 1]                    每个节点的 sizing field 预测值
```

输出经 `softplus` 变换保证为正值，再送入 GMSH 接口生成自适应网格。

---

## 参数量估算

| 模块 | 参数量（估算） |
|------|--------------|
| Input Embedding（节点 + 边） | ≈ 320 |
| 每个 Block — EdgeModule MLP | ≈ 24,832 |
| 每个 Block — NodeModule | ≈ 45,000 |
| × 20 个 Block | ≈ 1,397,000 |
| Decoder MLP + readout | ≈ 8,256 |
| **总计** | **≈ 140 万参数** |

---

## 关键配置速查

| 参数 | 当前值 | 含义 |
|------|--------|------|
| `latent_dimension` | 64 | 特征向量维度 |
| `num_steps` | 20 | Block 数量（信息传播跳数） |
| `num_heads` | 4 | 注意力头数 |
| `edge_dependent_attention` | True | 注意力打分依赖边特征 |
| `aggregate_edge` | True | 同时聚合邻居边特征 |
| `apply_attention_on_edge` | True | 边聚合使用注意力加权 |
| `self_loop` | True | self_loop 残差 |
| `layer_norm` | inner | 每个子模块后做 LayerNorm |
| `residual_connections` | inner | 每个子模块后加残差 |
| `edge_dropout` | 0.1 | 训练时随机删边比例 |
| `feat_drop` | 0.1 | 节点特征 Dropout |
| `attn_drop` | 0.1 | 注意力权重 Dropout |
| `edge_feat_drop` | 0.1 | 边特征 Dropout |

---

## 架构设计思路

```
传统 MPN（AMBER 原版）
  均匀聚合所有邻居，无差别对待
          ↓ 改进
边感知 GAT
  根据边的几何特征（长度、角度）动态决定
  "哪个邻居更重要"，同时聚合节点和边的信息
          ↓ 增强
FFN（Transformer 风格）
  注意力后加深度非线性，弥补 GAT
  聚合本身线性化的不足
```
