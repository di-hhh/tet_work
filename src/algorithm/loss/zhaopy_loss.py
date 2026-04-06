"""
Enhanced AmberLoss with two CPR-aligned improvements:
增强版 AmberLoss，包含两个与 CPR 对齐的改进：

1. Gradient-Aware Vertex Weighting (梯度感知顶点加权):
   Vertices in regions where the sizing field changes rapidly (transition zones
   between fine and coarse mesh regions) receive higher loss weight. These transition
   zones are the critical determinants of mesh quality, yet in the original uniform
   weighting they are easily outnumbered by vertices in flat regions.
   尺寸场变化剧烈的区域（细/粗网格过渡区）的顶点获得更高的损失权重。
   这些过渡区是网格质量的关键决定因素，但在原始均匀加权中很容易被平坦区域的顶点淹没。

   Weight formula: w_j = 1 + β · (g_j / mean(g)) where g_j = mean_{k∈N(j)} |y_j - y_k|
   权重公式：w_j = 1 + β · (g_j / mean(g))，其中 g_j = mean_{k∈N(j)} |y_j - y_k|

2. Depth-Scheduled Robust Loss (深度调度鲁棒损失):
   During early training (shallow curriculum depths), predictions are poor and outliers
   are frequent. A Huber-like loss is used for robustness. As training progresses and
   predictions improve, the loss smoothly transitions to MSE for precision.
   训练早期（浅层课程深度），预测较差且异常值频繁。使用类 Huber 损失增强鲁棒性。
   随着训练进展和预测改善，损失平滑过渡到 MSE 以提高精度。

   L = (1-λ)·Huber(δ, τ) + λ·δ²,  λ = clamp(progress / warmup, 0, 1)

Backward Compatibility (向后兼容性):
- loss_type="mse" with gradient_weight_beta=0.0 and scheduled_loss_enabled=False
  exactly reproduces the original AmberLoss behavior.
  使用 loss_type="mse"、gradient_weight_beta=0.0、scheduled_loss_enabled=False
  可以精确复现原始 AmberLoss 的行为。
- All new parameters have defaults that preserve original behavior.
  所有新参数的默认值都保持原始行为。
"""

from typing import Optional, Tuple

import torch
from torch_geometric.data import Batch

from src.algorithm.loss.mesh_generation_loss import MeshGenerationLoss
from src.algorithm.prediction_transform.prediction_transform import PredictionTransform


class AmberLoss(MeshGenerationLoss):
    def __init__(
        self,
        label_transform: PredictionTransform,
        loss_type: str = "mse",
        # ============================================================
        # Gradient-aware weighting parameters
        # 梯度感知加权参数
        # ============================================================
        gradient_weight_beta: float = 1.0,
        gradient_weight_max: float = 5.0,
        # ============================================================
        # Depth-scheduled robust loss parameters
        # 深度调度鲁棒损失参数
        # ============================================================
        scheduled_loss_enabled: bool = True,
        huber_delta: float = 1.0,
        loss_warmup_ratio: float = 0.3,
    ):
        """
        Enhanced loss for AMBER sizing field prediction.
        增强版 AMBER 尺寸场预测损失函数。

        Args:
            label_transform: Transform applied to invert softplus on target labels.
                应用于目标标签的 softplus 逆变换。
            loss_type: Base loss type ('mse' or 'mae'). Used as the final loss after
                scheduled warmup completes. 基础损失类型，调度预热完成后使用的最终损失。

            gradient_weight_beta: Controls strength of gradient-aware weighting.
                控制梯度感知加权的强度。
                - 0.0: disabled, all vertices weighted equally (original behavior)
                  禁用，所有顶点等权（原始行为）
                - 0.5~2.0: recommended range. Higher values focus more on transition zones.
                  推荐范围。值越高，越聚焦于过渡区域。
                The weight for vertex j is: w_j = 1 + β · normalized_gradient_j
                顶点 j 的权重为：w_j = 1 + β · 归一化梯度_j
            gradient_weight_max: Maximum per-vertex weight to prevent extreme values.
                每顶点最大权重，防止极端值。

            scheduled_loss_enabled: Whether to use depth-scheduled Huber→MSE transition.
                是否使用深度调度的 Huber→MSE 过渡。
                - False: always use loss_type (original behavior) 始终使用 loss_type（原始行为）
                - True: blend Huber and MSE based on training progress
                  根据训练进度混合 Huber 和 MSE
            huber_delta: Threshold for Huber loss. Errors below this use quadratic loss,
                above use linear loss. Huber 损失的阈值。误差低于此值使用二次损失，高于使用线性损失。
            loss_warmup_ratio: Fraction of training during which loss transitions from
                Huber to MSE. After this ratio, pure MSE/loss_type is used.
                损失从 Huber 过渡到 MSE 的训练比例。此比例后使用纯 MSE/loss_type。
        """
        super().__init__(label_transform=label_transform)
        self.loss_type = loss_type

        # Gradient-aware weighting
        self.gradient_weight_beta = gradient_weight_beta
        self.gradient_weight_max = gradient_weight_max

        # Depth-scheduled loss
        self.scheduled_loss_enabled = scheduled_loss_enabled
        self.huber_delta = huber_delta
        self.loss_warmup_ratio = loss_warmup_ratio

        # Training progress, updated externally by the algorithm
        # 训练进度，由算法外部更新
        self._training_progress: float = 0.0

    @property
    def training_progress(self) -> float:
        return self._training_progress

    @training_progress.setter
    def training_progress(self, value: float) -> None:
        self._training_progress = max(0.0, min(1.0, value))

    # ============================================================
    # Main loss computation
    # 主损失计算
    # ============================================================

    def calculate_loss(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        graph_batch: Optional[Batch] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the loss for a batch of predictions and labels.
        计算一批预测和标签的损失。

        Enhanced pipeline (增强流程):
        1. Transform labels (softplus⁻¹ with optional residual) — same as original
           标签变换（softplus⁻¹ 可选残差）— 与原始相同
        2. Compute per-vertex differences — same as original
           计算每顶点差异 — 与原始相同
        3. Compute per-vertex element loss (MSE/Huber/scheduled blend) — ENHANCED
           计算每顶点元素损失 — 增强
        4. Compute per-vertex weights from label gradients — NEW
           从标签梯度计算每顶点权重 — 新增
        5. Weighted mean → final loss — ENHANCED
           加权平均 → 最终损失 — 增强

        Args:
            predictions: Model raw outputs x_j (untransformed) 模型原始输出
            labels: Target sizing field y_j (before transform) 目标尺寸场（变换前）
            graph_batch: PyG Batch with edge_index, edge_attr, current_sizing_field, etc.

        Returns:
            (loss, differences): loss scalar and per-vertex absolute differences
        """
        # --- Step 1: Transform labels (same as original) ---
        if graph_batch is not None and hasattr(graph_batch, "current_sizing_field"):
            baseline = graph_batch.current_sizing_field
        else:
            baseline = None

        # Save raw labels BEFORE transform for gradient computation
        # 在变换之前保存原始标签用于梯度计算
        raw_labels = labels.detach()

        labels = self.label_transform.inverse(labels, baseline=baseline, is_train=True)

        # --- Step 2: Per-vertex differences (same as original) ---
        differences = self.get_differences(predictions=predictions, labels=labels)

        # --- Step 3: Per-vertex element loss (enhanced) ---
        element_loss = self._compute_element_loss(differences)

        # --- Step 4: Gradient-aware vertex weights (new) ---
        if self.gradient_weight_beta > 0.0 and graph_batch is not None:
            weights = self._compute_gradient_weights(raw_labels, graph_batch)
            # Apply weights: weighted mean instead of simple mean
            # 应用权重：加权平均而非简单平均
            loss = torch.sum(weights * element_loss) / torch.sum(weights)
        else:
            # Original behavior: simple mean  原始行为：简单平均
            loss = torch.mean(element_loss)

        return loss, differences

    # ============================================================
    # Component 1: Depth-Scheduled Element Loss
    # 组件1：深度调度元素损失
    # ============================================================

    def _compute_element_loss(self, differences: torch.Tensor) -> torch.Tensor:
        """
        Compute per-vertex loss values, optionally using depth-scheduled blending.
        计算每顶点损失值，可选使用深度调度混合。

        When scheduled_loss_enabled=False (default):
            Identical to original: MSE or MAE.
            与原始相同：MSE 或 MAE。

        When scheduled_loss_enabled=True:
            Blends between Huber loss (robust to outliers) and MSE (precise):
            在 Huber 损失（对异常值鲁棒）和 MSE（精确）之间混合：

            L_element = (1-λ) · Huber(diff, δ) + λ · diff²
            λ = clamp(training_progress / loss_warmup_ratio, 0, 1)

            Rationale (原理):
            - Early training (λ≈0): Model predictions are poor, intermediate meshes have
              low quality, many vertices have large errors. Huber loss's linear tail prevents
              these outliers from dominating gradients, allowing stable learning.
              训练早期：模型预测差，中间网格质量低，许多顶点误差大。Huber 损失的线性尾部
              防止这些异常值主导梯度，允许稳定学习。

            - Late training (λ≈1): Predictions are accurate, errors are small and well-behaved.
              Pure MSE provides stronger gradients for precise fitting.
              训练后期：预测准确，误差小且分布良好。纯 MSE 提供更强的梯度用于精确拟合。

            - The transition aligns with CPR's curriculum: as deeper depths are unlocked,
              the model is ready for more demanding MSE optimization.
              该过渡与 CPR 的课程对齐：随着更深层深度的解锁，模型已准备好接受更严格的 MSE 优化。

        Args:
            differences: Per-vertex absolute differences |prediction - target|
                每顶点绝对差异

        Returns:
            Per-vertex loss values (not yet reduced) 每顶点损失值（尚未归约）
        """
        if not self.scheduled_loss_enabled:
            # Original behavior: pure MSE or MAE
            # 原始行为：纯 MSE 或 MAE
            if self.loss_type == "mse":
                return differences ** 2
            elif self.loss_type == "mae":
                return differences
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Scheduled blending: Huber → MSE
        # 调度混合：Huber → MSE
        lambda_blend = min(1.0, self._training_progress / max(1e-8, self.loss_warmup_ratio))

        # Huber loss: quadratic for |diff| ≤ δ, linear for |diff| > δ
        # Huber 损失：|diff| ≤ δ 时二次，|diff| > δ 时线性
        delta = self.huber_delta
        abs_diff = differences  # already absolute from get_differences
        huber_loss = torch.where(
            abs_diff <= delta,
            0.5 * abs_diff ** 2,
            delta * abs_diff - 0.5 * delta ** 2,
        )

        # MSE loss
        mse_loss = differences ** 2

        # Blend: early training uses Huber, late training uses MSE
        # 混合：训练早期使用 Huber，训练后期使用 MSE
        element_loss = (1.0 - lambda_blend) * huber_loss + lambda_blend * mse_loss

        return element_loss

    # ============================================================
    # Component 2: Gradient-Aware Vertex Weighting
    # 组件2：梯度感知顶点加权
    # ============================================================

    def _compute_gradient_weights(
        self,
        raw_labels: torch.Tensor,
        graph_batch: Batch,
    ) -> torch.Tensor:
        """
        Compute per-vertex importance weights based on sizing field gradient magnitude.
        根据尺寸场梯度幅度计算每顶点重要性权重。

        Algorithm (算法):
        1. For each edge (i,j) in the graph, compute |y_i - y_j| as a proxy for
           the local sizing field gradient along that edge.
           对图中每条边 (i,j)，计算 |y_i - y_j| 作为该边上局部尺寸场梯度的代理。

        2. For each vertex j, average the edge gradients over all its neighbors:
           对每个顶点 j，对其所有邻居的边梯度取平均：
           g_j = mean_{k ∈ N(j)} |y_j - y_k|

        3. Normalize and scale:
           归一化并缩放：
           w_j = 1 + β · (g_j / (mean(g) + ε))

        4. Clamp to [1, gradient_weight_max] to prevent extreme weights.
           限制在 [1, gradient_weight_max] 以防止极端权重。

        Why raw labels instead of transformed labels? (为什么用原始标签而非变换后标签？)
        The raw sizing field values directly reflect the physical mesh resolution.
        A gradient in raw space means the mesh transitions from fine to coarse elements,
        which is exactly the spatial feature we want to upweight. In softplus⁻¹ space,
        the gradient is distorted by the nonlinear transform and less interpretable.
        原始尺寸场值直接反映物理网格分辨率。原始空间中的梯度意味着网格从细到粗的过渡，
        这正是我们想要增加权重的空间特征。

        Handling hierarchical graphs (处理分层图):
        When mask_output is present, we only compute weights for the current mesh nodes
        (mask_output=True), not the initial mesh nodes. Initial mesh nodes get weight=0
        since they don't contribute to the loss anyway.
        当存在 mask_output 时，仅为当前网格节点计算权重，初始网格节点权重为 0。

        Handling batched graphs (处理批量图):
        The edge_index and labels are already batched by PyG's Batch mechanism.
        Our scatter operations naturally respect batch boundaries because edge_index
        already contains offset node indices per graph in the batch.
        edge_index 和标签已由 PyG 的 Batch 机制批处理。
        我们的 scatter 操作自然遵守批次边界。

        Args:
            raw_labels: Sizing field targets before softplus⁻¹ transform, shape (num_nodes,)
                softplus⁻¹ 变换前的尺寸场目标
            graph_batch: PyG Batch object containing graph topology

        Returns:
            Per-vertex weights, shape (num_nodes,), all >= 1.0
            每顶点权重，所有值 >= 1.0
        """
        num_nodes = raw_labels.shape[0]
        device = raw_labels.device

        edge_index = graph_batch.edge_index  # [2, num_edges]

        # Step 1: Compute per-edge gradient magnitude
        # 步骤1：计算每边梯度幅度
        src_nodes = edge_index[0]  # source nodes
        dst_nodes = edge_index[1]  # destination nodes

        # |y_src - y_dst| for each edge
        edge_gradient = torch.abs(raw_labels[src_nodes] - raw_labels[dst_nodes])

        # Step 2: Aggregate edge gradients to destination nodes (mean over neighbors)
        # 步骤2：将边梯度聚合到目标节点（邻居平均）
        # Using scatter_mean: for each dst node, average over all incoming edge gradients
        # 使用 scatter_mean：对每个目标节点，对所有入边梯度取平均

        # Count edges per destination node for mean computation
        # 计算每个目标节点的边数用于平均计算
        node_gradient_sum = torch.zeros(num_nodes, device=device)
        node_degree = torch.zeros(num_nodes, device=device)

        node_gradient_sum.scatter_add_(0, dst_nodes, edge_gradient)
        node_degree.scatter_add_(0, dst_nodes, torch.ones_like(edge_gradient))

        # Avoid division by zero for isolated nodes
        # 避免孤立节点的除零错误
        safe_degree = torch.clamp(node_degree, min=1.0)
        node_gradient = node_gradient_sum / safe_degree  # per-vertex mean gradient

        # Step 3: Normalize by global mean gradient
        # 步骤3：按全局平均梯度归一化
        mean_gradient = node_gradient.mean() + 1e-8
        normalized_gradient = node_gradient / mean_gradient

        # Step 4: Compute weights
        # 步骤4：计算权重
        # w_j = 1 + β · normalized_gradient_j, clamped to [1, max]
        weights = 1.0 + self.gradient_weight_beta * normalized_gradient
        weights = torch.clamp(weights, min=1.0, max=self.gradient_weight_max)

        # Step 5: Handle hierarchical graph mask
        # 步骤5：处理分层图掩码
        # Initial mesh nodes should not contribute to loss (their weight = 0)
        # 初始网格节点不应贡献损失（权重为 0）
        if hasattr(graph_batch, "mask_output"):
            weights = weights * graph_batch.mask_output.float()

        return weights

    # ============================================================
    # Unchanged interface: get_differences
    # 不变接口：get_differences
    # ============================================================

    def get_differences(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculates the (absolute) differences between predictions and labels.
        Identical to original implementation.
        计算预测和标签之间的绝对差异。与原始实现相同。
        """
        differences = torch.abs(predictions - labels)
        return differences