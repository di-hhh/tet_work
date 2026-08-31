# AMBER 改动总结

本文档总结当前仓库相对原始仓库 [NiklasFreymuth/AMBER](https://github.com/NiklasFreymuth/AMBER) 的主要改动，重点包括：

- Console / Mold 上的物理加权模仿
- 模型侧 physics correction
- 训练目标、评估指标与实验入口扩展

## 1. 总体方向

相对原始 AMBER 的标准 sizing field imitation 流程，当前版本主要增加了两层能力：

1. 物理加权模仿
   - 在 Console 和 Mold 两个 3D 数据集上，为不同节点赋予不同的物理重要性权重。
   - 训练与评估不再只看全域平均误差，而是重点关注高物理重要区域。

2. 模型侧物理修正
   - 保留原有 expert-style 主预测路径。
   - 在主路径旁新增 physics correction branch 和 gate head。
   - 学习一个 `expert prior + gated physics residual` 的最小增量结构。

这两层改动的关系是：

- 第一层解决“loss 是否更关注重要区域”。
- 第二层解决“模型本身是否显式利用物理信息进行修正”。

## 2. 物理加权模仿

### 2.1 支持范围

当前只对以下两个数据集完整支持：

- `console`
- `mold`

对其他数据集：

- 不扩展物理加权功能
- 不报错
- 保持原始 AMBER 路径或安全退化

### 2.2 核心改动

在 Console / Mold 上，新增了基于参考物理场的节点重要性流程：

1. 在参考专家网格上计算物理重要性
2. 将重要性缓存到本地
3. 将参考重要性投影到当前训练/验证使用的节点上
4. 基于该重要性构造 weighted imitation loss
5. 在验证和测试中报告物理相关评估指标

当前验证效果最好的权重模式是：

- `linear`

### 2.3 主要实现位置

- `src/algorithm/util/console_mold_reference.py`
- `src/algorithm/util/linear_elasticity_reference_codex.py`
- `src/algorithm/util/fem_imitation_weights.py`
- `src/algorithm/loss/amber_loss.py`
- `src/algorithm/util/weighted_imitation_diagnostics_codex.py`
- `src/mesh_util/mesh_metrics.py`

### 2.4 物理加权带来的变化

相对原始 AMBER 的统一误差监督，当前版本新增了：

- 参考物理重要性缓存与加载
- 从参考网格到中间网格的权重投影
- weighted imitation loss
- 高重要区域评估指标
- two-stage importance finetune 的兼容支持

## 3. 模型架构侧 physics correction

### 3.1 设计目标

原始 AMBER 中，物理信息并不会作为模型内显式修正通道参与预测。当前版本在尽量少改 backbone 的前提下，增加了一个最小可用的模型侧物理修正结构。

目标是：

- 保留原始 expert prior 主路径
- 不让 physics branch 直接替代 expert prediction
- 只让 physics branch 学习“局部残差修正”
- 用 gate 控制修正强度，避免早期训练破坏 expert prior

### 3.2 新结构

在共享编码器之上，模型现在可以包含三个输出分支：

1. expert prior head
   - 输出 `delta_expert`
   - 语义与原有主头保持一致

2. physics correction head
   - 输出 `delta_phys`
   - 与 `delta_expert` 处于同一预测空间
   - 只负责 residual correction

3. gate head
   - 输出 gate logits
   - 经 `sigmoid` 得到 `alpha`

最终预测形式为：

```text
delta_total = delta_expert + alpha * delta_phys
```

然后继续沿用原有 prediction transform，将 `delta_total` 还原为最终 sizing prediction。

### 3.3 与原始 AMBER 的关系

这不是重写 AMBER backbone，而是在原有主干上增加最小增量：

- 共享 encoder / message passing / hierarchical graph 主干保留
- 原 expert-style 主预测头保留
- 只增加 correction head 和 gate head

当以下任一条件成立时，新结构应退化到原路径：

- `enable_physics_correction_branch=False`
- gate 接近 0
- 缺失 physics feature 且 fallback 为 `gate_zero` 或等价行为

### 3.4 主要实现位置

- `src/algorithm/architecture/physics_correction_branch_codex.py`
- `src/algorithm/architecture/supervised_mpn.py`
- `src/algorithm/architecture/graphmesh_gcn.py`
- `src/algorithm/architecture/edge_aware_gat.py`
- `src/algorithm/core/mesh_generation_algorithm.py`
- `src/algorithm/core/amber.py`

## 4. 物理信息如何进入模型

### 4.1 从 loss 端推进到输入端

相对原始 AMBER 的关键变化之一，是物理信息不再只存在于 loss 权重里，而是进入了模型输入。

对 Console / Mold：

- 复用当前已经存在的节点级物理重要性
- 将其标准化后作为额外节点特征拼接到原 node features
- 在图对象上显式保存 physics feature 与 availability 标记

### 4.2 安全回退

为了保证推理和多数据集兼容性，当前版本支持：

- physics feature 缺失时自动补零
- 或通过配置使 branch 失效
- 或让 gate 视为 0

因此不会因为某些场景缺失 physics feature 而导致模型崩溃。

### 4.3 主要实现位置

- `src/algorithm/dataloader/amber_data.py`
- `src/tasks/dataset_preparator.py`

## 5. 训练目标扩展

### 5.1 主监督

继续保留当前最有效的主监督：

- final prediction 上的 linear physics-weighted imitation loss

也就是说，physics correction 并没有替代原本成功的 weighted imitation，而是叠加在其上。

### 5.2 新增辅助项

为了稳定训练，当前版本又加入了几个轻量扩展项：

1. expert auxiliary loss
   - 直接监督 `delta_expert`
   - 防止 expert prior head 被 physics branch 拉偏

2. correction auxiliary loss
   - 约束 correction 学习“应该补多少”

3. correction regularization
   - 约束 `alpha * delta_phys` 的幅度
   - 防止 correction 一开始过大

总损失可以概括为：

```text
L_total =
    L_final_weighted
  + lambda_expert_aux * L_expert_aux
  + lambda_corr_aux * L_corr_aux
  + lambda_corr_reg * L_corr_reg
```

### 5.3 warmup 与 checkpoint 兼容

当前实现还支持：

- correction warmup
- 从已有 weighted baseline checkpoint 继续训练
- 对新增 head 的非严格加载与安全初始化

这保证 physics correction 是在已验证成功的 weighted baseline 基础上继续做增量训练。

### 5.4 主要实现位置

- `src/algorithm/loss/amber_loss.py`
- `src/algorithm/core/mesh_generation_algorithm.py`

## 6. 评估和诊断扩展

### 6.1 高重要区域评估

相对原始 AMBER 主要依赖全局误差，当前版本新增了一组更适合物理加权任务的指标：

- `weighted_size_l2`
- `topk_high_importance_l2`
- `bucket_low_size_l2`
- `bucket_high_size_l2`
- `bucket_high_low_ratio`
- `physics_weighted_projected_l2_error`

这些指标用于判断模型是否真的改善了高物理重要区域，而不是只改善全局均值。

### 6.2 physics correction 诊断

为了确认“物理信息真的进入了模型端”，新增了以下统计：

- gate 统计
  - `gate_mean`
  - `gate_std`
  - 高重要区域 gate mean
  - 低重要区域 gate mean

- correction 统计
  - `|delta_phys|` 均值
  - `|alpha * delta_phys|` 均值
  - 高低重要区域对比

- expert prior vs final prediction 对比
  - `expert_prior_weighted_size_l2`
  - `final_prediction_weighted_size_l2`
  - `expert_prior_topk_high_importance_l2`
  - `final_prediction_topk_high_importance_l2`

这些诊断用于判断：

- gate 是否在高重要区域更活跃
- physics correction 是否真的在起作用
- final prediction 是否优于 expert prior-only

### 6.3 主要实现位置

- `src/algorithm/util/weighted_imitation_diagnostics_codex.py`
- `src/mesh_util/mesh_metrics.py`
- `src/algorithm/core/mesh_generation_algorithm.py`

## 7. 配置与实验入口扩展

### 7.1 新增配置项

当前版本最小扩展了算法配置，主要包括：

- `enable_physics_correction_branch`
- `physics_feature_mode`
- `lambda_expert_aux`
- `lambda_corr_aux`
- `lambda_corr_reg`
- `gate_activation`
- `gate_max`
- `gate_init_bias`
- `physics_readout_init_std`
- `correction_warmup_epochs`
- `init_from_weighted_baseline_checkpoint`
- `inference_missing_physics_fallback`

主要配置文件：

- `config/algorithm/default_algorithm.yaml`

### 7.2 run 配置

除了原始和 weighted imitation 实验入口外，当前版本新增了 physics correction 对应的 run 预设，便于直接启动：

- `config/_runs/amber/_amber_physics_correction_codex.yaml`
- `config/_runs/amber/amber_console_physics_correction_codex.yaml`
- `config/_runs/amber/amber_mold_physics_correction_codex.yaml`

因此现在仓库里实际存在三类实验入口：

1. 原始 AMBER
2. weighted imitation
3. physics correction on top of weighted imitation

### 7.3 启用命令

如果需要重新生成 Console / Mold 的参考物理重要性缓存，可以先执行：

```bash
python precompute_console_mold_weights.py --datasets console mold
```

该命令会在 `data/weighted_imitation/` 下生成或更新缓存。若缓存已经存在，可直接跳过。

启用物理加权模仿实验：

```bash
python main.py +_runs/amber=amber_console_weighted
python main.py +_runs/amber=amber_mold_weighted
```

启用模型侧物理修正实验：

```bash
python main.py +_runs/amber=amber_console_physics_correction_codex
python main.py +_runs/amber=amber_mold_physics_correction_codex
```

如果要在已有 weighted baseline checkpoint 基础上继续训练 physics correction，可执行：

```bash
python main.py +_runs/amber=amber_console_physics_correction_codex \
  algorithm.init_from_weighted_baseline_checkpoint=/abs/path/to/weighted_baseline.ckpt
```

`mold` 数据集同理，只需将 run 入口替换为 `amber_mold_physics_correction_codex`。

## 8. 测试与工程配套

为了保证这些改动不是一次性 patch，当前版本补充了对应测试与兼容逻辑。

### 8.1 测试覆盖

主要测试包括：

- branch / gate 输出形状测试
- physics branch 关闭时的退化一致性测试
- physics feature 缺失时的安全回退测试
- loss 组合是否正确的测试
- checkpoint 兼容加载测试
- weighted imitation 诊断与线弹性参考流程测试

相关文件：

- `test_physics_correction_codex.py`
- `tests/test_weighted_imitation.py`
- `tests/test_weighted_imitation_diagnostics_codex.py`
- `tests/test_linear_elasticity_reference_codex.py`

### 8.2 工程兼容

还加入了一些环境和依赖兼容性处理，例如：

- `src/helpers/torch_scatter_compat_codex.py`

这类改动的目标是让当前增强版 AMBER 在现有环境下更稳定运行，不属于算法主线，但属于必要工程配套。

## 9. 一句话总结

相对原始 `NiklasFreymuth/AMBER`，当前仓库已经从“标准 sizing field imitation”推进到了一个面向 `console` / `mold` 的两层增强版本：

1. 先通过 physics-weighted imitation，让训练和评估更关注高物理重要区域。
2. 再通过 model-side physics correction，让模型学习 `expert prior + gated physics residual`，把物理信息真正推进到模型结构中。

这仍然不是端到端可微 FE-in-the-loop 框架，而是在保留 AMBER 原结构和原 expert 语义的前提下，做出的最小而完整的物理感知增量实现。
