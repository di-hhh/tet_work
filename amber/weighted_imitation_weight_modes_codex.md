[CodeX] 本文档由 CodeX 生成，用于说明 AMBER 中 `weighted_imitation` 的不同加权模式。

# Weighted Imitation 加权模式说明

本文档介绍 AMBER 当前针对 `console` / `mold` 任务实现的 `weighted_imitation` 加权模式。

这几种模式的共同目标是：在不改变模型输出语义的前提下，调整节点级监督损失的聚合权重，让训练更关注物理上更重要的区域。

## 1. 总体思路

当前链路中，参考物理求解会先生成节点级 `importance`，然后再把它转换成训练时使用的节点权重 `w_i`。

训练损失的聚合形式为：

```text
weighted_loss = sum(w_i * loss_i) / (sum(w_i) + eps)
```

其中：

- `loss_i`：第 `i` 个节点的逐点尺寸监督误差
- `w_i`：由物理重要性构造出的节点权重
- `eps`：数值稳定项

如果所有节点的权重都接近 `1`，那么它就几乎退化为原始未加权训练。

因此，不同加权模式的本质区别，就是“如何把归一化后的重要性映射成足够有区分度的训练权重”。

## 2. 统一输入：normalized importance

无论选择哪种模式，输入都不是原始的 FE 重要性，而是经过统一预处理后的 `normalized_importance`。

当前默认流程是：

1. 从参考线弹性求解中得到单元应变能密度；
2. 做单元到节点的体积加权平均；
3. 对节点重要性做 `log1p`；
4. 按样本内分位数做归一化，默认使用 `p5` 到 `p95`；
5. 截断到 `[0, 1]`。

因此，后面所有模式都可以理解为：

```text
输入: normalized_importance in [0, 1]
输出: positive node weights
```

## 3. 当前支持的加权模式

### 3.1 `linear`

公式：

```text
w_i = 1 + beta * normalized_importance_i
```

特点：

- 最温和、最平滑的模式；
- 不会突然抬高少数节点；
- 与当前 baseline 最接近，适合作为第一组可复现实验。

适用场景：

- 想先验证“轻度物理偏置”是否有收益；
- 不希望训练分布变化太激进；
- 作为其他更强模式的对照组。

风险：

- 如果 `beta` 不够大，或者投影后重要性本身已被压平，这个模式容易和全 1 权重差别很小；
- 在这种情况下，它可能不足以明显改变梯度分配。

关键配置：

- `algorithm.weighted_imitation.weight_mode=linear`
- `algorithm.weighted_imitation.beta`

示例：

```bash
python main.py +_runs/amber=amber_console_weighted \
  algorithm.weighted_imitation.weight_mode=linear \
  algorithm.weighted_imitation.beta=1.0
```

### 3.2 `power`

公式：

```text
w_i = 1 + beta * normalized_importance_i^gamma
```

特点：

- 在 `linear` 的基础上，通过 `gamma` 调整高重要区域的强化程度；
- `gamma > 1` 时，中低重要区域会被进一步压低，相对更强调高重要区域；
- 是“中等强度”的增强方案。

适用场景：

- `linear` 太弱，但又不想直接切换到二值硬阈值；
- 想保留连续权重，同时提升头部重要区域的影响力。

参数解释：

- `beta`：整体放大倍数；
- `gamma`：高重要区强化强度，越大越强调头部节点。

经验解释：

- `gamma = 1` 时，等价于 `linear`；
- `gamma = 2~4` 往往比 `linear` 更容易产生可观测差异；
- 如果 `gamma` 过大，容易让大部分节点权重都接近 `1`，训练可能过于集中到极少数区域。

关键配置：

- `algorithm.weighted_imitation.weight_mode=power`
- `algorithm.weighted_imitation.beta`
- `algorithm.weighted_imitation.gamma`

示例：

```bash
python main.py +_runs/amber=amber_console_weighted \
  algorithm.weighted_imitation.weight_mode=power \
  algorithm.weighted_imitation.gamma=3.0
```

### 3.3 `binary_topk`

规则：

- 样本内重要性最高的前 `topk_percent` 节点，权重设为 `lambda_high`
- 其余节点权重设为 `1`

可写成：

```text
if importance_i in top-k:
    w_i = lambda_high
else:
    w_i = 1
```

特点：

- 非常直接地把训练信号集中到高重要区域；
- 比 `linear` / `power` 更容易显著改变梯度分配；
- 适合回答一个直接问题：如果强行偏置高重要区，关键区域误差会不会下降。

适用场景：

- 诊断“当前连续加权是否太弱”；
- 想要明确拉开高重要区域与其余区域的损失贡献；
- 适合作为强对照实验。

风险：

- 如果 `lambda_high` 过大，低重要区域几乎不再参与优化，可能造成全域指标退化；
- 如果 `topk_percent` 太小，训练会过分依赖极少数点，稳定性变差。

关键配置：

- `algorithm.weighted_imitation.weight_mode=binary_topk`
- `algorithm.weighted_imitation.topk_percent`
- `algorithm.weighted_imitation.lambda_high`

示例：

```bash
python main.py +_runs/amber=amber_console_weighted \
  algorithm.weighted_imitation.weight_mode=binary_topk \
  algorithm.weighted_imitation.topk_percent=0.2 \
  algorithm.weighted_imitation.lambda_high=8.0
```

### 3.4 `ternary_quantile`

规则：

- 低重要区：`w = 1`
- 中重要区：`w = lambda_mid`
- 高重要区：`w = lambda_high`

阈值由两个样本内分位数控制：

- `ternary_low_quantile`
- `ternary_high_quantile`

可以理解为：

```text
importance < q_low       -> 1
q_low <= importance < q_high -> lambda_mid
importance >= q_high     -> lambda_high
```

特点：

- 介于连续加权和二值 top-k 之间；
- 能保留一定层次结构，不会像 `binary_topk` 那样只区分“重点 / 非重点”两类；
- 对重要性分布较复杂的样本通常更稳定。

适用场景：

- 想强调高重要区，但又不想完全忽略中等重要区；
- 希望训练信号具备“三段式资源分配”效果。

关键配置：

- `algorithm.weighted_imitation.weight_mode=ternary_quantile`
- `algorithm.weighted_imitation.lambda_mid`
- `algorithm.weighted_imitation.lambda_high`
- `algorithm.weighted_imitation.ternary_low_quantile`
- `algorithm.weighted_imitation.ternary_high_quantile`

示例：

```bash
python main.py +_runs/amber=amber_console_weighted \
  algorithm.weighted_imitation.weight_mode=ternary_quantile \
  algorithm.weighted_imitation.lambda_mid=2.0 \
  algorithm.weighted_imitation.lambda_high=6.0 \
  algorithm.weighted_imitation.ternary_low_quantile=0.5 \
  algorithm.weighted_imitation.ternary_high_quantile=0.8
```

## 4. 这些模式该怎么选

推荐顺序：

1. `linear`
2. `power`
3. `binary_topk`
4. `ternary_quantile`

这样做的原因是：

- `linear` 用来验证最小改动是否已足够；
- `power` 用来验证“是不是只是线性模式太弱”；
- `binary_topk` 用来做强诊断，判断把梯度强行集中后，关键区误差是否下降；
- `ternary_quantile` 用来在“过于温和”和“过于激进”之间找折中。

如果你的重点是诊断而不是保守训练，通常优先比较这三组最有信息量：

- baseline（不加权）
- `power`
- `binary_topk`

## 5. 与 Stage 2 微调的关系

`two_stage_importance_finetune` 不会引入新的权重家族，它只是把上面这些模式再用于微调阶段。

典型方式是：

- Stage 1：baseline 或较温和模式（如 `linear` / `power`）
- Stage 2：切换到更激进模式（如 `binary_topk`）

常见配置：

```yaml
algorithm:
  weighted_imitation:
    stage2_enable: true
    stage2_weight_mode: binary_topk
    stage2_topk_percent: 0.2
    stage2_lambda_high: 8.0
    stage2_high_importance_only: true
```

这类设置适合下面这种情况：

- 从头加权训练效果不明显；
- 但你希望在已有 checkpoint 基础上，进一步把误差压到高重要区。

## 6. 怎么判断某个模式是否有效

不要只看全域未加权 `projected_l2_error`。

建议至少同时看下面几类指标：

### 6.1 效果指标

- `projected_l2_error`
- `physics_weighted_projected_l2_error`
- `weighted_size_l2`
- `topk_high_importance_l2`
- `bucket_high_size_l2`
- `bucket_high_low_ratio`

推荐解读：

- 如果全域 `projected_l2_error` 基本不变，但 `topk_high_importance_l2` 下降，说明精度开始向高重要区域重新分配；
- 如果 `weighted_size_l2` 和 `bucket_high_size_l2` 都下降，说明物理重要区域的尺寸误差确实在改善；
- 如果这些指标都不改善，那么当前模式没有带来有效偏置。

### 6.2 诊断指标

- `imitation_weight_top20_ratio`
- `imitation_weight_effective_sample_ratio`
- `imitation_projection_q95_ratio`
- `imitation_projection_top20_ratio_delta`
- `imitation_weight_neg_log_size_spearman`

推荐解读：

- `top20_ratio` 还接近 `0.2`：说明权重太弱；
- `effective_sample_ratio` 还接近 `1.0`：说明整体仍接近均匀加权；
- `projection_q95_ratio` 很低：说明投影把高重要区域压平了；
- `weight_neg_log_size_spearman` 接近 `0`：说明物理重要性和尺寸目标可能错位。

## 7. 一个最小对比实验应该怎么跑

建议先重生成权重缓存：

```bash
python precompute_console_mold_weights.py --datasets console mold --overwrite
```

然后对同一个数据集至少比较下面几组：

### Baseline

```bash
python main.py +_runs/amber=amber_console \
  algorithm.weighted_imitation.enabled=False \
  algorithm.weighted_imitation.metric_use_physics_weights=True
```

### Linear

```bash
python main.py +_runs/amber=amber_console_weighted
```

### Power

```bash
python main.py +_runs/amber=amber_console_weighted \
  algorithm.weighted_imitation.weight_mode=power \
  algorithm.weighted_imitation.gamma=3.0
```

### Binary Top-k

```bash
python main.py +_runs/amber=amber_console_weighted \
  algorithm.weighted_imitation.weight_mode=binary_topk \
  algorithm.weighted_imitation.lambda_high=8.0 \
  algorithm.weighted_imitation.topk_percent=0.2
```

也可以先用缓存级快速对比脚本做筛选：

```bash
python compare_weighted_imitation_codex.py --dataset console
python compare_weighted_imitation_codex.py --dataset mold
```

## 8. 当前实践建议

结合当前实现与已有诊断逻辑，建议按下面顺序尝试：

1. 先确认缓存已重生成，不要继续使用旧缓存；
2. 先跑 baseline，打开 physics-aware 指标；
3. 再跑 `linear`；
4. 如果 `linear` 不明显，优先试 `power`；
5. 如果还不明显，优先试 `binary_topk`；
6. 如果 `binary_topk` 有效果，而从头训练仍然不稳定，再尝试 stage2 微调。

## 9. 对应配置项速查

| 配置项 | 作用 |
| --- | --- |
| `weight_mode` | 选择加权模式 |
| `beta` | 连续型模式的整体放大倍数 |
| `gamma` | `power` 模式的幂次 |
| `lambda_high` | 高重要区额外权重 |
| `lambda_mid` | `ternary_quantile` 的中重要区权重 |
| `topk_percent` | `binary_topk` 的高重要区比例 |
| `ternary_low_quantile` | 三段式低阈值分位数 |
| `ternary_high_quantile` | 三段式高阈值分位数 |
| `clip_min` / `clip_max` | 最终权重截断范围 |
| `stage2_enable` | 是否开启两阶段微调 |
| `stage2_weight_mode` | 第二阶段使用的权重模式 |
| `stage2_topk_percent` | 第二阶段 top-k 比例 |
| `stage2_lambda_high` | 第二阶段高重要区权重 |

## 10. 一句话总结

- `linear`：最稳妥，但可能太弱
- `power`：连续增强版，适合先试
- `binary_topk`：最强诊断模式，最容易看出是否真的把精度压向关键区
- `ternary_quantile`：折中方案，兼顾中高重要区

如果目标是快速判断“物理加权到底有没有改变误差分配”，优先比较：

```text
baseline vs power vs binary_topk
```
