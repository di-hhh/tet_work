# 四面体网格生成工作交接索引

这个目录是当前研究工作的交接入口，不是完整技术说明。下一位研究者应先通过本文档定位已有代码库和文档，再进入对应仓库继续实现。

当前工作主要由两个代码库组成：

- `dataest-pipeline/`：数据生产 pipeline，用真实 STEP/CAD 几何、物理条件和 budget 生成 condition-aware teacher mesh 数据集。
- `zhaopy2534116/amber`：基于 AMBER 改造的训练仓库，已经加入物理加权模仿和模型侧物理修正。GitHub 地址：[https://github.com/zhaopy2534116/amber](https://github.com/zhaopy2534116/amber)

## 建议阅读顺序

1. 先看路线背景和问题定义：
   - [文献调研-3-17.docx](./文献调研-3-17.docx)
   - [路线纠正4-8.docx](./路线纠正4-8.docx)
   - [4-7.docx](./4-7.docx)

2. 再看数据生产 pipeline：
   - [4-14-数据集生成pipeline.docx](./4-14-数据集生成pipeline.docx)
   - [dataest-pipeline/CONDITION_AWARE_DATA_PIPELINE.md](./dataest-pipeline/CONDITION_AWARE_DATA_PIPELINE.md)

3. 然后看 pipeline 如何接入 AMBER 训练：
   - [dataest-pipeline/AMBER_PHYSICS_TRAINING_WITH_PIPELINE_DATA.md](./dataest-pipeline/AMBER_PHYSICS_TRAINING_WITH_PIPELINE_DATA.md)

4. 最后看 AMBER 侧已经完成了哪些改动：
   - [zhaopy2534116/amber](https://github.com/zhaopy2534116/amber)
   - [AMBER_modifications_summary.md](https://github.com/zhaopy2534116/amber/blob/main/AMBER_modifications_summary.md)

## 文档之间的关系

`文献调研-3-17.docx` 主要记录相关工作和 AMBER / learned mesh generation 的背景，用来理解为什么要从 expert mesh imitation 出发。

`路线纠正4-8.docx` 说明早期路线中的关键问题：如果仍然使用 AMBER 原始 expert mesh 作为监督，却额外加入一套自己定义的物理权重，可能会出现“监督目标”和“物理权重来源”不完全一致的问题。后续引入 condition-aware dataset pipeline，就是为了解决这个一致性问题。

`4-7.docx` 记录从“物理加权模仿”推进到“物理修正模仿”的思路：物理信息不应只停留在 loss weight 中，还应进入模型结构，形成 `expert prior + gated physics residual`。

`4-14-数据集生成pipeline.docx` 和 `CONDITION_AWARE_DATA_PIPELINE.md` 说明数据生产端的目标：从 STEP/CAD 几何出发，采样物理条件，生成 teacher mesh、stage fields、error indicator、budget 诊断和 `sample_manifest.jsonl`。

`AMBER_PHYSICS_TRAINING_WITH_PIPELINE_DATA.md` 是两个代码库之间最关键的衔接文档：它说明 pipeline 负责造 teacher，AMBER 负责学 teacher，但两者之间目前还缺 dataset adapter / task adapter。

`AMBER_modifications_summary.md` 说明 AMBER 训练仓库已经完成的内容：Console / Mold 上的 physics-weighted imitation、linear 权重模式、模型侧 physics correction branch、gate head、expert auxiliary loss、correction regularization、诊断指标和实验启动命令。

## 当前状态

数据生产端已经具备 condition-aware teacher 数据生成能力，核心产物包括：

- `sample_manifest.jsonl`
- `teacher_records.jsonl`
- `initial_mesh`
- `final_target_mesh`
- `stage_fields`
- `error_indicator`
- `smoke_report`

AMBER 训练端已经具备：

- 原始 AMBER baseline
- Console / Mold 的 physics-weighted imitation
- physics correction branch
- gated residual correction
- expert prior + physics residual 的训练路径
- 高重要区域误差、weighted error、gate/correction 诊断指标

但两者还没有完全合并。当前还不能把 pipeline 产物零改动直接送进 AMBER 训练。

## 下一步工作

第一步是合并两个代码库的能力，但不建议简单复制文件。更稳的方式是在 AMBER 仓库中新增一个 pipeline dataset adapter / task adapter，让 AMBER 能读取 `dataest-pipeline` 生成的 `sample_manifest.jsonl`。

这个 adapter 至少需要完成：

- 读取 pipeline 输出目录和 `sample_manifest.jsonl`
- 按 `status`、`budget_status`、`smoke_report` 过滤可用样本
- 将 `initial_mesh` 或 `coarse_mesh` 映射为 AMBER 输入
- 将 `final_target_mesh` 映射为 AMBER expert target
- 将 `error_indicator` 或 `stage_fields` 映射为 physics weight / physics feature
- 为新数据集新增 Hydra task config 和 run config

第二步是用 ABC 数据集生产数据。注意 pipeline 读取的是解压后的 STEP 文件目录，不是 ABC 原始 chunk 压缩包目录。推荐数据布局参考 `CONDITION_AWARE_DATA_PIPELINE.md`：

```text
D:\condition-aware-meshing-data\
  abc\
    chunks\    # 原始 ABC chunk 压缩包
    step\      # 解压后的 STEP 文件，供 pipeline 读取
```

第三步是把 pipeline 生成的数据送入物理加权 / 物理修正 AMBER 跑实验。推荐顺序是：

1. 先用 pipeline 生成小规模 ABC teacher dataset，确认 manifest、mesh、stage fields、smoke report 都正常。
2. 接入 AMBER adapter 后，先训练普通 AMBER baseline。
3. 再训练 physics-weighted imitation baseline。
4. 最后从 weighted checkpoint 继续训练 physics correction。

第四步是处理兼容问题。当前最可能遇到的问题包括：

- pipeline 的 manifest 格式不是 AMBER 原生 `SourceData` / `AmberData` 格式
- pipeline 的 `final_target_mesh`、`stage_fields` 和 AMBER 当前 sizing-field target 语义需要对齐
- ABC 几何规模、STEP 质量、网格失败样本会影响 teacher 生成稳定性
- 物理重要性字段需要从 pipeline 产物投影或转换到 AMBER 当前训练图节点
- Console / Mold 已验证的物理加权逻辑不能直接假设对 ABC 全部有效，需要重新做小规模验证

第五步是设计和原始 AMBER 论文的对比实验。原始 AMBER 没有物理加权和物理修正，但它仍然能输出 sizing field / mesh prediction，因此可以用同一套测试集和同一套 evaluator 做公平比较。建议至少保留三组：

- 原始 AMBER：不启用物理加权，不启用物理修正
- AMBER + physics-weighted imitation：只启用物理加权
- AMBER + physics-weighted imitation + physics correction：完整方法

主表应比较所有方法都能计算的最终输出指标，例如：

- 全域 sizing / mesh error
- `weighted_size_l2`
- `topk_high_importance_l2`
- `bucket_high_size_l2`
- `bucket_high_low_ratio`
- final mesh quality / budget success rate

physics correction 的内部指标，例如 `gate_mean`、高低重要区域 gate 对比、`|alpha * delta_phys|`、`expert_prior` vs `final_prediction`，适合放在机制分析或消融实验中，不应强行要求原始 AMBER 也具备这些内部量。

## 最小可执行路线

最小可执行路线可以按下面顺序推进：

1. 在 `dataest-pipeline` 中用少量 ABC STEP 跑通 `run_full_pipeline`。
2. 检查 `sample_manifest.jsonl`、`stage_fields`、`final_target_mesh` 和 `smoke_report`。
3. 在 AMBER 中新增 pipeline dataset adapter。
4. 用 pipeline 数据先跑原始 AMBER baseline。
5. 用同一批数据跑 physics-weighted imitation。
6. 从 weighted checkpoint 继续跑 physics correction。
7. 用同一 evaluator 比较原始 AMBER、加权 AMBER、物理修正 AMBER。

这条路线的核心目标是把研究问题从“在 Console / Mold 上手工构造物理重要性”推进到“在 ABC 真实几何和样本级物理条件上学习 condition-aware、budget-aware 的 teacher mesh 分配规律”。
