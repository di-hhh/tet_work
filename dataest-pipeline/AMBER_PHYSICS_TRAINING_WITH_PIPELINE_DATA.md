<!-- Generated at 2026-06-22 22:32:57 +08:00 (Asia/Shanghai) -->

# 用数据 Pipeline 训练物理修正 AMBER 的方案

本文档说明如何把本仓库的 condition-aware 数据生产 pipeline，与 AMBER 的 `物理修正模仿` 分支结合起来。

目标不是在这里重新实现训练代码，而是把两者的职责、联系、接入步骤和预期收益讲清楚。

## 1. 两边分别负责什么

## 1.0 当前项目与数据路径

在你当前机器上，建议按下面的方式理解两套资产：

- 项目代码根目录：`D:\condition-aware-meshing`
- 数据集根目录：`D:\condition-aware-meshing-data`

因此：

- 本仓库负责代码与配置
- 外部数据目录负责 STEP、ABC、Console、Mold 等大体量数据

对于本方案来说，最重要的是：

> pipeline 生成的数据集输出在项目侧管理，而原始几何数据可以放在外部数据根目录中，通过配置文件传给 CLI。

### 1.1 本仓库的数据 pipeline 负责什么

本仓库的 `condition_aware_dataset_generation` 子系统负责：

- 从真实 STEP / CAD 几何出发
- 采样多个物理条件
- 生成 condition-aware teacher mesh
- 输出 AMR 轨迹、误差、尺寸场、budget 状态和 final mesh 诊断

参考：

- [CONDITION_AWARE_DATA_PIPELINE.md](D:/condition-aware-meshing/CONDITION_AWARE_DATA_PIPELINE.md)

### 1.2 `物理修正模仿` 分支负责什么

你提供的 AMBER 分支 `物理修正模仿`，从分支 README 和配置来看，已经在原始 AMBER 上做了两类扩展：

1. **weighted imitation**
   - 面向 `console` / `mold`
   - 用线弹性参考场计算物理重要性
   - 用重要性加权 imitation loss

2. **physics correction branch**
   - 保留 expert prior head
   - 新增 physics correction head
   - 新增 gate head
   - 学习 `expert prior + gated physics residual`

我们确认到的分支入口包括：

- `python main.py +_runs/amber=amber_console_weighted`
- `python main.py +_runs/amber=amber_mold_weighted`
- `python main.py +_runs/amber=amber_console_physics_correction_codex`
- `python main.py +_runs/amber=amber_mold_physics_correction_codex`

分支中和这件事直接相关的代码位置包括：

- `src/algorithm/util/console_mold_reference.py`
- `src/algorithm/util/linear_elasticity_reference_codex.py`
- `src/algorithm/util/fem_imitation_weights.py`
- `src/algorithm/loss/amber_loss.py`
- `src/algorithm/architecture/physics_correction_branch_codex.py`
- `src/algorithm/dataloader/amber_data.py`
- `src/tasks/dataset_preparator.py`

## 2. 两者为什么适合结合

现在的关系可以很直接地理解为：

- 数据 pipeline 负责 **生成 teacher**
- 物理修正 AMBER 分支负责 **学习 teacher**

原始 AMBER 或你当前分支里的 weighted imitation，重点还是在“如何利用已有 expert 数据更好地学”。  
而我们现在这个 pipeline 新增的是：

- 更真实的几何来源
- 更丰富的 condition
- 更强的 final mesh 监督
- 更明确的 budget 约束
- 更可诊断的 hotspot 分配

因此两者结合后的总体目标是：

> 让物理修正 AMBER 不只是学“已有 expert demo”，而是学“从真实几何和真实条件出发生成的 condition-aware teacher”。

## 3. 结合后，能获得原始 AMBER 不具备的什么能力

这是最关键的一段。

## 3.1 同一几何在不同 condition 下得到不同 target

原始 AMBER 的主线更偏向“在某个任务分布上学习 mesh/sizing 规律”。  
我们现在的 pipeline 会对同一 geometry 采多个 condition，并为每个 condition 单独生成 target mesh。

这意味着结合后，模型可以学习：

> 几何相同，但边界条件、载荷、材料或 source 变化时，mesh allocation 应该如何变化。

这是原始 AMBER 不具备或不强调的 condition-aware supervision。

## 3.2 学到 budget-aware 的资源分配，而不是只学局部细化趋势

本仓库 teacher 不是简单“哪里误差大就细一点”，而是显式受：

- `minimum_viable_budget`
- `desired_budget`
- `hard_max_budget`

以及 cheap budget growth 的约束。

结合后，模型学习到的不是纯局部指标，而是：

> 在给定总预算下，怎么把单元数优先分配到真正重要的位置。

这比原始 AMBER 的“单步 sizing imitation”更接近生产目标。

## 3.3 可用 final mesh 结果反向监督 physics weighting 和 correction

你当前分支里的 weighted imitation 主要依赖预先构造的物理重要性。  
而本仓库 pipeline 还能提供：

- `final_target_mesh`
- `error_indicator`
- `stage_fields`
- `final_allocation_diagnostics`

所以结合后不只是“loss 更关注高重要区”，而是：

> 模型可以直接对着最终 teacher mesh 的资源分配结果学习，验证物理加权和物理修正是否真的把最终网格推向了正确方向。

## 3.4 从 deterministic surrogate physics 升级到 sample-specific teacher physics

你当前分支中，`console` / `mold` 的 weighted imitation 已经比原始 AMBER 多了线弹性参考。  
但它仍然主要是固定的参考构造逻辑。

本仓库 pipeline 则能提供 sample-specific 的：

- condition spec
- PDE family
- source / load / traction
- stage-wise importance field
- final target mesh

这意味着未来 physics correction 可以从“固定工程载荷模板”升级到：

> 随样本条件变化的 teacher physics supervision。

## 4. 当前最推荐的结合方式

我建议把结合分成三个层次，从最稳妥的开始。

## 4.1 第一层：把 pipeline 数据作为新的 expert dataset

这是最小改动、最稳妥的方式。

思路：

- 用 pipeline 生成 dataset
- 把 `final_target_mesh` 视为新的 expert target
- 用 `initial_mesh` 或 `coarse mesh` 作为 learner 的输入起点
- 让 AMBER 先学会模仿这个更强的 teacher

这一层先不强求使用所有中间场，只先把“teacher 更强”这件事接上。

### 建议使用的字段

从 pipeline 输出目录下的 `manifests/sample_manifest.jsonl` 或同结构输出中读取：

- `geometry_artifact_paths.coarse_mesh_path`
- `initial_mesh_path`
- `final_target_mesh_path`
- `condition_spec`
- `pde_family`
- `budget`

其中：

- `initial_mesh_path`：最自然的 learner 输入 seed
- `final_target_mesh_path`：最自然的 expert target

## 4.2 第二层：把 pipeline 的物理/重要性中间场接到 weighted imitation

这是和你当前 weighted imitation 分支最契合的一层。

当前分支中的 weighted imitation 已经支持：

- 节点重要性
- 加权 loss
- top-k / bucket 诊断

而本仓库 teacher 侧已经有：

- `optional_error_indicator_path`
- `optional_stage_field_path`
- `teacher_metadata.final_allocation_diagnostics`
- `teacher_metadata.adaptive_error_history`

这意味着你可以把 weighted imitation 的权重来源，从“分支当前的参考线弹性缓存”，逐步扩展成：

1. `error_indicator`
2. `stage_fields['s_pde_raw']`
3. `stage_fields['h_pde_only']` 或其反比尺度
4. `stage_fields['importance']`（若在适配器中显式导出）
5. 最终 hotspot mask / final allocation summary

这一层的意义是：

> weighted imitation 不再只依赖预先手工定义的 Console/Mold 参考物理，而是可以直接利用 teacher 生成过程中的样本级物理重要性。

## 4.3 第三层：让 physics correction branch 学 sample-specific residual

这是最强的一层，也是最能体现你这条分支价值的一层。

你当前分支的 physics correction 本质是：

- `delta_expert` 先给出 expert prior
- `delta_phys` 再给出 physics residual
- `alpha` 控制 correction 强度

如果接入 pipeline 数据，这个 residual 的学习目标就不再只是一般性的“物理偏好”，而可以变成：

> 在给定几何、给定 condition、给定初始 mesh 的前提下，相对 expert prior 还需要怎样修正，才能逼近真正的 condition-aware teacher target。

这会让 `physics correction branch` 从“泛泛的物理残差修正”，变成“针对 teacher 的条件残差修正”。

## 5. 当前仓库与 AMBER 分支之间，还缺哪一层

这里要说实话：**目前还没有现成的即插即用训练桥接层**。

也就是说，今天仓库里已经有：

- 数据生产端
- AMBER 物理修正训练端

但还差中间这层：

> `sample_manifest / target_mesh / stage_fields -> AMBER SourceData / AmberData / DatasetPreparator`

这层适配器是接下来最值得补的地方。

## 6. 推荐的适配器设计

我建议新增一个新的 task / dataset 适配层，而不是硬改原始 Console/Mold 读取逻辑。

### 6.1 新增一个 pipeline dataset preparator

建议在 AMBER 分支里新增类似下面的入口：

- `src/tasks/pipeline_condition_aware_dataset_preparator.py`

职责：

- 读取 pipeline 的 `sample_manifest.jsonl`
- 为 train / val / test 过滤样本
- 加载 `initial_mesh`、`target_mesh`、`stage_fields`
- 构造 AMBER 所需的 `SourceData`
- 按当前分支的数据接口生成 `AmberData`

### 6.2 新增 task config

建议在 AMBER 分支中新增 task 配置，例如：

- `console_teacher_pipeline`
- `mold_teacher_pipeline`

配置项至少包括：

- `pipeline_output_root`
- `manifest_name`
- `input_mesh_mode`
  - `initial_mesh`
  - `coarse_mesh`
- `target_mode`
  - `final_target_mesh`
  - `stage_field_target`
- `physics_weight_source`
  - `pipeline_indicator`
  - `pipeline_stage_field`
  - `legacy_console_mold_reference`

### 6.3 推荐字段映射

最实用的一版映射如下：

| Pipeline 资产 | 在 AMBER 训练中的角色 |
| --- | --- |
| `coarse_mesh_path` | 几何参考 / 可选输入 seed |
| `initial_mesh_path` | 默认 learner 输入 seed |
| `final_target_mesh_path` | expert target |
| `optional_error_indicator_path` | weighted imitation 权重源之一 |
| `optional_stage_field_path` | stage supervision / physics feature / weight source |
| `condition_spec` | condition embedding 或元数据字段 |
| `teacher_metadata.final_allocation_diagnostics` | 样本筛选与 curriculum 信号 |
| `status` / `budget_status` | 样本可用性过滤 |

## 7. 推荐训练流程

下面这套流程最稳。

### Step 1：先用本仓库生成 teacher 数据

例如：

```powershell
python condition_aware_dataset_generation.py run_full_pipeline --config config/condition_aware_dataset_generation/smoke_console_layered.yaml
```

或者你自己的全量配置：

```powershell
python condition_aware_dataset_generation.py run_full_pipeline --config <your_pipeline_config>.yaml
```

推荐把 `<your_pipeline_config>.yaml` 中的几何根目录指向外部数据目录，例如：

- `D:/condition-aware-meshing-data/console`
- `D:/condition-aware-meshing-data/mold`
- `D:/condition-aware-meshing-data/abc/step`

注意：

- ABC 的 chunk 压缩包目录不是 teacher pipeline 直接读取的输入
- pipeline 真正读取的是解压后的 STEP 目录

### Step 2：用 `sample_manifest` 做一次样本过滤

建议优先保留：

- `status != failed`
- `budget_status in {success_budget_closed, success_near_desired_budget, success_partial_under_budget}`
- final mesh 质量过关的样本

更稳妥的初版可以优先保留：

- `success_budget_closed`
- `success_near_desired_budget`

并基于 smoke verdict 进一步做高质量子集。

### Step 3：先训练 weighted baseline

先让 AMBER 学会模仿 pipeline teacher，再加 physics correction。

在 AMBER 分支中，当前已有 weighted 训练入口是：

```bash
python main.py +_runs/amber=amber_console_weighted
python main.py +_runs/amber=amber_mold_weighted
```

接入 pipeline dataset 后，推荐变成“保留 weighted 训练入口，但把 task 切到新的 pipeline task”。

也就是说，逻辑上是：

- 保留 `amber_console_weighted` / `amber_mold_weighted`
- 把底层数据源从原生 Console/Mold expert dataset 改成 pipeline dataset adapter

### Step 4：再继续训练 physics correction

当前分支已支持从 weighted baseline checkpoint 继续：

```bash
python main.py +_runs/amber=amber_console_physics_correction_codex \
  algorithm.init_from_weighted_baseline_checkpoint=/abs/path/to/weighted_baseline.ckpt
```

接入 pipeline dataset 后，推荐继续使用同样思路：

1. 先训 weighted baseline
2. 再以该 checkpoint 初始化 physics correction
3. 保持 `expert prior + physics residual` 的渐进式训练方式

## 8. 结合后的推荐监督层次

如果只做最小可用版本，我建议监督层次按下面顺序加：

### 第一优先级

- final target mesh imitation

### 第二优先级

- weighted imitation using pipeline indicator / stage field

### 第三优先级

- stage field auxiliary supervision

### 第四优先级

- trajectory-based curriculum 或多步 imitation

这样做的原因是：

- final target mesh 是最直接的训练目标
- weighted imitation 是你现有分支最成熟的增强路径
- stage field 和 trajectory 很有价值，但可以晚一点再接，避免首版适配太重

## 9. 推荐文档中明确写清的现实边界

为了避免后面误解，我建议把下面这几点一直写清楚：

1. 当前仓库已经完成了 **teacher 数据生成**
2. `物理修正模仿` 分支已经完成了 **weighted imitation + physics correction learner**
3. 但两者之间还缺一个 **dataset adapter / task adapter**
4. 因此这不是“今天零改动就能直接训练”，而是“一条非常明确、工程量可控的接入路线”

## 10. 这套结合方案最终想达到什么

如果这条路线完全打通，最终得到的将不是原始 AMBER 意义上的普通 imitation learner，而是：

> 一个能够从真实几何和物理条件出发，学习 budget-aware、condition-aware mesh allocation，并通过 physics weighting 与 physics correction 强化高重要区域表现的 AMBER 变体。

相对原始 AMBER，它最核心的新能力是：

- 学同一几何下的多 condition mesh 差异
- 学 final teacher mesh 的预算分配规律
- 学样本级物理热点，而不是只学平均误差
- 用 physics correction 学“在 expert prior 之外还要补什么”

## 11. 当前已有命令与推荐命令

### 11.1 当前仓库已有的数据生成命令

```powershell
python condition_aware_dataset_generation.py run_full_pipeline --config config/condition_aware_dataset_generation/smoke_console_layered.yaml
python condition_aware_dataset_generation.py run_full_pipeline --config config/condition_aware_dataset_generation/smoke_mold_3d.yaml
```

### 11.2 `物理修正模仿` 分支当前已有训练命令

```bash
python main.py +_runs/amber=amber_console_weighted
python main.py +_runs/amber=amber_mold_weighted
python main.py +_runs/amber=amber_console_physics_correction_codex
python main.py +_runs/amber=amber_mold_physics_correction_codex
```

### 11.3 接入 pipeline dataset 后的推荐使用方式

原则上仍然是：

1. 先跑 weighted baseline
2. 再从 weighted checkpoint 继续 physics correction

只是底层数据读取从“原生 Console/Mold expert dataset”切换到“pipeline dataset adapter”。

## 12. 一句话结论

这两套系统的关系不是替代，而是前后衔接：

- 本仓库负责造更强的 teacher
- `物理修正模仿` 分支负责学更强的 teacher

两者结合后，最重要的增益不是“把 AMBER 再调一点点”，而是：

> 让 AMBER 从原本主要面向固定 expert 演示的 learned AMR 方法，升级为一个能够利用真实几何、真实条件和物理修正信号来学习 condition-aware mesh allocation 的训练体系。
