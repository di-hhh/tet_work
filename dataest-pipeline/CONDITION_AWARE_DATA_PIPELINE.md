<!-- Generated at 2026-06-22 22:32:57 +08:00 (Asia/Shanghai) -->

# Condition-Aware 数据生产 Pipeline

本文档介绍本仓库新增的 `condition_aware_dataset_generation` 子系统。它的目标不是训练模型，而是从真实几何和物理条件出发，稳定生成可用于训练的 teacher mesh 数据集。

## 1. 这条 pipeline 解决什么问题

原始 AMBER 更偏向“给定现成任务数据，训练/评估 learned AMR 方法”。本仓库新增的数据生产 pipeline 则负责：

- 读取真实几何，尤其是 STEP/CAD 几何
- 预处理几何并生成 `coarse mesh`
- 为同一几何采样多个物理条件
- 对每个 `geometry + condition + budget` 生成 teacher mesh
- 记录 AMR 轨迹、误差、尺寸场、预算状态和最终质量诊断
- 导出可直接消费的 `manifest / records / smoke report`

一句话概括：

> 它是一台面向 condition-aware meshing 的 teacher 数据生产机。

## 2. 代码入口

主入口文件：

- [condition_aware_dataset_generation.py](D:/condition-aware-meshing/condition_aware_dataset_generation.py)

核心模块：

- [cli.py](D:/condition-aware-meshing/src/condition_aware_dataset_generation/cli.py)
- [pipeline.py](D:/condition-aware-meshing/src/condition_aware_dataset_generation/pipeline.py)
- [prescreen.py](D:/condition-aware-meshing/src/condition_aware_dataset_generation/prescreen.py)
- [runtime_controls.py](D:/condition-aware-meshing/src/condition_aware_dataset_generation/runtime_controls.py)
- [records.py](D:/condition-aware-meshing/src/condition_aware_dataset_generation/records.py)
- [teacher.py](D:/condition-aware-meshing/src/condition_aware_dataset_generation/teacher_generation/teacher.py)

子模块划分：

- `geometry_sources/`：几何来源
- `geometry_preprocessing/`：几何预处理、feature 提取、coarse mesh
- `condition_sampling/`：条件采样
- `teacher_generation/`：teacher 生成、PDE 求解、AMR、budget growth
- `serialization/`：manifest / dataset reader / 布局约定
- `smoke_analysis.py`：最终质量分析与报告

## 2.1 本机路径约定

当前项目在你的机器上的实际路径约定是：

- 项目根目录：`D:\condition-aware-meshing`
- 数据集根目录：`D:\condition-aware-meshing-data`

这意味着：

- 代码在 `D:\condition-aware-meshing`
- 大体量数据建议放在 `D:\condition-aware-meshing-data`

文档中如果出现仓库内的相对示例路径，例如 `data/console`、`data/mold`、`data/abc/step`，它们表示的是一种默认布局示例，不代表代码必须把数据硬编码放在仓库内部。

## 2.2 路径是如何配置的

这条 pipeline 的数据路径原则上**不应该在代码里硬编码**。

当前实现里，路径主要由配置驱动：

- 通过 `--config` 选择 YAML 配置文件
- 在 YAML 中设置 `geometry_source.root`
- 通过 CLI 通用参数覆盖 `output_root`、`workers`、`overwrite`、`limit_geometries`

当前 CLI 直接支持的参数在 [cli.py](D:/condition-aware-meshing/src/condition_aware_dataset_generation/cli.py) 中可以看到，主要包括：

- `--config`
- `--output-root`
- `--workers`
- `--overwrite`
- `--limit-geometries`

也就是说，几何数据根目录通常是由你传入的配置文件控制，而不是写死在代码里。

## 3. 整体流程

这条 pipeline 的完整链路是：

1. `ingest_geometries`
2. `preprocess_geometries`
3. `sample_conditions`
4. `prescreen_conditions`
5. `generate_teacher_targets`
6. `build_dataset_manifest`
7. `build_smoke_report`

也可以直接一步跑完：

1. `run_full_pipeline`

## 4. 每一步做什么

### 4.1 ingest geometries

作用：

- 扫描几何源目录
- 生成 `GeometryRecord`
- 建立 `geometry_id`

输入通常是本地目录中的 `*.step` 或仓库支持的其他几何描述。

### 4.2 preprocess geometries

作用：

- 读取几何
- 估计 bounding box / centroid / principal axes
- 提取边界 patch、孔洞、sharp edge、feature edge
- 生成一张几何参考 `coarse mesh`

这里的 `coarse mesh` 有两个用途：

- 作为几何参考资产保存
- 在某些模式下直接作为 teacher 的 `initial mesh`

### 4.3 sample conditions

作用：

- 对同一几何采样多个 PDE condition
- 目前重点支持：
  - `scalar_elliptic`
  - `linear_elasticity`
- 同时采样 budget

这一步是 condition-aware 的核心，因为它让“同一几何在不同物理条件下应该得到不同 target mesh”成为可能。

### 4.4 prescreen conditions

作用：

- 在完整 teacher 生成前先做便宜的风险和价值评估
- 判断这个 condition 是否值得继续花 teacher 成本

如果 `enable_prescreen: false`，则默认全部放行。

### 4.5 generate teacher targets

这是整条线最核心的一步。每个样本大致会经历：

1. 生成 `initial mesh`
2. 在当前 mesh 上求解 PDE
3. 计算 indicator / importance
4. 做若干步 AMR
5. 构建 staged sizing fields
   - `s_pde_raw`
   - `h_pde_only`
   - `h_after_geometry_fusion`
   - `h_after_budget_calibration`
6. 做 cheap budget growth
7. 输出最终 `target_mesh`

### 4.6 build dataset manifest

作用：

- 把所有 geometry / preprocess / condition / prescreen / teacher / sample 记录统一汇总
- 形成后续训练和统计最方便读取的 `jsonl`

### 4.7 build smoke report

作用：

- 从 final mesh 角度评估 teacher 质量
- 输出 budget 状态、热点分配、condition separability、verdict

## 5. 当前 teacher 生成的关键机制

### 5.1 initial mesh

当前支持多种 `initial_mesh_generation_mode`，常见的有：

- `amber_uniform`
- `preprocess_coarse`

其中：

- `amber_uniform` 更偏吞吐优先
- `preprocess_coarse` 更偏几何一致性优先

### 5.2 layered budget

teacher 预算不是单一 target，而是三层：

- `minimum_viable_budget`
- `desired_budget`
- `hard_max_budget`

对应的 budget 状态包括：

- `success_budget_closed`
- `success_near_desired_budget`
- `success_partial_under_budget`
- `fail_budget_growth_stalled`
- `fail_budget_growth_timeout`
- `fail_budget_hard_cap_exceeded`

### 5.3 cheap budget growth loop

这是让 3D STEP teacher 从几千单元长到几万或十几万单元的关键。

思路是：

- 不每轮都 full CAD remesh
- 在当前 volume mesh 上 local refine
- 按 hotspot / desired-size mismatch / low-importance protection 分配新增单元
- 在受控成本下逐步逼近 `desired_budget`

### 5.4 final allocation diagnostics

最终不是只看中间 field，而是直接看 final mesh 本身的资源分配效果。关键指标包括：

- `final_hotspot_size_ratio`
- `final_hotspot_element_fraction`
- `final_hotspot_volume_fraction`
- `final_allocation_gain`

## 6. 主要产物

每个运行输出都放在：

- `output/condition_aware_dataset_generation/<run_name>/`

典型目录结构如下：

```text
output/condition_aware_dataset_generation/<run_name>/
  geometries/
  conditions/
  prescreens/
  teachers/
  manifests/
  reports/
```

最重要的文件有：

- `geometries/<geometry_id>/coarse_mesh.vtk`
- `teachers/<geometry_id>/<condition_id>/initial_mesh.vtk`
- `teachers/<geometry_id>/<condition_id>/trajectory/mesh_step_*.vtk`
- `teachers/<geometry_id>/<condition_id>/budgets/<budget>/target_mesh.vtk`
- `teachers/<geometry_id>/<condition_id>/budgets/<budget>/error_indicator.npy`
- `teachers/<geometry_id>/<condition_id>/budgets/<budget>/stage_fields.npz`
- `teachers/<geometry_id>/<condition_id>/teacher_record.json`
- `manifests/sample_manifest.jsonl`
- `manifests/teacher_records.jsonl`
- `reports/smoke_report.json`

## 7. manifest 里有什么

当前可以直接读样本 manifest：

- [dataset_reader.py](D:/condition-aware-meshing/src/condition_aware_dataset_generation/serialization/dataset_reader.py)

最小使用方式：

```python
from src.condition_aware_dataset_generation.serialization import ConditionAwareSampleDataset

dataset = ConditionAwareSampleDataset(
    output_root="output/condition_aware_dataset_generation/smoke_console_layered"
)

print(len(dataset))
print(dataset[0]["sample_id"])
```

`sample_manifest.jsonl` 中每条记录至少包含：

- `geometry_id`
- `condition_id`
- `pde_family`
- `budget`
- `condition_spec`
- `geometry_artifact_paths`
- `initial_mesh_path`
- `optional_intermediate_mesh_paths`
- `final_target_mesh_path`
- `optional_error_indicator_path`
- `optional_stage_field_path`
- `teacher_metadata`
- `status`

## 8. 常用配置

配置目录在：

- [config/condition_aware_dataset_generation](D:/condition-aware-meshing/config/condition_aware_dataset_generation)

常用配置文件：

- [default.yaml](D:/condition-aware-meshing/config/condition_aware_dataset_generation/default.yaml)
- [smoke_console_layered.yaml](D:/condition-aware-meshing/config/condition_aware_dataset_generation/smoke_console_layered.yaml)
- [smoke_mold_3d.yaml](D:/condition-aware-meshing/config/condition_aware_dataset_generation/smoke_mold_3d.yaml)

重点配置组：

- `geometry_source`
- `preprocessing`
- `condition_sampling`
- `prescreen`
- `teacher`
- `smoke`
- `split`

其中最关键的路径项是：

- `geometry_source.root`
- `output_root`

例如，如果你希望读取外部数据根目录下的 Console 数据，可以在 YAML 中写成：

```yaml
geometry_source:
  name: local_directory
  source_name: console_local
  root: D:/condition-aware-meshing-data/console
  patterns:
    - "*.step"
  recursive: true
```

同理，Mold 可以写成：

```yaml
geometry_source:
  name: local_directory
  source_name: mold_local
  root: D:/condition-aware-meshing-data/mold
  patterns:
    - "*.step"
  recursive: true
```

## 8.1 ABC 数据集与 chunk 位置

这里单独说明一下你刚才关心的 ABC 问题。

### pipeline 实际读取哪一层

当前 pipeline **不直接读取压缩 chunk 文件**。  
它实际读取的是：

- 解压后的 STEP 目录

也就是说，ABC chunk 只是原始存储格式；在进入 `preprocess` 和 `teacher generation` 之前，需要先把 STEP 文件解压出来。

### 推荐目录布局

在你当前机器上，建议这样放：

```text
D:\condition-aware-meshing-data\
  abc\
    chunks\        # 原始 chunk 压缩包
    step\          # 解压后的 STEP 文件目录，供 pipeline 真正读取
```

推荐约定：

- 原始 chunk：`D:\condition-aware-meshing-data\abc\chunks\`
- 解压后 STEP：`D:\condition-aware-meshing-data\abc\step\`

### 配置方式

如果使用 ABC，配置文件里的 `geometry_source.root` 应该指向：

- `D:/condition-aware-meshing-data/abc/step`

而不是 chunk 压缩包目录。

例如：

```yaml
geometry_source:
  name: abc_dataset
  source_name: abc_extracted
  root: D:/condition-aware-meshing-data/abc/step
  patterns:
    - "*.step"
  recursive: true
```

仓库里的示例文件是：

- [abc_extracted_example.yaml](D:/condition-aware-meshing/config/condition_aware_dataset_generation/abc_extracted_example.yaml)

它表达的是“读取解压后的 STEP 根目录”这个意思，只是示例里用的是仓库内相对路径。

## 9. 启动命令

### 9.1 一步跑完整 pipeline

```powershell
python condition_aware_dataset_generation.py run_full_pipeline --config config/condition_aware_dataset_generation/smoke_console_layered.yaml
```

### 9.2 逐阶段运行

```powershell
python condition_aware_dataset_generation.py ingest_geometries --config config/condition_aware_dataset_generation/default.yaml
python condition_aware_dataset_generation.py preprocess_geometries --config config/condition_aware_dataset_generation/default.yaml
python condition_aware_dataset_generation.py sample_conditions --config config/condition_aware_dataset_generation/default.yaml
python condition_aware_dataset_generation.py prescreen_conditions --config config/condition_aware_dataset_generation/default.yaml
python condition_aware_dataset_generation.py generate_teacher_targets --config config/condition_aware_dataset_generation/default.yaml
python condition_aware_dataset_generation.py build_dataset_manifest --config config/condition_aware_dataset_generation/default.yaml
python condition_aware_dataset_generation.py build_smoke_report --config config/condition_aware_dataset_generation/default.yaml
```

### 9.3 常见覆盖参数

```powershell
python condition_aware_dataset_generation.py run_full_pipeline `
  --config config/condition_aware_dataset_generation/smoke_mold_3d.yaml `
  --output-root output/condition_aware_dataset_generation/my_run `
  --workers 2 `
  --overwrite `
  --limit-geometries 10
```

### 9.4 已验证过的典型 smoke

Console：

```powershell
python condition_aware_dataset_generation.py run_full_pipeline --config config/condition_aware_dataset_generation/smoke_console_layered.yaml
```

Mold：

```powershell
python condition_aware_dataset_generation.py run_full_pipeline --config config/condition_aware_dataset_generation/smoke_mold_3d.yaml
```

### 9.5 使用外部数据根目录的推荐方式

由于你的数据集根目录在 `D:\condition-aware-meshing-data`，推荐做法是：

1. 复制一份配置文件
2. 把 `geometry_source.root` 改成外部数据路径
3. 再通过 `--config` 启动

例如：

```powershell
python condition_aware_dataset_generation.py run_full_pipeline `
  --config config/condition_aware_dataset_generation/smoke_console_layered.yaml `
  --output-root output/condition_aware_dataset_generation/console_external_run `
  --workers 2 `
  --overwrite
```

其中实际读取哪个数据目录，取决于该 YAML 文件中的 `geometry_source.root` 设置。

## 10. 这条 pipeline 相对原始 AMBER 新增了什么

相对原始 AMBER 的训练/评估主线，本仓库新增的是一整套“teacher 数据生产链”：

- STEP/CAD 几何入口
- 几何预处理与 coarse mesh
- condition-aware 条件采样
- teacher mesh 生成
- layered budget
- cheap budget growth
- AMR 轨迹与误差历史
- final mesh allocation diagnostics
- manifest / smoke report / runtime taxonomy

所以它的角色不是替代 AMBER，而是：

> 为 AMBER 或其变体提供更丰富、更真实、更可控的 teacher 数据。

## 11. 当前局限

这条 pipeline 已经能稳定产出 teacher 数据，但仍有一些边界：

- elasticity 路径仍比 scalar 更贵
- 高预算下 final mesh 质量与吞吐之间仍需取舍
- `sample_manifest` 目前是通用资产格式，不是原始 AMBER 训练脚本可直接即插即用的原生 task 格式

这也是为什么后续通常需要一层“训练适配器”把 pipeline 产物接入具体 learner。
