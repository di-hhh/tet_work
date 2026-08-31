# AGENTS.md

## 环境
在 AMBER_neurips 环境中，写代码、跑实验

**激活** conda 环境：
```powershell
conda activate AMBER_neurips
```
**禁止**跳过激活步骤、直接使用 AMBER_neurips 环境的解释器

**禁止**并行激活跑多个 AMBER_neurips 环境，只允许串行跑

## 目录结构
```text
tet_work/
|__amber/               # amber 模型源代码、数据适配器代码；拥有一个本地 Git 仓库
|   |__output/          # amber 模型训练过程的输出、模型权重 ckpt
|__dataest-pipeline/    # 数据生产线生产数据，供 amber 模型使用；拥有另一个本地 Git 仓库
|   |__output/          # 数据生产线输出的数据
...
```
**注意**：根目录 tet_work/ 内部的相对路径不变，但是根目录 tet_work/ 所在的实际位置可变
> 若根目录打包被复制到另一台机器上，实际位置是另一台机器上的绝对路径