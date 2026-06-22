# AMBER

Code for [AMBER: Adaptive Mesh Generation by Iterative Mesh Resolution Prediction](http://arxiv.org/abs/2505.23663).

For the earlier [workshop version of AMBER](https://arxiv.org/abs/2406.14161), see the ```workshop``` branch.
# Getting Started

## Setting up the environment

### Mamba
This project uses [mamba](https://github.com/conda-forge/miniforge) / [conda](https://docs.conda.io/en/latest/) and pip for handling packages and dependencies.
To install mamba on Linux-like OSes use one of the commands below.

```
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

For Windows please see the documentation in the link above or use (not recommended).
```
conda install -c conda-forge mamba
```

Afterward, you should be able to install all requirements using the commands below:

```
# for cpu use
mamba env create -f ./env/environment-cpu.yaml

# for gpu use
mamba env create -f ./env/environment-cuda.yaml

# Activate environment
mamba activate AMBER_neurips

wandb login  # login into wandb
pre-commit install  # install pre-commit for uniform formatting
```

### Test the environment
Test if everything works by running an experiment:

```bash
python main.py +_runs=debug
```


## Data
As part of AMBER, we propose six datasets, namely
* Poisson
* Laplace
* Airfoil
* Beam
* Console
* Mold

Poisson and Laplace are dynamically created during runtime.
The other datasets are provided under ./data/.


## Creating an experiment

Experiments are configured and distributed via hydra. For this, the folder `config` contains
a number of `.yaml` files that describe the configuration of the task to run.
The folder `_runs` contains individual *experiments*, each of which is a separate `.yaml`.

You can start an experiment, such as `debug` (in a corresponding `_runs/test/debug.yaml`), locally with the following command:

```bash
python main.py +_runs/test=debug
```

You can similarly run experiments on a cluster using Slurm by choosing an appropriate `platform`, e.g.,

```bash
python main.py +_runs/test=debug +platform=default_platform
```

When running the same experiment multiple times, e.g., for different hyperparameters or changes in the code, you can
use the `_version` flag (defaults to 1) to specify a version number.
This will append a f"v{version}" to the experiment name in the wandb logging.
We similarly have an `idx` parameter (defaults to 1000) that can be used to specify a unique identifier for the
experiment. We use this to identify semantic groupings of experiments ("Baseline X with feature Y") across tasks


## Model checkpointing and loading

Each experiment will be logged in `outputs/hydra/training/${YYYY-MM-DD}/${exp_name}/${run_name}/{$seed}`.
The logs contain folders
* `wandb` for the Weights and Biases logging, which includes metrics and figures
* `checkpoints` for the model checkpoints

To load a model from a checkpoint for evaluation, you can use the `evaluation.py` script, i.e.,
```
python evaluation.py +_evaluations=debug
```

This will load the algorithm config corresponding to that of the *checkpoint* specified in `_test/debug.yaml`, and
then run the evaluation with task settings as described in the test config. This allows loading a given algorithm and
evaluating it on novel task setups.
The results are written to disk as a `.json` file in `outputs/hydra/evaluation/${exp_name}/${run_name}/{$seed}`.
Additionally, plots for the testing data are saved in the same directory.

## Weighted imitation

AMBER now supports an optional weighted imitation loss for the 3D `console` and `mold` datasets. The implementation remains intentionally minimal, but the default reference physics is now a complete 3D linear-static elasticity solve on the expert tetrahedral mesh instead of the earlier harmonic surrogate:

- a cached reference solve is prepared once on the expert tetrahedral mesh
- `console` uses a deterministic cantilever-like setup: one end of the longest geometric axis is fully fixed, and the opposite end receives a distributed surface traction
- `mold` uses a deterministic gate-loading setup: a pressure load is applied on the inlet-side boundary patch, and the farthest stable boundary patch is fully fixed
- both datasets use the same isotropic small-strain material model with `young_modulus = 1.0` and `poisson_ratio = 0.3`
- the default physical importance is the element strain energy density `psi = 0.5 * epsilon^T sigma`
- element importance is converted to node importance through a volume-weighted element-to-node average, then mapped to positive training weights with `log1p + quantile normalization + w_i = 1 + beta * normalized_importance_i`
- those weights are projected to intermediate meshes with the same geometry mapping path used for expert sizing labels
- the training loss becomes `sum(w_i * loss_i) / (sum(w_i) + eps)` when enabled
- evaluation no longer relies only on global unweighted L2; it additionally reports `weighted_size_l2`, `topk_high_importance_l2`, `bucket_low_size_l2`, `bucket_high_size_l2`, `bucket_high_low_ratio`, and `physics_weighted_projected_l2_error`
- training logs additionally report weight spread, top-20% weight concentration, effective-sample ratio, weight/label correlations, and projection distortion statistics

Current support is complete only for `console` and `mold`. Other datasets keep the original behavior by falling back to all-one weights. If a Console/Mold cache file is missing, the code either prepares it before training when `weighted_imitation.auto_prepare=True`, or falls back to all-one weights when `weighted_imitation.fallback_to_ones=True`. If a stale harmonic cache is encountered while linear elasticity is requested, it is regenerated automatically when `auto_prepare=True`.

## Model-Side Physics Correction

AMBER now also includes a minimal model-side physics correction path for the validated `console` and `mold` weighted-imitation setup:

- the shared graph encoder/backbone is unchanged
- the original sizing head stays the expert prior head
- a new physics correction head predicts a residual in the same pre-transform prediction space
- a new gate head predicts `alpha in [0, gate_max]`
- the final pre-transform prediction is `delta_total = delta_expert + alpha * delta_phys`
- the existing `prediction_transform`, residual semantics, positivity handling, and mesh-generation path are reused unchanged

Physics information no longer stays only in the loss: when `algorithm.enable_physics_correction_branch=True`, the existing projected node-level physical importance for `console` / `mold` is appended to the node features and the model learns an `expert prior + physics residual` decomposition.

Training keeps the successful weighted imitation main loss on the final prediction and adds two conservative extensions:

- `lambda_expert_aux * L_expert_aux` keeps the expert prior head anchored to the original imitation target
- `lambda_corr_reg * L_corr_reg` keeps `alpha * delta_phys` small during early training
- `correction_warmup_epochs` linearly ramps the usable correction strength from 0 to 1

Checkpoint continuation from the current weighted baseline is supported through `algorithm.init_from_weighted_baseline_checkpoint=/path/to/old.ckpt`. Matching parameters are copied directly, new heads are freshly initialized, and first-layer tensors with one extra physics feature column are adapted by copying old columns and leaving the new column inactive initially.

If physics features are unavailable during inference, the fallback is controlled by `algorithm.inference_missing_physics_fallback`:

- `gate_zero`: keep the branch instantiated but force the correction to zero
- `zero_feature`: provide a zero physics feature and still run the branch
- `disable_branch`: equivalent safe fallback that suppresses the correction path

This remains a minimal expert-prior-plus-residual extension. It is not a differentiable FE-in-the-loop model and does not solve a new physics problem at every training step.

Why global L2 alone is not enough:

- a weighted model can intentionally trade small errors in low-importance regions for better accuracy in high-importance regions
- in that case, global unweighted L2 may stay flat while the physically important region improves
- the new top-k and bucketed metrics are designed to expose that tradeoff explicitly instead of hiding it inside a single global average

The weighting strategy is configurable:

- `linear`: `w = 1 + beta * importance`
- `power`: `w = 1 + beta * importance^gamma`
- `binary_topk`: only the top-k important region gets `lambda_high`
- `ternary_quantile`: low / mid / high importance regions receive `1 / lambda_mid / lambda_high`

AMBER also supports a minimal `two_stage_importance_finetune` workflow:

- stage 1 uses the baseline or normal weighted imitation objective
- stage 2 resumes from a checkpoint and switches the loss reduction to a more aggressive high-importance weighting mode
- if no checkpoint is supplied, the same run can switch to stage 2 during the last `stage2_epochs`

To precompute the reference caches explicitly, run:

```powershell
python precompute_console_mold_weights.py --datasets console mold
```

This regenerates the reference displacement/importance caches under `data/weighted_imitation/`. The current setup is still a geometry-driven simplified engineering load case, not a full industrial simulation workflow with sample-specific materials, supports, and process conditions.

To launch the weighted AMBER experiments through the existing Hydra run presets, use:

```bash
python main.py +_runs/amber=amber_console_weighted
python main.py +_runs/amber=amber_mold_weighted
```

To launch the model-side physics correction runs on top of the weighted baseline presets, use:

```bash
python main.py +_runs/amber=amber_console_physics_correction_codex
python main.py +_runs/amber=amber_mold_physics_correction_codex
```

To continue from an existing weighted baseline checkpoint, override:

```bash
python main.py +_runs/amber=amber_console_physics_correction_codex \
  algorithm.init_from_weighted_baseline_checkpoint=/abs/path/to/weighted_baseline.ckpt
```

Enable it through Hydra config, for example:

```yaml
algorithm:
  weighted_imitation:
    enabled: true
    datasets: [console, mold]
    weight_source_mode: console_mold_reference
    reference_physics_type: linear_elasticity
    importance_metric: strain_energy_density
    cache_dir: data/weighted_imitation
    auto_prepare: false
    metric_use_physics_weights: true
    young_modulus: 1.0
    poisson_ratio: 0.3
    weight_mode: linear
    beta: 1.0
    gamma: 2.0
    lambda_high: 4.0
    lambda_mid: 2.0
    topk_percent: 0.2
    bucket_count: 5
    epsilon: 1.0e-8
    normalization_lower_quantile: 0.05
    normalization_upper_quantile: 0.95
    clip_min: 1.0
    clip_max: 10.0
    stage2_enable: false
    stage2_epochs: 20
    stage2_weight_mode: binary_topk
    stage2_high_importance_only: true
    fallback_to_ones: true
```

To generate a minimal comparison plan or run a cache-level mode sweep, use:

```bash
python compare_weighted_imitation_codex.py --dataset console
python compare_weighted_imitation_codex.py --dataset mold
```

To continue from a baseline checkpoint with stage-2 importance fine-tuning, pass a checkpoint path:

```bash
python main.py +_runs/amber=amber_console_weighted \
  trainer.ckpt_path=/abs/path/to/checkpoints/last.ckpt \
  algorithm.weighted_imitation.stage2_enable=True
```

For a dedicated explanation of the available weighting modes, see [weighted_imitation_weight_modes_codex.md](./weighted_imitation_weight_modes_codex.md).

How to read the new results:

- if global unweighted L2 does not improve but `topk_high_importance_l2` or `bucket_high_size_l2` improves, the model is reallocating accuracy toward physically important regions
- if weight spread and `top20_ratio` stay close to the all-one baseline, the weighting is probably too weak
- if projection diagnostics collapse strongly from `reference_q95` / `reference_top20_ratio` to `projected_q95` / `projected_top20_ratio`, the importance map is being flattened during projection
- if weight/label correlations are near zero, the current physical importance and the sizing target may be misaligned
