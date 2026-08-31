# Condition-Aware Teacher Pipeline

The teacher pipeline now treats the initial mesh as a moderately coarse seed, grows budget cheaply inside teacher generation, and reports where condition differences collapse.
The current production bottleneck is no longer "all teachers die during adaptive refinement"; it is "teacher targets survive only as severely under-budget partial meshes".
The goal is to start coarse, keep topology valid, and spend the mesh budget where the condition-specific indicator says it matters without rerunning full CAD meshing in every budget step.

## Why M0 Must Stay Moderately Coarse

`M0` is no longer treated as an almost-final geometric mesh.
It should only guarantee:

- valid topology
- preserved STEP boundary, holes, feature edges, and key connectivity
- a stable seed for the teacher loop
- no broad PDE-driven refinement in the coarse seed

It should not already spend the full teacher budget, but it must still be a CAD-faithful seed.
The teacher `initial_mesh` is therefore generated with the same AMBER-style uniform STEP meshing path by default: `max_initial_element_volume` is converted into a target edge length, Gmsh receives `characteristic_length_max` and `characteristic_length_min`, and the mesh is generated directly from the STEP geometry.
`initial_sizing_field_scale` globally enlarges this target edge length when a coarser seed is needed; values above `1.0` make the initial mesh coarser without switching to topology-only geometry constraints.
This matters because later local AMR only refines the current boundary; it cannot recover a STEP feature that was simplified away in `M0`.
The pipeline now enforces this with explicit caps such as:

- `initial_target_num_elements`
- `initial_target_num_surface_faces`
- `initial_max_nodes`
- `initial_max_dofs`
- `initial_max_runtime_seconds`
- `initial_max_budget_fraction`
- `initial_mesh_generation_mode`
- `initial_sizing_field_scale`
- `initial_sizing_field_retry_factor`

Each teacher record carries:

- `initial_num_elements`
- `initial_num_nodes`
- `initial_surface_faces`
- `initial_mean_edge_length`
- `initial_hole_boundary_segments`
- `initial_budget_fraction`
- `initial_is_too_dense`
- `initial_mesh_generation_mode`
- `initial_sizing_field_scale`
- `initial_requested_element_volume`
- `initial_preserve_feature_edges`
- `initial_geometry_constraint_mode`
- `initial_mesh_source`

If the seed already consumes too much of the target budget, the sample is rejected as `reject_bad_initial_mesh`.
If the dedicated coarse CAD seed cannot be generated, the pipeline can fall back to the preprocessing coarse mesh as the minimum viable topology-preserving seed.

## Why The Priority Shifted

The first failure mode was easy to spot:

- every 3D STEP teacher died in `adaptive_refinement`
- no usable samples were emitted

That is now replaced by a quieter failure mode:

- prescreen could say two conditions were separable
- teacher could emit a coarse `M0` plus one local refinement
- the final mesh could contain only a few hundred elements against a desired budget such as 12000
- condition differences could still be weaker on the final mesh than in the stage fields

The pipeline now fixes that by separating the sizing logic into stages:

1. `s_pde_raw`
2. `h_pde_only`
3. `h_after_geometry_fusion`
4. `h_after_budget_calibration`

This makes it visible whether the condition difference was weak from the PDE side, erased by geometry fusion, erased by budget calibration, or erased when the final mesh was actually built.

## Cheap Budget Growth

Between coarse `M0` and any expensive CAD-aware final remesh, teacher generation now runs a cheap budget growth loop.
The loop uses the current volume mesh and local refinement instead of full CAD remeshing for each budget step.
Each round:

- rebuilds the condition-aware sizing field on the current mesh
- compares current element size with `h_after_geometry_fusion`
- ranks elements by desired-size mismatch, PDE/hotspot importance, and low-importance protection
- refines a capped top fraction of elements
- refuses a step that would exceed `hard_max_budget`
- records element count, node count, budget ratios, hotspot fraction, `hotspot_size_ratio`, `allocation_gain`, and step time

The loop stops when it reaches near-desired budget, hits `hard_max_budget`, exhausts `budget_growth_max_steps`, sees diminishing returns, nears timeout, or approaches DOF/matrix caps.
`budget_growth_cad_cleanup_interval` can enable occasional CAD-aware cleanup, but the default smoke path keeps it at `0` so growth remains cheap.

## Layered Budgets

Budgets are now interpreted as three layers:

- `minimum_viable_budget`: below this, a final target is too coarse for supervision
- `desired_budget`: the preferred teacher scale
- `hard_max_budget`: the absolute cap

Every sample now records:

- `target_budget`
- `minimum_viable_budget`
- `desired_budget`
- `hard_max_budget`
- `actual_budget`
- `budget_ratio`
- `desired_budget_ratio`
- `minimum_viable_budget_ratio`
- `hard_max_budget_ratio`
- `lambda`
- `calibration_iters`
- `calibration_converged`
- `budget_growth`
- `budget_status`
- `budget_closure_limiter`

Budget status is explicit:

- `success_budget_closed`: final count is within calibration tolerance of `desired_budget`
- `success_near_desired_budget`: final count is close enough to desired for production smoke
- `success_partial_under_budget`: final count reached `minimum_viable_budget` but not near desired
- `fail_budget_growth_stalled`: growth stopped below minimum viable
- `fail_budget_growth_timeout`: growth timed out below minimum viable
- `fail_budget_hard_cap_exceeded`: final target would exceed the hard cap

Under-budget partial success is therefore no longer mixed with true budget closure.

If CAD calibration still fails, the report tells you whether the limiter was:

- `geometry_floor`
- `hotspot_floor`
- `mesher_behavior`

## Final Mesh Allocation Diagnostics

The final mesh, not only the stage field, is now diagnosed.
Each sample reports:

- `final_hotspot_size_ratio`: median hotspot element size divided by median low-error element size
- `final_hotspot_element_fraction`
- `final_hotspot_volume_fraction`
- `final_allocation_gain`: hotspot element fraction divided by hotspot volume fraction
- budget progress against minimum viable, desired, and hard max

The smoke report compares accepted conditions on common probe points for all four stages and on the final mesh.
For each stage it outputs:

- Pearson
- Spearman
- relative L2 difference
- finest-20% region Jaccard
- hotspot region Jaccard

The report also emits collapse diagnostics:

- `condition_difference_collapsed_after_fusion`
- `condition_difference_collapsed_after_budget`
- `condition_difference_collapsed_at_mesh_stage`

This lets you distinguish:

- weak PDE signal
- geometry fusion flattening
- budget calibration flattening
- final mesh execution flattening

Verdicts are final-mesh oriented:

- `PASS_STRONG`
- `PASS_WEAK`
- `FAIL_GLOBAL_OVERREFINE`
- `FAIL_PARTIAL_TOO_COARSE`
- `FAIL_CONDITION_COLLAPSE_AT_FINAL_MESH`
- `FAIL_TOO_EXPENSIVE`

## Timeout Wall Clock

Worker status files still contain worker-reported elapsed time, but timeout records now prefer the parent process wall clock.
Timeout failures distinguish:

- `parent_observed_elapsed_seconds`
- `worker_reported_elapsed_seconds`
- `parent_started_at`
- `parent_kill_at`
- `parent_finish_at`

Use `elapsed_seconds` as parent-observed wall clock for throughput analysis.

## Scalar Smoke Versus Elasticity Smoke

`scalar_elliptic` is the primary smoke layer.
It answers the main questions quickly:

- is the mesh still globally coarse outside hotspots
- are hotspots actually denser than the far field
- do two accepted conditions move the refinement region
- does the final mesh stay near the requested budget

`linear_elasticity` is a secondary layer, but it no longer has to use the full expensive reference solve in smoke.
With `elasticity_smoke_mode: cheap_reference` and `elasticity_smoke_reference_level: 0`, the teacher solves elasticity on the current mesh and builds a cheap hotspot indicator from displacement variation, traction-side proximity, and body-force scale.
This is lower fidelity, but it is enough to test whether different elasticity conditions move the hotspot.
If even the cheap path exceeds caps, the sample is rejected as too expensive and scalar smoke continues.

## How to run the new smoke pipeline

Console layered smoke:

```powershell
python condition_aware_dataset_generation.py run_full_pipeline --config config/condition_aware_dataset_generation/smoke_console_layered.yaml
```

Smoke outputs are written under `output/condition_aware_dataset_generation/<run_name>/` at the repository root.

The main outputs are:

- `teachers/<geometry>/<condition>/teacher_record.json`
- `teachers/<geometry>/<condition>/budgets/<budget>/stage_fields.npz`
- `samples/*.json`
- `reports/smoke_report.json`

## How to read the new report

The report is split into:

- `scalar_smoke`
- `elasticity_smoke`

Inside each geometry report, focus on:

- `sample_metrics`
- `pairwise_condition_metrics`
- `field_stage_pairwise_metrics`
- `collapse_diagnostics`
- `verdict`

Useful indicators:

- low `hotspot_size_ratio`: hotspot is finer than the far field
- high `allocation_gain`: elements are concentrated in the hotspot instead of everywhere
- low pairwise hotspot Jaccard on final sizes: different conditions move refinement to different places
- `success_budget_closed` versus `success_partial_under_budget`: true budget closure versus usable partial target
- a collapse flag after fusion, budget, or mesh stage: the pipeline is still flattening condition differences at that stage
