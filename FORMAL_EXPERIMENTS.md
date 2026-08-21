# Formal Mold-7000 experiment protocol

The executable freeze is `config/formal_experiment_plan.yaml`. It fixes the dataset fingerprint, method identities,
seeds, dependencies, metrics, budget rules, geometry-level statistics, and planned contrasts before any formal test
result is read.

Hard boundary: this protocol does not modify the AMBER architecture, forward equations, or loss. The stage-field
adapter projects `probe_points` directly to each current learner mesh. M3 consumes the final teacher indicator and is
therefore labeled oracle-only. M1-FT remains `method_id: M1`; it reloads base-M1 weights into an identical M1 model
and starts a fresh optimizer/scheduler for another 101 epochs.

Run from the `AMBER_neurips` conda environment:

```powershell
python run_formal_experiments.py validate
python run_formal_experiments.py train
python run_formal_experiments.py evaluate
python run_formal_experiments.py analyze
```

`train` never invokes the test split. `evaluate` refuses to start until every preregistered fit has a complete
`training_summary.json` and `checkpoints/last.ckpt`. It then exports all test predictions, the M4 same-checkpoint
expert-only predictions, the preregistered within-geometry condition-shuffle diagnostics, R0-Initial/R1-Teacher, and
offline PDE results. Mesh/PDE failures remain rows and the primary success definition is their conjunction.

The primary PDE quantity is discrepancy relative to the frozen strong reference, not ground truth. Energy/H1 is not
reported. Legacy sizing columns ending in `_size_l2` are mean squared errors and have explicit `_size_mse` aliases.
