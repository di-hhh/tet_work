# AMBER_neurips environment freeze

Freeze date: 2026-08-20.  Platform: `win-64`.

The files in this directory describe the environment named by the repository
`AGENTS.md`; the older `environment-cuda.yaml` is not the formal experiment
lock.

- `AMBER_neurips-win64-conda-explicit.txt`: exact conda artifacts.
- `AMBER_neurips-pip-only.txt`: packages recorded by conda as `pypi` installs.
- `AMBER_neurips-pip-freeze.txt`: unmodified `python -m pip freeze` audit output.
- `AMBER_neurips-pip-normalized.txt`: version-only inventory for comparison.
- `AMBER_neurips-pip-portable.txt`: ordinary PyPI subset used by the recreation script.
- `environment-summary.json`: critical runtime and CUDA/PyG compatibility facts.

Recreate the Windows base environment with:

```powershell
conda create -n AMBER_neurips --file env/locks/AMBER_neurips-win64-conda-explicit.txt
conda activate AMBER_neurips
```

The explicit conda file intentionally cannot contain packages whose conda
channel is `pypi`.  `recreate_AMBER_neurips.ps1` installs the CUDA/PyG wheels
from their matching wheel indexes and then the portable PyPI subset.  The
authoritative versions are in `AMBER_neurips-pip-only.txt`; in particular this
freeze uses PyTorch `2.10.0+cu128` and the `pt210cu128` PyG extensions.  The
complete recreation command is:

```powershell
powershell -ExecutionPolicy Bypass -File env/locks/recreate_AMBER_neurips.ps1
```

Finally run:

```powershell
python env/verify_AMBER_neurips.py --require-cuda
```

Do not use `pip freeze`'s local `file:///...` build URLs as installation
sources; that file is retained as the required audit snapshot, while the
normalized and pip-only inventories are the portable version records.
