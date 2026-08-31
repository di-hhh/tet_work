$ErrorActionPreference = "Stop"

$lockRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$amberRoot = (Resolve-Path (Join-Path $lockRoot "..\..")).Path

conda create -n AMBER_neurips --file (Join-Path $lockRoot "AMBER_neurips-win64-conda-explicit.txt")
conda run -n AMBER_neurips python -m pip install `
  --index-url https://download.pytorch.org/whl/cu128 `
  torch==2.10.0+cu128 torchvision==0.25.0+cu128
conda run -n AMBER_neurips python -m pip install `
  --find-links https://data.pyg.org/whl/torch-2.10.0+cu128.html `
  torch-scatter==2.1.2+pt210cu128 `
  torch-cluster==1.6.3+pt210cu128 `
  torch-sparse==0.6.18+pt210cu128
conda run -n AMBER_neurips python -m pip install `
  --requirement (Join-Path $lockRoot "AMBER_neurips-pip-portable.txt")
conda run -n AMBER_neurips python (Join-Path $amberRoot "env\verify_AMBER_neurips.py") --require-cuda
