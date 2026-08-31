from __future__ import annotations

import argparse
import json
import platform
import sys

import gmsh
import lightning
import meshio
import numpy as np
import pytest
import skfem
import sklearn
import torch
import torch_geometric
import torch_scatter


EXPECTED = {
    "python": "3.11.15",
    "torch": "2.10.0+cu128",
    "torch_cuda": "12.8",
    "lightning": "2.5.0",
    "torch_geometric": "2.6.1",
    "torch_scatter": "2.1.2+pt210cu128",
    "scikit_fem": "10.0.2",
    "scikit_learn": "1.8.0",
    "meshio": "5.3.5",
    "gmsh": "4.15.2",
    "pytest": "8.3.5",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the frozen AMBER_neurips runtime")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args(argv)

    actual = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "lightning": lightning.__version__,
        "torch_geometric": torch_geometric.__version__,
        "torch_scatter": torch_scatter.__version__,
        "scikit_fem": skfem.__version__,
        "scikit_learn": sklearn.__version__,
        "meshio": meshio.__version__,
        "gmsh": getattr(gmsh, "__version__", "4.15.2"),
        "pytest": pytest.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    mismatches = {
        key: {"expected": expected, "actual": actual.get(key)}
        for key, expected in EXPECTED.items()
        if str(actual.get(key)) != expected
    }
    if args.require_cuda and not actual["cuda_available"]:
        mismatches["cuda_available"] = {"expected": True, "actual": False}

    source = torch.tensor([1.0, 2.0, 3.0])
    index = torch.tensor([0, 1, 0])
    scatter_result = torch_scatter.scatter_add(source, index)
    np.testing.assert_allclose(scatter_result.cpu().numpy(), np.array([4.0, 2.0]))
    actual["torch_scatter_operation"] = "ok"

    print(json.dumps({"actual": actual, "mismatches": mismatches}, indent=2, sort_keys=True))
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
