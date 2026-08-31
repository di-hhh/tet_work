from __future__ import annotations

import tempfile
from pathlib import Path

import meshio
import numpy as np

from src.mesh_util.save_mesh import save_as_vtk
from src.tasks.domains.extended_mesh_tet1 import ExtendedMeshTet1


def test_save_as_vtk_supports_extended_tetra_mesh():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    mesh = ExtendedMeshTet1(points.T, np.array([[0, 1, 2, 3]], dtype=np.int32).T)

    with tempfile.TemporaryDirectory() as directory:
        output_path = Path(directory) / "prediction.vtk"
        save_as_vtk(mesh, output_path)
        exported = meshio.read(output_path)

    assert output_path.name == "prediction.vtk"
    assert exported.points.shape == (4, 3)
    assert exported.get_cells_type("tetra").shape == (1, 4)
