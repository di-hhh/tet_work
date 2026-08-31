import os
import tempfile
from pathlib import Path

import gmsh
import meshio
from skfem import Mesh
from skfem.io.meshio import to_meshio

from src.tasks.domains.mesh_wrapper import MeshWrapper


def save_as_vtk(mesh: Mesh | MeshWrapper, output_vtk_path: str | Path, verbose: bool = False) -> None:
    """
    Convert a scikit FEM mesh to a legacy VTK file using Gmsh for format conversion.

    Args:
        mesh (meshio.Mesh): The input meshio mesh object.
        output_vtk_path (str): Path to the output VTK file.
    """
    if isinstance(output_vtk_path, Path):
        output_vtk_path = str(output_vtk_path)
    if isinstance(mesh, MeshWrapper):
        mesh = mesh.mesh

    if hasattr(mesh, "base_mesh_class"):
        mesh = mesh.base_mesh_class(mesh.p, mesh.t)

    mesh = to_meshio(mesh)
    save_meshio_as_vtk(mesh, output_vtk_path=output_vtk_path, verbose=verbose)


def save_meshio_as_vtk(mesh: meshio.Mesh, output_vtk_path: str | Path, verbose: bool = False) -> None:
    if isinstance(output_vtk_path, Path):
        output_vtk_path = str(output_vtk_path)

    with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        meshio.write(tmp_path, mesh, file_format="gmsh22")
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", int(verbose))
        gmsh.option.setNumber("General.Verbosity", int(verbose))
        gmsh.open(tmp_path)
        gmsh.write(output_vtk_path)
        gmsh.finalize()
    finally:
        os.remove(tmp_path)
