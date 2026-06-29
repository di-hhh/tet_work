from typing import Dict, Optional

import gmsh


class gmsh_session:
    """
    A context manager for a Gmsh session. This ensures that the Gmsh session is properly initialized and finalized.
    """

    def __init__(self, gmsh_kwargs: Optional[Dict] = None, verbose: bool = False):
        if gmsh_kwargs is None:
            gmsh_kwargs = {}
        self.gmsh_kwargs = gmsh_kwargs
        self._verbose = verbose

    def __enter__(self):
        gmsh.initialize()

        gmsh.option.setNumber("General.Terminal", int(self._verbose))
        gmsh.option.setNumber("General.Verbosity", int(self._verbose))
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMin", self.gmsh_kwargs.get("min_sizing_field", 1e-6))
        gmsh.option.setNumber("Mesh.MeshSizeMax", self.gmsh_kwargs.get("max_sizing_field", 10))

        gmsh.model.add("model")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        cleanup_errors = []
        try:
            try:
                gmsh.model.remove()
            except Exception as cleanup_exc:
                cleanup_errors.append(cleanup_exc)
        finally:
            try:
                gmsh.finalize()
            except Exception as finalize_exc:
                cleanup_errors.append(finalize_exc)

        if exc_type:
            print(f"An exception occurred: {exc_val}")
            return False

        if cleanup_errors and self._verbose:
            print(f"Ignored gmsh cleanup errors: {cleanup_errors}")
        return False
