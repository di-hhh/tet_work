from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_RUNS_ROOT = REPO_ROOT / ".test_runs"


def pytest_configure():
    deps_dir = REPO_ROOT / ".deps"
    if deps_dir.exists():
        deps_path = str(deps_dir)
        if deps_path not in sys.path:
            sys.path.insert(0, deps_path)
    repo_path = str(REPO_ROOT)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    TEST_RUNS_ROOT.mkdir(parents=True, exist_ok=True)


@pytest.fixture()
def case_root() -> Path:
    root = TEST_RUNS_ROOT / f"case_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture()
def geometry_root(case_root: Path) -> Path:
    root = case_root / "geometries"
    root.mkdir()
    (root / "square.json").write_text(
        """
{
  "geometry_type": "polygon2d",
  "boundary_nodes": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
}
""".strip(),
        encoding="utf-8",
    )
    (root / "rectangle.json").write_text(
        """
{
  "geometry_type": "rectangle2d",
  "width": 1.5,
  "height": 0.5
}
""".strip(),
        encoding="utf-8",
    )
    (root / "broken.json").write_text("", encoding="utf-8")
    return root


@pytest.fixture()
def step_geometry_root(case_root: Path) -> Path:
    import gmsh

    root = case_root / "step_geometries"
    root.mkdir()
    step_path = root / "box_with_hole.step"

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("box_with_hole")
        box = gmsh.model.occ.addBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        cylinder = gmsh.model.occ.addCylinder(0.5, 0.5, 0.0, 0.0, 0.0, 1.0, 0.15)
        gmsh.model.occ.cut([(3, box)], [(3, cylinder)], removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()
        gmsh.write(str(step_path))
    finally:
        gmsh.finalize()
    return root
