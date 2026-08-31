import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import meshio
import numpy as np

from src.formal_reference_manifests import build_static_reference_manifests


class FormalReferenceManifestTests(unittest.TestCase):
    def test_r0_and_r1_have_frozen_distinct_claim_boundaries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            initial = root / "initial.vtk"
            target = root / "target.vtk"
            _write_tet(initial)
            _write_tet(target)
            record = {
                "sample_id": "sample0",
                "geometry_id": "geometry0",
                "condition_id": "condition0",
                "pde_family": "scalar_elliptic",
                "budget": 7000,
                "_resolved_paths": {
                    "input_mesh_path": str(initial),
                    "target_mesh_path": str(target),
                },
            }
            audit = SimpleNamespace(records_by_split={"test": [record]})

            result = build_static_reference_manifests(
                audit_result=audit,
                output_dir=root / "references",
                protocol_id="protocol-test",
                dataset_fingerprint_sha256="fingerprint",
                manifest_sha256="manifest",
                code_versions={
                    "amber": {"commit": "amber-commit"},
                    "dataest-pipeline": {"commit": "pipeline-commit"},
                },
            )

            r0 = _read_one(Path(result["references"]["R0-Initial"]["prediction_manifest"]))
            r1 = _read_one(Path(result["references"]["R1-Teacher"]["prediction_manifest"]))
            self.assertIn("Not a budget-matched", r0["claim_boundary"])
            self.assertIn("not a strict performance upper bound", r1["claim_boundary"])
            self.assertEqual(r0["analysis_id"], "R0-Initial")
            self.assertEqual(r1["analysis_id"], "R1-Teacher")
            self.assertEqual(int(r0["predicted_vertices"]), 4)
            self.assertEqual(int(r1["predicted_elements"]), 1)


def _write_tet(path: Path) -> None:
    meshio.write(
        path,
        meshio.Mesh(
            points=np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            ),
            cells=[("tetra", np.array([[0, 1, 2, 3]], dtype=np.int32))],
        ),
    )


def _read_one(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return next(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
