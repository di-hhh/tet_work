from __future__ import annotations

from pathlib import Path

import gmsh
import numpy as np
import pygmsh

from src.condition_aware_dataset_generation.geometry_preprocessing.feature_extractor import extract_geometry_features
from src.condition_aware_dataset_generation.records import FailureRecord, GeometryPreprocessRecord, GeometryRecord
from src.condition_aware_dataset_generation.runtime_controls import PipelineAbort, RuntimeTracker
from src.condition_aware_dataset_generation.serialization.layout import PipelineLayout
from src.condition_aware_dataset_generation.utils import dump_json, load_json, now_iso
from src.mesh_util.save_mesh import save_as_vtk
from src.tasks.domains.extended_mesh_tet1 import ExtendedMeshTet1
from src.tasks.domains.extended_mesh_tri1 import ExtendedMeshTri1
from src.tasks.domains.gmsh_geometries import polygon_geom
from src.tasks.domains.gmsh_session import gmsh_session
from src.tasks.domains.gmsh_util import geom_fn_from_file


STEP_SUFFIXES = {'.step', '.stp', '.brep', '.iges', '.igs'}


def geometry_fn_from_path(source_path: str):
    path = Path(source_path)
    suffix = path.suffix.lower()
    if suffix in STEP_SUFFIXES:
        return geom_fn_from_file(str(path))
    if suffix == '.json':
        geometry_spec = load_json(path)
        geometry_type = geometry_spec['geometry_type']
        if geometry_type == 'polygon2d':
            boundary_nodes = np.asarray(geometry_spec['boundary_nodes'], dtype=float)
            return lambda: polygon_geom(boundary_nodes=boundary_nodes)
        if geometry_type == 'rectangle2d':
            width = float(geometry_spec.get('width', 1.0))
            height = float(geometry_spec.get('height', 1.0))
            boundary_nodes = np.array(
                [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]],
                dtype=float,
            )
            return lambda: polygon_geom(boundary_nodes=boundary_nodes)
        if geometry_type == 'box3d':
            size = np.asarray(geometry_spec.get('size', [1.0, 1.0, 1.0]), dtype=float)

            def _box_geom() -> pygmsh.occ.Geometry:
                geom = pygmsh.occ.Geometry()
                geom.add_box([0.0, 0.0, 0.0], size.tolist())
                return geom

            return _box_geom
        raise ValueError(f'Unsupported JSON geometry type: {geometry_type}')
    raise ValueError(f'Unsupported geometry suffix: {suffix}')


def _compute_principal_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    centered = points - center
    if np.allclose(centered, 0.0):
        axes = np.eye(points.shape[1])
    else:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axes = vh.T
        if np.linalg.det(axes) < 0:
            axes[:, -1] *= -1.0
    return center, axes


def _inspect_geometry(geometry_fn, sharp_dihedral_deg: float):
    with gmsh_session(verbose=False):
        geometry_fn()
        gmsh.model.occ.synchronize()
        dimension = 2 if len(gmsh.model.getEntities(3)) == 0 else 3
        top_entities = gmsh.model.getEntities(dimension)
        if not top_entities:
            raise RuntimeError('No top-level entities were found after geometry import')

        mins = np.full(3, np.inf)
        maxs = np.full(3, -np.inf)
        for entity_dim, tag in top_entities:
            bbox = np.asarray(gmsh.model.getBoundingBox(entity_dim, tag), dtype=float)
            mins = np.minimum(mins, bbox[:3])
            maxs = np.maximum(maxs, bbox[3:6])
        bounding_box = np.concatenate((mins, maxs))
        if dimension == 2:
            bounding_box = bounding_box[[0, 1, 3, 4]]

        boundary_entities = gmsh.model.getBoundary(top_entities, oriented=False, recursive=False)
        boundary_dimension = dimension - 1
        patch_lookup = {}
        for entity_dim, tag in boundary_entities:
            if entity_dim != boundary_dimension:
                continue
            patch_lookup[(entity_dim, tag)] = np.asarray(gmsh.model.getBoundingBox(entity_dim, tag), dtype=float)

        patches = []
        for patch_index, ((entity_dim, tag), bbox) in enumerate(sorted(patch_lookup.items())):
            if dimension == 2:
                bbox = bbox[[0, 1, 3, 4]]
            center = 0.5 * (bbox[:dimension] + bbox[dimension:])
            patches.append(
                {
                    'patch_id': f'patch_{patch_index:03d}',
                    'gmsh_dim': int(entity_dim),
                    'gmsh_tag': int(tag),
                    'bbox': bbox.tolist(),
                    'center': center.tolist(),
                }
            )

        geometry_features = extract_geometry_features(
            dimension=dimension,
            bounding_box=bounding_box.astype(float),
            sharp_dihedral_deg=sharp_dihedral_deg,
        )

    return dimension, bounding_box.astype(float), patches, len(top_entities) > 0, geometry_features


class GeometryPreprocessor:
    def __init__(self, preprocess_config: dict):
        self.preprocess_config = preprocess_config
        self.coarse_element_volume = float(preprocess_config.get('coarse_element_volume', 0.05))
        self.min_extent = float(preprocess_config.get('min_extent', 1.0e-6))
        self.sharp_dihedral_deg = float(preprocess_config.get('sharp_dihedral_deg', 40.0))
        self.save_geometry_feature_metadata = bool(preprocess_config.get('save_geometry_feature_metadata', True))

    def preprocess(
        self,
        geometry_record: GeometryRecord,
        layout: PipelineLayout,
        overwrite: bool = False,
        runtime_tracker: RuntimeTracker | None = None,
    ) -> tuple[GeometryPreprocessRecord | None, FailureRecord | None]:
        cache_path = layout.preprocess_record_path(geometry_record.geometry_id)
        if cache_path.exists() and not overwrite:
            cached_record = load_json(cache_path)
            return GeometryPreprocessRecord(**cached_record), None

        started_at = runtime_tracker.started_at if runtime_tracker is not None else now_iso()
        try:
            if runtime_tracker is not None:
                runtime_tracker.check_soft_limits()
            geometry_fn = geometry_fn_from_path(geometry_record.source_path)
            dimension, bounding_box, patches, has_topology, geometry_features = _inspect_geometry(
                geometry_fn,
                sharp_dihedral_deg=self.sharp_dihedral_deg,
            )
            if runtime_tracker is not None:
                runtime_tracker.check_soft_limits()
            mesh_cls = ExtendedMeshTri1 if dimension == 2 else ExtendedMeshTet1
            coarse_mesh = mesh_cls.init_from_geom_fn(geom_fn=geometry_fn, max_element_volume=self.coarse_element_volume)
            coarse_mesh.geom_fn = geometry_fn
            if runtime_tracker is not None:
                runtime_tracker.check_soft_limits()

            points = coarse_mesh.p.T
            centroid, principal_axes = _compute_principal_axes(points)
            oriented_points = (points - centroid) @ principal_axes
            oriented_bbox_min = oriented_points.min(axis=0)
            oriented_bbox_max = oriented_points.max(axis=0)
            extents = oriented_bbox_max - oriented_bbox_min

            validation = {
                'is_meshable': True,
                'is_closed_like': bool(has_topology and (dimension == 2 or len(patches) > 0)),
                'is_degenerate': bool(np.any(extents < self.min_extent)),
                'max_extent': float(extents.max()),
                'min_extent': float(extents.min()),
                'num_boundary_patches': len(patches),
                'num_hole_features': int(geometry_features.get('statistics', {}).get('num_hole_features', 0)),
                'num_sharp_edges': int(geometry_features.get('statistics', {}).get('num_sharp_edges', 0)),
            }

            coarse_mesh_path = layout.geometry_mesh_path(geometry_record.geometry_id)
            save_as_vtk(coarse_mesh, coarse_mesh_path)

            geometry_feature_metadata_path = layout.geometry_feature_metadata_path(geometry_record.geometry_id)
            if self.save_geometry_feature_metadata:
                dump_json(geometry_feature_metadata_path, geometry_features)
                metadata_path = str(geometry_feature_metadata_path)
            else:
                metadata_path = None

            preprocess_record = GeometryPreprocessRecord(
                geometry_id=geometry_record.geometry_id,
                source_path=geometry_record.source_path,
                dimension=dimension,
                bounding_box=bounding_box.tolist(),
                centroid=centroid.tolist(),
                principal_axes=principal_axes.tolist(),
                oriented_bbox_min=oriented_bbox_min.tolist(),
                oriented_bbox_max=oriented_bbox_max.tolist(),
                boundary_patches=patches,
                validation=validation,
                coarse_mesh_path=str(coarse_mesh_path),
                coarse_mesh_num_vertices=int(coarse_mesh.nvertices),
                coarse_mesh_num_elements=int(coarse_mesh.t.shape[1]),
                status='success',
                geometry_feature_metadata_path=metadata_path,
                geometry_features=geometry_features,
            )
            dump_json(cache_path, preprocess_record.to_dict())
            return preprocess_record, None
        except PipelineAbort as exc:
            failure = FailureRecord(
                stage='preprocess',
                item_id=geometry_record.geometry_id,
                source_path=geometry_record.source_path,
                reason=str(exc),
                category=exc.category,
                started_at=started_at,
                finished_at=now_iso(),
                stage_where_stopped=exc.stage,
                partial_output_available=False,
            )
            return None, failure
        except Exception as exc:
            failure = FailureRecord(
                stage='preprocess',
                item_id=geometry_record.geometry_id,
                source_path=geometry_record.source_path,
                reason=str(exc),
                category='invalid_geometry',
                started_at=started_at,
                finished_at=now_iso(),
                stage_where_stopped='geometry_preprocessing',
                partial_output_available=False,
            )
            return None, failure
