# Generated at 2026-04-08 22:33:28 +08:00 (Asia/Shanghai)
from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any

import gmsh
import meshio
import numpy as np
from skfem.io import from_meshio

from src.condition_aware_dataset_generation.geometry_preprocessing.feature_extractor import fit_circle_3d
from src.mesh_util.save_mesh import save_meshio_as_vtk
from src.tasks.domains.extended_mesh_tet1 import ExtendedMeshTet1
from src.tasks.domains.extended_mesh_tri1 import ExtendedMeshTri1
from src.tasks.domains.geometry_util import get_simplex_volumes_from_indices, volume_to_edge_length
from src.tasks.domains.gmsh_session import gmsh_session
from src.tasks.domains.update_mesh import write_sizing_field_to_tmpfile


def combine_geometry_constraints(
    geometry_features: dict[str, Any],
    base_size: float,
    config: dict[str, Any],
    attempt_index: int = 0,
) -> dict[str, Any]:
    if not bool(config.get("enable_geometry_fidelity_constraints", True)):
        return {
            "enabled": False,
            "base_size": float(base_size),
            "min_size": float(base_size * 0.25),
            "max_size": float(base_size),
            "holes": [],
            "curved_surfaces": [],
            "sharp_edges": [],
            "thin_regions": [],
            "fillets": [],
            "retry_level": int(attempt_index),
        }

    constraint_mode = str(config.get("geometry_constraint_mode", "full")).lower()
    topology_only = constraint_mode == "topology_only"
    locality_scale = float(config.get("geometry_constraint_locality_scale", 1.0))
    hole_band_distance_scale = float(config.get("hole_band_distance_scale", 1.0))
    apply_transfinite_hole_curves = bool(config.get("enable_transfinite_hole_curves", True))
    min_circle_segments = int(config.get("min_circle_segments", 20)) + (0 if topology_only else 4 * int(attempt_index))
    hole_edge_length_ratio = float(config.get("hole_edge_length_ratio", 0.24)) * ((0.92 if topology_only else 0.82) ** int(attempt_index))
    hole_radial_layers = max(1, int(config.get("hole_radial_refinement_layers", 3)))
    hole_growth = max(1.05, float(config.get("hole_radial_growth_rate", 1.45)))
    curvature_strength = float(config.get("curvature_refinement_strength", 1.5)) * (1.15 ** int(attempt_index))
    feature_strength = float(config.get("feature_size_refinement_strength", 1.5)) * (1.15 ** int(attempt_index))
    global_scale = float(geometry_features.get("statistics", {}).get("global_scale", 1.0))

    surface_lookup = {record["tag"]: record for record in geometry_features.get("surface_patches", [])}
    default_min_size_ratio = 0.45 if topology_only else 0.015
    min_size_ratio = float(config.get("geometry_min_size_ratio", default_min_size_ratio))
    constraints = {
        "enabled": True,
        "constraint_mode": constraint_mode,
        "apply_transfinite_hole_curves": apply_transfinite_hole_curves,
        "base_size": float(base_size),
        "min_size": float(max(base_size * min_size_ratio, 1.0e-6 * max(global_scale, 1.0))),
        "max_size": float(base_size),
        "holes": [],
        "curved_surfaces": [],
        "sharp_edges": [],
        "thin_regions": [],
        "fillets": [],
        "retry_level": int(attempt_index),
        "effective_config": {
            "min_circle_segments": min_circle_segments,
            "hole_edge_length_ratio": hole_edge_length_ratio,
            "hole_radial_refinement_layers": hole_radial_layers,
            "hole_radial_growth_rate": hole_growth,
            "curvature_refinement_strength": curvature_strength,
            "feature_size_refinement_strength": feature_strength,
            "geometry_constraint_locality_scale": locality_scale,
            "hole_band_distance_scale": hole_band_distance_scale,
            "apply_transfinite_hole_curves": apply_transfinite_hole_curves,
            "geometry_min_size_ratio": min_size_ratio,
        },
    }

    for hole in geometry_features.get("feature_anchors", {}).get("hole_curve_loops", []):
        radius = float(hole.get("radius", np.inf))
        if not np.isfinite(radius) or radius <= 1.0e-9:
            continue
        min_segments = max(min_circle_segments, int(math.ceil((2.0 * math.pi) / max(hole_edge_length_ratio, 1.0e-6))))
        target_size = min(base_size, hole_edge_length_ratio * radius, 2.0 * math.pi * radius / max(min_segments, 3))
        target_size = max(target_size, constraints["min_size"])
        band_sizes = [target_size]
        band_distances = []
        base_ring_distance = max(radius * hole_band_distance_scale, target_size)
        for layer in range(hole_radial_layers):
            band_sizes.append(min(base_size, target_size * (hole_growth ** (layer + 1))))
            band_distances.append(base_ring_distance * float(layer + 1))
        cylinder_surface_tags = [int(tag) for tag in hole.get("associated_cylinder_surface_tags", []) if int(tag) in surface_lookup]
        surface_anchor_points = []
        for surface_tag in cylinder_surface_tags:
            surface_anchor_points.extend(surface_lookup[surface_tag].get("sample_points", []))
        constraints["holes"].append(
            {
                "feature_id": hole["feature_id"],
                "curve_tags": [int(hole["curve_tag"])],
                "surface_tags": cylinder_surface_tags,
                "radius": radius,
                "center": hole.get("center", [0.0, 0.0, 0.0]),
                "normal": hole.get("normal", [0.0, 0.0, 1.0]),
                "min_segments": int(min_segments),
                "target_size": float(target_size),
                "band_sizes": [float(value) for value in band_sizes],
                "band_distances": [float(value) for value in band_distances],
                "curve_anchor_points": hole.get("anchor_points", []),
                "surface_anchor_points": surface_anchor_points,
                "sampling": max(80, 4 * min_segments),
            }
        )

    if not topology_only:
        for surface in geometry_features.get("surface_patches", []):
            surface_type = surface.get("surface_type")
            if surface_type == "plane":
                continue
            radius = float(surface.get("curvature_radius", np.inf))
            if not np.isfinite(radius):
                continue
            target_size = min(base_size, radius / max(1.0 + curvature_strength, 1.0))
            target_size = max(target_size, constraints["min_size"])
            constraints["curved_surfaces"].append(
                {
                    "surface_tag": int(surface["tag"]),
                    "surface_type": surface_type,
                    "target_size": float(target_size),
                    "influence_distance": float(max(radius, target_size) * locality_scale),
                    "anchor_points": surface.get("sample_points", []),
                }
            )

        for record in geometry_features.get("feature_anchors", {}).get("fillet_neighborhoods", []):
            radius = float(record.get("radius", np.inf))
            if not np.isfinite(radius):
                continue
            target_size = min(base_size, radius / max(1.0 + feature_strength, 1.0))
            target_size = max(target_size, constraints["min_size"])
            constraints["fillets"].append(
                {
                    "surface_tag": int(record["surface_tag"]),
                    "target_size": float(target_size),
                    "influence_distance": float(max(radius, target_size) * locality_scale),
                    "anchor_points": record.get("anchor_points", []),
                }
            )

        for record in geometry_features.get("feature_anchors", {}).get("sharp_edges", []):
            feature_size = float(record.get("feature_size", global_scale))
            target_size = min(base_size, feature_size / max(1.0 + feature_strength, 1.0))
            target_size = max(target_size, constraints["min_size"])
            constraints["sharp_edges"].append(
                {
                    "curve_tag": int(record["curve_tag"]),
                    "target_size": float(target_size),
                    "influence_distance": float(max(0.5 * feature_size, target_size) * locality_scale),
                    "anchor_points": record.get("anchor_points", []),
                }
            )

        for record in geometry_features.get("feature_anchors", {}).get("thin_regions", []):
            thickness = float(record.get("distance", np.inf))
            if not np.isfinite(thickness):
                continue
            target_size = min(base_size, 0.5 * thickness / max(1.0 + feature_strength, 1.0))
            target_size = max(target_size, constraints["min_size"])
            constraints["thin_regions"].append(
                {
                    "surface_tags": [int(tag) for tag in record.get("surface_tags", [])],
                    "target_size": float(target_size),
                    "influence_distance": float(max(thickness, target_size) * locality_scale),
                    "anchor_points": [point for center in record.get("centers", []) for point in [center]],
                }
            )

    min_targets = [constraints["min_size"]]
    for group_name in ["holes", "curved_surfaces", "sharp_edges", "thin_regions", "fillets"]:
        min_targets.extend(record.get("target_size", constraints["min_size"]) for record in constraints[group_name])
    constraints["min_size"] = float(max(min(min_targets), 1.0e-8))
    constraints["summary"] = {
        "num_holes": len(constraints["holes"]),
        "num_curved_surfaces": len(constraints["curved_surfaces"]),
        "num_sharp_edges": len(constraints["sharp_edges"]),
        "num_thin_regions": len(constraints["thin_regions"]),
        "num_fillets": len(constraints["fillets"]),
    }
    return constraints


def evaluate_geometry_sizing(
    points: np.ndarray,
    geometry_features: dict[str, Any],
    constraint_summary: dict[str, Any],
    base_size: float,
    config: dict[str, Any],
) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError(f"Expected points with shape (N, dim), got {points.shape}")
    if not constraint_summary.get("enabled", False):
        return np.full(points.shape[0], float(base_size), dtype=float)

    sizes = np.full(points.shape[0], float(base_size), dtype=float)

    for hole in constraint_summary.get("holes", []):
        circle_distance = _distance_to_circle(
            points,
            center=np.asarray(hole.get("center", [0.0, 0.0, 0.0]), dtype=float),
            normal=np.asarray(hole.get("normal", [0.0, 0.0, 1.0]), dtype=float),
            radius=float(hole.get("radius", 1.0)),
        )
        surface_points = np.asarray(hole.get("surface_anchor_points", []), dtype=float)
        if surface_points.size:
            circle_distance = np.minimum(circle_distance, _distance_to_samples(points, surface_points))
        sizes = np.minimum(sizes, _banded_size(circle_distance, hole["band_distances"], hole["band_sizes"], base_size))

    for group_name in ["curved_surfaces", "fillets", "sharp_edges", "thin_regions"]:
        for record in constraint_summary.get(group_name, []):
            anchor_points = np.asarray(record.get("anchor_points", []), dtype=float)
            if anchor_points.size == 0:
                continue
            distances = _distance_to_samples(points, anchor_points)
            influence = float(record.get("influence_distance", base_size))
            target = float(record.get("target_size", base_size))
            sizes = np.minimum(sizes, _linear_transition_size(distances, target, base_size, influence))

    return np.clip(sizes, constraint_summary["min_size"], constraint_summary["max_size"])


def fuse_sizing_fields(
    geometry_sizes: np.ndarray,
    pde_sizes: np.ndarray,
    budget_sizes: np.ndarray,
    mode: str = "min",
) -> np.ndarray:
    geometry_sizes = np.asarray(geometry_sizes, dtype=float)
    pde_sizes = np.asarray(pde_sizes, dtype=float)
    budget_sizes = np.asarray(budget_sizes, dtype=float)
    mode = mode.lower()
    if mode == "min":
        return np.minimum(np.minimum(geometry_sizes, pde_sizes), budget_sizes)
    if mode == "harmonic":
        fused = 3.0 / (
            1.0 / np.maximum(geometry_sizes, 1.0e-12)
            + 1.0 / np.maximum(pde_sizes, 1.0e-12)
            + 1.0 / np.maximum(budget_sizes, 1.0e-12)
        )
        return np.minimum(geometry_sizes, fused)
    if mode == "geometric_mean":
        fused = np.cbrt(np.maximum(geometry_sizes, 1.0e-12) * np.maximum(pde_sizes, 1.0e-12) * np.maximum(budget_sizes, 1.0e-12))
        return np.minimum(geometry_sizes, fused)
    raise ValueError(f"Unsupported geometry/PDE fusion mode: {mode}")

def generate_cad_aware_mesh(
    *,
    geometry_fn,
    preprocess_record,
    max_element_volume: float,
    config: dict[str, Any],
    attempt_index: int = 0,
    surface_mesh_path: str | None = None,
    additional_background: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    dimension = int(preprocess_record.dimension)
    geometry_features = preprocess_record.geometry_features or {}
    base_size = float(volume_to_edge_length(float(max_element_volume), dim=dimension))
    constraints = combine_geometry_constraints(geometry_features, base_size=base_size, config=config, attempt_index=attempt_index)
    gmsh_kwargs = {"min_sizing_field": constraints["min_size"], "max_sizing_field": constraints["max_size"]}
    target_class = ExtendedMeshTri1 if dimension == 2 else ExtendedMeshTet1

    if dimension == 3 and bool(config.get("surface_first_meshing", True)):
        surface_pass = _run_gmsh_meshing_pass(
            geometry_fn=geometry_fn,
            preprocess_record=preprocess_record,
            constraints=constraints,
            gmsh_kwargs=gmsh_kwargs,
            target_class=target_class,
            target_dimension=2,
            surface_mesh_path=surface_mesh_path,
            additional_background=additional_background,
            metric_config=config,
        )
        if surface_pass["status"] != "success":
            return {
                "mesh": None,
                "surface_mesh": None,
                "surface_metrics": {"status": "failed", "reasons": [surface_pass["error"]]},
                "volume_metrics": {},
                "constraint_summary": constraints,
                "status": "surface_failed",
            }
        if surface_pass["surface_metrics"].get("status") == "failed":
            return {
                "mesh": None,
                "surface_mesh": surface_pass.get("surface_mesh"),
                "surface_metrics": surface_pass["surface_metrics"],
                "volume_metrics": {},
                "constraint_summary": constraints,
                "status": "surface_failed",
            }
        volume_pass = _run_gmsh_meshing_pass(
            geometry_fn=geometry_fn,
            preprocess_record=preprocess_record,
            constraints=constraints,
            gmsh_kwargs=gmsh_kwargs,
            target_class=target_class,
            target_dimension=3,
            surface_mesh_path=None,
            additional_background=additional_background,
            metric_config=config,
        )
        if volume_pass["status"] != "success":
            return {
                "mesh": None,
                "surface_mesh": surface_pass.get("surface_mesh"),
                "surface_metrics": surface_pass["surface_metrics"],
                "volume_metrics": {"status": "failed", "reasons": [volume_pass["error"]]},
                "constraint_summary": constraints,
                "status": "volume_failed",
            }
        if volume_pass["volume_metrics"].get("status") == "failed":
            return {
                "mesh": volume_pass.get("mesh"),
                "surface_mesh": surface_pass.get("surface_mesh"),
                "surface_metrics": surface_pass["surface_metrics"],
                "volume_metrics": volume_pass["volume_metrics"],
                "constraint_summary": constraints,
                "status": "volume_failed",
            }
        return {
            "mesh": volume_pass["mesh"],
            "surface_mesh": surface_pass.get("surface_mesh"),
            "surface_metrics": surface_pass["surface_metrics"],
            "volume_metrics": volume_pass["volume_metrics"],
            "constraint_summary": constraints,
            "status": "success",
        }

    single_pass = _run_gmsh_meshing_pass(
        geometry_fn=geometry_fn,
        preprocess_record=preprocess_record,
        constraints=constraints,
        gmsh_kwargs=gmsh_kwargs,
        target_class=target_class,
        target_dimension=dimension,
        surface_mesh_path=surface_mesh_path,
        additional_background=additional_background,
        metric_config=config,
    )
    if single_pass["status"] != "success":
        return {
            "mesh": None,
            "surface_mesh": None,
            "surface_metrics": {"status": "failed", "reasons": [single_pass["error"]]},
            "volume_metrics": {},
            "constraint_summary": constraints,
            "status": "surface_failed",
        }
    if single_pass.get("surface_metrics", {}).get("status") == "failed":
        return {
            "mesh": single_pass.get("mesh"),
            "surface_mesh": single_pass.get("surface_mesh"),
            "surface_metrics": single_pass["surface_metrics"],
            "volume_metrics": single_pass.get("volume_metrics", {}),
            "constraint_summary": constraints,
            "status": "surface_failed",
        }
    if single_pass.get("volume_metrics", {}).get("status") == "failed":
        return {
            "mesh": single_pass.get("mesh"),
            "surface_mesh": single_pass.get("surface_mesh"),
            "surface_metrics": single_pass.get("surface_metrics", {}),
            "volume_metrics": single_pass["volume_metrics"],
            "constraint_summary": constraints,
            "status": "volume_failed",
        }
    return {
        "mesh": single_pass.get("mesh"),
        "surface_mesh": single_pass.get("surface_mesh"),
        "surface_metrics": single_pass.get("surface_metrics", {}),
        "volume_metrics": single_pass.get("volume_metrics", {}),
        "constraint_summary": constraints,
        "status": "success",
    }


def _run_gmsh_meshing_pass(
    *,
    geometry_fn,
    preprocess_record,
    constraints: dict[str, Any],
    gmsh_kwargs: dict[str, float],
    target_class,
    target_dimension: int,
    surface_mesh_path: str | None,
    additional_background: dict[str, np.ndarray] | None,
    metric_config: dict[str, Any],
) -> dict[str, Any]:
    final_dimension = int(preprocess_record.dimension)
    geometry_features = preprocess_record.geometry_features or {}
    with gmsh_session(gmsh_kwargs=gmsh_kwargs, verbose=False):
        geometry_fn()
        gmsh.model.occ.synchronize()
        _configure_global_mesh_options(constraints=constraints, dimension=final_dimension, base_size=constraints["base_size"])
        field_ids, temp_files = _apply_all_background_fields(constraints=constraints, additional_background=additional_background)
        if field_ids:
            _activate_background_field(field_ids)
        try:
            gmsh.model.mesh.generate(target_dimension)
        except Exception as exc:
            _cleanup_temp_files(temp_files)
            return {"status": "generation_failed", "error": f"gmsh mesh generation failed: {exc}"}

        if target_dimension == 2:
            surface_metrics = _compute_surface_metrics(
                geometry_features=geometry_features,
                config=metric_config,
                dimension=final_dimension,
            )
            surface_mesh = _current_meshio_from_gmsh()
            if surface_mesh_path is not None:
                save_meshio_as_vtk(surface_mesh, surface_mesh_path)
            if final_dimension == 2:
                mesh = _meshio_to_extended_mesh(meshio_mesh=surface_mesh, dimension=2, target_class=target_class)
                mesh.geom_fn = geometry_fn
                volume_metrics = _compute_volume_metrics(mesh=mesh, dimension=2)
                _cleanup_temp_files(temp_files)
                return {
                    "status": "success",
                    "surface_mesh": surface_mesh,
                    "surface_metrics": surface_metrics,
                    "mesh": mesh,
                    "volume_metrics": volume_metrics,
                }
            _cleanup_temp_files(temp_files)
            return {
                "status": "success",
                "surface_mesh": surface_mesh,
                "surface_metrics": surface_metrics,
            }

        meshio_mesh = _current_meshio_from_gmsh()
        mesh = _meshio_to_extended_mesh(meshio_mesh=meshio_mesh, dimension=final_dimension, target_class=target_class)
        mesh.geom_fn = geometry_fn
        volume_metrics = _compute_volume_metrics(mesh=mesh, dimension=final_dimension)
        _cleanup_temp_files(temp_files)
        return {"status": "success", "mesh": mesh, "volume_metrics": volume_metrics}

def _configure_global_mesh_options(*, constraints: dict[str, Any], dimension: int, base_size: float) -> None:
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 5)
    if dimension == 3:
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), float(base_size))

    if not bool(constraints.get("apply_transfinite_hole_curves", True)):
        return

    for hole in constraints.get("holes", []):
        for curve_tag in hole.get("curve_tags", []):
            try:
                gmsh.model.mesh.setTransfiniteCurve(int(curve_tag), int(hole["min_segments"]))
            except Exception:
                continue


def _apply_all_background_fields(
    *,
    constraints: dict[str, Any],
    additional_background: dict[str, np.ndarray] | None,
) -> tuple[list[int], list[str]]:
    field_ids: list[int] = []
    temp_files: list[str] = []

    for hole in constraints.get("holes", []):
        curve_tags = [int(tag) for tag in hole.get("curve_tags", [])]
        surface_tags = [int(tag) for tag in hole.get("surface_tags", [])]
        band_sizes = hole.get("band_sizes", [])
        band_distances = hole.get("band_distances", [])
        for layer_index, outer_distance in enumerate(band_distances):
            dist_min = 0.0 if layer_index == 0 else float(band_distances[layer_index - 1])
            dist_max = float(outer_distance)
            size_min = float(band_sizes[layer_index])
            size_max = float(band_sizes[layer_index + 1])
            if curve_tags:
                field_ids.append(
                    _add_distance_threshold_field(
                        curve_tags=curve_tags,
                        surface_tags=None,
                        size_min=size_min,
                        size_max=size_max,
                        dist_min=dist_min,
                        dist_max=dist_max,
                        sampling=int(hole.get("sampling", 100)),
                    )
                )
            if surface_tags:
                field_ids.append(
                    _add_distance_threshold_field(
                        curve_tags=None,
                        surface_tags=surface_tags,
                        size_min=size_min,
                        size_max=size_max,
                        dist_min=dist_min,
                        dist_max=dist_max,
                        sampling=int(hole.get("sampling", 100)),
                    )
                )

    for record in constraints.get("curved_surfaces", []):
        field_ids.append(
            _add_distance_threshold_field(
                curve_tags=None,
                surface_tags=[int(record["surface_tag"])],
                size_min=float(record["target_size"]),
                size_max=float(constraints["max_size"]),
                dist_min=0.0,
                dist_max=float(record["influence_distance"]),
                sampling=60,
            )
        )

    for record in constraints.get("fillets", []):
        field_ids.append(
            _add_distance_threshold_field(
                curve_tags=None,
                surface_tags=[int(record["surface_tag"])],
                size_min=float(record["target_size"]),
                size_max=float(constraints["max_size"]),
                dist_min=0.0,
                dist_max=float(record["influence_distance"]),
                sampling=60,
            )
        )

    for record in constraints.get("sharp_edges", []):
        field_ids.append(
            _add_distance_threshold_field(
                curve_tags=[int(record["curve_tag"])],
                surface_tags=None,
                size_min=float(record["target_size"]),
                size_max=float(constraints["max_size"]),
                dist_min=0.0,
                dist_max=float(record["influence_distance"]),
                sampling=60,
            )
        )

    for record in constraints.get("thin_regions", []):
        if record.get("surface_tags"):
            field_ids.append(
                _add_distance_threshold_field(
                    curve_tags=None,
                    surface_tags=[int(tag) for tag in record["surface_tags"]],
                    size_min=float(record["target_size"]),
                    size_max=float(constraints["max_size"]),
                    dist_min=0.0,
                    dist_max=float(record["influence_distance"]),
                    sampling=40,
                )
            )

    if additional_background is not None:
        positions = np.asarray(additional_background["positions"], dtype=float)
        sizes = np.asarray(additional_background["sizes"], dtype=float)
        if len(positions) != len(sizes):
            raise ValueError("Additional background sizing positions and sizes must have the same length")
        fd, tmp_path = tempfile.mkstemp(suffix=".pos", dir=str(_workspace_tmp_dir()))
        os.close(fd)
        with open(tmp_path, "wb") as handle:
            write_sizing_field_to_tmpfile(
                sizing_field_positions=positions,
                sizing_field=sizes,
                tmpfile=handle,
            )
        gmsh.merge(tmp_path)
        post_view_field = gmsh.model.mesh.field.add("PostView")
        gmsh.model.mesh.field.setNumber(post_view_field, "ViewIndex", 0)
        field_ids.append(post_view_field)
        temp_files.append(tmp_path)

    return field_ids, temp_files


def _add_distance_threshold_field(
    *,
    curve_tags: list[int] | None,
    surface_tags: list[int] | None,
    size_min: float,
    size_max: float,
    dist_min: float,
    dist_max: float,
    sampling: int,
) -> int:
    distance_field = gmsh.model.mesh.field.add("Distance")
    if curve_tags:
        gmsh.model.mesh.field.setNumbers(distance_field, "CurvesList", curve_tags)
    if surface_tags:
        gmsh.model.mesh.field.setNumbers(distance_field, "SurfacesList", surface_tags)
    gmsh.model.mesh.field.setNumber(distance_field, "Sampling", int(sampling))
    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "InField", distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin", float(size_min))
    gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax", float(size_max))
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", float(dist_min))
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", float(max(dist_max, dist_min + 1.0e-9)))
    return threshold_field


def _activate_background_field(field_ids: list[int]) -> None:
    if not field_ids:
        return
    if len(field_ids) == 1:
        gmsh.model.mesh.field.setAsBackgroundMesh(field_ids[0])
        return
    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", field_ids)
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

def _current_meshio_from_gmsh() -> meshio.Mesh:
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    points = np.asarray(coords, dtype=float).reshape(-1, 3)
    tag_to_index = {int(tag): index for index, tag in enumerate(node_tags)}
    cell_map: dict[str, list[np.ndarray]] = {}
    element_types, _, node_blocks = gmsh.model.mesh.getElements()
    for element_type, node_block in zip(element_types, node_blocks):
        name, _, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(element_type)
        cell_type = _meshio_cell_type(name=name, num_nodes=num_nodes)
        if cell_type is None:
            continue
        connectivity_tags = np.asarray(node_block, dtype=int).reshape(-1, num_nodes)
        connectivity = np.vectorize(lambda value: tag_to_index[int(value)], otypes=[int])(connectivity_tags)
        cell_map.setdefault(cell_type, []).append(connectivity)
    cells = [(cell_type, np.concatenate(blocks, axis=0)) for cell_type, blocks in cell_map.items()]
    if not cells:
        raise RuntimeError("Current gmsh model does not contain exportable linear mesh cells")
    return meshio.Mesh(points=points, cells=cells)


def _meshio_to_extended_mesh(meshio_mesh: meshio.Mesh, dimension: int, target_class):
    if dimension == 3:
        cells = _extract_cells(meshio_mesh, "tetra")
        mesh = meshio.Mesh(points=meshio_mesh.points, cells=[("tetra", cells)])
    else:
        cells = _extract_cells(meshio_mesh, "triangle")
        mesh = meshio.Mesh(points=meshio_mesh.points, cells=[("triangle", cells)])
    converted = from_meshio(mesh)
    return target_class(converted.p, converted.t)


def _extract_cells(meshio_mesh: meshio.Mesh, cell_type: str) -> np.ndarray:
    blocks = [cell_block.data for cell_block in meshio_mesh.cells if cell_block.type == cell_type]
    if not blocks:
        raise RuntimeError(f"Mesh does not contain any '{cell_type}' cells")
    return np.concatenate(blocks, axis=0)


def _meshio_cell_type(*, name: str, num_nodes: int) -> str | None:
    lowered = name.lower()
    if "line" in lowered and num_nodes == 2:
        return "line"
    if "triangle" in lowered and num_nodes == 3:
        return "triangle"
    if "tetra" in lowered and num_nodes == 4:
        return "tetra"
    return None


def _compute_surface_metrics(*, geometry_features: dict[str, Any], config: dict[str, Any], dimension: int) -> dict[str, Any]:
    global_scale = float(geometry_features.get("statistics", {}).get("global_scale", 1.0))
    hole_records = []
    max_edge_ratio = 0.0
    max_circle_fit_error = 0.0
    min_segments = math.inf

    for hole in geometry_features.get("feature_anchors", {}).get("hole_curve_loops", []):
        curve_tag = int(hole["curve_tag"])
        node_tags, coords, params = gmsh.model.mesh.getNodes(1, curve_tag, includeBoundary=True)
        if len(node_tags) == 0:
            continue
        points = np.asarray(coords, dtype=float).reshape(-1, 3)
        if len(params) == len(node_tags):
            order = np.argsort(np.asarray(params, dtype=float))
            points = points[order]
        fit = fit_circle_3d(points)
        closed = True
        segment_lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1) if closed else np.linalg.norm(np.diff(points, axis=0), axis=1)
        segments = int(len(points) if closed else max(len(points) - 1, 0))
        radius = float(hole.get("radius", fit.get("radius", 1.0)))
        edge_ratio = float(segment_lengths.max() / max(radius, 1.0e-12)) if len(segment_lengths) else float("inf")
        fit_error = float(fit.get("max_relative_error", float("inf")))
        hole_records.append(
            {
                "curve_tag": curve_tag,
                "segments": segments,
                "radius": radius,
                "edge_length_stats": {
                    "min": float(segment_lengths.min()) if len(segment_lengths) else float("inf"),
                    "mean": float(segment_lengths.mean()) if len(segment_lengths) else float("inf"),
                    "max": float(segment_lengths.max()) if len(segment_lengths) else float("inf"),
                },
                "max_edge_length_over_radius": edge_ratio,
                "circle_fit": fit,
            }
        )
        min_segments = min(min_segments, segments)
        max_edge_ratio = max(max_edge_ratio, edge_ratio)
        max_circle_fit_error = max(max_circle_fit_error, fit_error)

    if dimension == 3:
        boundary_stats = _compute_surface_boundary_and_normal_stats(global_scale=global_scale)
    else:
        boundary_stats = _compute_curve_boundary_stats(global_scale=global_scale)

    reasons = []
    required_segments = int(config.get("min_circle_segments", 20))
    if hole_records and min_segments < required_segments:
        reasons.append(f"hole segments {min_segments} below required {required_segments}")
    if hole_records and max_edge_ratio > float(config.get("hole_edge_length_ratio", 0.24)) * 1.05:
        reasons.append("hole edge length ratio exceeded")
    if hole_records and max_circle_fit_error > float(config.get("max_circle_fit_error", 0.04)):
        reasons.append("circle fit error exceeded")
    if boundary_stats["boundary_deviation"]["max_normalized"] > float(config.get("max_boundary_deviation", 0.02)):
        reasons.append("boundary deviation exceeded")
    if boundary_stats["normal_deviation"]["max_deg"] > float(config.get("max_normal_deviation", 22.5)):
        reasons.append("normal deviation exceeded")

    return {
        "status": "failed" if reasons else "success",
        "reasons": reasons,
        "hole_sampling": {
            "detected_holes": len(geometry_features.get("feature_anchors", {}).get("hole_curve_loops", [])),
            "measured_holes": len(hole_records),
            "min_segments": int(min_segments) if hole_records else 0,
            "max_edge_length_over_radius": float(max_edge_ratio),
            "max_circle_fit_error": float(max_circle_fit_error),
            "records": hole_records,
        },
        "boundary_deviation": boundary_stats["boundary_deviation"],
        "normal_deviation": boundary_stats["normal_deviation"],
        "feature_preservation": {
            "all_holes_measured": len(hole_records) == len(geometry_features.get("feature_anchors", {}).get("hole_curve_loops", [])),
            "num_sharp_edges": int(geometry_features.get("statistics", {}).get("num_sharp_edges", 0)),
            "num_fillets": int(geometry_features.get("statistics", {}).get("num_fillet_features", 0)),
        },
    }


def _compute_surface_boundary_and_normal_stats(*, global_scale: float) -> dict[str, dict[str, float]]:
    node_lookup = _node_lookup()
    deviation_values = []
    normal_angles = []
    for _, surface_tag in gmsh.model.getEntities(2):
        for connectivity in _surface_connectivity_for_entity(surface_tag):
            vertices = np.asarray([node_lookup[int(tag)] for tag in connectivity], dtype=float)
            centroid = vertices.mean(axis=0)
            closest_points, params = gmsh.model.getClosestPoint(2, surface_tag, centroid.tolist())
            closest = np.asarray(closest_points, dtype=float).reshape(-1, 3)[0]
            deviation_values.append(float(np.linalg.norm(centroid - closest)))
            if len(params) >= 2:
                cad_normal = np.asarray(gmsh.model.getNormal(surface_tag, params[:2]), dtype=float).reshape(-1, 3)[0]
                face_normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
                cad_norm = np.linalg.norm(cad_normal)
                face_norm = np.linalg.norm(face_normal)
                if cad_norm > 1.0e-12 and face_norm > 1.0e-12:
                    cad_normal = cad_normal / cad_norm
                    face_normal = face_normal / face_norm
                    angle = float(np.degrees(np.arccos(np.clip(np.abs(np.dot(face_normal, cad_normal)), -1.0, 1.0))))
                    normal_angles.append(angle)
    if not deviation_values:
        deviation_values = [0.0]
    if not normal_angles:
        normal_angles = [0.0]
    return {
        "boundary_deviation": {
            "max": float(max(deviation_values)),
            "mean": float(np.mean(deviation_values)),
            "max_normalized": float(max(deviation_values) / max(global_scale, 1.0e-12)),
            "mean_normalized": float(np.mean(deviation_values) / max(global_scale, 1.0e-12)),
        },
        "normal_deviation": {
            "max_deg": float(max(normal_angles)),
            "mean_deg": float(np.mean(normal_angles)),
        },
    }


def _compute_curve_boundary_stats(*, global_scale: float) -> dict[str, dict[str, float]]:
    node_lookup = _node_lookup()
    deviation_values = []
    for _, curve_tag in gmsh.model.getEntities(1):
        element_types, _, node_blocks = gmsh.model.mesh.getElements(1, curve_tag)
        for element_type, node_block in zip(element_types, node_blocks):
            _, _, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(element_type)
            if num_nodes < 2:
                continue
            connectivity = np.asarray(node_block, dtype=int).reshape(-1, num_nodes)
            for edge in connectivity:
                midpoint = np.asarray([node_lookup[int(edge[0])], node_lookup[int(edge[1])]], dtype=float).mean(axis=0)
                closest_points, _ = gmsh.model.getClosestPoint(1, curve_tag, midpoint.tolist())
                closest = np.asarray(closest_points, dtype=float).reshape(-1, 3)[0]
                deviation_values.append(float(np.linalg.norm(midpoint - closest)))
    if not deviation_values:
        deviation_values = [0.0]
    return {
        "boundary_deviation": {
            "max": float(max(deviation_values)),
            "mean": float(np.mean(deviation_values)),
            "max_normalized": float(max(deviation_values) / max(global_scale, 1.0e-12)),
            "mean_normalized": float(np.mean(deviation_values) / max(global_scale, 1.0e-12)),
        },
        "normal_deviation": {"max_deg": 0.0, "mean_deg": 0.0},
    }

def _compute_volume_metrics(*, mesh, dimension: int) -> dict[str, Any]:
    points = mesh.p.T
    simplices = mesh.t.T
    simplex_volumes = get_simplex_volumes_from_indices(points, simplices)
    edge_ratios = []
    radius_ratios = []

    if dimension == 3:
        min_dihedrals = []
        for simplex in simplices:
            tetra = points[simplex]
            edge_lengths = _tetra_edge_lengths(tetra)
            edge_ratios.append(float(edge_lengths.max() / max(edge_lengths.min(), 1.0e-12)))
            radius_ratios.append(float(_tetra_radius_ratio(tetra)))
            min_dihedrals.append(float(_tetra_min_dihedral(tetra)))
        reasons = []
        if np.any(simplex_volumes <= 0.0):
            reasons.append("non-positive tetra volume")
        if radius_ratios and min(radius_ratios) < 1.0e-4:
            reasons.append("tetra radius ratio too small")
        if edge_ratios and np.percentile(edge_ratios, 95) > 25.0:
            reasons.append("tetra edge ratio too large")
        return {
            "status": "failed" if reasons else "success",
            "reasons": reasons,
            "num_elements": int(len(simplices)),
            "min_volume": float(simplex_volumes.min()) if len(simplex_volumes) else 0.0,
            "mean_volume": float(simplex_volumes.mean()) if len(simplex_volumes) else 0.0,
            "min_radius_ratio": float(min(radius_ratios)) if radius_ratios else 0.0,
            "mean_radius_ratio": float(np.mean(radius_ratios)) if radius_ratios else 0.0,
            "min_dihedral_deg": float(min(min_dihedrals)) if min_dihedrals else 0.0,
            "max_edge_ratio": float(max(edge_ratios)) if edge_ratios else 0.0,
            "sliver_indicator": float(1.0 - np.mean(radius_ratios)) if radius_ratios else 1.0,
        }

    for simplex in simplices:
        triangle = points[simplex]
        edge_lengths = np.linalg.norm(np.roll(triangle, -1, axis=0) - triangle, axis=1)
        edge_ratios.append(float(edge_lengths.max() / max(edge_lengths.min(), 1.0e-12)))
    reasons = []
    if np.any(simplex_volumes <= 0.0):
        reasons.append("non-positive triangle area")
    if edge_ratios and np.percentile(edge_ratios, 95) > 20.0:
        reasons.append("triangle edge ratio too large")
    return {
        "status": "failed" if reasons else "success",
        "reasons": reasons,
        "num_elements": int(len(simplices)),
        "min_area": float(simplex_volumes.min()) if len(simplex_volumes) else 0.0,
        "mean_area": float(simplex_volumes.mean()) if len(simplex_volumes) else 0.0,
        "max_edge_ratio": float(max(edge_ratios)) if edge_ratios else 0.0,
        "sliver_indicator": float(max(edge_ratios)) if edge_ratios else 0.0,
    }


def _banded_size(distances: np.ndarray, band_distances: list[float], band_sizes: list[float], base_size: float) -> np.ndarray:
    result = np.full(len(distances), float(base_size), dtype=float)
    if not band_distances:
        return np.minimum(result, float(band_sizes[0]) if band_sizes else base_size)
    previous = 0.0
    for layer_index, outer in enumerate(band_distances):
        inner_size = float(band_sizes[layer_index])
        outer_size = float(band_sizes[layer_index + 1])
        mask = distances <= outer
        if not np.any(mask):
            previous = outer
            continue
        local = np.clip((distances[mask] - previous) / max(outer - previous, 1.0e-12), 0.0, 1.0)
        interpolated = inner_size + (outer_size - inner_size) * local
        result[mask] = np.minimum(result[mask], interpolated)
        previous = outer
    return result


def _linear_transition_size(distances: np.ndarray, target: float, base_size: float, influence: float) -> np.ndarray:
    if influence <= 1.0e-12:
        return np.full(len(distances), target, dtype=float)
    alpha = np.clip(distances / influence, 0.0, 1.0)
    return target + (base_size - target) * alpha


def _distance_to_circle(points: np.ndarray, center: np.ndarray, normal: np.ndarray, radius: float) -> np.ndarray:
    normal = np.asarray(normal, dtype=float)
    norm = np.linalg.norm(normal)
    if norm <= 1.0e-12:
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        normal = normal / norm
    offsets = points - center[None, :]
    axial = offsets @ normal
    planar = offsets - axial[:, None] * normal[None, :]
    planar_radius = np.linalg.norm(planar, axis=1)
    return np.sqrt((planar_radius - radius) ** 2 + axial**2)


def _distance_to_samples(points: np.ndarray, samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 1:
        samples = samples[None, :]
    diff = points[:, None, :] - samples[None, :, :]
    return np.linalg.norm(diff, axis=2).min(axis=1)


def _cleanup_temp_files(paths: list[str]) -> None:
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def _workspace_tmp_dir() -> Path:
    path = Path.cwd() / ".gmsh_tmp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _node_lookup() -> dict[int, np.ndarray]:
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    coordinates = np.asarray(coords, dtype=float).reshape(-1, 3)
    return {int(tag): coordinates[index] for index, tag in enumerate(node_tags)}


def _surface_connectivity_for_entity(surface_tag: int) -> list[np.ndarray]:
    connectivity_blocks: list[np.ndarray] = []
    element_types, _, node_blocks = gmsh.model.mesh.getElements(2, surface_tag)
    for element_type, node_block in zip(element_types, node_blocks):
        _, _, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(element_type)
        if num_nodes < 3:
            continue
        connectivity = np.asarray(node_block, dtype=int).reshape(-1, num_nodes)
        if len(connectivity) > 120:
            stride = max(1, len(connectivity) // 120)
            connectivity = connectivity[::stride]
        connectivity_blocks.extend(connectivity)
    return connectivity_blocks


def _tetra_edge_lengths(tetra: np.ndarray) -> np.ndarray:
    edge_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    return np.asarray([np.linalg.norm(tetra[i] - tetra[j]) for i, j in edge_pairs], dtype=float)


def _tetra_radius_ratio(tetra: np.ndarray) -> float:
    volume = abs(np.dot(tetra[1] - tetra[0], np.cross(tetra[2] - tetra[0], tetra[3] - tetra[0]))) / 6.0
    if volume <= 1.0e-16:
        return 0.0
    faces = [
        tetra[[1, 2, 3]],
        tetra[[0, 2, 3]],
        tetra[[0, 1, 3]],
        tetra[[0, 1, 2]],
    ]
    areas = [0.5 * np.linalg.norm(np.cross(face[1] - face[0], face[2] - face[0])) for face in faces]
    surface_area = float(sum(areas))
    if surface_area <= 1.0e-16:
        return 0.0
    inradius = 3.0 * volume / surface_area
    a = 2.0 * np.vstack((tetra[1] - tetra[0], tetra[2] - tetra[0], tetra[3] - tetra[0]))
    b = np.asarray([
        np.dot(tetra[1], tetra[1]) - np.dot(tetra[0], tetra[0]),
        np.dot(tetra[2], tetra[2]) - np.dot(tetra[0], tetra[0]),
        np.dot(tetra[3], tetra[3]) - np.dot(tetra[0], tetra[0]),
    ])
    try:
        circumcenter = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return 0.0
    circumradius = np.linalg.norm(circumcenter - tetra[0])
    if circumradius <= 1.0e-16:
        return 0.0
    return float(3.0 * inradius / circumradius)


def _tetra_min_dihedral(tetra: np.ndarray) -> float:
    faces = [
        tetra[[1, 2, 3]],
        tetra[[0, 2, 3]],
        tetra[[0, 1, 3]],
        tetra[[0, 1, 2]],
    ]
    normals = []
    for face in faces:
        normal = np.cross(face[1] - face[0], face[2] - face[0])
        norm = np.linalg.norm(normal)
        normals.append(normal / max(norm, 1.0e-12))
    min_dihedral = 180.0
    face_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for i, j in face_pairs:
        plane_angle = np.degrees(np.arccos(np.clip(np.abs(np.dot(normals[i], normals[j])), -1.0, 1.0)))
        dihedral = 180.0 - plane_angle
        min_dihedral = min(min_dihedral, dihedral)
    return float(min_dihedral)


