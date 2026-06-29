# Generated at 2026-04-08 22:33:28 +08:00 (Asia/Shanghai)
from __future__ import annotations

import math
from itertools import combinations
from typing import Any

import gmsh
import numpy as np


def extract_geometry_features(
    *,
    dimension: int,
    bounding_box: np.ndarray,
    sharp_dihedral_deg: float = 40.0,
) -> dict[str, Any]:
    global_scale = _bbox_diagonal(bounding_box, dimension)
    surfaces = _extract_surface_records(dimension=dimension, global_scale=global_scale)
    curves = _extract_curve_records(global_scale=global_scale)
    point_records = _extract_point_records()

    surface_lookup = {record["tag"]: record for record in surfaces}
    curve_lookup = {record["tag"]: record for record in curves}

    sharp_edges = _detect_sharp_edges(
        curves=curves,
        surface_lookup=surface_lookup,
        sharp_dihedral_deg=sharp_dihedral_deg,
    )
    hole_features = _detect_hole_features(curves=curves, surface_lookup=surface_lookup)
    hole_surface_tags = {
        surface_tag
        for feature in hole_features
        for surface_tag in feature.get("associated_cylinder_surface_tags", [])
    }
    fillet_features = _detect_fillet_features(
        surfaces=surfaces,
        hole_surface_tags=hole_surface_tags,
        global_scale=global_scale,
    )
    thin_regions = _detect_thin_regions(surfaces=surfaces, global_scale=global_scale)
    key_points = _detect_key_points(curves=curves, point_records=point_records)

    local_curvatures = [record.get("curvature_max", 0.0) for record in surfaces + curves]
    finite_surface_radii = [record["curvature_radius"] for record in surfaces if np.isfinite(record.get("curvature_radius", np.inf))]
    finite_curve_radii = [record["radius"] for record in curves if np.isfinite(record.get("radius", np.inf))]

    return {
        "surface_patches": surfaces,
        "curves": curves,
        "points": point_records,
        "feature_anchors": {
            "hole_curve_loops": hole_features,
            "cylinder_patches": [
                {
                    "surface_tag": record["tag"],
                    "radius": record.get("curvature_radius"),
                    "anchor_points": record.get("sample_points", []),
                    "adjacent_curve_tags": record.get("adjacent_curve_tags", []),
                }
                for record in surfaces
                if record.get("surface_type") == "cylinder"
            ],
            "fillet_neighborhoods": fillet_features,
            "thin_regions": thin_regions,
            "sharp_edges": sharp_edges,
            "key_points": key_points,
        },
        "statistics": {
            "num_surfaces": len(surfaces),
            "num_curves": len(curves),
            "num_circle_curves": sum(record.get("curve_type") == "circle" for record in curves),
            "num_arc_curves": sum(record.get("curve_type") == "arc" for record in curves),
            "num_cylinders": sum(record.get("surface_type") == "cylinder" for record in surfaces),
            "num_hole_features": len(hole_features),
            "num_fillet_features": len(fillet_features),
            "num_sharp_edges": len(sharp_edges),
            "max_curvature": float(max(local_curvatures) if local_curvatures else 0.0),
            "mean_curvature": float(np.mean(local_curvatures) if local_curvatures else 0.0),
            "min_surface_radius": float(min(finite_surface_radii) if finite_surface_radii else np.inf),
            "min_curve_radius": float(min(finite_curve_radii) if finite_curve_radii else np.inf),
            "min_thin_wall_thickness": float(
                min((record["distance"] for record in thin_regions), default=np.inf)
            ),
            "global_scale": float(global_scale),
        },
    }


def fit_circle_3d(points: np.ndarray) -> dict[str, Any]:
    points = np.asarray(points, dtype=float)
    if points.shape[0] < 3:
        return {
            "success": False,
            "center": points.mean(axis=0).tolist() if len(points) else [0.0, 0.0, 0.0],
            "radius": float("inf"),
            "normal": [0.0, 0.0, 1.0],
            "mean_error": float("inf"),
            "max_error": float("inf"),
            "max_relative_error": float("inf"),
        }

    center = points.mean(axis=0)
    centered = points - center
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if singular_values.shape[0] < 2 or singular_values[1] <= 1.0e-12:
        return {
            "success": False,
            "center": center.tolist(),
            "radius": float("inf"),
            "normal": [0.0, 0.0, 1.0],
            "mean_error": float("inf"),
            "max_error": float("inf"),
            "max_relative_error": float("inf"),
        }

    basis_u = vh[0]
    basis_v = vh[1]
    normal = np.cross(basis_u, basis_v)
    normal_norm = np.linalg.norm(normal)
    if normal_norm <= 1.0e-12:
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        normal = normal / normal_norm

    projected = np.stack((centered @ basis_u, centered @ basis_v), axis=1)
    system = np.column_stack((2.0 * projected[:, 0], 2.0 * projected[:, 1], np.ones(projected.shape[0])))
    rhs = np.sum(projected**2, axis=1)
    try:
        solution, *_ = np.linalg.lstsq(system, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return {
            "success": False,
            "center": center.tolist(),
            "radius": float("inf"),
            "normal": normal.tolist(),
            "mean_error": float("inf"),
            "max_error": float("inf"),
            "max_relative_error": float("inf"),
        }

    circle_center_2d = solution[:2]
    radius = math.sqrt(max(solution[2] + np.dot(circle_center_2d, circle_center_2d), 0.0))
    circle_center_3d = center + circle_center_2d[0] * basis_u + circle_center_2d[1] * basis_v
    radial_distances = np.linalg.norm(projected - circle_center_2d[None, :], axis=1)
    errors = np.abs(radial_distances - radius)
    return {
        "success": bool(radius > 1.0e-12),
        "center": circle_center_3d.tolist(),
        "radius": float(radius),
        "normal": normal.tolist(),
        "mean_error": float(np.mean(errors)),
        "max_error": float(np.max(errors)),
        "max_relative_error": float(np.max(errors) / max(radius, 1.0e-12)),
    }


def _extract_surface_records(*, dimension: int, global_scale: float) -> list[dict[str, Any]]:
    if dimension < 2:
        return []

    records: list[dict[str, Any]] = []
    for _, tag in gmsh.model.getEntities(2):
        raw_type = gmsh.model.getType(2, tag)
        sample = _sample_surface(tag)
        bbox = np.asarray(gmsh.model.getBoundingBox(2, tag), dtype=float)
        up, down = gmsh.model.getAdjacencies(2, tag)
        curvature_values = np.abs(np.asarray(sample.get("curvature_samples", []), dtype=float))
        principal_max = np.abs(np.asarray(sample.get("principal_curvature_max", []), dtype=float))
        principal_min = np.abs(np.asarray(sample.get("principal_curvature_min", []), dtype=float))
        curvature_radius = _radius_from_curvatures(np.concatenate((principal_max, principal_min), axis=0))
        bbox_size = bbox[3:6] - bbox[:3]
        area = float(gmsh.model.occ.getMass(2, tag))
        center = np.asarray(gmsh.model.occ.getCenterOfMass(2, tag), dtype=float)
        feature_size = float(min(max(np.linalg.norm(bbox_size), 1.0e-12), max(math.sqrt(max(area, 1.0e-12)), 1.0e-12)))
        record = {
            "surface_id": f"surface_{tag:04d}",
            "tag": int(tag),
            "raw_type": raw_type,
            "surface_type": _classify_surface_type(raw_type),
            "bbox": bbox.tolist(),
            "center": center.tolist(),
            "area": area,
            "adjacent_curve_tags": [int(curve_tag) for curve_tag in down.tolist()],
            "adjacent_volume_tags": [int(volume_tag) for volume_tag in up.tolist()],
            "sample_points": sample.get("points", np.zeros((0, 3), dtype=float)).tolist(),
            "sample_normals": sample.get("normals", np.zeros((0, 3), dtype=float)).tolist(),
            "parametric_samples": sample.get("params", np.zeros((0, 2), dtype=float)).tolist(),
            "curvature_max": float(curvature_values.max() if curvature_values.size else 0.0),
            "curvature_mean": float(curvature_values.mean() if curvature_values.size else 0.0),
            "principal_curvature_max": float(principal_max.max() if principal_max.size else 0.0),
            "principal_curvature_min": float(principal_min.max() if principal_min.size else 0.0),
            "curvature_radius": float(curvature_radius),
            "feature_size": feature_size,
            "global_scale_ratio": float(feature_size / max(global_scale, 1.0e-12)),
        }
        records.append(record)
    return records


def _extract_curve_records(*, global_scale: float) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _, tag in gmsh.model.getEntities(1):
        raw_type = gmsh.model.getType(1, tag)
        sample = _sample_curve(tag)
        bbox = np.asarray(gmsh.model.getBoundingBox(1, tag), dtype=float)
        up, down = gmsh.model.getAdjacencies(1, tag)
        length = float(gmsh.model.occ.getMass(1, tag))
        points = sample.get("points", np.zeros((0, 3), dtype=float))
        curvatures = np.abs(np.asarray(sample.get("curvatures", []), dtype=float))
        boundary_points = gmsh.model.getBoundary([(1, tag)], oriented=False, recursive=False)
        is_closed = len(boundary_points) == 0
        fit = fit_circle_3d(points) if raw_type.lower().startswith(("circle", "ellipse")) and len(points) >= 3 else None
        radius_from_curvature = _radius_from_curvatures(curvatures)
        radius = float(fit["radius"] if fit and fit.get("success") else radius_from_curvature)
        center = np.asarray(gmsh.model.occ.getCenterOfMass(1, tag), dtype=float)
        feature_size = float(min(max(length, 1.0e-12), max(2.0 * radius if np.isfinite(radius) else length, 1.0e-12)))
        record = {
            "curve_id": f"curve_{tag:04d}",
            "tag": int(tag),
            "raw_type": raw_type,
            "curve_type": _classify_curve_type(raw_type=raw_type, is_closed=is_closed),
            "bbox": bbox.tolist(),
            "center": center.tolist(),
            "length": length,
            "is_closed": bool(is_closed),
            "endpoint_tags": [int(point_tag) for _, point_tag in boundary_points],
            "adjacent_surface_tags": [int(surface_tag) for surface_tag in up.tolist()],
            "adjacent_point_tags": [int(point_tag) for point_tag in down.tolist()],
            "sample_points": points.tolist(),
            "parametric_samples": np.asarray(sample.get("params", []), dtype=float).reshape(-1).tolist(),
            "curvature_max": float(curvatures.max() if curvatures.size else 0.0),
            "curvature_mean": float(curvatures.mean() if curvatures.size else 0.0),
            "radius": radius,
            "feature_size": feature_size,
            "global_scale_ratio": float(feature_size / max(global_scale, 1.0e-12)),
            "fit_normal": fit.get("normal") if fit else None,
            "fit_center": fit.get("center") if fit else None,
            "fit_max_relative_error": float(fit.get("max_relative_error", 0.0)) if fit else 0.0,
        }
        if fit is not None:
            record["fit_radius"] = float(fit["radius"])
            record["fit_max_error"] = float(fit["max_error"])
            record["fit_mean_error"] = float(fit["mean_error"])
        records.append(record)
    return records


def _extract_point_records() -> list[dict[str, Any]]:
    records = []
    for _, tag in gmsh.model.getEntities(0):
        bbox = np.asarray(gmsh.model.getBoundingBox(0, tag), dtype=float)
        center = 0.5 * (bbox[:3] + bbox[3:6])
        up, down = gmsh.model.getAdjacencies(0, tag)
        records.append(
            {
                "point_id": f"point_{tag:04d}",
                "tag": int(tag),
                "coordinates": center.tolist(),
                "adjacent_curve_tags": [int(curve_tag) for curve_tag in up.tolist()],
                "downstream_tags": [int(child_tag) for child_tag in down.tolist()],
            }
        )
    return records


def _detect_hole_features(curves: list[dict[str, Any]], surface_lookup: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    hole_features: list[dict[str, Any]] = []
    for curve in curves:
        if curve.get("curve_type") != "circle":
            continue
        cylinder_surface_tags = [
            int(surface_tag)
            for surface_tag in curve.get("adjacent_surface_tags", [])
            if surface_lookup.get(surface_tag, {}).get("surface_type") == "cylinder"
        ]
        if not cylinder_surface_tags and not curve.get("is_closed", False):
            continue
        radius = float(curve.get("radius", np.inf))
        if not np.isfinite(radius):
            continue
        hole_features.append(
            {
                "feature_id": f"hole_{curve['tag']:04d}",
                "curve_tag": int(curve["tag"]),
                "radius": radius,
                "center": curve.get("fit_center") or curve.get("center"),
                "normal": curve.get("fit_normal") or [0.0, 0.0, 1.0],
                "anchor_points": curve.get("sample_points", []),
                "associated_cylinder_surface_tags": cylinder_surface_tags,
                "adjacent_surface_tags": list(curve.get("adjacent_surface_tags", [])),
                "feature_size": float(curve.get("feature_size", 2.0 * radius)),
            }
        )
    return hole_features


def _detect_fillet_features(
    *,
    surfaces: list[dict[str, Any]],
    hole_surface_tags: set[int],
    global_scale: float,
) -> list[dict[str, Any]]:
    threshold = max(0.2 * global_scale, 1.0e-6)
    fillets: list[dict[str, Any]] = []
    for surface in surfaces:
        if surface["tag"] in hole_surface_tags:
            continue
        if surface.get("surface_type") not in {"cylinder", "cone", "torus", "sphere", "bspline", "generic_curved"}:
            continue
        radius = float(surface.get("curvature_radius", np.inf))
        if np.isfinite(radius) and radius <= threshold:
            fillets.append(
                {
                    "feature_id": f"fillet_{surface['tag']:04d}",
                    "surface_tag": int(surface["tag"]),
                    "surface_type": surface.get("surface_type"),
                    "radius": radius,
                    "anchor_points": surface.get("sample_points", []),
                    "adjacent_curve_tags": list(surface.get("adjacent_curve_tags", [])),
                }
            )
    return fillets


def _detect_thin_regions(*, surfaces: list[dict[str, Any]], global_scale: float) -> list[dict[str, Any]]:
    plane_surfaces = []
    for surface in surfaces:
        if surface.get("surface_type") != "plane":
            continue
        normals = np.asarray(surface.get("sample_normals", []), dtype=float)
        if normals.size == 0:
            continue
        normal = normals.mean(axis=0)
        norm = np.linalg.norm(normal)
        if norm <= 1.0e-12:
            continue
        plane_surfaces.append((surface, normal / norm))

    candidates: list[dict[str, Any]] = []
    for (surface_a, normal_a), (surface_b, normal_b) in combinations(plane_surfaces, 2):
        alignment = float(np.abs(np.dot(normal_a, normal_b)))
        if alignment < 0.95:
            continue
        center_a = np.asarray(surface_a["center"], dtype=float)
        center_b = np.asarray(surface_b["center"], dtype=float)
        distance = float(abs(np.dot(center_b - center_a, normal_a)))
        if distance <= 1.0e-9:
            continue
        if distance > max(0.3 * global_scale, 1.0e-5):
            continue
        candidates.append(
            {
                "feature_id": f"thin_{surface_a['tag']:04d}_{surface_b['tag']:04d}",
                "surface_tags": [int(surface_a["tag"]), int(surface_b["tag"])],
                "distance": distance,
                "normal": normal_a.tolist(),
                "centers": [surface_a["center"], surface_b["center"]],
            }
        )
    candidates.sort(key=lambda item: item["distance"])
    return candidates[:8]


def _detect_sharp_edges(
    *,
    curves: list[dict[str, Any]],
    surface_lookup: dict[int, dict[str, Any]],
    sharp_dihedral_deg: float,
) -> list[dict[str, Any]]:
    sharp_edges: list[dict[str, Any]] = []
    for curve in curves:
        adjacent = [surface_lookup[tag] for tag in curve.get("adjacent_surface_tags", []) if tag in surface_lookup]
        if len(adjacent) < 2:
            continue
        sample_points = np.asarray(curve.get("sample_points", []), dtype=float)
        if sample_points.size == 0:
            continue
        midpoint = sample_points[len(sample_points) // 2]
        for surface_a, surface_b in combinations(adjacent, 2):
            normal_a = _surface_normal_at_point(surface_a["tag"], midpoint)
            normal_b = _surface_normal_at_point(surface_b["tag"], midpoint)
            if normal_a is None or normal_b is None:
                continue
            angle_deg = float(np.degrees(np.arccos(np.clip(np.abs(np.dot(normal_a, normal_b)), -1.0, 1.0))))
            if angle_deg >= sharp_dihedral_deg:
                sharp_edges.append(
                    {
                        "feature_id": f"sharp_{curve['tag']:04d}_{surface_a['tag']:04d}_{surface_b['tag']:04d}",
                        "curve_tag": int(curve["tag"]),
                        "surface_tags": [int(surface_a["tag"]), int(surface_b["tag"])],
                        "angle_deg": angle_deg,
                        "anchor_points": curve.get("sample_points", []),
                        "feature_size": float(curve.get("feature_size", 0.0)),
                    }
                )
    sharp_edges.sort(key=lambda item: item["angle_deg"], reverse=True)
    return sharp_edges


def _detect_key_points(*, curves: list[dict[str, Any]], point_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    point_lookup = {record["tag"]: record for record in point_records}
    key_points: list[dict[str, Any]] = []
    for record in point_records:
        valence = len(record.get("adjacent_curve_tags", []))
        if valence != 2:
            key_points.append(
                {
                    "point_tag": int(record["tag"]),
                    "coordinates": record["coordinates"],
                    "valence": valence,
                }
            )

    for curve in curves:
        if curve.get("curve_type") not in {"arc", "ellipse", "generic_spline"}:
            continue
        for point_tag in curve.get("endpoint_tags", []):
            record = point_lookup.get(point_tag)
            if record is None:
                continue
            key_points.append(
                {
                    "point_tag": int(point_tag),
                    "coordinates": record["coordinates"],
                    "valence": len(record.get("adjacent_curve_tags", [])),
                }
            )

    unique = {}
    for record in key_points:
        unique[record["point_tag"]] = record
    return list(unique.values())


def _surface_normal_at_point(surface_tag: int, point: np.ndarray) -> np.ndarray | None:
    try:
        _, params = gmsh.model.getClosestPoint(2, surface_tag, point.astype(float).tolist())
        if len(params) < 2:
            return None
        normal = np.asarray(gmsh.model.getNormal(surface_tag, params[:2]), dtype=float).reshape(-1, 3)[0]
        norm = np.linalg.norm(normal)
        if norm <= 1.0e-12:
            return None
        return normal / norm
    except Exception:
        return None


def _sample_curve(tag: int, num_samples: int = 9) -> dict[str, Any]:
    try:
        param_min, param_max = gmsh.model.getParametrizationBounds(1, tag)
    except Exception:
        return {"params": np.zeros((0,), dtype=float), "points": np.zeros((0, 3), dtype=float), "curvatures": np.zeros((0,), dtype=float)}

    start = float(param_min[0])
    stop = float(param_max[0])
    if abs(stop - start) <= 1.0e-12:
        params = np.asarray([start], dtype=float)
    else:
        params = np.linspace(start, stop, num=num_samples, dtype=float)
    points = np.asarray(gmsh.model.getValue(1, tag, params.tolist()), dtype=float).reshape(-1, 3)
    try:
        curvatures = np.asarray(gmsh.model.getCurvature(1, tag, params.tolist()), dtype=float)
    except Exception:
        curvatures = np.zeros(len(params), dtype=float)
    return {"params": params, "points": points, "curvatures": curvatures}


def _sample_surface(tag: int, num_u: int = 4, num_v: int = 4) -> dict[str, Any]:
    try:
        param_min, param_max = gmsh.model.getParametrizationBounds(2, tag)
    except Exception:
        empty = np.zeros((0, 3), dtype=float)
        return {
            "params": np.zeros((0, 2), dtype=float),
            "points": empty,
            "normals": empty,
            "curvature_samples": np.zeros((0,), dtype=float),
            "principal_curvature_max": np.zeros((0,), dtype=float),
            "principal_curvature_min": np.zeros((0,), dtype=float),
        }

    u_values = _linspace_inside(float(param_min[0]), float(param_max[0]), num_u)
    v_values = _linspace_inside(float(param_min[1]), float(param_max[1]), num_v)
    grid = np.asarray([[u, v] for u in u_values for v in v_values], dtype=float)
    flat_params = grid.reshape(-1).tolist()
    points = np.asarray(gmsh.model.getValue(2, tag, flat_params), dtype=float).reshape(-1, 3)
    try:
        normals = np.asarray(gmsh.model.getNormal(tag, flat_params), dtype=float).reshape(-1, 3)
    except Exception:
        normals = np.zeros_like(points)
    try:
        curvatures = np.asarray(gmsh.model.getCurvature(2, tag, flat_params), dtype=float)
    except Exception:
        curvatures = np.zeros(len(grid), dtype=float)
    try:
        principal_max, principal_min, _, _ = gmsh.model.getPrincipalCurvatures(tag, flat_params)
        principal_max = np.asarray(principal_max, dtype=float)
        principal_min = np.asarray(principal_min, dtype=float)
    except Exception:
        principal_max = np.zeros(len(grid), dtype=float)
        principal_min = np.zeros(len(grid), dtype=float)
    return {
        "params": grid,
        "points": points,
        "normals": normals,
        "curvature_samples": curvatures,
        "principal_curvature_max": principal_max,
        "principal_curvature_min": principal_min,
    }


def _classify_surface_type(raw_type: str) -> str:
    lowered = raw_type.lower()
    if "plane" in lowered:
        return "plane"
    if "cylinder" in lowered:
        return "cylinder"
    if "cone" in lowered:
        return "cone"
    if "sphere" in lowered:
        return "sphere"
    if "torus" in lowered:
        return "torus"
    if "bspline" in lowered or "spline" in lowered:
        return "bspline"
    if "bezier" in lowered or "surface" in lowered:
        return "generic_curved"
    return "generic_curved"


def _classify_curve_type(*, raw_type: str, is_closed: bool) -> str:
    lowered = raw_type.lower()
    if "line" in lowered:
        return "line"
    if "circle" in lowered:
        return "circle" if is_closed else "arc"
    if "ellipse" in lowered:
        return "ellipse" if is_closed else "arc"
    if "spline" in lowered or "bezier" in lowered or "bspline" in lowered:
        return "generic_spline"
    return "generic_spline"


def _linspace_inside(start: float, stop: float, num: int) -> np.ndarray:
    if abs(stop - start) <= 1.0e-12:
        return np.asarray([start], dtype=float)
    margin = 0.05 * (stop - start)
    return np.linspace(start + margin, stop - margin, num=num, dtype=float)


def _radius_from_curvatures(curvatures: np.ndarray) -> float:
    curvatures = np.abs(np.asarray(curvatures, dtype=float))
    valid = curvatures > 1.0e-10
    if not np.any(valid):
        return float(np.inf)
    radii = 1.0 / curvatures[valid]
    return float(np.median(radii))


def _bbox_diagonal(bounding_box: np.ndarray, dimension: int) -> float:
    bbox = np.asarray(bounding_box, dtype=float)
    if dimension == 2 and bbox.shape[0] >= 4:
        mins = bbox[:2]
        maxs = bbox[2:4]
    else:
        mins = bbox[:3]
        maxs = bbox[3:6]
    return float(np.linalg.norm(maxs - mins))
