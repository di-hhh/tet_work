from __future__ import annotations

from copy import deepcopy
from pathlib import Path, PureWindowsPath
from typing import Any


PATH_PROTOCOL_SCHEMA_VERSION = 1

# These are protocol fields, not a heuristic such as ``key.endswith("path")``.
# Keeping the list explicit prevents ordinary strings in condition/solver metadata
# from being silently rewritten.
OUTPUT_PATH_FIELDS = {
    "adaptive_error_history_path",
    "coarse_mesh_path",
    "final_allocation_diagnostics_path",
    "final_target_mesh_path",
    "geometry_feature_metadata_path",
    "geometry_record_path",
    "indicator_path",
    "initial_mesh_path",
    "initial_surface_mesh_path",
    "mesh_path",
    "optional_error_indicator_path",
    "optional_reference_solution_path",
    "optional_stage_field_path",
    "optional_stage_probe_points_path",
    "preprocess_record_path",
    "probe_field_path",
    "probe_points_path",
    "reference_solution_path",
    "solution_path",
    "stage_field_path",
    "stage_probe_points_path",
    "target_mesh_path",
}

OUTPUT_PATH_LIST_FIELDS = {
    "optional_intermediate_mesh_paths",
    "trajectory_indicator_paths",
    "trajectory_mesh_paths",
    "trajectory_solution_paths",
}

GEOMETRY_SOURCE_PATH_FIELDS = {"source_path"}


class PathProtocolError(ValueError):
    pass


def anchor_repo_path(value: str | Path, repo_root: Path) -> Path:
    """Resolve a config path against its repository, never the shell CWD."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def serialize_record_paths(
    payload: dict[str, Any],
    *,
    pipeline_output_root: Path,
    geometry_source_root: Path,
) -> dict[str, Any]:
    """Return a copy whose declared protocol paths are anchor-relative."""
    return _transform_mapping(
        deepcopy(payload),
        output_root=pipeline_output_root.resolve(),
        geometry_root=geometry_source_root.resolve(),
        serialize=True,
    )


def resolve_record_paths(
    payload: dict[str, Any],
    *,
    pipeline_output_root: Path,
    geometry_source_root: Path,
) -> dict[str, Any]:
    """Return a copy whose declared protocol paths are absolute runtime paths."""
    return _transform_mapping(
        deepcopy(payload),
        output_root=pipeline_output_root.resolve(),
        geometry_root=geometry_source_root.resolve(),
        serialize=False,
    )


def _transform_mapping(
    value: Any,
    *,
    output_root: Path,
    geometry_root: Path,
    serialize: bool,
) -> Any:
    if isinstance(value, dict):
        transformed: dict[str, Any] = {}
        for key, child in value.items():
            if key in GEOMETRY_SOURCE_PATH_FIELDS:
                transformed[key] = _transform_path(child, geometry_root, serialize)
            elif key in OUTPUT_PATH_FIELDS:
                transformed[key] = _transform_path(child, output_root, serialize)
            elif key in OUTPUT_PATH_LIST_FIELDS:
                transformed[key] = [
                    _transform_path(item, output_root, serialize) for item in (child or [])
                ]
            elif key == "metadata" and isinstance(value.get("relative_source_path"), str):
                # GeometryRecord.metadata.root is the geometry source anchor.
                metadata = deepcopy(child)
                if isinstance(metadata, dict) and "root" in metadata:
                    metadata["root"] = "." if serialize else str(geometry_root)
                transformed[key] = _transform_mapping(
                    metadata,
                    output_root=output_root,
                    geometry_root=geometry_root,
                    serialize=serialize,
                )
            else:
                transformed[key] = _transform_mapping(
                    child,
                    output_root=output_root,
                    geometry_root=geometry_root,
                    serialize=serialize,
                )
        return transformed
    if isinstance(value, list):
        return [
            _transform_mapping(
                item,
                output_root=output_root,
                geometry_root=geometry_root,
                serialize=serialize,
            )
            for item in value
        ]
    return value


def _transform_path(value: Any, root: Path, serialize: bool) -> Any:
    if value in {None, ""}:
        return value
    if not isinstance(value, (str, Path)):
        raise PathProtocolError(f"Protocol path must be a string, got {type(value)!r}")
    if serialize:
        return _serialize_path(value, root)
    return str(_resolve_path(value, root))


def _serialize_path(value: str | Path, root: Path) -> str:
    raw = str(value)
    path = Path(raw).resolve() if _is_absolute(raw) else (root / Path(raw)).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise PathProtocolError(
            f"Path '{path}' is outside its declared portable anchor '{root}'. "
            "Move the artifact under tet_work or declare the correct anchor."
        ) from exc
    return relative.as_posix()


def _resolve_path(value: str | Path, root: Path) -> Path:
    raw = str(value)
    if _is_absolute(raw):
        return Path(raw).resolve()
    path = (root / Path(raw)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PathProtocolError(
            f"Relative protocol path '{value}' escapes its anchor '{root}'."
        ) from exc
    return path


def _is_absolute(value: str) -> bool:
    return Path(value).is_absolute() or PureWindowsPath(value).is_absolute()
