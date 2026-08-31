from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import meshio
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr

from src.condition_aware_dataset_generation.records import PrescreenRecord, SampleRecord
from src.condition_aware_dataset_generation.utils import dump_json
from src.tasks.domains.geometry_util import get_simplex_volumes_from_indices, volume_to_edge_length


HOTSPOT_EPS = 1.0e-12
SUPPORTED_CELL_TYPES = ('triangle', 'tetra')
STAGE_FIELD_MODES = {
    's_pde_raw': 'high',
    'h_pde_only': 'low',
    'h_after_geometry_fusion': 'low',
    'h_after_budget_calibration': 'low',
}


def _is_success_status(status: str | None) -> bool:
    return str(status or '').startswith('success') or status == 'success'


def _simplex_connectivity(mesh: meshio.Mesh) -> np.ndarray:
    for block in mesh.cells:
        if block.type in SUPPORTED_CELL_TYPES:
            return np.asarray(block.data, dtype=np.int64)
    raise ValueError('Only triangle and tetra target meshes are currently supported for smoke analysis')


def _mesh_points_and_elements(mesh_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = meshio.read(str(mesh_path))
    return np.asarray(mesh.points, dtype=float), _simplex_connectivity(mesh)


def _element_sizes(points: np.ndarray, simplices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dim = int(simplices.shape[1] - 1)
    volumes = get_simplex_volumes_from_indices(points[:, :dim], simplices)
    return volumes, volume_to_edge_length(volumes, dim=dim)


def _vertex_average(points: np.ndarray, simplices: np.ndarray, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    accum = np.zeros(points.shape[0], dtype=float)
    counts = np.zeros(points.shape[0], dtype=float)
    np.add.at(accum, simplices.reshape(-1), np.repeat(values, simplices.shape[1]))
    np.add.at(counts, simplices.reshape(-1), 1.0)
    return accum / np.maximum(counts, 1.0)


def _safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
    if np.allclose(left, left[0]) and np.allclose(right, right[0]):
        return 1.0 if np.isclose(left[0], right[0]) else 0.0
    return float(pearsonr(left, right).statistic)


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if np.allclose(left, left[0]) and np.allclose(right, right[0]):
        return 1.0 if np.isclose(left[0], right[0]) else 0.0
    return float(spearmanr(left, right).statistic)


def _relative_l2(left: np.ndarray, right: np.ndarray) -> float:
    diff = np.linalg.norm(left - right)
    denom = max(np.linalg.norm(left), np.linalg.norm(right), HOTSPOT_EPS)
    return float(diff / denom)


def _jaccard(mask_left: np.ndarray, mask_right: np.ndarray) -> float:
    union = np.logical_or(mask_left, mask_right).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(mask_left, mask_right).sum() / union)


def _load_stage_bundle(path: str | Path | None) -> dict[str, np.ndarray]:
    if not path:
        return {}
    payload = np.load(path)
    return {key: np.asarray(payload[key], dtype=float) for key in payload.files}


def analyze_sample_mesh(
    sample_record: SampleRecord,
    *,
    hotspot_quantile: float,
    low_error_quantile: float = 0.5,
) -> dict[str, Any]:
    points, simplices = _mesh_points_and_elements(sample_record.final_target_mesh_path)
    volumes, sizes = _element_sizes(points, simplices)
    indicator = np.asarray(np.load(sample_record.optional_error_indicator_path), dtype=float)
    if indicator.shape[0] != simplices.shape[0]:
        raise ValueError(f'Indicator shape mismatch for {sample_record.sample_id}')

    hotspot_threshold = float(np.quantile(indicator, hotspot_quantile))
    low_threshold = float(np.quantile(indicator, low_error_quantile))
    hotspot_mask = indicator >= hotspot_threshold
    low_mask = indicator <= low_threshold
    error_mass = indicator * volumes
    hotspot_concentration = float(error_mass[hotspot_mask].sum() / max(error_mass.sum(), HOTSPOT_EPS))
    hotspot_element_fraction = float(np.mean(hotspot_mask))
    hotspot_volume_fraction = float(volumes[hotspot_mask].sum() / max(volumes.sum(), HOTSPOT_EPS))
    hotspot_mean_size = float(np.mean(sizes[hotspot_mask])) if np.any(hotspot_mask) else float(np.mean(sizes))
    low_importance_mean_size = float(np.mean(sizes[low_mask])) if np.any(low_mask) else float(np.mean(sizes))
    hotspot_size_ratio = float(np.median(sizes[hotspot_mask]) / max(np.median(sizes[low_mask]), HOTSPOT_EPS))
    allocation_gain = float(hotspot_element_fraction / max(hotspot_volume_fraction, HOTSPOT_EPS))

    teacher_metadata = dict(sample_record.teacher_metadata or {})
    budget_diagnostics = dict(teacher_metadata.get('budget_diagnostics', {}))
    initial_mesh_diagnostics = dict(teacher_metadata.get('initial_mesh_diagnostics', {}))
    final_allocation_diagnostics = dict(teacher_metadata.get('final_allocation_diagnostics', {}))

    return {
        'sample_id': sample_record.sample_id,
        'geometry_id': sample_record.geometry_id,
        'condition_id': sample_record.condition_id,
        'pde_family': sample_record.pde_family,
        'budget': sample_record.budget,
        'num_elements': int(simplices.shape[0]),
        'hotspot_quantile': float(hotspot_quantile),
        'hotspot_threshold': hotspot_threshold,
        'hotspot_size_ratio': hotspot_size_ratio,
        'final_hotspot_size_ratio': hotspot_size_ratio,
        'hotspot_element_fraction': hotspot_element_fraction,
        'final_hotspot_element_fraction': hotspot_element_fraction,
        'hotspot_volume_fraction': hotspot_volume_fraction,
        'final_hotspot_volume_fraction': hotspot_volume_fraction,
        'allocation_gain': allocation_gain,
        'final_allocation_gain': allocation_gain,
        'hotspot_concentration': hotspot_concentration,
        'indicator_mean': float(np.mean(indicator)),
        'indicator_q90': float(np.quantile(indicator, 0.9)),
        'indicator_q99': float(np.quantile(indicator, 0.99)),
        'size_q10': float(np.quantile(sizes, 0.1)),
        'size_q50': float(np.quantile(sizes, 0.5)),
        'size_q90': float(np.quantile(sizes, 0.9)),
        'size_nonuniform_ratio': float(np.quantile(sizes, 0.9) / max(np.quantile(sizes, 0.1), HOTSPOT_EPS)),
        'hotspot_mean_size': hotspot_mean_size,
        'low_importance_mean_size': low_importance_mean_size,
        'mesh_path': sample_record.final_target_mesh_path,
        'indicator_path': sample_record.optional_error_indicator_path,
        'stage_field_path': sample_record.optional_stage_field_path,
        'stage_probe_points_path': sample_record.optional_stage_probe_points_path,
        'probe_vertex_sizes': _vertex_average(points, simplices, sizes).tolist(),
        'probe_vertex_hotspots': _vertex_average(points, simplices, indicator).tolist(),
        'budget_diagnostics': budget_diagnostics,
        'budget_status': budget_diagnostics.get('budget_status', budget_diagnostics.get('status', sample_record.status)),
        'budget_growth_diagnostics': dict(teacher_metadata.get('budget_growth_diagnostics', {})),
        'final_allocation_diagnostics': final_allocation_diagnostics,
        'final_allocation_diagnostics_path': teacher_metadata.get('final_allocation_diagnostics_path'),
        'initial_mesh_diagnostics': initial_mesh_diagnostics,
        'target_budget': int(budget_diagnostics.get('target_budget', sample_record.budget)),
        'actual_budget': int(budget_diagnostics.get('actual_budget', simplices.shape[0])),
        'budget_ratio': float(budget_diagnostics.get('budget_ratio', simplices.shape[0] / max(float(sample_record.budget), 1.0))),
        'lambda': float(budget_diagnostics.get('lambda', 1.0)),
        'calibration_iters': int(budget_diagnostics.get('calibration_iters', 0)),
        'calibration_converged': bool(budget_diagnostics.get('calibration_converged', False)),
        'initial_num_elements': int(initial_mesh_diagnostics.get('initial_num_elements', 0) or 0),
        'initial_num_nodes': int(initial_mesh_diagnostics.get('initial_num_nodes', 0) or 0),
        'initial_budget_fraction': float(initial_mesh_diagnostics.get('initial_budget_fraction', 0.0) or 0.0),
        'initial_is_too_dense': bool(initial_mesh_diagnostics.get('initial_is_too_dense', False)),
    }


def compare_condition_fields(
    *,
    geometry_id: str,
    sample_metrics: list[dict[str, Any]],
    probe_points: np.ndarray,
    hotspot_quantile: float,
    finest_fraction: float = 0.2,
) -> dict[str, dict[str, Any]]:
    pointwise_sizes: dict[str, np.ndarray] = {}
    pointwise_hotspots: dict[str, np.ndarray] = {}
    for metrics in sample_metrics:
        points, simplices = _mesh_points_and_elements(metrics['mesh_path'])
        _, sizes = _element_sizes(points, simplices)
        indicator = np.asarray(np.load(metrics['indicator_path']), dtype=float)
        vertex_sizes = _vertex_average(points, simplices, sizes)
        vertex_hotspots = _vertex_average(points, simplices, indicator)
        tree = cKDTree(points)
        _, indices = tree.query(probe_points)
        pointwise_sizes[metrics['condition_id']] = vertex_sizes[indices]
        pointwise_hotspots[metrics['condition_id']] = vertex_hotspots[indices]

    comparisons: dict[str, dict[str, Any]] = {}
    condition_ids = list(pointwise_sizes)
    for left_index, left_condition in enumerate(condition_ids):
        for right_condition in condition_ids[left_index + 1 :]:
            left_sizes = pointwise_sizes[left_condition]
            right_sizes = pointwise_sizes[right_condition]
            left_hotspots = pointwise_hotspots[left_condition]
            right_hotspots = pointwise_hotspots[right_condition]
            left_fine_mask = left_sizes <= np.quantile(left_sizes, finest_fraction)
            right_fine_mask = right_sizes <= np.quantile(right_sizes, finest_fraction)
            left_hot_mask = left_hotspots >= np.quantile(left_hotspots, hotspot_quantile)
            right_hot_mask = right_hotspots >= np.quantile(right_hotspots, hotspot_quantile)
            left_density = 1.0 / np.maximum(left_sizes, HOTSPOT_EPS)
            right_density = 1.0 / np.maximum(right_sizes, HOTSPOT_EPS)
            union_hot_mask = np.logical_or(left_hot_mask, right_hot_mask)
            if not np.any(union_hot_mask):
                union_hot_mask = np.ones_like(left_hot_mask, dtype=bool)
            density_difference = _relative_l2(left_density[union_hot_mask], right_density[union_hot_mask])
            key = f'{left_condition}__vs__{right_condition}'
            comparisons[key] = {
                'geometry_id': geometry_id,
                'left_condition_id': left_condition,
                'right_condition_id': right_condition,
                'probe_wise_sizing_pearson': _safe_pearson(left_sizes, right_sizes),
                'final_sizing_field_pearson': _safe_pearson(left_sizes, right_sizes),
                'probe_wise_sizing_spearman': _safe_spearman(left_sizes, right_sizes),
                'probe_wise_sizing_relative_l2_difference': _relative_l2(left_sizes, right_sizes),
                'final_sizing_field_relative_l2': _relative_l2(left_sizes, right_sizes),
                'median_relative_size_difference': float(np.median(np.abs(left_sizes - right_sizes) / np.maximum(np.minimum(left_sizes, right_sizes), HOTSPOT_EPS))),
                'finest_20_region_jaccard': _jaccard(left_fine_mask, right_fine_mask),
                'hotspot_region_jaccard': _jaccard(left_hot_mask, right_hot_mask),
                'region_wise_density_difference': density_difference,
            }
    return comparisons


def _compare_aligned_fields(
    *,
    geometry_id: str,
    stage_name: str,
    field_lookup: dict[str, np.ndarray],
    hotspot_quantile: float,
    finest_fraction: float,
    mode: str,
) -> dict[str, dict[str, Any]]:
    comparisons: dict[str, dict[str, Any]] = {}
    condition_ids = list(field_lookup)
    hotspot_size_fraction = max(1.0 - hotspot_quantile, 0.05)
    for left_index, left_condition in enumerate(condition_ids):
        for right_condition in condition_ids[left_index + 1 :]:
            left = field_lookup[left_condition]
            right = field_lookup[right_condition]
            if mode == 'low':
                left_fine_mask = left <= np.quantile(left, finest_fraction)
                right_fine_mask = right <= np.quantile(right, finest_fraction)
                left_hot_mask = left <= np.quantile(left, hotspot_size_fraction)
                right_hot_mask = right <= np.quantile(right, hotspot_size_fraction)
            else:
                left_fine_mask = left >= np.quantile(left, 1.0 - finest_fraction)
                right_fine_mask = right >= np.quantile(right, 1.0 - finest_fraction)
                left_hot_mask = left >= np.quantile(left, hotspot_quantile)
                right_hot_mask = right >= np.quantile(right, hotspot_quantile)
            key = f'{left_condition}__vs__{right_condition}'
            comparisons[key] = {
                'geometry_id': geometry_id,
                'stage_name': stage_name,
                'left_condition_id': left_condition,
                'right_condition_id': right_condition,
                'pearson': _safe_pearson(left, right),
                'spearman': _safe_spearman(left, right),
                'relative_l2_difference': _relative_l2(left, right),
                'finest_20_region_jaccard': _jaccard(left_fine_mask, right_fine_mask),
                'hotspot_region_jaccard': _jaccard(left_hot_mask, right_hot_mask),
            }
    return comparisons


def compare_stage_fields(
    *,
    geometry_id: str,
    sample_metrics: list[dict[str, Any]],
    hotspot_quantile: float,
    finest_fraction: float = 0.2,
) -> dict[str, dict[str, dict[str, Any]]]:
    stage_payloads = {stage_name: {} for stage_name in STAGE_FIELD_MODES}
    for metrics in sample_metrics:
        bundle = _load_stage_bundle(metrics.get('stage_field_path'))
        if not bundle:
            continue
        for stage_name in STAGE_FIELD_MODES:
            if stage_name in bundle:
                stage_payloads[stage_name][metrics['condition_id']] = np.asarray(bundle[stage_name], dtype=float)

    comparisons: dict[str, dict[str, dict[str, Any]]] = {}
    for stage_name, payload in stage_payloads.items():
        if len(payload) < 2:
            comparisons[stage_name] = {}
            continue
        comparisons[stage_name] = _compare_aligned_fields(
            geometry_id=geometry_id,
            stage_name=stage_name,
            field_lookup=payload,
            hotspot_quantile=hotspot_quantile,
            finest_fraction=finest_fraction,
            mode=STAGE_FIELD_MODES[stage_name],
        )
    return comparisons


def detect_collapse_stage(
    stage_pairwise_metrics: dict[str, dict[str, dict[str, Any]]],
    final_pairwise_metrics: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    collapse_flags: list[dict[str, Any]] = []
    pde_pairs = stage_pairwise_metrics.get('s_pde_raw', {})
    fusion_pairs = stage_pairwise_metrics.get('h_after_geometry_fusion', {})
    budget_pairs = stage_pairwise_metrics.get('h_after_budget_calibration', {})
    final_pairwise_metrics = final_pairwise_metrics or {}
    for key, pde_metric in pde_pairs.items():
        fusion_metric = fusion_pairs.get(key)
        budget_metric = budget_pairs.get(key)
        final_metric = final_pairwise_metrics.get(key)
        if fusion_metric is not None:
            if pde_metric['relative_l2_difference'] >= 0.12 and fusion_metric['relative_l2_difference'] <= 0.06 and fusion_metric['hotspot_region_jaccard'] >= 0.88:
                collapse_flags.append(
                    {
                        'pair_key': key,
                        'collapse_stage': 'after_fusion',
                        'taxonomy_category': 'condition_difference_collapsed_after_fusion',
                    }
                )
        if fusion_metric is not None and budget_metric is not None:
            if fusion_metric['relative_l2_difference'] >= 0.10 and budget_metric['relative_l2_difference'] <= 0.06 and budget_metric['hotspot_region_jaccard'] >= 0.88:
                collapse_flags.append(
                    {
                        'pair_key': key,
                        'collapse_stage': 'after_budget',
                        'taxonomy_category': 'condition_difference_collapsed_after_budget',
                    }
                )
        if budget_metric is not None and final_metric is not None:
            if (
                budget_metric['relative_l2_difference'] >= 0.10
                and final_metric['final_sizing_field_relative_l2'] <= 0.06
                and final_metric['hotspot_region_jaccard'] >= 0.88
            ):
                collapse_flags.append(
                    {
                        'pair_key': key,
                        'collapse_stage': 'mesh_stage',
                        'taxonomy_category': 'condition_difference_collapsed_at_mesh_stage',
                        'before_budget_relative_l2': float(budget_metric['relative_l2_difference']),
                        'after_final_relative_l2': float(final_metric['final_sizing_field_relative_l2']),
                        'final_hotspot_region_jaccard': float(final_metric['hotspot_region_jaccard']),
                    }
                )
    if not collapse_flags:
        return {'collapse_stage': None, 'flags': []}
    if any(flag['collapse_stage'] == 'after_fusion' for flag in collapse_flags):
        return {'collapse_stage': 'after_fusion', 'flags': collapse_flags}
    if any(flag['collapse_stage'] == 'mesh_stage' for flag in collapse_flags):
        return {'collapse_stage': 'mesh_stage', 'flags': collapse_flags}
    return {'collapse_stage': 'after_budget', 'flags': collapse_flags}


def verdict_for_geometry(
    *,
    sample_metrics: list[dict[str, Any]],
    pairwise_metrics: dict[str, dict[str, Any]],
    stage_pairwise_metrics: dict[str, dict[str, dict[str, Any]]],
    prescreen_records: list[PrescreenRecord],
    target_hotspot_size_ratio: float,
    overlap_threshold: float,
) -> tuple[str, list[str], dict[str, Any]]:
    reasons: list[str] = []
    collapse = detect_collapse_stage(stage_pairwise_metrics, pairwise_metrics)
    if any(record.label == 'reject_too_expensive' for record in prescreen_records) and not sample_metrics:
        return 'FAIL_TOO_EXPENSIVE', ['all candidate conditions were rejected as too expensive'], collapse
    if not sample_metrics:
        return 'FAIL_TIMEOUT', ['no successful teacher samples were generated'], collapse

    if any(str(metrics.get('budget_status', '')).startswith('fail_budget') for metrics in sample_metrics):
        return 'FAIL_PARTIAL_TOO_COARSE', ['budget growth failed before a usable final target was produced'], collapse
    if any(metrics.get('budget_status') == 'success_partial_under_budget' and metrics.get('actual_budget', 0) < metrics.get('target_budget', 1) for metrics in sample_metrics):
        if min(metrics.get('budget_diagnostics', {}).get('minimum_viable_budget_ratio', 0.0) for metrics in sample_metrics) < 1.0:
            return 'FAIL_PARTIAL_TOO_COARSE', ['partial target stayed below minimum viable budget'], collapse

    hotspot_size_values = [metrics['hotspot_size_ratio'] for metrics in sample_metrics]
    allocation_values = [metrics['allocation_gain'] for metrics in sample_metrics]
    if max(hotspot_size_values) >= 1.05 and max(allocation_values) <= 1.15:
        return 'FAIL_GLOBAL_OVERREFINE', ['hotspot regions are not significantly denser than the far field'], collapse
    if collapse['collapse_stage'] == 'after_fusion':
        return 'FAIL_CONDITION_COLLAPSE_AT_FINAL_MESH', ['condition difference is present in the PDE field but collapses after geometry fusion'], collapse
    if collapse['collapse_stage'] == 'after_budget':
        return 'FAIL_CONDITION_COLLAPSE_AT_FINAL_MESH', ['condition difference survives fusion but collapses after budget calibration'], collapse
    if collapse['collapse_stage'] == 'mesh_stage':
        return 'FAIL_CONDITION_COLLAPSE_AT_FINAL_MESH', ['condition difference survives sizing fields but collapses on the final mesh'], collapse

    if pairwise_metrics:
        max_hotspot_overlap = max(metric['hotspot_region_jaccard'] for metric in pairwise_metrics.values())
        min_size_diff = min(metric['median_relative_size_difference'] for metric in pairwise_metrics.values())
        min_density_diff = min(metric.get('region_wise_density_difference', 0.0) for metric in pairwise_metrics.values())
        if max_hotspot_overlap >= overlap_threshold and min_size_diff <= 0.08:
            return 'FAIL_CONDITION_COLLAPSE_AT_FINAL_MESH', ['different conditions still produce overly similar final hotspots and size fields'], collapse
        if max(hotspot_size_values) <= target_hotspot_size_ratio and min(allocation_values) >= 1.20 and max_hotspot_overlap < overlap_threshold and min_density_diff >= 0.08:
            reasons.append('hotspot regions are strongly refined and cross-condition overlap stays low')
            return 'PASS_STRONG', reasons, collapse
        reasons.append('local refinement is present, but cross-condition separation remains moderate')
        return 'PASS_WEAK', reasons, collapse

    if max(hotspot_size_values) <= target_hotspot_size_ratio and min(allocation_values) >= 1.15:
        return 'PASS_WEAK', ['single-condition result shows non-global refinement'], collapse
    if min(allocation_values) >= 1.20 and max(hotspot_size_values) <= 1.10:
        return 'PASS_WEAK', ['single-condition result allocates more elements to the hotspot, but size contrast is still weak'], collapse
    return 'FAIL_GLOBAL_OVERREFINE', ['single-condition result did not reach the target hotspot contrast'], collapse


def _family_report(
    *,
    family_name: str,
    pde_family: str,
    sample_records: list[SampleRecord],
    prescreen_records: list[PrescreenRecord],
    hotspot_quantile: float,
    target_hotspot_size_ratio: float,
    overlap_threshold: float,
    enabled: bool,
) -> dict[str, Any]:
    by_geometry: dict[str, list[SampleRecord]] = defaultdict(list)
    prescreen_by_geometry: dict[str, list[PrescreenRecord]] = defaultdict(list)
    for sample_record in sample_records:
        if sample_record.pde_family != pde_family:
            continue
        if not _is_success_status(sample_record.status) or not sample_record.final_target_mesh_path or not sample_record.optional_error_indicator_path:
            continue
        by_geometry[sample_record.geometry_id].append(sample_record)
    for prescreen_record in prescreen_records:
        if prescreen_record.pde_family == pde_family:
            prescreen_by_geometry[prescreen_record.geometry_id].append(prescreen_record)

    geometry_reports = []
    verdict_counts: dict[str, int] = defaultdict(int)
    geometry_ids = sorted(set(by_geometry) | set(prescreen_by_geometry))
    for geometry_id in geometry_ids:
        geometry_samples = by_geometry.get(geometry_id, [])
        sample_metrics = [analyze_sample_mesh(record, hotspot_quantile=hotspot_quantile) for record in geometry_samples]
        pairwise_metrics: dict[str, dict[str, Any]] = {}
        stage_pairwise_metrics: dict[str, dict[str, dict[str, Any]]] = {stage_name: {} for stage_name in STAGE_FIELD_MODES}
        if sample_metrics:
            coarse_mesh_path = geometry_samples[0].geometry_artifact_paths['coarse_mesh_path']
            coarse_mesh = meshio.read(coarse_mesh_path)
            probe_points = np.asarray(coarse_mesh.points, dtype=float)
            pairwise_metrics = compare_condition_fields(
                geometry_id=geometry_id,
                sample_metrics=sample_metrics,
                probe_points=probe_points,
                hotspot_quantile=hotspot_quantile,
            )
            stage_pairwise_metrics = compare_stage_fields(
                geometry_id=geometry_id,
                sample_metrics=sample_metrics,
                hotspot_quantile=hotspot_quantile,
            )
        verdict, reasons, collapse = verdict_for_geometry(
            sample_metrics=sample_metrics,
            pairwise_metrics=pairwise_metrics,
            stage_pairwise_metrics=stage_pairwise_metrics,
            prescreen_records=prescreen_by_geometry.get(geometry_id, []),
            target_hotspot_size_ratio=target_hotspot_size_ratio,
            overlap_threshold=overlap_threshold,
        )
        verdict_counts[verdict] += 1
        geometry_reports.append(
            {
                'geometry_id': geometry_id,
                'family_name': family_name,
                'sample_metrics': sample_metrics,
                'pairwise_condition_metrics': pairwise_metrics,
                'field_stage_pairwise_metrics': stage_pairwise_metrics,
                'collapse_diagnostics': collapse,
                'prescreen_labels': {record.condition_id: record.label for record in prescreen_by_geometry.get(geometry_id, [])},
                'verdict': verdict,
                'verdict_reasons': reasons,
            }
        )
    return {
        'enabled': bool(enabled),
        'summary': {
            'family': pde_family,
            'num_samples': sum(len(report['sample_metrics']) for report in geometry_reports),
            'num_geometries': len(geometry_reports),
            'verdict_counts': dict(verdict_counts),
            'budget_status_counts': {
                status: sum(
                    1
                    for report in geometry_reports
                    for metrics in report['sample_metrics']
                    if str(metrics.get('budget_status') or 'unknown') == status
                )
                for status in sorted({
                    str(metrics.get('budget_status') or 'unknown')
                    for report in geometry_reports
                    for metrics in report['sample_metrics']
                })
            },
        },
        'geometry_reports': geometry_reports,
    }


def build_smoke_report(
    *,
    report_path: Path,
    sample_records: list[SampleRecord],
    prescreen_records: list[PrescreenRecord],
    hotspot_quantile: float,
    target_hotspot_size_ratio: float,
    overlap_threshold: float,
    scalar_smoke_enable: bool = True,
    elasticity_smoke_enable: bool = True,
) -> dict[str, Any]:
    scalar_report = _family_report(
        family_name='scalar_smoke',
        pde_family='scalar_elliptic',
        sample_records=sample_records,
        prescreen_records=prescreen_records,
        hotspot_quantile=hotspot_quantile,
        target_hotspot_size_ratio=target_hotspot_size_ratio,
        overlap_threshold=overlap_threshold,
        enabled=scalar_smoke_enable,
    )
    elasticity_report = _family_report(
        family_name='elasticity_smoke',
        pde_family='linear_elasticity',
        sample_records=sample_records,
        prescreen_records=prescreen_records,
        hotspot_quantile=hotspot_quantile,
        target_hotspot_size_ratio=target_hotspot_size_ratio,
        overlap_threshold=overlap_threshold,
        enabled=elasticity_smoke_enable,
    )

    rejected_conditions = []
    for record in prescreen_records:
        if record.label == 'accept_for_smoke':
            continue
        taxonomy = record.taxonomy_category
        if taxonomy is None:
            if record.label == 'reject_low_contrast':
                taxonomy = 'reject_low_contrast_prescreen'
            elif record.label == 'reject_too_expensive':
                taxonomy = 'reject_too_expensive_prescreen'
            else:
                taxonomy = record.failure_category or record.label
        rejected_conditions.append(
            {
                'geometry_id': record.geometry_id,
                'condition_id': record.condition_id,
                'pde_family': record.pde_family,
                'label': record.label,
                'taxonomy_category': taxonomy,
                'selected_reason': record.selected_reason,
                'failure_reason': record.failure_reason,
            }
        )

    failed_teacher_samples = []
    for record in sample_records:
        if _is_success_status(record.status):
            continue
        runtime_observation = dict((record.teacher_metadata or {}).get('runtime_observation', {}))
        failed_teacher_samples.append(
            {
                'sample_id': record.sample_id,
                'geometry_id': record.geometry_id,
                'condition_id': record.condition_id,
                'pde_family': record.pde_family,
                'failure_category': record.failure_category,
                'stage_where_stopped': record.stage_where_stopped,
                'elapsed_seconds': float(record.elapsed_seconds),
                'parent_observed_elapsed_seconds': runtime_observation.get('parent_observed_elapsed_seconds'),
                'worker_reported_elapsed_seconds': runtime_observation.get('worker_reported_elapsed_seconds'),
            }
        )

    report = {
        'summary': {
            'scalar': scalar_report['summary'],
            'elasticity': elasticity_report['summary'],
            'num_rejected_conditions': len(rejected_conditions),
            'num_failed_teacher_samples': len(failed_teacher_samples),
        },
        'scalar_smoke': scalar_report,
        'elasticity_smoke': elasticity_report,
        'geometry_reports': scalar_report['geometry_reports'],
        'rejected_conditions': rejected_conditions,
        'failed_teacher_samples': failed_teacher_samples,
    }
    dump_json(report_path, report)
    return report
