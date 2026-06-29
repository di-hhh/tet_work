from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.condition_aware_dataset_generation.records import ConditionRecord, GeometryPreprocessRecord, GeometryRecord
from src.condition_aware_dataset_generation.utils import numpy_random_state, stable_identifier


class ConditionSampler:
    def __init__(self, sampler_config: dict):
        self.sampler_config = sampler_config
        self.default_conditions_per_geometry = int(sampler_config.get("default_conditions_per_geometry", 4))
        self.pde_families = list(sampler_config.get("pde_families", ["scalar_elliptic", "linear_elasticity"]))
        self.default_budgets = list(sampler_config.get("budgets", [100, 180]))

    def sample_for_geometry(self, geometry_record: GeometryRecord, preprocess_record: GeometryPreprocessRecord, seed: int) -> list[ConditionRecord]:
        families = self._build_family_schedule()
        conditions: list[ConditionRecord] = []
        for condition_index, family in enumerate(families):
            rng = numpy_random_state(seed, geometry_record.geometry_id, family, condition_index)
            if family == "scalar_elliptic":
                condition_spec = self._sample_scalar_elliptic(preprocess_record, rng, condition_index)
            elif family == "linear_elasticity":
                condition_spec = self._sample_linear_elasticity(preprocess_record, rng, condition_index)
            else:
                raise ValueError(f"Unsupported PDE family: {family}")
            condition_id = stable_identifier(
                prefix=f"{family}_{condition_index}",
                text=f"{geometry_record.geometry_id}::{family}::{condition_index}",
            )
            conditions.append(
                ConditionRecord(
                    condition_id=condition_id,
                    geometry_id=geometry_record.geometry_id,
                    pde_family=family,
                    condition_index=condition_index,
                    condition_spec=condition_spec,
                    budget_or_tolerance_spec={"budgets": list(self.default_budgets)},
                    source_name=geometry_record.source_name,
                )
            )
        return conditions

    def _build_family_schedule(self) -> list[str]:
        families = []
        while len(families) < self.default_conditions_per_geometry:
            for family in self.pde_families:
                families.append(family)
                if len(families) >= self.default_conditions_per_geometry:
                    break
        return families

    def _selector_spec(
        self,
        preprocess_record: GeometryPreprocessRecord,
        axis_index: int,
        side: str,
        band_fraction: float,
    ) -> dict[str, Any]:
        matched_patch_ids = []
        oriented_min = np.asarray(preprocess_record.oriented_bbox_min, dtype=float)
        oriented_max = np.asarray(preprocess_record.oriented_bbox_max, dtype=float)
        extent = oriented_max[axis_index] - oriented_min[axis_index]
        lower = oriented_min[axis_index]
        upper = oriented_max[axis_index]
        band_width = max(extent * band_fraction, 1.0e-8)
        for patch in preprocess_record.boundary_patches:
            patch_center = np.asarray(patch["center"], dtype=float)
            centroid = np.asarray(preprocess_record.centroid, dtype=float)
            principal_axes = np.asarray(preprocess_record.principal_axes, dtype=float)
            local_patch_center = (patch_center - centroid) @ principal_axes
            coordinate = float(local_patch_center[axis_index])
            if side == "min" and coordinate <= lower + band_width:
                matched_patch_ids.append(patch["patch_id"])
            elif side == "max" and coordinate >= upper - band_width:
                matched_patch_ids.append(patch["patch_id"])
        return {
            "selector_type": "oriented_band",
            "axis_index": axis_index,
            "side": side,
            "band_fraction": band_fraction,
            "matched_patch_ids": matched_patch_ids,
        }

    def _random_local_center(self, preprocess_record: GeometryPreprocessRecord, rng: np.random.RandomState) -> list[float]:
        local_min = np.asarray(preprocess_record.oriented_bbox_min, dtype=float)
        local_max = np.asarray(preprocess_record.oriented_bbox_max, dtype=float)
        span = local_max - local_min
        center = local_min + 0.2 * span + rng.uniform(size=span.shape[0]) * 0.6 * span
        return center.tolist()

    def _sample_scalar_elliptic(
        self,
        preprocess_record: GeometryPreprocessRecord,
        rng: np.random.RandomState,
        condition_index: int,
    ) -> dict[str, Any]:
        config = self.sampler_config.get("scalar_elliptic", {})
        band_fraction = float(config.get("band_fraction", 0.12))
        diffusion_low, diffusion_high = config.get("diffusion_range", [0.5, 2.0])
        source_low, source_high = config.get("source_amplitude_range", [-2.0, 2.0])
        dirichlet_low, dirichlet_high = config.get("dirichlet_value_range", [0.0, 1.0])
        axis_index = condition_index % preprocess_record.dimension
        secondary_axis = (axis_index + 1) % preprocess_record.dimension
        low_value = float(rng.uniform(dirichlet_low, dirichlet_high))
        high_value = float(low_value + rng.uniform(0.4, 1.4))
        diffusion = float(rng.uniform(diffusion_low, diffusion_high))
        source_amplitude = float(rng.uniform(source_low, source_high))
        flux_probability = float(config.get("flux_probability", 0.5))
        include_flux = bool(rng.rand() < flux_probability)
        flux_value = float(rng.uniform(-0.6, 0.6)) if include_flux else 0.0
        source_sigma = [float(0.12 * (mx - mn)) for mn, mx in zip(preprocess_record.oriented_bbox_min, preprocess_record.oriented_bbox_max)]

        boundary_role_spec = [
            {"role": "dirichlet_low", "selector": self._selector_spec(preprocess_record, axis_index, "min", band_fraction), "value": low_value},
            {"role": "dirichlet_high", "selector": self._selector_spec(preprocess_record, axis_index, "max", band_fraction), "value": high_value},
        ]
        if include_flux:
            boundary_role_spec.append(
                {
                    "role": "neumann_flux",
                    "selector": self._selector_spec(preprocess_record, secondary_axis, "max", band_fraction),
                    "value": flux_value,
                }
            )

        return {
            "pde_family": "scalar_elliptic",
            "boundary_role_spec": boundary_role_spec,
            "coefficient_spec": {"diffusion": diffusion},
            "source_or_load_spec": {
                "internal_source": {
                    "type": "gaussian",
                    "center_local": self._random_local_center(preprocess_record, rng),
                    "sigma": source_sigma,
                    "amplitude": source_amplitude,
                }
            },
            "budget_or_tolerance_spec": {"budgets": list(self.default_budgets)},
            "qoi_spec": {"type": "boundary_average", "selector": self._selector_spec(preprocess_record, axis_index, "max", band_fraction)},
        }

    def _sample_linear_elasticity(
        self,
        preprocess_record: GeometryPreprocessRecord,
        rng: np.random.RandomState,
        condition_index: int,
    ) -> dict[str, Any]:
        config = self.sampler_config.get("linear_elasticity", {})
        band_fraction = float(config.get("band_fraction", 0.12))
        axis_index = condition_index % preprocess_record.dimension
        load_axis = (axis_index + 1) % preprocess_record.dimension if preprocess_record.dimension > 1 and (condition_index % 2 == 1) else axis_index
        support_selector = self._selector_spec(preprocess_record, axis_index, "min", band_fraction)
        load_selector = self._selector_spec(preprocess_record, load_axis, "max", band_fraction)
        youngs_low, youngs_high = config.get("youngs_modulus_range", [50.0, 200.0])
        nu_low, nu_high = config.get("poissons_ratio_range", [0.2, 0.35])
        traction_low, traction_high = config.get("traction_magnitude_range", [0.2, 1.0])
        body_force_probability = float(config.get("body_force_probability", 0.3))

        traction_magnitude = float(rng.uniform(traction_low, traction_high))
        direction = np.zeros(preprocess_record.dimension, dtype=float)
        direction[load_axis] = -1.0 if load_axis == axis_index else 1.0
        if not np.any(direction):
            direction[0] = 1.0
        direction /= np.linalg.norm(direction)
        body_force = np.zeros(preprocess_record.dimension, dtype=float)
        if rng.rand() < body_force_probability:
            body_force = rng.uniform(-0.1, 0.1, size=preprocess_record.dimension)

        return {
            "pde_family": "linear_elasticity",
            "boundary_role_spec": [
                {"role": "support", "selector": support_selector, "components": list(range(preprocess_record.dimension)), "value": 0.0},
                {"role": "traction", "selector": load_selector, "vector": (traction_magnitude * direction).tolist()},
            ],
            "coefficient_spec": {
                "youngs_modulus": float(rng.uniform(youngs_low, youngs_high)),
                "poissons_ratio": float(rng.uniform(nu_low, nu_high)),
                "constitutive_model": "plane_stress" if preprocess_record.dimension == 2 else "linear_elasticity",
            },
            "source_or_load_spec": {"body_force": body_force.tolist()},
            "budget_or_tolerance_spec": {"budgets": list(self.default_budgets)},
            "qoi_spec": {"type": "mean_displacement_norm", "selector": load_selector},
        }
