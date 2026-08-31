from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch_geometric.data import Batch

from src.algorithm.architecture.mlp import MLP


class PhysicsCorrectionHeadsCodex(nn.Module):
    def __init__(
        self,
        *,
        latent_dimension: int,
        mlp_config,
        direct_physics_feature_dim: int = 0,
        gate_activation: str = "sigmoid",
        gate_max: float = 1.0,
        gate_init_bias: float = -2.5,
        physics_readout_init_std: float = 1.0e-3,
        inference_missing_physics_fallback: str = "gate_zero",
    ):
        super().__init__()
        self.direct_physics_feature_dim = max(int(direct_physics_feature_dim), 0)
        self.gate_activation = str(gate_activation).lower()
        self.gate_max = float(gate_max)
        self.gate_init_bias = float(gate_init_bias)
        self.physics_readout_init_std = float(physics_readout_init_std)
        self.inference_missing_physics_fallback = str(inference_missing_physics_fallback)

        branch_input_dimension = latent_dimension + self.direct_physics_feature_dim
        self.physics_decoder_mlp = MLP(
            in_features=branch_input_dimension,
            mlp_config=mlp_config,
            latent_dimension=latent_dimension,
        )
        self.physics_readout = nn.Linear(latent_dimension, 1)

        self.gate_decoder_mlp = MLP(
            in_features=branch_input_dimension,
            mlp_config=mlp_config,
            latent_dimension=latent_dimension,
        )
        self.gate_readout = nn.Linear(latent_dimension, 1)

        self.reset_parameters_codex()

    def reset_parameters_codex(self) -> None:
        nn.init.normal_(self.physics_readout.weight, mean=0.0, std=self.physics_readout_init_std)
        nn.init.zeros_(self.physics_readout.bias)
        nn.init.zeros_(self.gate_readout.weight)
        nn.init.constant_(self.gate_readout.bias, self.gate_init_bias)

    def forward(
        self,
        *,
        latent_features: torch.Tensor,
        expert_outputs: torch.Tensor,
        observations: Batch,
        correction_warmup_factor: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        branch_inputs = self._get_branch_inputs(
            observations=observations,
            latent_features=latent_features,
        )
        physics_outputs = self.physics_readout(self.physics_decoder_mlp(branch_inputs))
        gate_logits = self.gate_readout(self.gate_decoder_mlp(branch_inputs))
        gate = self._apply_gate_activation(gate_logits)
        gate = gate * float(max(0.0, min(1.0, correction_warmup_factor)))

        availability = self._get_output_level_availability(
            observations=observations,
            reference_tensor=gate,
        )
        if self.inference_missing_physics_fallback in {"gate_zero", "disable_branch"}:
            gate = gate * availability
        elif self.inference_missing_physics_fallback == "zero_feature":
            pass
        else:
            raise ValueError(
                f"Unsupported inference_missing_physics_fallback '{self.inference_missing_physics_fallback}'"
            )

        return {
            "expert_output": expert_outputs,
            "physics_output": physics_outputs,
            "gate_logits": gate_logits,
            "gate": gate,
            "physics_feature_available": availability,
        }

    def _get_branch_inputs(
        self,
        *,
        observations: Batch,
        latent_features: torch.Tensor,
    ) -> torch.Tensor:
        if self.direct_physics_feature_dim <= 0:
            return latent_features

        direct_physics_feature = self._get_direct_physics_feature(
            observations=observations,
            reference_tensor=latent_features,
        )
        return torch.cat([latent_features, direct_physics_feature], dim=-1)

    def _apply_gate_activation(self, gate_logits: torch.Tensor) -> torch.Tensor:
        if self.gate_activation == "sigmoid":
            gate = torch.sigmoid(gate_logits)
        elif self.gate_activation == "hard_sigmoid":
            gate = torch.nn.functional.hardsigmoid(gate_logits)
        else:
            raise ValueError(f"Unsupported gate_activation '{self.gate_activation}'")
        return gate * self.gate_max

    def _get_direct_physics_feature(
        self,
        *,
        observations: Batch,
        reference_tensor: torch.Tensor,
    ) -> torch.Tensor:
        feature_shape = (reference_tensor.shape[0], self.direct_physics_feature_dim)
        if not hasattr(observations, "physics_feature"):
            return torch.zeros(feature_shape, device=reference_tensor.device, dtype=reference_tensor.dtype)

        direct_feature = observations.physics_feature.to(
            device=reference_tensor.device,
            dtype=reference_tensor.dtype,
        )
        if direct_feature.ndim == 1:
            direct_feature = direct_feature.unsqueeze(-1)
        if hasattr(observations, "mask_output"):
            direct_feature = direct_feature[observations.mask_output]
        if direct_feature.shape[0] != reference_tensor.shape[0]:
            return torch.zeros(feature_shape, device=reference_tensor.device, dtype=reference_tensor.dtype)
        if direct_feature.shape[1] < self.direct_physics_feature_dim:
            pad_width = self.direct_physics_feature_dim - direct_feature.shape[1]
            padding = torch.zeros((direct_feature.shape[0], pad_width), device=direct_feature.device, dtype=direct_feature.dtype)
            direct_feature = torch.cat([direct_feature, padding], dim=-1)
        elif direct_feature.shape[1] > self.direct_physics_feature_dim:
            direct_feature = direct_feature[:, : self.direct_physics_feature_dim]
        return direct_feature

    @staticmethod
    def _get_output_level_availability(*, observations: Batch, reference_tensor: torch.Tensor) -> torch.Tensor:
        if not hasattr(observations, "physics_feature_available"):
            return torch.ones_like(reference_tensor)

        availability = observations.physics_feature_available.to(
            device=reference_tensor.device,
            dtype=reference_tensor.dtype,
        )
        if availability.ndim == 1:
            availability = availability.unsqueeze(-1)
        if hasattr(observations, "mask_output"):
            availability = availability[observations.mask_output]
        if availability.shape != reference_tensor.shape:
            availability = availability.reshape(reference_tensor.shape)
        return availability
