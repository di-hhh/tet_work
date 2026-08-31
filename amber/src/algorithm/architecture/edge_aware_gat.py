import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch_geometric.data import Batch, Data

from src.algorithm.architecture.mlp import MLP
from src.algorithm.architecture.physics_correction_branch_codex import PhysicsCorrectionHeadsCodex
from src.ours_gat.get_gat_base import get_gat_from_graph


class EdgeAwareGat(nn.Module):
    def __init__(self, architecture_config: DictConfig, example_graph: Data):
        super().__init__()

        self._node_type = "node"
        latent_dimension = architecture_config.latent_dimension
        self.gat = get_gat_from_graph(
            example_graph=example_graph,
            latent_dimension=latent_dimension,
            node_name=self._node_type,
            base_config=architecture_config,
        )

        mlp_config = architecture_config.decoder
        self.decoder_mlp = MLP(
            in_features=latent_dimension,
            mlp_config=mlp_config,
            latent_dimension=latent_dimension,
        )
        self.readout = nn.Linear(latent_dimension, 1)
        self.physics_correction_heads = None
        if architecture_config.get("enable_physics_correction_branch", False):
            direct_physics_feature_dim = _get_direct_physics_feature_dim(example_graph=example_graph)
            self.physics_correction_heads = PhysicsCorrectionHeadsCodex(
                latent_dimension=latent_dimension,
                mlp_config=mlp_config,
                direct_physics_feature_dim=direct_physics_feature_dim,
                gate_activation=architecture_config.get("gate_activation", "sigmoid"),
                gate_max=float(architecture_config.get("gate_max", 1.0)),
                gate_init_bias=float(architecture_config.get("gate_init_bias", -2.5)),
                physics_readout_init_std=float(architecture_config.get("physics_readout_init_std", 1.0e-3)),
                inference_missing_physics_fallback=architecture_config.get(
                    "inference_missing_physics_fallback",
                    "gate_zero",
                ),
            )

    def forward(self, observations: Batch, **kwargs) -> torch.Tensor:
        node_features, _, _ = self.gat(observations)
        node_features = node_features.get(self._node_type)

        if hasattr(observations, "mask_output"):
            node_features = node_features[observations.mask_output]

        decoded_node_features = self.decoder_mlp(node_features)
        outputs = self.readout(decoded_node_features)
        if self.physics_correction_heads is None:
            return outputs
        return self.physics_correction_heads(
            latent_features=node_features,
            expert_outputs=outputs,
            observations=observations,
            correction_warmup_factor=kwargs.get("correction_warmup_factor", 1.0),
        )


def _get_direct_physics_feature_dim(*, example_graph: Data) -> int:
    if not hasattr(example_graph, "physics_feature"):
        return 0
    direct_physics_feature = example_graph.physics_feature
    if direct_physics_feature.ndim <= 1:
        return 1
    return int(direct_physics_feature.shape[-1])
