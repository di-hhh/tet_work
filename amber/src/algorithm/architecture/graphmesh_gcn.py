import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.nn import Dropout, LayerNorm
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import GCNConv

from src.algorithm.architecture.mlp import MLP
from src.algorithm.architecture.physics_correction_branch_codex import PhysicsCorrectionHeadsCodex


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, activation_function="leakyrelu", dropout=0.0, layer_norm=False):
        super().__init__()

        if activation_function == "leakyrelu":
            self.activation_function = F.leaky_relu
        elif activation_function == "relu":
            self.activation_function = F.relu
        elif activation_function == "tanh":
            self.activation_function = F.tanh
        else:
            raise ValueError(f"Unknown activation function {activation_function}")

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.dropout = Dropout(dropout)

        self.convs.append(GCNConv(in_channels, hidden_channels))

        if layer_norm:
            self.norms.append(LayerNorm(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

            if layer_norm:
                self.norms.append(LayerNorm(hidden_channels))

        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if len(self.norms) > 0:
                x = self.norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)

        x = self.convs[-1](x, edge_index)
        return x


class ResidualGCN(GCN):
    def forward(self, x, residual, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index) + residual
            if len(self.norms) > 0:
                x = self.norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)

        x = self.convs[-1](x, edge_index)
        return x


class GraphmeshGCN(nn.Module):
    def __init__(self, architecture_config: DictConfig, example_graph: Data):
        super().__init__()

        self._node_type = "node"
        latent_dimension = architecture_config.latent_dimension

        if "boundary_vertex_graphs" in example_graph:
            boundary_vertex_graphs = example_graph.boundary_vertex_graphs
            boundary_feature_dim = boundary_vertex_graphs[0].x.shape[1]
            self.boundary_gcn = GCN(
                in_channels=boundary_feature_dim,
                hidden_channels=latent_dimension,
                out_channels=latent_dimension,
                num_layers=1,
                activation_function=architecture_config.activation_function,
            )
            self.main_gcn = ResidualGCN(
                in_channels=latent_dimension,
                hidden_channels=latent_dimension,
                out_channels=latent_dimension,
                num_layers=architecture_config.num_steps,
                activation_function=architecture_config.activation_function,
            )
            self.embedding_layer = nn.Linear(
                in_features=example_graph.num_node_features,
                out_features=latent_dimension,
            )
        else:
            self.main_gcn = GCN(
                in_channels=example_graph.num_node_features,
                hidden_channels=latent_dimension,
                out_channels=latent_dimension,
                num_layers=architecture_config.num_steps,
                activation_function=architecture_config.activation_function,
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
        if hasattr(observations, "boundary_vertex_graphs"):
            all_boundary = []
            for _, graphs in enumerate(observations.boundary_vertex_graphs):
                all_boundary.extend(graphs)

            batched = Batch.from_data_list(all_boundary)
            feat = self.boundary_gcn(batched.x, batched.edge_index)
            graph_emb = global_mean_pool(feat, batched.batch)

            obs_emb = self.embedding_layer(observations.x)
            node_features = self.main_gcn(graph_emb, residual=obs_emb, edge_index=observations.edge_index)
        else:
            node_features = self.main_gcn(observations.x, observations.edge_index)

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
