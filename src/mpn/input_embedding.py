from typing import Dict, Optional

from torch import nn
from torch_geometric.data import Data

from src.mpn.common.embedding import Embedding
from src.mpn.common.embedding_v1 import Embedding_v1


class MessagePassingInputEmbedding(nn.Module):
    def __init__(self, *, in_node_features: int, in_edge_features: int, embedding_config: Optional[Dict], latent_dimension: int, device: Optional):
        """
        Builds and returns an input embedding for a graph.
        Args:
            in_node_features:
                number of input node features
            in_edge_features:
                number of input edge features
            latent_dimension:
                dimension of the latent space.
            device:
                torch.device to use
        """
        super().__init__()

        self.node_input_embedding = Embedding(
            in_features=in_node_features, latent_dimension=latent_dimension, embedding_config=embedding_config, device=device
        )

        self.edge_input_embedding = Embedding(
            in_features=in_edge_features, latent_dimension=latent_dimension, embedding_config=embedding_config, device=device
        )

        # self.node_input_embedding = Embedding_v1(
        #     in_features = in_node_features,
        #     out_features = latent_dimension,
        #     is_vertex = True,
        #     num_freqs = 8,  # 低频少一点，因为维度低
        # ).to(device)
        # self.edge_input_embedding = Embedding_v1(
        #     in_features=in_edge_features,
        #     out_features=latent_dimension,
        #     is_vertex=True,
        #     num_freqs=10,  # 边特征更少，可以增加频率数以丰富表达
        # ).to(device)

    def forward(self, graph: Data):
        """
        Computes the forward pass for this input embedding inplace
        Args:
            graph: torch_geometric.data.Batch, represents a batch of graphs
        Returns:
            None
        """
        graph.__setattr__("x", self.node_input_embedding(graph.x))
        graph.__setattr__("edge_attr", self.edge_input_embedding(graph.edge_attr))
