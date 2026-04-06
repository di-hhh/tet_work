from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data.batch import Batch
from torch_geometric.utils import softmax
from torch_scatter import scatter_add

from src.mpn.common.latent_mlp import LatentMLP


class EdgeAwareGATEdgeModule(nn.Module):
    """
    Edge update:
        e_new = MLP([x_src, x_dst, e])
    """

    def __init__(self, *, in_features: int, latent_dimension: int, stack_config: Dict):
        super().__init__()
        mlp_config = stack_config.get("mlp")
        self._mlp = LatentMLP(
            in_features=in_features,
            latent_dimension=latent_dimension,
            config=mlp_config,
        )

    def forward(self, graph: Batch) -> None:
        src, dst = graph.edge_index
        x_src = graph.x[src]
        x_dst = graph.x[dst]
        e = graph.edge_attr
        aggregated = torch.cat([x_src, x_dst, e], dim=1)
        graph.edge_attr = self._mlp(aggregated)


class EdgeAwareGATNodeModule(nn.Module):
    """
    Node update aligned with edge-aware GAT behavior.
    """

    def __init__(self, d: int, stack_config: Dict[str, Any]):
        super().__init__()

        self.apply_attention = bool(stack_config.get("apply_attention"))
        self.transform_edge_for_att_calc = bool(stack_config.get("transform_edge_for_att_calc"))
        self.apply_attention_on_edge = bool(stack_config.get("apply_attention_on_edge"))
        self.aggregate_edge = bool(stack_config.get("aggregate_edge"))
        self.edge_dependent_attention = bool(stack_config.get("edge_dependent_attention"))
        self.use_edge_in_value = bool(stack_config.get("use_edge_in_value", False))
        self.self_loop = bool(stack_config.get("self_loop"))
        self.self_node_transform = bool(stack_config.get("self_node_transform")) and self.self_loop

        self.feat_drop = nn.Dropout(float(stack_config.get("feat_drop", 0.0)))
        self.attn_drop = nn.Dropout(float(stack_config.get("attn_drop", 0.0)))
        self.edge_feat_drop = nn.Dropout(float(stack_config.get("edge_feat_drop", 0.0)))
        self.use_batch_norm = bool(stack_config.get("use_batch_norm"))

        self.num_heads = int(stack_config.get("num_heads", 4))
        if d % self.num_heads != 0:
            raise ValueError(f"d={d} must be divisible by num_heads={self.num_heads}")
        self.d_head = d // self.num_heads
        self.d = d

        self.fc = nn.Linear(d, d, bias=bool(stack_config.get("use_bias", True)))

        att_in_dim = 2 * self.d_head + (self.d_head if self.edge_dependent_attention else 0)
        # codex: each head gets its own attention scorer instead of sharing one linear layer.
        self.attn_weight = nn.Parameter(torch.empty(self.num_heads, att_in_dim))
        self.attn_bias = nn.Parameter(torch.empty(self.num_heads)) if bool(stack_config.get("use_bias", True)) else None

        if self.transform_edge_for_att_calc:
            self.fc_edge_for_att_calc = nn.Linear(d, d, bias=bool(stack_config.get("use_bias", True)))
        else:
            self.fc_edge_for_att_calc = None

        if self.aggregate_edge:
            self.fc_edge = nn.Linear(d, d, bias=bool(stack_config.get("use_bias", True)))
        else:
            self.fc_edge = None

        if self.self_node_transform:
            self.fc_self = nn.Linear(d, d, bias=bool(stack_config.get("use_bias", True)))
        else:
            self.fc_self = None

        self.out_proj = nn.Linear(d, d, bias=bool(stack_config.get("use_bias", True)))
        # codex: wire up use_batch_norm so the config is not a no-op.
        self.out_proj_norm = nn.BatchNorm1d(d) if self.use_batch_norm else nn.LayerNorm(d)

        self.use_ffn = bool(stack_config.get("use_ffn", True))
        ffn_mult = float(stack_config.get("ffn_hidden_mult", 2.0))
        ffn_hidden = int(max(d, int(ffn_mult * d)))
        ffn_drop = float(stack_config.get("ffn_drop", 0.0))

        self.ffn_ln = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, ffn_hidden),
            nn.SiLU(),
            nn.Dropout(ffn_drop),
            nn.Linear(ffn_hidden, d),
            nn.Dropout(ffn_drop),
        )

        self.activation = stack_config.get("activation", None)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")

        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

        nn.init.xavier_normal_(self.attn_weight, gain=gain)
        if self.attn_bias is not None:
            nn.init.zeros_(self.attn_bias)

        if self.fc_edge_for_att_calc is not None:
            nn.init.xavier_normal_(self.fc_edge_for_att_calc.weight, gain=gain)
            if self.fc_edge_for_att_calc.bias is not None:
                nn.init.zeros_(self.fc_edge_for_att_calc.bias)

        if self.fc_edge is not None:
            nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)
            if self.fc_edge.bias is not None:
                nn.init.zeros_(self.fc_edge.bias)

        if self.fc_self is not None:
            nn.init.xavier_normal_(self.fc_self.weight, gain=gain)
            if self.fc_self.bias is not None:
                nn.init.zeros_(self.fc_self.bias)

        nn.init.xavier_normal_(self.out_proj.weight, gain=gain)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

        for layer in self.ffn:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        if isinstance(self.out_proj_norm, nn.BatchNorm1d):
            self.out_proj_norm.reset_parameters()
        else:
            self.out_proj_norm.reset_parameters()
        self.ffn_ln.reset_parameters()

    def forward(self, graph: Batch) -> None:
        x = graph.x
        e = graph.edge_attr
        src, dst = graph.edge_index
        n_nodes = x.size(0)

        x_in = x
        x = self.feat_drop(x)
        e = self.edge_feat_drop(e)

        z = self.fc(x).view(n_nodes, self.num_heads, self.d_head)

        if self.edge_dependent_attention:
            if self.fc_edge_for_att_calc is not None:
                e_att = self.fc_edge_for_att_calc(e).view(-1, self.num_heads, self.d_head)
            else:
                e_att = e.view(-1, self.num_heads, self.d_head)
        else:
            e_att = None

        z_src = z[src]
        z_dst = z[dst]
        if self.edge_dependent_attention:
            att_in = torch.cat([z_src, z_dst, e_att], dim=-1)
        else:
            att_in = torch.cat([z_src, z_dst], dim=-1)

        # codex: compute per-head logits with head-specific parameters.
        logits = (att_in * self.attn_weight.unsqueeze(0)).sum(dim=-1)
        if self.attn_bias is not None:
            logits = logits + self.attn_bias.unsqueeze(0)
        logits = F.leaky_relu(logits, negative_slope=0.2)

        if self.apply_attention:
            alpha = self.attn_drop(softmax(logits, dst))
        else:
            alpha = torch.ones_like(logits)

        msg_node = z_src * alpha.unsqueeze(-1)
        h = scatter_add(msg_node, dst, dim=0, dim_size=n_nodes)

        if self.aggregate_edge and self.use_edge_in_value:
            ez = self.fc_edge(e).view(-1, self.num_heads, self.d_head)
            if self.apply_attention_on_edge and self.apply_attention:
                msg_edge = ez * alpha.unsqueeze(-1)
            else:
                msg_edge = ez
            h = h + scatter_add(msg_edge, dst, dim=0, dim_size=n_nodes)

        h = h.reshape(n_nodes, self.d)
        h = self.out_proj_norm(self.out_proj(h))

        if self.self_loop:
            if self.fc_self is not None:
                h = h + self.fc_self(x_in)
            else:
                h = h + x_in

        if self.activation is not None:
            h = self.activation(h)

        # codex: enable the FFN residual branch described by the architecture notes.
        if self.use_ffn:
            h = h + self.ffn(self.ffn_ln(h))

        graph.x = h
