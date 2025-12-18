from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, global_max_pool, global_mean_pool, SAGEConv


class GCNClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 96, layers: int = 3, dropout: float = 0.25):
        super().__init__()
        self.dropout = dropout
        convs = []
        for i in range(layers):
            input_dim = in_channels if i == 0 else hidden
            convs.append(GCNConv(input_dim, hidden, normalize=True, add_self_loops=True))
        self.convs = nn.ModuleList(convs)
        self.node_head = nn.Linear(hidden, 1)
        self.graph_head = nn.Linear(hidden * 2, 1)

    def forward(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        node_logits = self.node_head(x).view(-1)
        pooled = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        graph_logits = self.graph_head(pooled).view(-1)
        return graph_logits, node_logits


class GATClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 64, heads: int = 4, layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        convs = []
        for i in range(layers):
            input_dim = in_channels if i == 0 else hidden * heads
            out_dim = hidden
            convs.append(GATConv(input_dim, out_dim, heads=heads, dropout=dropout, concat=True))
        self.convs = nn.ModuleList(convs)
        self.node_head = nn.Linear(hidden * heads, 1)
        self.graph_head = nn.Linear(hidden * heads * 2, 1)

    def forward(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        node_logits = self.node_head(x).view(-1)
        pooled = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        graph_logits = self.graph_head(pooled).view(-1)
        return graph_logits, node_logits


class H2GNNClassifier(nn.Module):
    """Simple two-hop GCN-style model: combines 1-hop and 2-hop representations."""

    def __init__(self, in_channels: int, hidden: int = 96, dropout: float = 0.25):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, hidden, normalize=True, add_self_loops=True)
        self.conv2 = GCNConv(hidden, hidden, normalize=True, add_self_loops=True)
        self.node_head = nn.Linear(hidden * 2, 1)
        self.graph_head = nn.Linear(hidden * 4, 1)

    def forward(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        h1 = F.relu(self.conv1(x, edge_index))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h2 = F.relu(self.conv2(h1, edge_index))
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        h = torch.cat([h1, h2], dim=1)
        node_logits = self.node_head(h).view(-1)
        pooled = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=1)
        graph_logits = self.graph_head(pooled).view(-1)
        return graph_logits, node_logits


class GraphSAGEClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 96, layers: int = 3, dropout: float = 0.25):
        super().__init__()
        self.dropout = dropout
        convs = []
        for i in range(layers):
            input_dim = in_channels if i == 0 else hidden
            convs.append(SAGEConv(input_dim, hidden, normalize=True))
        self.convs = nn.ModuleList(convs)
        self.node_head = nn.Linear(hidden, 1)
        self.graph_head = nn.Linear(hidden * 2, 1)

    def forward(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        node_logits = self.node_head(x).view(-1)
        pooled = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        graph_logits = self.graph_head(pooled).view(-1)
        return graph_logits, node_logits