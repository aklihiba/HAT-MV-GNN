import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class TypedAttributeEncoder(nn.Module):
    """
    Typed attribute encoder that generates initial embeddings for each node type.
    Uses separate MLPs for different node types as described in Eq. 3.1.
    """
    def __init__(self, node_types: List[str], input_dims: Dict[str, int], 
                 hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        self.node_types = node_types
        self.output_dim = output_dim
        
        # Create separate MLPs for each node type
        self.type_encoders = nn.ModuleDict()
        for node_type in node_types:
            layers = []
            input_dim = input_dims[node_type]
            
            for i in range(num_layers):
                if i == 0:
                    layers.append(nn.Linear(input_dim, hidden_dim))
                elif i == num_layers - 1:
                    layers.append(nn.Linear(hidden_dim, output_dim))
                else:
                    layers.append(nn.Linear(hidden_dim, hidden_dim))
                
                if i < num_layers - 1:  # No activation after last layer
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(0.1))
            
            self.type_encoders[node_type] = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, node_types: torch.Tensor) -> torch.Tensor:
        """
        Encode node features based on their types.
        
        Args:
            x: Node features [num_nodes, max_feat_dim]
            node_types: Node type indices [num_nodes]
        
        Returns:
            Encoded embeddings [num_nodes, output_dim]
        """
        batch_size = x.size(0)
        embeddings = torch.zeros(batch_size, self.output_dim, device=x.device)
        
        for type_idx, node_type in enumerate(self.node_types):
            mask = (node_types == type_idx)
            if mask.sum() > 0:
                type_features = x[mask]
                # Handle different input dimensions by slicing
                input_dim = list(self.type_encoders[node_type].parameters())[0].size(1)
                type_features = type_features[:, :input_dim]
                embeddings[mask] = self.type_encoders[node_type](type_features)
        
        return embeddings


class TemporalEncoder(nn.Module):
    """
    Temporal encoder implementing both positional encoding (Eq. 3.2) 
    and learnable time embedding (Eq. 3.3).
    """
    def __init__(self, d_t: int = 64):
        super().__init__()
        self.d_t = d_t
        
        # Learnable time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_t // 2),
            nn.ReLU(),
            nn.Linear(d_t // 2, d_t // 2)
        )
        
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Encode timestamps using dual encoding strategy.
        
        Args:
            timestamps: Edge timestamps [num_edges]
        
        Returns:
            Temporal embeddings [num_edges, d_t]
        """
        batch_size = timestamps.size(0)
        device = timestamps.device
        
        # Positional encoding (Eq. 3.2)
        pos_encoding = torch.zeros(batch_size, self.d_t // 2, device=device)
        
        for i in range(0, self.d_t // 2, 2):
            div_term = math.pow(10000.0, 2 * i / self.d_t)
            pos_encoding[:, i] = torch.sin(timestamps / div_term)
            if i + 1 < self.d_t // 2:
                pos_encoding[:, i + 1] = torch.cos(timestamps / div_term)
        
        # Learnable time embedding
        timestamps_normalized = timestamps.unsqueeze(-1).float()
        learnable_encoding = self.time_mlp(timestamps_normalized)
        
        # Concatenate both encodings (Eq. 3.3)
        temporal_embedding = torch.cat([pos_encoding, learnable_encoding], dim=1)
        
        return temporal_embedding


class DecoupledAttentionAggregation(nn.Module):
    """
    Decoupled attention aggregation that separates neighbors into three groups:
    same-class, different-class, and unlabeled (based on Algorithm 1).
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1, top_k: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.top_k = top_k
        self.head_dim = hidden_dim // num_heads
        
        # Attention parameters for each relation type
        self.attention_weights = nn.ParameterDict()
        self.w_h = nn.ModuleDict()  # Weight matrices for node embeddings
        self.w_e = nn.ModuleDict()  # Weight matrices for edge embeddings
        
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def add_relation(self, relation_type: str, edge_dim: int):
        """Add parameters for a new relation type."""
        # Attention parameters (Eq. 3.5)
        self.attention_weights[relation_type] = nn.Parameter(
            torch.randn(2 * self.hidden_dim + edge_dim, self.num_heads)
        )
        
        # Message construction parameters (Eq. 3.4)
        self.w_h[relation_type] = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.w_e[relation_type] = nn.Linear(edge_dim, self.hidden_dim)
    
    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor, relation_type: str, 
                node_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Perform decoupled attention aggregation for a specific relation type.
        
        Args:
            h: Node embeddings [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes including temporal encoding [num_edges, edge_dim]
            relation_type: Type of relation
            node_labels: Node labels for grouping [num_nodes] (None for unlabeled)
        
        Returns:
            Aggregated embeddings [num_nodes, 3 * hidden_dim] (concatenated groups)
        """
        row, col = edge_index
        num_nodes = h.size(0)
        
        # Construct messages (Eq. 3.4)
        h_src = self.w_h[relation_type](h[row])
        e_transformed = self.w_e[relation_type](edge_attr)
        messages = F.relu(h_src + e_transformed)
        
        # Compute attention scores (Eq. 3.5)
        h_concat = torch.cat([h[row], h[col], edge_attr], dim=1)
        attention_scores = torch.matmul(h_concat, self.attention_weights[relation_type])
        attention_scores = self.leaky_relu(attention_scores).mean(dim=1)  # Average over heads
        
        # Apply softmax attention per target node
        attention_weights = torch.zeros_like(attention_scores)
        for node_idx in range(num_nodes):
            neighbor_mask = (col == node_idx)
            if neighbor_mask.sum() > 0:
                neighbor_scores = attention_scores[neighbor_mask]
                neighbor_weights = F.softmax(neighbor_scores, dim=0)
                attention_weights[neighbor_mask] = neighbor_weights
        
        # Apply top-k neighbor selection
        if self.top_k > 0:
            for node_idx in range(num_nodes):
                neighbor_mask = (col == node_idx)
                if neighbor_mask.sum() > self.top_k:
                    neighbor_indices = torch.where(neighbor_mask)[0]
                    neighbor_scores = attention_weights[neighbor_indices]
                    _, top_k_indices = torch.topk(neighbor_scores, self.top_k)
                    
                    # Zero out non-top-k neighbors
                    keep_mask = torch.zeros_like(neighbor_mask)
                    keep_mask[neighbor_indices[top_k_indices]] = True
                    attention_weights = attention_weights * keep_mask.float()
        
        # Group neighbors and aggregate separately
        group_aggregations = []
        
        if node_labels is not None:
            # Same-class neighbors
            same_class_messages = self._aggregate_group(
                messages, attention_weights, edge_index, node_labels, 'same', num_nodes
            )
            group_aggregations.append(same_class_messages)
            
            # Different-class neighbors  
            diff_class_messages = self._aggregate_group(
                messages, attention_weights, edge_index, node_labels, 'diff', num_nodes
            )
            group_aggregations.append(diff_class_messages)
            
            # Unlabeled neighbors (treat as separate group)
            unlabeled_messages = self._aggregate_group(
                messages, attention_weights, edge_index, node_labels, 'unlabeled', num_nodes
            )
            group_aggregations.append(unlabeled_messages)
        else:
            # If no labels available, treat all as one group
            all_messages = torch.zeros(num_nodes, self.hidden_dim, device=h.device)
            all_messages.index_add_(0, col, messages * attention_weights.unsqueeze(1))
            group_aggregations = [all_messages, all_messages, all_messages]
        
        # Concatenate group aggregations (Eq. 3.7)
        return torch.cat(group_aggregations, dim=1)
    
    def _aggregate_group(self, messages: torch.Tensor, attention_weights: torch.Tensor,
                        edge_index: torch.Tensor, node_labels: torch.Tensor, 
                        group_type: str, num_nodes: int) -> torch.Tensor:
        """Helper function to aggregate messages for a specific neighbor group."""
        row, col = edge_index
        aggregated = torch.zeros(num_nodes, self.hidden_dim, device=messages.device)
        
        if group_type == 'same':
            # Same class neighbors
            mask = (node_labels[row] == node_labels[col]) & (node_labels[row] != -1)
        elif group_type == 'diff':
            # Different class neighbors
            mask = (node_labels[row] != node_labels[col]) & (node_labels[row] != -1) & (node_labels[col] != -1)
        else:  # unlabeled
            # Unlabeled neighbors
            mask = (node_labels[row] == -1) | (node_labels[col] == -1)
        
        if mask.sum() > 0:
            masked_messages = messages[mask] * attention_weights[mask].unsqueeze(1)
            masked_col = col[mask]
            aggregated.index_add_(0, masked_col, masked_messages)
        
        return aggregated