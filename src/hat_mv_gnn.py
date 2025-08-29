import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from torch_geometric.data import HeteroData
from model_components import TypedAttributeEncoder, TemporalEncoder, DecoupledAttentionAggregation


class MultiViewAttentionFusion(nn.Module):
    """
    Multi-view attention fusion mechanism (Eq. 3.8) that combines 
    information from different relation types using learnable attention.
    """
    def __init__(self, hidden_dim: int, num_relation_types: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_relation_types = num_relation_types
        
        # Learnable attention parameters for relation types
        self.relation_attention = nn.Parameter(torch.randn(num_relation_types, hidden_dim))
        self.attention_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, view_embeddings: Dict[str, torch.Tensor], 
                relation_types: List[str]) -> torch.Tensor:
        """
        Fuse embeddings from multiple views using attention.
        
        Args:
            view_embeddings: Dictionary of embeddings per relation type
            relation_types: List of relation type names
            
        Returns:
            Fused embeddings [num_nodes, hidden_dim]
        """
        if not view_embeddings:
            raise ValueError("No view embeddings provided")
        
        # Get dimensions from first view
        first_view = list(view_embeddings.values())[0]
        num_nodes = first_view.size(0)
        device = first_view.device
        
        # Stack all view embeddings
        view_stack = []
        valid_relations = []
        
        for i, rel_type in enumerate(relation_types):
            if rel_type in view_embeddings:
                view_stack.append(view_embeddings[rel_type])
                valid_relations.append(i)
        
        if not view_stack:
            return torch.zeros(num_nodes, self.hidden_dim, device=device)
        
        # Shape: [num_nodes, num_valid_relations, hidden_dim]
        stacked_views = torch.stack(view_stack, dim=1)
        
        # Compute attention weights for each relation type
        relation_weights = []
        for rel_idx in valid_relations:
            # Use relation-specific attention parameter
            rel_attention = self.relation_attention[rel_idx].unsqueeze(0).expand(num_nodes, -1)
            attention_score = self.attention_mlp(rel_attention)  # [num_nodes, 1]
            relation_weights.append(attention_score)
        
        # Stack and apply softmax: [num_nodes, num_valid_relations]
        attention_weights = torch.cat(relation_weights, dim=1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention weights and sum: [num_nodes, hidden_dim]
        fused_embedding = torch.sum(
            stacked_views * attention_weights.unsqueeze(-1), dim=1
        )
        
        return fused_embedding


class HATMVGNNLayer(nn.Module):
    """
    Single layer of the Heterophily-Aware Temporal Multi-View GNN.
    Implements the complete message passing framework from Algorithm 1.
    """
    def __init__(self, hidden_dim: int, relation_types: List[str], 
                 edge_dims: Dict[str, int], temporal_dim: int = 64,
                 num_heads: int = 4, dropout: float = 0.1, top_k: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.relation_types = relation_types
        self.temporal_dim = temporal_dim
        
        # Decoupled attention aggregation for each relation type
        self.aggregators = nn.ModuleDict()
        for rel_type in relation_types:
            aggregator = DecoupledAttentionAggregation(
                hidden_dim, num_heads, dropout, top_k
            )
            # Add relation with edge dimension + temporal dimension
            aggregator.add_relation(rel_type, edge_dims[rel_type] + temporal_dim)
            self.aggregators[rel_type] = aggregator
        
        # Multi-view attention fusion
        self.multi_view_fusion = MultiViewAttentionFusion(
            3 * hidden_dim,  # 3x because of decoupled aggregation
            len(relation_types)
        )
        
        # GRU for temporal update (Eq. 3.9)
        self.gru = nn.GRUCell(3 * hidden_dim, hidden_dim)
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, h: torch.Tensor, edge_data: Dict[str, Tuple], 
                temporal_encodings: Dict[str, torch.Tensor],
                node_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of HAT-MV-GNN layer.
        
        Args:
            h: Current node embeddings [num_nodes, hidden_dim]
            edge_data: Dict of (edge_index, edge_attr) for each relation type
            temporal_encodings: Dict of temporal encodings for each relation type
            node_labels: Node labels for decoupled aggregation
            
        Returns:
            Updated node embeddings [num_nodes, hidden_dim]
        """
        view_embeddings = {}
        
        # Process each relation type (view)
        for rel_type in self.relation_types:
            if rel_type in edge_data:
                edge_index, edge_attr = edge_data[rel_type]
                temporal_encoding = temporal_encodings.get(rel_type, 
                    torch.zeros(edge_attr.size(0), self.temporal_dim, device=h.device))
                
                # Combine edge attributes with temporal encoding
                enhanced_edge_attr = torch.cat([edge_attr, temporal_encoding], dim=1)
                
                # Apply decoupled attention aggregation
                view_embedding = self.aggregators[rel_type](
                    h, edge_index, enhanced_edge_attr, rel_type, node_labels
                )
                view_embeddings[rel_type] = view_embedding
        
        # Multi-view fusion (Eq. 3.8)
        if view_embeddings:
            fused_embedding = self.multi_view_fusion(view_embeddings, self.relation_types)
        else:
            fused_embedding = torch.zeros(h.size(0), 3 * self.hidden_dim, device=h.device)
        
        # Temporal update with GRU (Eq. 3.9)
        updated_h = self.gru(fused_embedding, h)
        
        # Apply layer normalization and dropout
        updated_h = self.layer_norm(updated_h)
        updated_h = self.dropout(updated_h)
        
        return updated_h


class HATMVGNN(nn.Module):
    """
    Complete Heterophily-Aware Temporal Multi-View GNN model for fraud detection.
    Implements the full architecture described in your research paper.
    """
    def __init__(self, node_types: List[str], input_dims: Dict[str, int],
                 relation_types: List[str], edge_dims: Dict[str, int],
                 hidden_dim: int = 128, num_layers: int = 3, 
                 temporal_dim: int = 64, num_heads: int = 4,
                 dropout: float = 0.1, top_k: int = 10, num_classes: int = 2):
        super().__init__()
        
        self.node_types = node_types
        self.relation_types = relation_types
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Input encoding
        self.node_encoder = TypedAttributeEncoder(
            node_types, input_dims, hidden_dim, hidden_dim, num_layers=2
        )
        self.temporal_encoder = TemporalEncoder(temporal_dim)
        
        # HAT-MV-GNN layers
        self.gnn_layers = nn.ModuleList([
            HATMVGNNLayer(
                hidden_dim, relation_types, edge_dims, temporal_dim,
                num_heads, dropout, top_k
            ) for _ in range(num_layers)
        ])
        
        # Classification head (Eq. 3.10)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize parameters
        self.init_parameters()
    
    def init_parameters(self):
        """Initialize model parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Parameter):
                nn.init.xavier_uniform_(module)
    
    def forward(self, node_features: torch.Tensor, node_types: torch.Tensor,
                edge_data: Dict[str, Tuple], timestamps: Dict[str, torch.Tensor],
                node_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the complete HAT-MV-GNN model.
        
        Args:
            node_features: Raw node features [num_nodes, max_feat_dim]
            node_types: Node type indices [num_nodes]
            edge_data: Dict of (edge_index, edge_attr) for each relation type
            timestamps: Dict of edge timestamps for each relation type
            node_labels: Node labels for training (optional)
            
        Returns:
            Node predictions [num_nodes, num_classes]
        """
        # Input encoding (Eq. 3.1)
        h = self.node_encoder(node_features, node_types)
        
        # Temporal encoding for all relation types (Eq. 3.2, 3.3)
        temporal_encodings = {}
        for rel_type in self.relation_types:
            if rel_type in timestamps:
                temporal_encodings[rel_type] = self.temporal_encoder(timestamps[rel_type])
        
        # Apply HAT-MV-GNN layers
        for layer in self.gnn_layers:
            h = layer(h, edge_data, temporal_encodings, node_labels)
        
        # Classification (Eq. 3.10)
        predictions = self.classifier(h)
        
        return predictions
    
    def compute_loss(self, predictions: torch.Tensor, labels: torch.Tensor,
                     labeled_mask: torch.Tensor, class_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute cross-entropy loss over labeled nodes (Eq. 3.11).
        
        Args:
            predictions: Model predictions [num_nodes, num_classes]
            labels: Ground truth labels [num_nodes]
            labeled_mask: Boolean mask for labeled nodes [num_nodes]
            class_weights: Optional class weights for handling imbalanced data
            
        Returns:
            Cross-entropy loss
        """
        if labeled_mask.sum() == 0:
            return torch.tensor(0.0, device=predictions.device)
        
        labeled_predictions = predictions[labeled_mask]
        labeled_targets = labels[labeled_mask]
        
        if class_weights is not None:
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            loss_fn = nn.CrossEntropyLoss()
        
        return loss_fn(labeled_predictions, labeled_targets)
    
    def predict(self, node_features: torch.Tensor, node_types: torch.Tensor,
                edge_data: Dict[str, Tuple], timestamps: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Make predictions without requiring labels.
        
        Returns:
            Predicted class probabilities [num_nodes, num_classes]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(node_features, node_types, edge_data, timestamps)
            probabilities = F.softmax(logits, dim=1)
        return probabilities