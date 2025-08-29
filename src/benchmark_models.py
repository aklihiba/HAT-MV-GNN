import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, HeteroConv, to_hetero
from typing import Dict, List, Tuple, Optional


class GCNBaseline(nn.Module):
    """
    Simple GCN baseline for fraud detection.
    Uses homogeneous graph (all relations combined).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, 
                 dropout: float = 0.1, num_classes: int = 2):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout_layer(x)
        
        return self.classifier(x)


class GATBaseline(nn.Module):
    """
    GAT baseline for fraud detection.
    Uses attention mechanism but single relation type.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 num_heads: int = 4, dropout: float = 0.1, num_classes: int = 2):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout))
        
        if num_layers > 1:
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout))
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = self.dropout_layer(x)
        
        return self.classifier(x)


class GraphSAGEBaseline(nn.Module):
    """
    GraphSAGE baseline for fraud detection.
    Uses sampling-based aggregation.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.1, num_classes: int = 2):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout_layer(x)
        
        return self.classifier(x)


class HeteroGNNBaseline(nn.Module):
    """
    Heterogeneous GNN baseline using PyTorch Geometric's HeteroConv.
    Processes multiple relation types but without temporal or heterophily awareness.
    """
    def __init__(self, input_dims: Dict[str, int], relation_types: List[str], 
                 hidden_dim: int = 64, num_layers: int = 2, 
                 dropout: float = 0.1, num_classes: int = 2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.relation_types = relation_types
        
        # Create heterogeneous convolutions
        self.convs = nn.ModuleList()
        
        # First layer
        conv_dict = {}
        for rel_type in relation_types:
            # Assume all node types have same input dim for simplicity
            input_dim = list(input_dims.values())[0]
            conv_dict[rel_type] = SAGEConv(input_dim, hidden_dim)
        
        self.convs.append(HeteroConv(conv_dict))
        
        # Additional layers
        for _ in range(num_layers - 1):
            conv_dict = {}
            for rel_type in relation_types:
                conv_dict[rel_type] = SAGEConv(hidden_dim, hidden_dim)
            self.convs.append(HeteroConv(conv_dict))
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x_dict: Dict[str, torch.Tensor], 
                edge_index_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for heterogeneous graph.
        
        Args:
            x_dict: Node features for each node type
            edge_index_dict: Edge indices for each relation type
            
        Returns:
            Predictions for target node type
        """
        # Process through hetero convolutions
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if i < len(self.convs) - 1:
                x_dict = {key: F.relu(x) for key, x in x_dict.items()}
                x_dict = {key: self.dropout(x) for key, x in x_dict.items()}
        
        # Get predictions for target node type (assume first type is target)
        target_type = list(x_dict.keys())[0]
        target_embeddings = x_dict[target_type]
        
        return self.classifier(target_embeddings)


class SimpleMLP(nn.Module):
    """
    Simple MLP baseline that ignores graph structure.
    Uses only node features for prediction.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.1, num_classes: int = 2):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor = None) -> torch.Tensor:
        # Ignore edge_index, use only node features
        return self.network(x)


def create_homogeneous_graph(edge_data: Dict[str, Tuple], num_nodes: int) -> torch.Tensor:
    """
    Combine all relations into a single homogeneous graph.
    
    Args:
        edge_data: Dictionary of (edge_index, edge_attr) for each relation
        num_nodes: Total number of nodes
        
    Returns:
        Combined edge index tensor
    """
    all_edges = []
    
    for rel_type, (edge_index, _) in edge_data.items():
        all_edges.append(edge_index)
    
    if all_edges:
        combined_edges = torch.cat(all_edges, dim=1)
        # Remove duplicate edges
        combined_edges = torch.unique(combined_edges, dim=1)
        return combined_edges
    else:
        return torch.empty((2, 0), dtype=torch.long)


def create_benchmark_model(model_name: str, input_dim: int, 
                          relation_types: Optional[List[str]] = None,
                          input_dims: Optional[Dict[str, int]] = None,
                          **kwargs) -> nn.Module:
    """
    Factory function to create benchmark models.
    
    Args:
        model_name: Name of the model ('gcn', 'gat', 'sage', 'hetero', 'mlp')
        input_dim: Input feature dimension
        relation_types: List of relation types (for hetero model)
        input_dims: Input dimensions per node type (for hetero model)
        **kwargs: Additional model parameters
        
    Returns:
        Initialized model
    """
    model_name = model_name.lower()
    
    default_kwargs = {
        'hidden_dim': 64,
        'num_layers': 2,
        'dropout': 0.1,
        'num_classes': 2
    }
    default_kwargs.update(kwargs)
    
    if model_name == 'gcn':
        return GCNBaseline(input_dim, **default_kwargs)
    elif model_name == 'gat':
        return GATBaseline(input_dim, num_heads=kwargs.get('num_heads', 4), **default_kwargs)
    elif model_name == 'sage':
        return GraphSAGEBaseline(input_dim, **default_kwargs)
    elif model_name == 'hetero':
        if relation_types is None or input_dims is None:
            raise ValueError("relation_types and input_dims required for hetero model")
        return HeteroGNNBaseline(input_dims, relation_types, **default_kwargs)
    elif model_name == 'mlp':
        return SimpleMLP(input_dim, **default_kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


class BaselineModelWrapper:
    """
    Wrapper class to standardize baseline model interface.
    Handles data format conversion for different model types.
    """
    def __init__(self, model: nn.Module, model_type: str):
        self.model = model
        self.model_type = model_type.lower()
    
    def forward(self, node_features: torch.Tensor, node_types: torch.Tensor,
                edge_data: Dict[str, Tuple], **kwargs) -> torch.Tensor:
        """
        Standardized forward pass for all baseline models.
        
        Args:
            node_features: Node feature matrix
            node_types: Node type indices (may be ignored)
            edge_data: Dictionary of edge data per relation type
            
        Returns:
            Model predictions
        """
        if self.model_type in ['gcn', 'gat', 'sage', 'mlp']:
            # For homogeneous models, combine all edges
            if self.model_type == 'mlp':
                return self.model(node_features)
            else:
                combined_edges = create_homogeneous_graph(edge_data, node_features.size(0))
                return self.model(node_features, combined_edges)
        
        elif self.model_type == 'hetero':
            # For heterogeneous model, need to convert format
            # This is a simplified version - real implementation would be more complex
            x_dict = {'user': node_features}  # Simplified - assume single node type
            edge_index_dict = {}
            for rel_type, (edge_index, _) in edge_data.items():
                edge_index_dict[rel_type] = edge_index
            
            return self.model(x_dict, edge_index_dict)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def to(self, device):
        """Move model to device."""
        self.model = self.model.to(device)
        return self
    
    def parameters(self):
        """Return model parameters."""
        return self.model.parameters()
    
    def train(self, mode=True):
        """Set training mode."""
        self.model.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode."""
        self.model.eval()
        return self