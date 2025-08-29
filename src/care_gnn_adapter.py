import scipy.io as sio
import numpy as np
import torch
import pickle
import os
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
import logging


class CAREGNNDataAdapter:
    """
    Adapter to convert CARE-GNN datasets to HAT-MV-GNN format.
    Handles Amazon.mat and YelpChi.mat files from the CARE-GNN repository.
    """
    
    def __init__(self, dataset_name: str):
        """
        Initialize adapter for CARE-GNN datasets.
        
        Args:
            dataset_name: Either 'amazon' or 'yelp'
        """
        self.dataset_name = dataset_name.lower()
        self.logger = logging.getLogger(__name__)
        
        # CARE-GNN dataset configurations
        self.care_configs = {
            'amazon': {
                'mat_file': 'Amazon.mat',
                'node_types': ['user', 'product', 'vendor'],  # Based on U-P-U, U-S-U, U-V-U relations
                'relation_types': ['user_product', 'user_same', 'user_vendor', 'user_user_homo'],
                'relation_keys': ['net_upu', 'net_usu', 'net_uvu', 'homo'],  # Keys in .mat file
                'feature_key': 'features',
                'label_key': 'label',
                'temporal_relations': ['user_product'],  # Relations that might have temporal info
                'target_type': 'user'
            },
            'yelp': {
                'mat_file': 'YelpChi.mat', 
                'node_types': ['user', 'review', 'restaurant', 'service'],  # Based on R-U-R, R-T-R, R-S-R relations
                'relation_types': ['review_user', 'review_restaurant', 'review_service', 'user_user_homo'],
                'relation_keys': ['net_rur', 'net_rtr', 'net_rsr', 'homo'],  # Keys in .mat file
                'feature_key': 'features',
                'label_key': 'label', 
                'temporal_relations': ['review_user', 'review_restaurant'],
                'target_type': 'user'  # Actually reviews in Yelp, but we predict user fraud
            }
        }
    
    def load_care_gnn_data(self, data_path: str) -> Dict:
        """
        Load CARE-GNN .mat dataset.
        
        Args:
            data_path: Path to the directory containing .mat file
            
        Returns:
            Dictionary containing loaded data
        """
        config = self.care_configs[self.dataset_name]
        mat_file_path = os.path.join(data_path, config['mat_file'])
        
        if not os.path.exists(mat_file_path):
            raise FileNotFoundError(f"Dataset file not found: {mat_file_path}")
        
        self.logger.info(f"Loading CARE-GNN {self.dataset_name} data from {mat_file_path}")
        
        # Load .mat file
        mat_data = sio.loadmat(mat_file_path)
        
        # Extract features and labels
        features = mat_data[config['feature_key']]
        labels = mat_data[config['label_key']].flatten()
        
        # Convert sparse matrices if needed
        if hasattr(features, 'toarray'):
            features = features.toarray()
        
        # Extract relation matrices
        relations = {}
        for i, rel_key in enumerate(config['relation_keys']):
            if rel_key in mat_data:
                adj_matrix = mat_data[rel_key]
                if hasattr(adj_matrix, 'toarray'):
                    adj_matrix = adj_matrix.toarray()
                relations[config['relation_types'][i]] = adj_matrix
            else:
                self.logger.warning(f"Relation {rel_key} not found in dataset")
        
        return {
            'features': features,
            'labels': labels,
            'relations': relations,
            'config': config
        }
    
    def convert_to_hat_mv_gnn_format(self, care_data: Dict) -> Dict:
        """
        Convert CARE-GNN data format to HAT-MV-GNN format.
        
        Args:
            care_data: Data loaded from CARE-GNN .mat file
            
        Returns:
            Data in HAT-MV-GNN format
        """
        features = care_data['features']
        labels = care_data['labels']
        relations = care_data['relations']
        config = care_data['config']
        
        num_nodes = features.shape[0]
        feature_dim = features.shape[1]
        
        # Create node features tensor
        node_features = torch.FloatTensor(features)
        
        # Create node types (for now, treat all nodes as same type since CARE-GNN doesn't distinguish)
        # In CARE-GNN, all nodes are typically users, but relations connect them through different paths
        node_types = torch.zeros(num_nodes, dtype=torch.long)  # All nodes as type 0 (user)
        
        # Convert labels (CARE-GNN uses 0/1, HAT-MV-GNN expects same)
        labels_tensor = torch.LongTensor(labels)
        
        # Convert relation matrices to edge lists
        edge_data = {}
        timestamps = {}
        
        for rel_name, adj_matrix in relations.items():
            edge_index, edge_attr = self._adj_matrix_to_edge_list(adj_matrix, rel_name)
            if edge_index.size(1) > 0:  # Only add if there are edges
                edge_data[rel_name] = (edge_index, edge_attr)
                
                # Generate dummy timestamps for temporal relations
                num_edges = edge_index.size(1)
                if rel_name in config['temporal_relations']:
                    # Generate random timestamps (you can replace with actual timestamps if available)
                    timestamps[rel_name] = torch.randint(0, 1000, (num_edges,)).float()
                else:
                    timestamps[rel_name] = torch.zeros(num_edges).float()
        
        # Create node type mapping
        node_type_mapping = {node_type: i for i, node_type in enumerate(config['node_types'])}
        
        # Create global node mapping (identity mapping since all nodes are in one space)
        global_node_mapping = {i: i for i in range(num_nodes)}
        
        return {
            'node_features': node_features,
            'node_types': node_types,
            'labels': labels_tensor,
            'edge_data': edge_data,
            'timestamps': timestamps,
            'node_type_mapping': node_type_mapping,
            'global_node_mapping': global_node_mapping,
            'config': {
                'node_types': config['node_types'],
                'relation_types': list(edge_data.keys()),
                'temporal_relations': config['temporal_relations'],
                'target_type': config['target_type']
            }
        }
    
    def _adj_matrix_to_edge_list(self, adj_matrix: np.ndarray, relation_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert adjacency matrix to edge list format.
        
        Args:
            adj_matrix: Adjacency matrix
            relation_name: Name of the relation
            
        Returns:
            Tuple of (edge_index, edge_attr)
        """
        # Find non-zero entries
        rows, cols = np.nonzero(adj_matrix)
        
        if len(rows) == 0:
            # Return empty tensors if no edges
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1), dtype=torch.float)
        
        # Create edge index
        edge_index = torch.LongTensor([rows, cols])
        
        # Create edge attributes (use the values from adjacency matrix)
        edge_values = adj_matrix[rows, cols]
        if hasattr(edge_values, 'toarray'):
            edge_values = edge_values.toarray().flatten()
        
        # If binary adjacency matrix, create simple edge features
        if np.all(np.isin(edge_values, [0, 1])):
            edge_attr = torch.ones((len(rows), 1), dtype=torch.float)
        else:
            edge_attr = torch.FloatTensor(edge_values).unsqueeze(1)
        
        return edge_index, edge_attr
    
    def create_train_test_split(self, graph_data: Dict, 
                               test_size: float = 0.2, val_size: float = 0.1,
                               random_state: int = 42) -> Dict:
        """
        Create train/validation/test splits for CARE-GNN data.
        
        Args:
            graph_data: Processed graph data
            test_size: Proportion of data for testing
            val_size: Proportion of data for validation
            random_state: Random seed
            
        Returns:
            Updated graph_data with train/val/test masks
        """
        labels = graph_data['labels']
        num_nodes = len(labels)
        
        # Get labeled nodes (CARE-GNN typically has all nodes labeled)
        labeled_indices = np.arange(num_nodes)
        labeled_targets = labels.numpy()
        
        # Create stratified splits
        train_idx, temp_idx, train_y, temp_y = train_test_split(
            labeled_indices, labeled_targets, 
            test_size=test_size + val_size, 
            random_state=random_state,
            stratify=labeled_targets
        )
        
        val_idx, test_idx, _, _ = train_test_split(
            temp_idx, temp_y,
            test_size=test_size / (test_size + val_size),
            random_state=random_state,
            stratify=temp_y
        )
        
        # Create masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        
        graph_data['train_mask'] = train_mask
        graph_data['val_mask'] = val_mask
        graph_data['test_mask'] = test_mask
        
        # Log class distribution
        unique_labels, counts = torch.unique(labels, return_counts=True)
        self.logger.info(f"Class distribution: {dict(zip(unique_labels.tolist(), counts.tolist()))}")
        self.logger.info(f"Data split - Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
        
        return graph_data
    
    def process_care_gnn_dataset(self, data_path: str, output_path: Optional[str] = None) -> Dict:
        """
        Complete preprocessing pipeline for CARE-GNN datasets.
        
        Args:
            data_path: Path to directory containing .mat file
            output_path: Optional path to save processed data
            
        Returns:
            Processed graph data ready for HAT-MV-GNN
        """
        self.logger.info(f"Processing CARE-GNN {self.dataset_name} dataset")
        
        # Load CARE-GNN data
        care_data = self.load_care_gnn_data(data_path)
        
        # Convert to HAT-MV-GNN format
        graph_data = self.convert_to_hat_mv_gnn_format(care_data)
        
        # Create splits
        graph_data = self.create_train_test_split(graph_data)
        
        # Log dataset statistics
        self.logger.info(f"Dataset statistics:")
        self.logger.info(f"  Nodes: {graph_data['node_features'].size(0)}")
        self.logger.info(f"  Features: {graph_data['node_features'].size(1)}")
        self.logger.info(f"  Relations: {list(graph_data['edge_data'].keys())}")
        self.logger.info(f"  Total edges: {sum(edge_data[0].size(1) for edge_data in graph_data['edge_data'].values())}")
        
        # Save processed data
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(graph_data, output_path)
            self.logger.info(f"Processed data saved to {output_path}")
        
        self.logger.info("CARE-GNN dataset processing completed successfully")
        return graph_data


def load_care_gnn_dataset(dataset_name: str, data_path: str, output_path: Optional[str] = None) -> Dict:
    """
    Convenience function to load and process CARE-GNN datasets.
    
    Args:
        dataset_name: 'amazon' or 'yelp'
        data_path: Path to directory containing .mat files
        output_path: Optional path to save processed data
        
    Returns:
        Processed graph data ready for HAT-MV-GNN training
    """
    adapter = CAREGNNDataAdapter(dataset_name)
    return adapter.process_care_gnn_dataset(data_path, output_path)