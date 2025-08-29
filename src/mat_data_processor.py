import scipy.io as sio
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import os


class MatDataProcessor:
    """
    Data processor for .mat fraud detection datasets.
    Handles Amazon.mat and YelpChi.mat files directly.
    """
    
    def __init__(self, dataset_name: str):
        """
        Initialize data processor for .mat datasets.
        
        Args:
            dataset_name: Either 'amazon' or 'yelp'
        """
        self.dataset_name = dataset_name.lower()
        self.scalers = {}
        self.logger = logging.getLogger(__name__)
        
        # Dataset configurations based on common .mat file structures
        self.dataset_configs = {
            'amazon': {
                'mat_file': 'amazon.mat',
                'node_types': ['user', 'product', 'vendor'],
                'relation_types': ['user_product', 'user_same', 'user_vendor', 'homo'],
                'relation_keys': ['net_upu', 'net_usu', 'net_uvu', 'homo'],  # Common keys in Amazon.mat
                'feature_key': 'features',
                'label_key': 'label',
                'temporal_relations': ['user_product'],
                'target_type': 'user'
            },
            'yelp': {
                'mat_file': 'yelpChi.mat',
                'node_types': ['user', 'review', 'restaurant', 'service'], 
                'relation_types': ['review_user', 'review_restaurant', 'review_service', 'homo'],
                'relation_keys': ['net_rur', 'net_rtr', 'net_rsr', 'homo'],  # Common keys in YelpChi.mat
                'feature_key': 'features',
                'label_key': 'label',
                'temporal_relations': ['review_user', 'review_restaurant'],
                'target_type': 'user'
            }
        }
    
    def load_mat_data(self, data_path: str) -> Dict:
        """
        Load .mat dataset and extract all available information.
        
        Args:
            data_path: Path to the directory containing .mat file
            
        Returns:
            Dictionary containing loaded data
        """
        config = self.dataset_configs[self.dataset_name]
        mat_file_path = os.path.join(data_path, config['mat_file'])
        
        if not os.path.exists(mat_file_path):
            raise FileNotFoundError(f"Dataset file not found: {mat_file_path}")
        
        self.logger.info(f"Loading {self.dataset_name} data from {mat_file_path}")
        
        # Load .mat file
        mat_data = sio.loadmat(mat_file_path)
        
        # Log all keys in the .mat file for debugging
        mat_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        self.logger.info(f"Available keys in .mat file: {mat_keys}")
        
        # Extract features
        if config['feature_key'] in mat_data:
            features = mat_data[config['feature_key']]
            if hasattr(features, 'toarray'):
                features = features.toarray()
            self.logger.info(f"Features shape: {features.shape}")
        else:
            self.logger.warning(f"Feature key '{config['feature_key']}' not found, using identity features")
            # Create identity features if not available
            num_nodes = self._estimate_num_nodes(mat_data, config)
            features = np.eye(num_nodes)
        
        # Extract labels
        if config['label_key'] in mat_data:
            labels = mat_data[config['label_key']]
            if labels.ndim > 1:
                labels = labels.flatten()
            self.logger.info(f"Labels shape: {labels.shape}, unique values: {np.unique(labels)}")
        else:
            self.logger.error(f"Label key '{config['label_key']}' not found in .mat file")
            raise KeyError(f"Labels not found in dataset")
        
        # Extract relation matrices
        relations = {}
        for i, rel_key in enumerate(config['relation_keys']):
            if rel_key in mat_data:
                adj_matrix = mat_data[rel_key]
                if hasattr(adj_matrix, 'toarray'):
                    adj_matrix = adj_matrix.toarray()
                relations[config['relation_types'][i]] = adj_matrix
                self.logger.info(f"Relation '{rel_key}' shape: {adj_matrix.shape}, "
                               f"edges: {np.count_nonzero(adj_matrix)}")
            else:
                self.logger.warning(f"Relation key '{rel_key}' not found in .mat file")
        
        if not relations:
            self.logger.error("No relations found in .mat file")
            raise ValueError("No valid relations found in dataset")
        
        return {
            'features': features,
            'labels': labels,
            'relations': relations,
            'config': config,
            'mat_keys': mat_keys  # For debugging
        }
    
    def _estimate_num_nodes(self, mat_data: Dict, config: Dict) -> int:
        """Estimate number of nodes from available matrices."""
        for rel_key in config['relation_keys']:
            if rel_key in mat_data:
                matrix = mat_data[rel_key]
                if hasattr(matrix, 'shape'):
                    return matrix.shape[0]
        return 1000  # Fallback
    
    def convert_to_hat_mv_gnn_format(self, mat_data: Dict) -> Dict:
        """
        Convert .mat data to HAT-MV-GNN format.
        
        Args:
            mat_data: Data loaded from .mat file
            
        Returns:
            Data in HAT-MV-GNN format
        """
        features = mat_data['features']
        labels = mat_data['labels']
        relations = mat_data['relations']
        config = mat_data['config']
        
        num_nodes = features.shape[0]
        
        # Ensure labels are in correct format
        if labels.min() < 0:
            self.logger.info("Converting labels from {-1, 1} to {0, 1} format")
            labels = (labels + 1) // 2
        
        # Standardize features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)
        self.scalers[self.dataset_name] = scaler
        
        # Create tensors
        node_features = torch.FloatTensor(features_normalized)
        labels_tensor = torch.LongTensor(labels.astype(int))
        
        # For now, treat all nodes as the same type (can be refined later)
        node_types = torch.zeros(num_nodes, dtype=torch.long)
        
        # Convert relation matrices to edge lists
        edge_data = {}
        timestamps = {}
        
        for rel_name, adj_matrix in relations.items():
            edge_index, edge_attr = self._adj_matrix_to_edge_list(adj_matrix)
            if edge_index.size(1) > 0:  # Only add if there are edges
                edge_data[rel_name] = (edge_index, edge_attr)
                
                # Generate synthetic timestamps for temporal relations
                num_edges = edge_index.size(1)
                if rel_name in config['temporal_relations']:
                    # Create realistic timestamps (e.g., spread over a year)
                    timestamps[rel_name] = torch.randint(0, 365*24*3600, (num_edges,)).float()
                else:
                    timestamps[rel_name] = torch.zeros(num_edges).float()
        
        # Create node type mapping
        node_type_mapping = {node_type: i for i, node_type in enumerate(config['node_types'])}
        
        # Create global node mapping
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
    
    def _adj_matrix_to_edge_list(self, adj_matrix: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert adjacency matrix to edge list format.
        
        Args:
            adj_matrix: Adjacency matrix (can be weighted)
            
        Returns:
            Tuple of (edge_index, edge_attr)
        """
        # Find non-zero entries
        rows, cols = np.nonzero(adj_matrix)
        
        if len(rows) == 0:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1), dtype=torch.float)
        
        # Create edge index
        edge_index = torch.LongTensor([rows, cols])
        
        # Create edge attributes from matrix values
        edge_values = adj_matrix[rows, cols]
        if hasattr(edge_values, 'A1'):  # Handle sparse matrix
            edge_values = edge_values.A1
        
        # Normalize edge weights if they're not binary
        if not np.all(np.isin(edge_values, [0, 1])):
            edge_values = (edge_values - edge_values.min()) / (edge_values.max() - edge_values.min() + 1e-8)
        
        edge_attr = torch.FloatTensor(edge_values.reshape(-1, 1))
        
        return edge_index, edge_attr
    
    def create_train_test_split(self, graph_data: Dict, 
                               test_size: float = 0.2, val_size: float = 0.1,
                               random_state: int = 42) -> Dict:
        """
        Create stratified train/validation/test splits.
        
        Args:
            graph_data: Processed graph data
            test_size: Proportion for testing
            val_size: Proportion for validation
            random_state: Random seed
            
        Returns:
            Updated graph_data with masks
        """
        labels = graph_data['labels']
        num_nodes = len(labels)
        
        # Get all node indices and labels
        all_indices = np.arange(num_nodes)
        all_labels = labels.numpy()
        
        # Check class distribution
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        self.logger.info(f"Class distribution: {dict(zip(unique_labels, counts))}")
        
        # Ensure we have enough samples for stratification
        min_class_size = counts.min()
        if min_class_size < 3:
            self.logger.warning("Very small minority class, using random split instead of stratified")
            # Random split
            train_idx, temp_idx = train_test_split(
                all_indices, test_size=test_size + val_size, random_state=random_state
            )
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=test_size / (test_size + val_size), random_state=random_state
            )
        else:
            # Stratified split
            train_idx, temp_idx, train_y, temp_y = train_test_split(
                all_indices, all_labels, 
                test_size=test_size + val_size, 
                random_state=random_state,
                stratify=all_labels
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
        
        # Log split statistics
        train_labels = labels[train_mask]
        val_labels = labels[val_mask]
        test_labels = labels[test_mask]
        
        self.logger.info(f"Train set: {train_mask.sum()} nodes, "
                        f"pos: {(train_labels == 1).sum()}, neg: {(train_labels == 0).sum()}")
        self.logger.info(f"Val set: {val_mask.sum()} nodes, "
                        f"pos: {(val_labels == 1).sum()}, neg: {(val_labels == 0).sum()}")
        self.logger.info(f"Test set: {test_mask.sum()} nodes, "
                        f"pos: {(test_labels == 1).sum()}, neg: {(test_labels == 0).sum()}")
        
        return graph_data
    
    def process_dataset(self, data_path: str, output_path: Optional[str] = None) -> Dict:
        """
        Complete preprocessing pipeline for .mat datasets.
        
        Args:
            data_path: Path to directory containing .mat file
            output_path: Optional path to save processed data
            
        Returns:
            Processed graph data ready for HAT-MV-GNN
        """
        self.logger.info(f"Processing {self.dataset_name} .mat dataset from {data_path}")
        
        # Load .mat data
        mat_data = self.load_mat_data(data_path)
        
        # Convert to HAT-MV-GNN format
        graph_data = self.convert_to_hat_mv_gnn_format(mat_data)
        
        # Create splits
        graph_data = self.create_train_test_split(graph_data)
        
        # Log final statistics
        self.logger.info(f"Final dataset statistics:")
        self.logger.info(f"  Nodes: {graph_data['node_features'].size(0)}")
        self.logger.info(f"  Features: {graph_data['node_features'].size(1)}")
        self.logger.info(f"  Relations: {list(graph_data['edge_data'].keys())}")
        total_edges = sum(edge_data[0].size(1) for edge_data in graph_data['edge_data'].values())
        self.logger.info(f"  Total edges: {total_edges}")
        
        # Save processed data
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(graph_data, output_path)
            self.logger.info(f"Processed data saved to {output_path}")
        
        self.logger.info("Dataset processing completed successfully")
        return graph_data


def load_mat_dataset(dataset_name: str, data_path: str, output_path: Optional[str] = None) -> Dict:
    """
    Convenience function to load and process .mat datasets.
    
    Args:
        dataset_name: 'amazon' or 'yelp'
        data_path: Path to directory containing .mat file
        output_path: Optional path to save processed data
        
    Returns:
        Processed graph data ready for HAT-MV-GNN training
    """
    processor = MatDataProcessor(dataset_name)
    return processor.process_dataset(data_path, output_path)