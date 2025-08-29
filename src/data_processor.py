import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import networkx as nx
from collections import defaultdict
import logging


class FraudDataProcessor:
    """
    Data preprocessing pipeline for Amazon and Yelp fraud detection datasets.
    Creates heterogeneous temporal graphs from raw data.
    """
    
    def __init__(self, dataset_name: str = 'amazon'):
        """
        Initialize data processor.
        
        Args:
            dataset_name: Either 'amazon' or 'yelp'
        """
        self.dataset_name = dataset_name.lower()
        self.scalers = {}
        self.label_encoders = {}
        self.node_mappings = {}
        self.logger = logging.getLogger(__name__)
        
        # Dataset-specific configurations
        self.dataset_configs = {
            'amazon': {
                'node_types': ['user', 'product', 'category'],
                'relation_types': ['user_product', 'product_category', 'user_user'],
                'temporal_relations': ['user_product'],  # Relations with timestamps
                'target_type': 'user'  # Node type to predict
            },
            'yelp': {
                'node_types': ['user', 'business', 'review'],
                'relation_types': ['user_business', 'user_review', 'review_business', 'user_user'],
                'temporal_relations': ['user_business', 'user_review'],
                'target_type': 'user'
            }
        }
    
    def load_data(self, data_path: str) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load raw fraud detection data.
        
        Args:
            data_path: Path to dataset directory
            
        Returns:
            Tuple of (node_data, edge_data) DataFrames
        """
        if self.dataset_name == 'amazon':
            return self._load_amazon_data(data_path)
        elif self.dataset_name == 'yelp':
            return self._load_yelp_data(data_path)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
    
    def _load_amazon_data(self, data_path: str) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Load Amazon fraud dataset."""
        try:
            # Load main data file
            df = pd.read_csv(f"{data_path}/Amazon.csv")
            
            # Basic preprocessing
            df = df.dropna()
            
            # Create node features from user and product information
            node_data = pd.DataFrame()
            
            # User nodes
            user_features = df.groupby('user_id').agg({
                'rating': ['mean', 'std', 'count'],
                'helpful': 'sum',
                'verified': lambda x: (x == 'Y').sum() / len(x)
            }).reset_index()
            user_features.columns = ['node_id', 'rating_mean', 'rating_std', 'rating_count', 
                                   'helpful_total', 'verified_ratio']
            user_features['node_type'] = 'user'
            user_features['label'] = -1  # Will be filled if labels exist
            
            # Product nodes  
            product_features = df.groupby('prod_id').agg({
                'rating': ['mean', 'std', 'count'],
                'helpful': 'mean'
            }).reset_index()
            product_features.columns = ['node_id', 'rating_mean', 'rating_std', 'rating_count', 'helpful_mean']
            product_features['node_type'] = 'product'
            product_features['label'] = -1
            
            # Combine node data
            node_data = pd.concat([user_features, product_features], ignore_index=True)
            
            # Create edge data
            edge_data = df[['user_id', 'prod_id', 'rating', 'helpful', 'time']].copy()
            edge_data['relation_type'] = 'user_product'
            edge_data = edge_data.rename(columns={'user_id': 'source', 'prod_id': 'target', 'time': 'timestamp'})
            
            return node_data, edge_data
            
        except Exception as e:
            self.logger.error(f"Error loading Amazon data: {e}")
            raise
    
    def _load_yelp_data(self, data_path: str) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Load Yelp fraud dataset."""
        try:
            # Load main data files
            df = pd.read_csv(f"{data_path}/YelpChi.csv")
            
            # Basic preprocessing
            df = df.dropna()
            
            # Create node features
            node_data = pd.DataFrame()
            
            # User nodes
            user_features = df.groupby('user_id').agg({
                'stars': ['mean', 'std', 'count'],
                'useful': 'sum',
                'cool': 'sum',
                'funny': 'sum'
            }).reset_index()
            user_features.columns = ['node_id', 'stars_mean', 'stars_std', 'review_count', 
                                   'useful_total', 'cool_total', 'funny_total']
            user_features['node_type'] = 'user'
            user_features['label'] = -1
            
            # Business nodes
            business_features = df.groupby('business_id').agg({
                'stars': ['mean', 'std', 'count']
            }).reset_index()
            business_features.columns = ['node_id', 'stars_mean', 'stars_std', 'review_count']
            business_features['node_type'] = 'business'
            business_features['label'] = -1
            
            # Combine node data
            node_data = pd.concat([user_features, business_features], ignore_index=True)
            
            # Create edge data
            edge_data = df[['user_id', 'business_id', 'stars', 'useful', 'date']].copy()
            edge_data['relation_type'] = 'user_business'
            edge_data = edge_data.rename(columns={'user_id': 'source', 'business_id': 'target', 
                                                'date': 'timestamp'})
            
            return node_data, edge_data
            
        except Exception as e:
            self.logger.error(f"Error loading Yelp data: {e}")
            raise
    
    def preprocess_features(self, node_data: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Preprocess node features by type.
        
        Args:
            node_data: DataFrame with node information
            
        Returns:
            Dictionary of processed features by node type
        """
        processed_features = {}
        node_type_mapping = {}
        
        config = self.dataset_configs[self.dataset_name]
        
        for i, node_type in enumerate(config['node_types']):
            # Filter nodes by type
            type_mask = node_data['node_type'] == node_type
            type_data = node_data[type_mask]
            
            if len(type_data) == 0:
                continue
            
            # Extract numeric features (exclude node_id, node_type, label)
            feature_columns = [col for col in type_data.columns 
                             if col not in ['node_id', 'node_type', 'label'] and 
                             pd.api.types.is_numeric_dtype(type_data[col])]
            
            if not feature_columns:
                # Create dummy features if no numeric features
                features = np.ones((len(type_data), 1))
            else:
                features = type_data[feature_columns].values
                
                # Handle missing values
                features = np.nan_to_num(features, nan=0.0)
                
                # Standardize features
                scaler = StandardScaler()
                features = scaler.fit_transform(features)
                self.scalers[node_type] = scaler
            
            processed_features[node_type] = torch.FloatTensor(features)
            
            # Create node mapping
            node_ids = type_data['node_id'].values
            node_mapping = {node_id: idx for idx, node_id in enumerate(node_ids)}
            self.node_mappings[node_type] = node_mapping
            
            # Store type mapping
            node_type_mapping[node_type] = i
        
        return processed_features, node_type_mapping
    
    def create_heterogeneous_graph(self, node_data: pd.DataFrame, 
                                  edge_data: pd.DataFrame) -> Dict:
        """
        Create heterogeneous temporal graph from node and edge data.
        
        Returns:
            Dictionary containing graph data in HAT-MV-GNN format
        """
        config = self.dataset_configs[self.dataset_name]
        
        # Preprocess node features
        node_features, node_type_mapping = self.preprocess_features(node_data)
        
        # Create global node mapping and features
        global_node_mapping = {}
        all_node_features = []
        node_types_list = []
        node_labels = []
        current_idx = 0
        
        for node_type in config['node_types']:
            if node_type not in self.node_mappings:
                continue
                
            type_mapping = self.node_mappings[node_type]
            
            for original_id, local_idx in type_mapping.items():
                global_node_mapping[original_id] = current_idx
                current_idx += 1
                
                # Add node features (pad if necessary)
                features = node_features[node_type][local_idx]
                all_node_features.append(features)
                
                # Add node type
                node_types_list.append(node_type_mapping[node_type])
                
                # Add label (if available)
                node_row = node_data[node_data['node_id'] == original_id].iloc[0]
                node_labels.append(node_row['label'] if node_row['label'] != -1 else -1)
        
        # Pad features to same dimension
        max_feature_dim = max(feat.size(0) for feat in all_node_features)
        padded_features = []
        for feat in all_node_features:
            if feat.size(0) < max_feature_dim:
                padding = torch.zeros(max_feature_dim - feat.size(0))
                feat = torch.cat([feat, padding])
            padded_features.append(feat)
        
        # Create tensors
        node_features_tensor = torch.stack(padded_features)
        node_types_tensor = torch.LongTensor(node_types_list)
        labels_tensor = torch.LongTensor(node_labels)
        
        # Process edges by relation type
        edge_data_dict = {}
        timestamps_dict = {}
        
        for relation_type in config['relation_types']:
            rel_edges = edge_data[edge_data['relation_type'] == relation_type]
            
            if len(rel_edges) == 0:
                continue
            
            # Map edge indices to global node indices
            source_indices = []
            target_indices = []
            edge_features = []
            timestamps = []
            
            for _, row in rel_edges.iterrows():
                if row['source'] in global_node_mapping and row['target'] in global_node_mapping:
                    source_indices.append(global_node_mapping[row['source']])
                    target_indices.append(global_node_mapping[row['target']])
                    
                    # Extract edge features (exclude source, target, relation_type, timestamp)
                    edge_feat_cols = [col for col in rel_edges.columns 
                                    if col not in ['source', 'target', 'relation_type', 'timestamp']]
                    
                    if edge_feat_cols:
                        edge_feat = [row[col] for col in edge_feat_cols]
                        edge_feat = np.nan_to_num(edge_feat, nan=0.0)
                    else:
                        edge_feat = [1.0]  # Dummy feature
                    
                    edge_features.append(edge_feat)
                    
                    # Handle timestamps
                    if 'timestamp' in row and relation_type in config['temporal_relations']:
                        timestamps.append(row['timestamp'])
                    else:
                        timestamps.append(0.0)  # Default timestamp
            
            if source_indices:
                edge_index = torch.LongTensor([source_indices, target_indices])
                edge_attr = torch.FloatTensor(edge_features)
                
                edge_data_dict[relation_type] = (edge_index, edge_attr)
                timestamps_dict[relation_type] = torch.FloatTensor(timestamps)
        
        return {
            'node_features': node_features_tensor,
            'node_types': node_types_tensor,
            'labels': labels_tensor,
            'edge_data': edge_data_dict,
            'timestamps': timestamps_dict,
            'node_type_mapping': node_type_mapping,
            'global_node_mapping': global_node_mapping,
            'config': config
        }
    
    def create_train_test_split(self, graph_data: Dict, 
                               test_size: float = 0.2, val_size: float = 0.1,
                               random_state: int = 42) -> Dict:
        """
        Create train/validation/test splits for node classification.
        
        Args:
            graph_data: Processed graph data
            test_size: Proportion of data for testing
            val_size: Proportion of data for validation
            random_state: Random seed
            
        Returns:
            Updated graph_data with train/val/test masks
        """
        labels = graph_data['labels']
        config = graph_data['config']
        
        # Only split labeled nodes
        labeled_mask = labels != -1
        labeled_indices = torch.where(labeled_mask)[0].numpy()
        labeled_targets = labels[labeled_mask].numpy()
        
        if len(labeled_indices) == 0:
            raise ValueError("No labeled nodes found for splitting")
        
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
        num_nodes = len(labels)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        
        graph_data['train_mask'] = train_mask
        graph_data['val_mask'] = val_mask
        graph_data['test_mask'] = test_mask
        
        self.logger.info(f"Data split - Train: {train_mask.sum()}, "
                        f"Val: {val_mask.sum()}, Test: {test_mask.sum()}")
        
        return graph_data
    
    def process_dataset(self, data_path: str, output_path: Optional[str] = None) -> Dict:
        """
        Complete preprocessing pipeline.
        
        Args:
            data_path: Path to raw dataset
            output_path: Optional path to save processed data
            
        Returns:
            Processed graph data ready for HAT-MV-GNN
        """
        self.logger.info(f"Processing {self.dataset_name} dataset from {data_path}")
        
        # Load raw data
        node_data, edge_data = self.load_data(data_path)
        
        # Create heterogeneous graph
        graph_data = self.create_heterogeneous_graph(node_data, edge_data)
        
        # Create splits
        graph_data = self.create_train_test_split(graph_data)
        
        # Save processed data
        if output_path:
            torch.save(graph_data, output_path)
            self.logger.info(f"Processed data saved to {output_path}")
        
        self.logger.info("Dataset processing completed successfully")
        return graph_data