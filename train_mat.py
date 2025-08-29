#!/usr/bin/env python3
"""
Training script for HAT-MV-GNN with .mat datasets.
Handles amazon.mat and yelpChi.mat files directly.
"""

import os
import sys
import yaml
import argparse
import logging
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from hat_mv_gnn import HATMVGNN
from trainer import FraudDetectionTrainer, compute_class_weights
from mat_data_processor import load_mat_dataset


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('mat_training.log')
        ]
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def prepare_mat_data(config: dict, force_reprocess: bool = False):
    """Prepare .mat dataset for training."""
    dataset_config = config['dataset']
    processed_path = dataset_config['processed_path']
    
    # Check if processed data exists
    if os.path.exists(processed_path) and not force_reprocess:
        logging.info(f"Loading processed data from {processed_path}")
        graph_data = torch.load(processed_path)
    else:
        # Process .mat data
        logging.info(f"Processing {dataset_config['name']} .mat dataset")
        graph_data = load_mat_dataset(
            dataset_config['name'],
            dataset_config['data_path'],
            processed_path
        )
    
    return graph_data


def create_model_for_mat_data(config: dict, graph_data: dict) -> HATMVGNN:
    """Create HAT-MV-GNN model for .mat data."""
    model_config = config['model']
    dataset_config = graph_data['config']
    
    # Get feature dimension from data
    node_features = graph_data['node_features']
    feature_dim = node_features.size(1)
    
    # Input dimensions for each node type
    input_dims = {node_type: feature_dim for node_type in dataset_config['node_types']}
    
    # Edge dimensions for each relation type
    edge_dims = {}
    for rel_type, (edge_index, edge_attr) in graph_data['edge_data'].items():
        edge_dims[rel_type] = edge_attr.size(1)
    
    # Create model
    model = HATMVGNN(
        node_types=dataset_config['node_types'],
        input_dims=input_dims,
        relation_types=dataset_config['relation_types'],
        edge_dims=edge_dims,
        hidden_dim=model_config['hidden_dim'],
        num_layers=model_config['num_layers'],
        temporal_dim=model_config['temporal_dim'],
        num_heads=model_config['num_heads'],
        dropout=model_config['dropout'],
        top_k=model_config['top_k'],
        num_classes=model_config['num_classes']
    )
    
    return model


def validate_data_files(config: dict):
    """Validate that required .mat files exist."""
    dataset_config = config['dataset']
    dataset_name = dataset_config['name']
    data_path = dataset_config['data_path']
    
    expected_file = f"{dataset_name}.mat"
    if dataset_name == 'yelp':
        expected_file = "yelpChi.mat"
    
    mat_file_path = os.path.join(data_path, expected_file)
    
    if not os.path.exists(mat_file_path):
        raise FileNotFoundError(
            f"Dataset file not found: {mat_file_path}\n"
            f"Please ensure the {expected_file} file is in {data_path}"
        )
    
    return mat_file_path


def main():
    parser = argparse.ArgumentParser(description='Train HAT-MV-GNN on .mat datasets')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--force-reprocess', action='store_true',
                       help='Force reprocessing of data even if processed version exists')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu), overrides config')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging for data processing')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    log_level = "DEBUG" if args.debug else config.get('log_level', 'INFO')
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    # Set device
    device = args.device or config.get('device', 'cuda')
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set random seed
    set_seed(args.seed)
    
    # Create checkpoint directory
    checkpoint_dir = config.get('checkpoint_dir', './experiments/mat/')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    logger.info("Starting HAT-MV-GNN training on .mat dataset")
    
    # Validate data files exist
    try:
        mat_file_path = validate_data_files(config)
        logger.info(f"Found dataset file: {mat_file_path}")
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    
    # Prepare data
    logger.info("Loading and processing .mat dataset...")
    try:
        graph_data = prepare_mat_data(config, args.force_reprocess)
    except Exception as e:
        logger.error(f"Error processing .mat data: {e}")
        logger.info("Common issues:")
        logger.info("1. Make sure scipy is installed: pip install scipy")
        logger.info("2. Check if .mat file is corrupted")
        logger.info("3. Verify file permissions")
        if args.debug:
            import traceback
            traceback.print_exc()
        return
    
    # Log dataset statistics
    dataset_name = config['dataset']['name']
    logger.info(f"{dataset_name.capitalize()} dataset statistics:")
    logger.info(f"  Nodes: {graph_data['node_features'].size(0)}")
    logger.info(f"  Features per node: {graph_data['node_features'].size(1)}")
    logger.info(f"  Relations: {list(graph_data['edge_data'].keys())}")
    
    total_edges = sum(edge_data[0].size(1) for edge_data in graph_data['edge_data'].values())
    logger.info(f"  Total edges: {total_edges}")
    
    for rel_type, (edge_index, _) in graph_data['edge_data'].items():
        logger.info(f"    {rel_type}: {edge_index.size(1)} edges")
    
    logger.info(f"  Train nodes: {graph_data['train_mask'].sum()}")
    logger.info(f"  Val nodes: {graph_data['val_mask'].sum()}")
    logger.info(f"  Test nodes: {graph_data['test_mask'].sum()}")
    
    # Check class distribution
    train_labels = graph_data['labels'][graph_data['train_mask']]
    unique_labels, counts = torch.unique(train_labels, return_counts=True)
    class_distribution = dict(zip(unique_labels.tolist(), counts.tolist()))
    logger.info(f"  Class distribution in training set: {class_distribution}")
    
    if len(unique_labels) == 2:
        imbalance_ratio = max(counts) / min(counts)
        logger.info(f"  Class imbalance ratio: {imbalance_ratio:.2f}:1")
    
    # Create model
    logger.info("Creating HAT-MV-GNN model...")
    try:
        model = create_model_for_mat_data(config, graph_data)
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return
    
    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Compute class weights if configured
    class_weights = None
    if config.get('data_processing', {}).get('use_class_weights', True):
        class_weights = compute_class_weights(graph_data['labels'], graph_data['train_mask'])
        logger.info(f"Using class weights: {class_weights}")
    
    # Create trainer
    training_config = config['training']
    trainer = FraudDetectionTrainer(
        model=model,
        device=device,
        learning_rate=float(training_config['learning_rate']),
        weight_decay=float(training_config['weight_decay']),
        scheduler_step=int(training_config['scheduler_step']),
        scheduler_gamma=float(training_config['scheduler_gamma'])
    )
    
    # Train model
    logger.info("Starting training...")
    try:
        training_history = trainer.train(
            node_features=graph_data['node_features'],
            node_types=graph_data['node_types'],
            edge_data=graph_data['edge_data'],
            timestamps=graph_data['timestamps'],
            labels=graph_data['labels'],
            train_mask=graph_data['train_mask'],
            val_mask=graph_data['val_mask'],
            num_epochs=training_config['num_epochs'],
            early_stopping_patience=training_config['early_stopping_patience'],
            class_weights=class_weights,
            eval_every=training_config['eval_every']
        )
    except Exception as e:
        logger.error(f"Error during training: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return
    
    # Test model
    logger.info("Evaluating on test set...")
    try:
        test_metrics = trainer.test(
            node_features=graph_data['node_features'],
            node_types=graph_data['node_types'],
            edge_data=graph_data['edge_data'],
            timestamps=graph_data['timestamps'],
            labels=graph_data['labels'],
            test_mask=graph_data['test_mask']
        )
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return
    
    # Save model and results
    if config.get('save_best_model', True):
        model_path = os.path.join(checkpoint_dir, 'best_model.pt')
        trainer.save_model(model_path)
        
        # Save comprehensive results
        results_path = os.path.join(checkpoint_dir, 'test_results.yaml')
        results = {
            'dataset': dataset_name,
            'source': 'mat',
            'test_metrics': test_metrics,
            'training_history': {
                'best_val_f1': training_history['best_val_f1'],
                'final_train_loss': training_history['train_losses'][-1] if training_history['train_losses'] else None,
                'final_val_loss': training_history['val_losses'][-1] if training_history['val_losses'] else None,
                'total_epochs': len(training_history['train_losses'])
            },
            'dataset_stats': {
                'num_nodes': graph_data['node_features'].size(0),
                'num_features': graph_data['node_features'].size(1),
                'num_relations': len(graph_data['edge_data']),
                'total_edges': total_edges,
                'class_distribution': class_distribution,
                'imbalance_ratio': imbalance_ratio if len(unique_labels) == 2 else None
            },
            'model_stats': {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params
            },
            'config': config
        }
        
        with open(results_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        
        logger.info(f"Model and results saved to {checkpoint_dir}")
    
    logger.info("Training completed successfully!")
    logger.info(f"Final test results on {dataset_name} dataset:")
    for metric, value in test_metrics.items():
        logger.info(f"  {metric.capitalize()}: {value:.4f}")


if __name__ == '__main__':
    main()