#!/usr/bin/env python3
"""
Training script for HAT-MV-GNN fraud detection model.
Implements the complete training pipeline from your research paper.
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
from data_processor import FraudDataProcessor


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
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


def prepare_data(config: dict, force_reprocess: bool = False):
    """Prepare dataset for training."""
    dataset_config = config['dataset']
    data_processor = FraudDataProcessor(dataset_config['name'])
    
    processed_path = dataset_config['processed_path']
    
    # Check if processed data exists
    if os.path.exists(processed_path) and not force_reprocess:
        logging.info(f"Loading processed data from {processed_path}")
        graph_data = torch.load(processed_path)
    else:
        # Process raw data
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        graph_data = data_processor.process_dataset(
            dataset_config['data_path'], 
            processed_path
        )
    
    return graph_data


def create_model(config: dict, graph_data: dict) -> HATMVGNN:
    """Create HAT-MV-GNN model based on configuration."""
    model_config = config['model']
    dataset_config = graph_data['config']
    
    # Determine input dimensions for each node type
    input_dims = {}
    node_features = graph_data['node_features']
    node_types = graph_data['node_types']
    
    for i, node_type in enumerate(dataset_config['node_types']):
        type_mask = node_types == i
        if type_mask.sum() > 0:
            input_dims[node_type] = node_features.size(1)
    
    # Determine edge dimensions for each relation type
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


def main():
    parser = argparse.ArgumentParser(description='Train HAT-MV-GNN for fraud detection')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--force-reprocess', action='store_true',
                       help='Force reprocessing of data even if processed version exists')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu), overrides config')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(config.get('log_level', 'INFO'))
    logger = logging.getLogger(__name__)
    
    # Set device
    device = args.device or config.get('device', 'cuda')
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set random seed
    set_seed(args.seed)
    
    # Create checkpoint directory
    checkpoint_dir = config.get('checkpoint_dir', './experiments/')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    logger.info("Starting HAT-MV-GNN training pipeline")
    
    # Prepare data
    logger.info("Preparing dataset...")
    graph_data = prepare_data(config, args.force_reprocess)
    
    # Log dataset statistics
    logger.info(f"Dataset statistics:")
    logger.info(f"  Nodes: {graph_data['node_features'].size(0)}")
    logger.info(f"  Features: {graph_data['node_features'].size(1)}")
    logger.info(f"  Relations: {list(graph_data['edge_data'].keys())}")
    logger.info(f"  Train nodes: {graph_data['train_mask'].sum()}")
    logger.info(f"  Val nodes: {graph_data['val_mask'].sum()}")
    logger.info(f"  Test nodes: {graph_data['test_mask'].sum()}")
    
    # Check for class imbalance
    train_labels = graph_data['labels'][graph_data['train_mask']]
    unique_labels, counts = torch.unique(train_labels, return_counts=True)
    logger.info(f"  Class distribution: {dict(zip(unique_labels.tolist(), counts.tolist()))}")
    
    # Create model
    logger.info("Creating HAT-MV-GNN model...")
    model = create_model(config, graph_data)
    
    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Compute class weights for imbalanced data
    class_weights = compute_class_weights(graph_data['labels'], graph_data['train_mask'])
    logger.info(f"Class weights: {class_weights}")
    
    # Create trainer
    training_config = config['training']
    trainer = FraudDetectionTrainer(
        model=model,
        device=device,
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        scheduler_step=training_config['scheduler_step'],
        scheduler_gamma=training_config['scheduler_gamma']
    )
    
    # Train model
    logger.info("Starting training...")
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
    
    # Test model
    logger.info("Evaluating on test set...")
    test_metrics = trainer.test(
        node_features=graph_data['node_features'],
        node_types=graph_data['node_types'],
        edge_data=graph_data['edge_data'],
        timestamps=graph_data['timestamps'],
        labels=graph_data['labels'],
        test_mask=graph_data['test_mask']
    )
    
    # Save model and results
    if config.get('save_best_model', True):
        model_path = os.path.join(checkpoint_dir, 'best_model.pt')
        trainer.save_model(model_path)
        
        # Save test results
        results_path = os.path.join(checkpoint_dir, 'test_results.yaml')
        with open(results_path, 'w') as f:
            yaml.dump({
                'test_metrics': test_metrics,
                'training_history': {
                    'best_val_f1': training_history['best_val_f1'],
                    'final_train_loss': training_history['train_losses'][-1] if training_history['train_losses'] else None,
                    'final_val_loss': training_history['val_losses'][-1] if training_history['val_losses'] else None
                },
                'config': config
            }, f, default_flow_style=False)
        
        logger.info(f"Model and results saved to {checkpoint_dir}")
    
    logger.info("Training completed successfully!")
    logger.info(f"Final test results:")
    for metric, value in test_metrics.items():
        logger.info(f"  {metric.capitalize()}: {value:.4f}")


if __name__ == '__main__':
    main()