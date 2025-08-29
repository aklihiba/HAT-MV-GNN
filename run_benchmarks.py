#!/usr/bin/env python3
"""
Comprehensive benchmark runner for comparing HAT-MV-GNN against baseline models.
Runs multiple models on the same dataset and compares results.
"""

import os
import sys
import yaml
import argparse
import logging
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import time
from typing import Dict, List

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from hat_mv_gnn import HATMVGNN
from benchmark_models import create_benchmark_model, BaselineModelWrapper
from trainer import FraudDetectionTrainer, compute_class_weights
from mat_data_processor import load_mat_dataset


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('benchmark_results.log')
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
    """Prepare dataset for benchmarking."""
    dataset_config = config['dataset']
    processed_path = dataset_config['processed_path']
    
    # Check if processed data exists
    if os.path.exists(processed_path) and not force_reprocess:
        logging.info(f"Loading processed data from {processed_path}")
        graph_data = torch.load(processed_path)
    else:
        # Process data
        logging.info(f"Processing {dataset_config['name']} dataset")
        graph_data = load_mat_dataset(
            dataset_config['name'],
            dataset_config['data_path'],
            processed_path
        )
    
    return graph_data


def create_hat_mv_gnn_model(config: dict, graph_data: dict) -> HATMVGNN:
    """Create HAT-MV-GNN model."""
    model_config = config['model']
    dataset_config = graph_data['config']
    
    # Get dimensions
    node_features = graph_data['node_features']
    feature_dim = node_features.size(1)
    
    input_dims = {node_type: feature_dim for node_type in dataset_config['node_types']}
    
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


def create_baseline_models(config: dict, graph_data: dict) -> Dict[str, BaselineModelWrapper]:
    """Create all baseline models for comparison."""
    model_config = config['model']
    node_features = graph_data['node_features']
    feature_dim = node_features.size(1)
    
    # Common parameters for all models
    common_params = {
        'hidden_dim': model_config['hidden_dim'],
        'num_layers': min(model_config['num_layers'], 3),  # Limit baseline layers
        'dropout': model_config['dropout'],
        'num_classes': model_config['num_classes']
    }
    
    models = {}
    
    # GCN baseline
    models['GCN'] = BaselineModelWrapper(
        create_benchmark_model('gcn', feature_dim, **common_params),
        'gcn'
    )
    
    # GAT baseline
    models['GAT'] = BaselineModelWrapper(
        create_benchmark_model('gat', feature_dim, 
                             num_heads=model_config.get('num_heads', 4), **common_params),
        'gat'
    )
    
    # GraphSAGE baseline
    models['GraphSAGE'] = BaselineModelWrapper(
        create_benchmark_model('sage', feature_dim, **common_params),
        'sage'
    )
    
    # MLP baseline (no graph structure)
    models['MLP'] = BaselineModelWrapper(
        create_benchmark_model('mlp', feature_dim, **common_params),
        'mlp'
    )
    
    return models


def train_model(model, model_name: str, graph_data: dict, config: dict, 
                device: torch.device, class_weights: torch.Tensor = None) -> Dict:
    """Train a single model and return results."""
    logger = logging.getLogger(__name__)
    logger.info(f"Training {model_name}...")
    
    # Create trainer
    training_config = config['training']
    
    if model_name == 'HAT-MV-GNN':
        # Use full trainer for HAT-MV-GNN
        trainer = FraudDetectionTrainer(
            model=model,
            device=device,
            learning_rate=training_config['learning_rate'],
            weight_decay=training_config['weight_decay'],
            scheduler_step=training_config['scheduler_step'],
            scheduler_gamma=training_config['scheduler_gamma']
        )
        
        start_time = time.time()
        
        # Train model
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
        
        training_time = time.time() - start_time
        
        # Test model
        test_metrics = trainer.test(
            node_features=graph_data['node_features'],
            node_types=graph_data['node_types'],
            edge_data=graph_data['edge_data'],
            timestamps=graph_data['timestamps'],
            labels=graph_data['labels'],
            test_mask=graph_data['test_mask']
        )
        
    else:
        # Use simplified trainer for baselines
        trainer = FraudDetectionTrainer(
            model=model,
            device=device,
            learning_rate=training_config['learning_rate'] * 2,  # Higher LR for simpler models
            weight_decay=training_config['weight_decay']
        )
        
        start_time = time.time()
        
        # Create dummy timestamps for baselines
        dummy_timestamps = {rel_type: torch.zeros(edge_data[0].size(1)) 
                           for rel_type, edge_data in graph_data['edge_data'].items()}
        
        # Train with reduced epochs for baselines
        training_history = trainer.train(
            node_features=graph_data['node_features'],
            node_types=graph_data['node_types'],
            edge_data=graph_data['edge_data'],
            timestamps=dummy_timestamps,
            labels=graph_data['labels'],
            train_mask=graph_data['train_mask'],
            val_mask=graph_data['val_mask'],
            num_epochs=min(training_config['num_epochs'], 100),  # Fewer epochs
            early_stopping_patience=training_config['early_stopping_patience'] // 2,
            class_weights=class_weights,
            eval_every=training_config['eval_every']
        )
        
        training_time = time.time() - start_time
        
        # Test model
        test_metrics = trainer.test(
            node_features=graph_data['node_features'],
            node_types=graph_data['node_types'],
            edge_data=graph_data['edge_data'],
            timestamps=dummy_timestamps,
            labels=graph_data['labels'],
            test_mask=graph_data['test_mask']
        )
    
    # Add training time to metrics
    test_metrics['training_time'] = training_time
    test_metrics['best_val_f1'] = training_history['best_val_f1']
    test_metrics['total_epochs'] = len(training_history['train_losses'])
    
    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters())
    test_metrics['total_parameters'] = total_params
    
    logger.info(f"{model_name} - Test F1: {test_metrics['f1']:.4f}, "
               f"AUC: {test_metrics['auc']:.4f}, Time: {training_time:.2f}s")
    
    return test_metrics


def run_benchmark_suite(config: dict, graph_data: dict, device: torch.device,
                       models_to_run: List[str] = None) -> pd.DataFrame:
    """Run complete benchmark suite."""
    logger = logging.getLogger(__name__)
    
    # Compute class weights
    class_weights = None
    if config.get('data_processing', {}).get('use_class_weights', True):
        class_weights = compute_class_weights(graph_data['labels'], graph_data['train_mask'])
    
    # Create models
    baseline_models = create_baseline_models(config, graph_data)
    hat_mv_gnn = create_hat_mv_gnn_model(config, graph_data)
    
    all_models = {'HAT-MV-GNN': hat_mv_gnn, **baseline_models}
    
    # Filter models if specified
    if models_to_run:
        all_models = {k: v for k, v in all_models.items() if k in models_to_run}
    
    # Run benchmarks
    results = {}
    
    for model_name, model in all_models.items():
        try:
            # Move model to device
            if hasattr(model, 'to'):
                model = model.to(device)
            
            # Train and evaluate
            metrics = train_model(model, model_name, graph_data, config, device, class_weights)
            results[model_name] = metrics
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            # Add placeholder results for failed model
            results[model_name] = {
                'accuracy': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0,
                'auc': 0.5, 'training_time': 0.0, 'best_val_f1': 0.0,
                'total_epochs': 0, 'total_parameters': 0, 'status': 'failed'
            }
    
    # Convert to DataFrame
    results_df = pd.DataFrame.from_dict(results, orient='index')
    
    # Round numeric columns
    numeric_cols = ['accuracy', 'f1', 'precision', 'recall', 'auc', 'best_val_f1']
    results_df[numeric_cols] = results_df[numeric_cols].round(4)
    results_df['training_time'] = results_df['training_time'].round(2)
    
    return results_df


def save_benchmark_results(results_df: pd.DataFrame, config: dict, output_dir: str):
    """Save benchmark results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'benchmark_results.csv')
    results_df.to_csv(csv_path, index=True)
    
    # Save detailed YAML report
    yaml_path = os.path.join(output_dir, 'benchmark_report.yaml')
    report = {
        'dataset': config['dataset']['name'],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': config,
        'results': results_df.to_dict('index')
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(report, f, default_flow_style=False)
    
    logging.info(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Run HAT-MV-GNN benchmark comparison')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--force-reprocess', action='store_true',
                       help='Force reprocessing of data')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--models', type=str, nargs='+', 
                       default=['HAT-MV-GNN', 'GCN', 'GAT', 'GraphSAGE', 'MLP'],
                       help='Models to benchmark')
    parser.add_argument('--output-dir', type=str, default='./benchmark_results/',
                       help='Output directory for results')
    
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
    
    logger.info("Starting HAT-MV-GNN benchmark comparison")
    logger.info(f"Models to benchmark: {args.models}")
    
    # Prepare data
    logger.info("Preparing dataset...")
    graph_data = prepare_data(config, args.force_reprocess)
    
    # Log dataset info
    dataset_name = config['dataset']['name']
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"  Nodes: {graph_data['node_features'].size(0)}")
    logger.info(f"  Features: {graph_data['node_features'].size(1)}")
    logger.info(f"  Relations: {list(graph_data['edge_data'].keys())}")
    logger.info(f"  Train/Val/Test: {graph_data['train_mask'].sum()}/{graph_data['val_mask'].sum()}/{graph_data['test_mask'].sum()}")
    
    # Run benchmarks
    logger.info("Running benchmark suite...")
    start_time = time.time()
    
    results_df = run_benchmark_suite(config, graph_data, device, args.models)
    
    total_time = time.time() - start_time
    
    # Display results
    logger.info(f"\n{'='*80}")
    logger.info(f"BENCHMARK RESULTS - {dataset_name.upper()} DATASET")
    logger.info(f"{'='*80}")
    
    # Sort by F1 score
    results_sorted = results_df.sort_values('f1', ascending=False)
    
    print("\nRanking by F1 Score:")
    print("-" * 80)
    print(f"{'Rank':<5} {'Model':<15} {'F1':<8} {'AUC':<8} {'Precision':<10} {'Recall':<8} {'Time(s)':<10}")
    print("-" * 80)
    
    for rank, (model_name, row) in enumerate(results_sorted.iterrows(), 1):
        print(f"{rank:<5} {model_name:<15} {row['f1']:<8.4f} {row['auc']:<8.4f} "
              f"{row['precision']:<10.4f} {row['recall']:<8.4f} {row['training_time']:<10.2f}")
    
    print("\nDetailed Results:")
    print(results_df.to_string())
    
    # Save results
    save_benchmark_results(results_df, config, args.output_dir)
    
    logger.info(f"\nBenchmark completed in {total_time:.2f} seconds")
    logger.info(f"Best model: {results_sorted.index[0]} (F1: {results_sorted.iloc[0]['f1']:.4f})")


if __name__ == '__main__':
    main()