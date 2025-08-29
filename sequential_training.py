#!/usr/bin/env python3
"""
Memory-efficient sequential training script for HAT-MV-GNN and baseline models.
Trains models one by one to avoid memory issues, with comprehensive logging.
"""

import os
import sys
import yaml
import json
import argparse
import logging
import torch
import numpy as np
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from hat_mv_gnn import HATMVGNN
from trainer import FraudDetectionTrainer, compute_class_weights
from data_processor import FraudDataProcessor
from mat_data_processor import MatDataProcessor
from benchmark_models import create_benchmark_model, BaselineModelWrapper


class SequentialTrainer:
    """Sequential trainer with memory management and comprehensive logging."""
    
    def __init__(self, config_path: str, results_dir: str = "./results/sequential/"):
        self.config = self.load_config(config_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup experiment ID based on timestamp and dataset
        self.experiment_id = f"{self.config['dataset']['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir = self.results_dir / self.experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Results storage
        self.all_results = {
            'experiment_id': self.experiment_id,
            'config': self.config,
            'models': {},
            'summary': {}
        }
    
    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_logging(self):
        """Setup comprehensive logging."""
        log_file = self.experiment_dir / "training.log"
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Setup logger
        self.logger = logging.getLogger('SequentialTrainer')
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Log experiment start
        self.logger.info(f"Starting sequential training experiment: {self.experiment_id}")
        self.logger.info(f"Results will be saved to: {self.experiment_dir}")
    
    def clear_memory(self):
        """Clear GPU and system memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        self.logger.debug("Memory cleared")
    
    def log_memory_usage(self, stage: str):
        """Log current memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3  # GB
            self.logger.info(f"{stage} - GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")
    
    def prepare_data(self):
        """Prepare dataset for training with memory optimization."""
        self.logger.info("Preparing dataset...")
        
        dataset_config = self.config['dataset']
        
        # Use appropriate data processor
        if dataset_config.get('source') == 'mat':
            data_processor = MatDataProcessor(dataset_config['name'])
        else:
            data_processor = FraudDataProcessor(dataset_config['name'])
        
        processed_path = dataset_config['processed_path']
        
        # Load or process data
        if os.path.exists(processed_path):
            self.logger.info(f"Loading processed data from {processed_path}")
            graph_data = torch.load(processed_path, map_location='cpu')
        else:
            self.logger.info("Processing raw data...")
            os.makedirs(os.path.dirname(processed_path), exist_ok=True)
            graph_data = data_processor.process_dataset(
                dataset_config['data_path'], 
                processed_path
            )
        
        # Log dataset statistics
        self.logger.info("Dataset Statistics:")
        self.logger.info(f"  Nodes: {graph_data['node_features'].size(0):,}")
        self.logger.info(f"  Features: {graph_data['node_features'].size(1)}")
        self.logger.info(f"  Relations: {list(graph_data['edge_data'].keys())}")
        self.logger.info(f"  Train nodes: {graph_data['train_mask'].sum():,}")
        self.logger.info(f"  Val nodes: {graph_data['val_mask'].sum():,}")
        self.logger.info(f"  Test nodes: {graph_data['test_mask'].sum():,}")
        
        # Class distribution
        train_labels = graph_data['labels'][graph_data['train_mask']]
        unique_labels, counts = torch.unique(train_labels, return_counts=True)
        class_dist = dict(zip(unique_labels.tolist(), counts.tolist()))
        self.logger.info(f"  Class distribution: {class_dist}")
        
        return graph_data
    
    def train_proposed_model(self, graph_data: dict) -> dict:
        """Train the proposed HAT-MV-GNN model."""
        self.logger.info("=" * 60)
        self.logger.info("TRAINING PROPOSED HAT-MV-GNN MODEL")
        self.logger.info("=" * 60)
        
        self.clear_memory()
        self.log_memory_usage("Before HAT-MV-GNN training")
        
        # Create model
        model_config = self.config['model']
        dataset_config = graph_data['config']
        
        # Determine input dimensions (only for actually present node types)
        input_dims = {}
        node_features = graph_data['node_features']
        node_types = graph_data['node_types']
        
        unique_types = torch.unique(node_types)
        actual_node_types = []
        
        for i, node_type in enumerate(dataset_config['node_types']):
            if i in unique_types:
                type_mask = node_types == i
                if type_mask.sum() > 0:
                    input_dims[node_type] = node_features.size(1)
                    actual_node_types.append(node_type)
        
        # If no types found, default to treating all as first type
        if not input_dims:
            input_dims[dataset_config['node_types'][0]] = node_features.size(1)
            actual_node_types = [dataset_config['node_types'][0]]
        
        self.logger.info(f"Active node types: {actual_node_types}")
        self.logger.info(f"Input dimensions: {input_dims}")
        
        # Determine edge dimensions
        edge_dims = {}
        for rel_type, (edge_index, edge_attr) in graph_data['edge_data'].items():
            edge_dims[rel_type] = edge_attr.size(1)
        
        # Create HAT-MV-GNN model (use only active node types)
        model = HATMVGNN(
            node_types=actual_node_types,
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
        
        # Log model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"HAT-MV-GNN parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Setup device
        device = torch.device(self.config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {device}")
        
        # Compute class weights
        class_weights = compute_class_weights(graph_data['labels'], graph_data['train_mask'])
        self.logger.info(f"Class weights: {class_weights}")
        
        # Create trainer
        training_config = self.config['training']
        trainer = FraudDetectionTrainer(
            model=model,
            device=device,
            learning_rate=training_config['learning_rate'],
            weight_decay=training_config['weight_decay'],
            scheduler_step=training_config['scheduler_step'],
            scheduler_gamma=training_config['scheduler_gamma']
        )
        
        # Train model
        self.logger.info("Starting HAT-MV-GNN training...")
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
        self.logger.info("Evaluating HAT-MV-GNN on test set...")
        test_metrics = trainer.test(
            node_features=graph_data['node_features'],
            node_types=graph_data['node_types'],
            edge_data=graph_data['edge_data'],
            timestamps=graph_data['timestamps'],
            labels=graph_data['labels'],
            test_mask=graph_data['test_mask']
        )
        
        # Log results
        self.logger.info("HAT-MV-GNN Test Results:")
        for metric, value in test_metrics.items():
            self.logger.info(f"  {metric.capitalize()}: {value:.4f}")
        
        # Save model
        model_path = self.experiment_dir / "hat_mv_gnn_model.pt"
        trainer.save_model(str(model_path))
        
        # Store results
        results = {
            'model_type': 'HAT-MV-GNN',
            'test_metrics': test_metrics,
            'training_history': {
                'best_val_f1': training_history['best_val_f1'],
                'final_train_loss': training_history['train_losses'][-1] if training_history['train_losses'] else None,
                'final_val_loss': training_history['val_losses'][-1] if training_history['val_losses'] else None
            },
            'model_params': {
                'total_params': total_params,
                'trainable_params': trainable_params
            }
        }
        
        self.log_memory_usage("After HAT-MV-GNN training")
        self.clear_memory()
        
        return results
    
    def train_benchmark_models(self, graph_data: dict) -> dict:
        """Train all benchmark models sequentially."""
        self.logger.info("=" * 60)
        self.logger.info("TRAINING BENCHMARK MODELS")
        self.logger.info("=" * 60)
        
        benchmark_models = ['mlp', 'gcn', 'gat', 'sage']
        benchmark_results = {}
        
        for model_name in benchmark_models:
            self.logger.info(f"\n{'-' * 40}")
            self.logger.info(f"Training {model_name.upper()} baseline")
            self.logger.info(f"{'-' * 40}")
            
            self.clear_memory()
            self.log_memory_usage(f"Before {model_name} training")
            
            try:
                # Create model
                input_dim = graph_data['node_features'].size(1)
                model = create_benchmark_model(
                    model_name=model_name,
                    input_dim=input_dim,
                    hidden_dim=64,
                    num_layers=2,
                    dropout=0.1,
                    num_classes=2
                )
                
                # Wrap model
                wrapped_model = BaselineModelWrapper(model, model_name)
                
                # Log model parameters
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                self.logger.info(f"{model_name.upper()} parameters: {total_params:,} total, {trainable_params:,} trainable")
                
                # Setup device and move model
                device = torch.device('cpu')  # Use CPU to avoid memory issues
                wrapped_model = wrapped_model.to(device)
                
                # Simple training loop for benchmarks
                results = self.train_baseline_model(
                    wrapped_model, graph_data, device, model_name
                )
                
                benchmark_results[model_name] = results
                
                self.logger.info(f"{model_name.upper()} Test Results:")
                for metric, value in results['test_metrics'].items():
                    self.logger.info(f"  {metric.capitalize()}: {value:.4f}")
                
                self.log_memory_usage(f"After {model_name} training")
                
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {str(e)}")
                benchmark_results[model_name] = {
                    'error': str(e),
                    'test_metrics': {'f1': 0.0, 'accuracy': 0.0, 'auc': 0.5}
                }
            
            self.clear_memory()
        
        return benchmark_results
    
    def train_baseline_model(self, model, graph_data: dict, device: torch.device, model_name: str) -> dict:
        """Train a single baseline model."""
        from torch.optim import Adam
        from torch.optim.lr_scheduler import StepLR
        from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
        import torch.nn.functional as F
        from tqdm import tqdm
        
        # Move data to device
        node_features = graph_data['node_features'].to(device)
        labels = graph_data['labels'].to(device)
        train_mask = graph_data['train_mask'].to(device)
        val_mask = graph_data['val_mask'].to(device)
        test_mask = graph_data['test_mask'].to(device)
        
        # Setup optimizer
        optimizer = Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.9)
        
        # Training parameters
        num_epochs = 100  # Reduced for benchmarks
        early_stopping_patience = 15
        eval_every = 5
        
        best_val_f1 = 0.0
        patience_counter = 0
        
        # Convert edge data format for homogeneous models
        if model_name != 'hetero':
            from benchmark_models import create_homogeneous_graph
            combined_edges = create_homogeneous_graph(graph_data['edge_data'], node_features.size(0))
            combined_edges = combined_edges.to(device)
        
        # Training loop
        for epoch in tqdm(range(num_epochs), desc=f"Training {model_name}"):
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            if model_name == 'mlp':
                predictions = model.forward(node_features)
            else:
                predictions = model.forward(node_features, combined_edges)
            
            # Compute loss
            train_predictions = predictions[train_mask]
            train_labels = labels[train_mask]
            loss = F.cross_entropy(train_predictions, train_labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Validation
            if epoch % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    if model_name == 'mlp':
                        val_predictions = model.forward(node_features)
                    else:
                        val_predictions = model.forward(node_features, combined_edges)
                    
                    val_pred_labels = val_predictions[val_mask].argmax(dim=1)
                    val_true_labels = labels[val_mask]
                    
                    if val_mask.sum() > 0:
                        val_f1 = f1_score(
                            val_true_labels.cpu().numpy(),
                            val_pred_labels.cpu().numpy(),
                            average='binary',
                            zero_division=0
                        )
                        
                        if val_f1 > best_val_f1:
                            best_val_f1 = val_f1
                            patience_counter = 0
                        else:
                            patience_counter += eval_every
                        
                        if patience_counter >= early_stopping_patience:
                            self.logger.info(f"Early stopping for {model_name} at epoch {epoch}")
                            break
        
        # Test evaluation
        model.eval()
        with torch.no_grad():
            if model_name == 'mlp':
                test_predictions = model.forward(node_features)
            else:
                test_predictions = model.forward(node_features, combined_edges)
            
            test_pred_labels = test_predictions[test_mask].argmax(dim=1)
            test_true_labels = labels[test_mask]
            test_pred_probs = F.softmax(test_predictions[test_mask], dim=1)[:, 1]
            
            # Compute metrics
            test_accuracy = accuracy_score(
                test_true_labels.cpu().numpy(),
                test_pred_labels.cpu().numpy()
            )
            test_f1 = f1_score(
                test_true_labels.cpu().numpy(),
                test_pred_labels.cpu().numpy(),
                average='binary',
                zero_division=0
            )
            
            try:
                test_auc = roc_auc_score(
                    test_true_labels.cpu().numpy(),
                    test_pred_probs.cpu().numpy()
                )
            except:
                test_auc = 0.5
            
            test_metrics = {
                'accuracy': test_accuracy,
                'f1': test_f1,
                'auc': test_auc
            }
        
        return {
            'model_type': model_name,
            'test_metrics': test_metrics,
            'best_val_f1': best_val_f1,
            'model_params': {
                'total_params': sum(p.numel() for p in model.parameters()),
                'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
            }
        }
    
    def save_results(self):
        """Save comprehensive results."""
        # Save detailed results as JSON
        results_file = self.experiment_dir / "complete_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.all_results, f, indent=2, default=str)
        
        # Save summary CSV for easy analysis
        summary_file = self.experiment_dir / "results_summary.csv"
        import pandas as pd
        
        rows = []
        for model_name, results in self.all_results['models'].items():
            if 'test_metrics' in results:
                row = {
                    'Model': model_name,
                    'Dataset': self.config['dataset']['name'],
                    'Accuracy': results['test_metrics'].get('accuracy', 0),
                    'F1': results['test_metrics'].get('f1', 0),
                    'AUC': results['test_metrics'].get('auc', 0.5),
                    'Parameters': results.get('model_params', {}).get('total_params', 0)
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(summary_file, index=False)
        
        self.logger.info(f"Results saved to {self.experiment_dir}")
        self.logger.info("\n" + "="*60)
        self.logger.info("EXPERIMENT SUMMARY")
        self.logger.info("="*60)
        print(df.to_string(index=False))
    
    def run_complete_training(self):
        """Run the complete sequential training pipeline."""
        try:
            # Prepare data
            graph_data = self.prepare_data()
            
            # Train proposed model
            proposed_results = self.train_proposed_model(graph_data)
            self.all_results['models']['HAT-MV-GNN'] = proposed_results
            
            # Train benchmark models
            benchmark_results = self.train_benchmark_models(graph_data)
            self.all_results['models'].update(benchmark_results)
            
            # Create summary
            self.all_results['summary'] = {
                'best_model': max(
                    self.all_results['models'].items(),
                    key=lambda x: x[1].get('test_metrics', {}).get('f1', 0)
                )[0],
                'dataset': self.config['dataset']['name'],
                'total_models_trained': len(self.all_results['models'])
            }
            
            # Save results
            self.save_results()
            
            self.logger.info("Sequential training completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Sequential training for HAT-MV-GNN and baselines')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--results-dir', type=str, default='./results/sequential/',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create trainer and run
    trainer = SequentialTrainer(args.config, args.results_dir)
    trainer.run_complete_training()


if __name__ == '__main__':
    main()