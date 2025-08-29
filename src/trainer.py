import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm
import logging
from hat_mv_gnn import HATMVGNN


class FraudDetectionTrainer:
    """
    Training framework for HAT-MV-GNN fraud detection model.
    Implements the training procedure from Algorithm 1.
    """
    
    def __init__(self, model: HATMVGNN, device: torch.device,
                 learning_rate: float = 0.001, weight_decay: float = 5e-4,
                 scheduler_step: int = 50, scheduler_gamma: float = 0.9):
        self.model = model.to(device)
        self.device = device
        
        # Optimizer and scheduler
        self.optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, node_features: torch.Tensor, node_types: torch.Tensor,
                   edge_data: Dict[str, Tuple], timestamps: Dict[str, torch.Tensor],
                   labels: torch.Tensor, train_mask: torch.Tensor,
                   class_weights: Optional[torch.Tensor] = None) -> float:
        """
        Train for one epoch following Algorithm 1.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        predictions = self.model(node_features, node_types, edge_data, timestamps, labels)
        
        # Compute loss (Eq. 3.11)
        loss = self.model.compute_loss(predictions, labels, train_mask, class_weights)
        
        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, node_features: torch.Tensor, node_types: torch.Tensor,
                edge_data: Dict[str, Tuple], timestamps: Dict[str, torch.Tensor],
                labels: torch.Tensor, eval_mask: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate model performance on validation/test set.
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get predictions
            predictions = self.model(node_features, node_types, edge_data, timestamps)
            probabilities = F.softmax(predictions, dim=1)
            
            # Extract evaluation data
            eval_predictions = predictions[eval_mask]
            eval_probabilities = probabilities[eval_mask]
            eval_labels = labels[eval_mask]
            
            if eval_mask.sum() == 0:
                return {'loss': 0.0, 'accuracy': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'auc': 0.5}
            
            # Compute loss
            loss = F.cross_entropy(eval_predictions, eval_labels)
            
            # Convert to numpy for sklearn metrics
            pred_labels = eval_predictions.argmax(dim=1).cpu().numpy()
            true_labels = eval_labels.cpu().numpy()
            pred_probs = eval_probabilities[:, 1].cpu().numpy()  # Probability of fraud class
            
            # Compute metrics
            accuracy = accuracy_score(true_labels, pred_labels)
            f1 = f1_score(true_labels, pred_labels, average='binary', zero_division=0)
            precision = precision_score(true_labels, pred_labels, average='binary', zero_division=0)
            recall = recall_score(true_labels, pred_labels, average='binary', zero_division=0)
            
            # AUC score (handle edge cases)
            try:
                auc = roc_auc_score(true_labels, pred_probs)
            except ValueError:
                auc = 0.5  # Default for edge cases
            
            return {
                'loss': loss.item(),
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'auc': auc
            }
    
    def train(self, node_features: torch.Tensor, node_types: torch.Tensor,
              edge_data: Dict[str, Tuple], timestamps: Dict[str, torch.Tensor],
              labels: torch.Tensor, train_mask: torch.Tensor, val_mask: torch.Tensor,
              num_epochs: int = 200, early_stopping_patience: int = 20,
              class_weights: Optional[torch.Tensor] = None,
              eval_every: int = 5) -> Dict[str, List]:
        """
        Complete training loop with early stopping and validation.
        
        Returns:
            Training history dictionary
        """
        # Move data to device
        node_features = node_features.to(self.device)
        node_types = node_types.to(self.device)
        labels = labels.to(self.device)
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)
        
        # Move edge data to device
        edge_data_device = {}
        timestamps_device = {}
        for rel_type in edge_data:
            edge_index, edge_attr = edge_data[rel_type]
            edge_data_device[rel_type] = (edge_index.to(self.device), edge_attr.to(self.device))
            if rel_type in timestamps:
                timestamps_device[rel_type] = timestamps[rel_type].to(self.device)
        
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        
        best_val_f1 = 0.0
        patience_counter = 0
        best_model_state = None
        
        self.logger.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in tqdm(range(num_epochs), desc="Training"):
            # Training step
            train_loss = self.train_epoch(
                node_features, node_types, edge_data_device, timestamps_device,
                labels, train_mask, class_weights
            )
            self.train_losses.append(train_loss)
            
            # Validation step
            if epoch % eval_every == 0:
                val_metrics = self.evaluate(
                    node_features, node_types, edge_data_device, timestamps_device,
                    labels, val_mask
                )
                
                self.val_losses.append(val_metrics['loss'])
                self.val_metrics.append(val_metrics)
                
                # Early stopping based on F1 score
                if val_metrics['f1'] > best_val_f1:
                    best_val_f1 = val_metrics['f1']
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += eval_every
                
                # Log progress
                self.logger.info(
                    f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | Val F1: {val_metrics['f1']:.4f} | "
                    f"Val AUC: {val_metrics['auc']:.4f}"
                )
                
                # Early stopping
                if patience_counter >= early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch} (best F1: {best_val_f1:.4f})")
                    break
            
            # Update learning rate
            self.scheduler.step()
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.logger.info(f"Loaded best model with validation F1: {best_val_f1:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics,
            'best_val_f1': best_val_f1
        }
    
    def test(self, node_features: torch.Tensor, node_types: torch.Tensor,
             edge_data: Dict[str, Tuple], timestamps: Dict[str, torch.Tensor],
             labels: torch.Tensor, test_mask: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Returns:
            Test metrics
        """
        # Move data to device
        node_features = node_features.to(self.device)
        node_types = node_types.to(self.device)
        labels = labels.to(self.device)
        test_mask = test_mask.to(self.device)
        
        # Move edge data to device
        edge_data_device = {}
        timestamps_device = {}
        for rel_type in edge_data:
            edge_index, edge_attr = edge_data[rel_type]
            edge_data_device[rel_type] = (edge_index.to(self.device), edge_attr.to(self.device))
            if rel_type in timestamps:
                timestamps_device[rel_type] = timestamps[rel_type].to(self.device)
        
        test_metrics = self.evaluate(
            node_features, node_types, edge_data_device, timestamps_device,
            labels, test_mask
        )
        
        self.logger.info("Test Results:")
        for metric, value in test_metrics.items():
            self.logger.info(f"  {metric.capitalize()}: {value:.4f}")
        
        return test_metrics
    
    def save_model(self, path: str):
        """Save model state dict."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics
        }, path)
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model state dict."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_metrics = checkpoint.get('val_metrics', [])
        self.logger.info(f"Model loaded from {path}")


def compute_class_weights(labels: torch.Tensor, train_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute class weights for handling imbalanced datasets.
    
    Args:
        labels: All node labels
        train_mask: Boolean mask for training nodes
        
    Returns:
        Class weights tensor
    """
    train_labels = labels[train_mask]
    unique_labels, counts = torch.unique(train_labels, return_counts=True)
    
    # Compute inverse frequency weights
    total_samples = len(train_labels)
    weights = total_samples / (len(unique_labels) * counts.float())
    
    # Create weight tensor for all classes
    class_weights = torch.ones(labels.max().item() + 1)
    for label, weight in zip(unique_labels, weights):
        class_weights[label] = weight
    
    return class_weights