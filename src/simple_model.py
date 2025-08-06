# simple_model.py
"""
Simple Multi-Target Model for BERT vs SciBERT Comparison
Clean architecture without unnecessary complexity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class SimpleMultiTargetModel(nn.Module):
    """
    Simple multi-target classification model for BERT/SciBERT comparison
    Clean architecture with direct classification heads
    """
    
    def __init__(self,
                 base_model,
                 target_configs: Dict[str, int],
                 dropout: float = 0.1):
        """
        Initialize multi-target model
        
        Args:
            base_model: Pretrained BERT or SciBERT model
            target_configs: Dict mapping target names to number of classes
                          e.g., {'themes': 27, 'objectives': 9, 'zones': 3, 'areas': 17}
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        self.base_model = base_model
        self.target_configs = target_configs
        self.dropout = nn.Dropout(dropout)
        
        # Get hidden size from base model
        self.hidden_size = base_model.config.hidden_size
        
        # Create simple classification heads for each target
        self.classifiers = nn.ModuleDict()
        for target_name, num_classes in target_configs.items():
            # Simple linear layer for each target
            self.classifiers[target_name] = nn.Linear(self.hidden_size, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights with Xavier initialization"""
        for classifier in self.classifiers.values():
            nn.init.xavier_uniform_(classifier.weight)
            nn.init.zeros_(classifier.bias)
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass
        
        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask from tokenizer
            
        Returns:
            Dict of logits for each target
        """
        # Get BERT/SciBERT outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use CLS token representation (first token)
        # Shape: (batch_size, hidden_size)
        cls_representation = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout
        cls_representation = self.dropout(cls_representation)
        
        # Get predictions for each target
        predictions = {}
        for target_name, classifier in self.classifiers.items():
            # Simple linear transformation
            # Shape: (batch_size, num_classes)
            predictions[target_name] = classifier(cls_representation)
        
        return predictions
    
    def get_model_info(self) -> Dict:
        """Get information about the model"""
        return {
            'model_type': 'SimpleMultiTargetModel',
            'base_model': self.base_model.config._name_or_path,
            'hidden_size': self.hidden_size,
            'dropout': self.dropout.p,
            'targets': self.target_configs,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Simplified version without unnecessary complexity
    """
    
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        """
        Initialize Focal Loss
        
        Args:
            gamma: Focusing parameter (typically 2.0)
            alpha: Class weights tensor
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate focal loss
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels (binary)
            
        Returns:
            Scalar loss value
        """
        # Binary cross entropy with logits
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate focal term
        pt = torch.exp(-bce_loss)  # probability of correct class
        focal_loss = (1 - pt) ** self.gamma * bce_loss
        
        # Apply class weights if provided
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            focal_loss = self.alpha.unsqueeze(0) * focal_loss
        
        return focal_loss.mean()


class MultiTargetLoss(nn.Module):
    """
    Combined loss for multi-target classification
    Simple weighted sum of individual target losses
    """
    
    def __init__(self,
                 target_configs: Dict[str, int],
                 loss_weights: Optional[Dict[str, float]] = None,
                 class_weights: Optional[Dict[str, torch.Tensor]] = None,
                 focal_gamma: float = 2.0):
        """
        Initialize multi-target loss
        
        Args:
            target_configs: Dict mapping target names to number of classes
            loss_weights: Weights for each target's loss (default: equal)
            class_weights: Class weights for each target (for imbalance)
            focal_gamma: Focal loss gamma parameter
        """
        super().__init__()
        
        self.target_configs = target_configs
        self.loss_weights = loss_weights or {target: 1.0 for target in target_configs}
        
        # Create focal loss for each target
        self.losses = nn.ModuleDict()
        for target in target_configs:
            alpha = class_weights.get(target) if class_weights else None
            self.losses[target] = FocalLoss(gamma=focal_gamma, alpha=alpha)
    
    def forward(self,
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate combined loss
        
        Args:
            predictions: Dict of model predictions
            targets: Dict of ground truth labels
            
        Returns:
            Tuple of (total_loss, individual_losses_dict)
        """
        individual_losses = {}
        total_loss = 0.0
        
        for target_name in self.target_configs:
            # Calculate loss for this target
            loss = self.losses[target_name](predictions[target_name], targets[target_name])
            
            # Apply target weight
            weighted_loss = loss * self.loss_weights[target_name]
            
            # Track individual losses
            individual_losses[target_name] = loss.item()
            total_loss += weighted_loss
        
        return total_loss, individual_losses


def calculate_class_weights(labels: np.ndarray, min_weight: float = 0.1) -> torch.Tensor:
    """
    Calculate class weights for handling imbalance
    
    Args:
        labels: Binary label matrix (n_samples, n_classes)
        min_weight: Minimum weight to avoid extreme values
        
    Returns:
        Tensor of class weights
    """
    # Calculate class frequencies
    class_counts = labels.sum(axis=0)
    total_samples = len(labels)
    
    # Calculate inverse frequency weights
    # Add 1 to avoid division by zero
    weights = total_samples / (labels.shape[1] * (class_counts + 1))
    
    # Clip weights to avoid extreme values
    weights = np.clip(weights, min_weight, 1/min_weight)
    
    return torch.FloatTensor(weights)


def get_optimizer(model, learning_rate: float = 2e-5, weight_decay: float = 0.01):
    """
    Get AdamW optimizer with reasonable defaults
    
    Args:
        model: The model to optimize
        learning_rate: Learning rate
        weight_decay: L2 regularization
        
    Returns:
        AdamW optimizer
    """
    # Don't apply weight decay to bias terms
    no_decay = ['bias', 'LayerNorm.weight']
    
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters()
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters()
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        }
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    return optimizer


class EarlyStopping:
    """Simple early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        """
        Initialize early stopping
        
        Args:
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """
        Check if should stop training
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if should stop training
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


def save_model(model,
               tokenizer_name: str,
               save_path: str,
               target_configs: Dict[str, int],
               training_args: Dict,
               evaluation_results: Dict):
    """
    Save model with all necessary information for loading
    
    Args:
        model: Trained model
        tokenizer_name: Name of tokenizer used
        save_path: Where to save
        target_configs: Target configuration
        training_args: Training arguments used
        evaluation_results: Evaluation metrics
    """
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'base_model': model.base_model.config._name_or_path,
            'target_configs': target_configs,
            'dropout': model.dropout.p,
            'model_class': 'SimpleMultiTargetModel'
        },
        'tokenizer_name': tokenizer_name,
        'training_args': training_args,
        'evaluation_results': evaluation_results,
        'model_info': model.get_model_info()
    }
    
    torch.save(save_dict, save_path)
    print(f" Model saved to: {save_path}")


def load_model(model_path: str, device: str = 'cpu'):
    """
    Load a saved model
    
    Args:
        model_path: Path to saved model
        device: Device to load on
        
    Returns:
        Tuple of (model, model_config, tokenizer_name)
    """
    from transformers import AutoModel
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract config
    model_config = checkpoint['model_config']
    
    # Load base model
    base_model = AutoModel.from_pretrained(model_config['base_model'])
    
    # Create model
    model = SimpleMultiTargetModel(
        base_model=base_model,
        target_configs=model_config['target_configs'],
        dropout=model_config.get('dropout', 0.1)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, model_config, checkpoint.get('tokenizer_name')


if __name__ == "__main__":
    """Test the model architecture"""
    print("Testing Simple Multi-Target Model")
    print("=" * 70)
    
    # Test with dummy BERT model
    from transformers import AutoModel
    
    # Load small BERT for testing
    print("Loading test BERT model...")
    base_model = AutoModel.from_pretrained('bert-base-uncased')
    
    # Create multi-target model
    target_configs = {
        'themes': 27,
        'objectives': 9,
        'zones': 3,
        'areas': 17
    }
    
    model = SimpleMultiTargetModel(base_model, target_configs, dropout=0.1)
    print(f"\n Model created successfully")
    
    # Print model info
    info = model.get_model_info()
    print(f"\nModel Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test forward pass
    print(f"\n Testing forward pass...")
    batch_size = 2
    seq_length = 128
    
    # Create dummy inputs
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    
    print(f"\n Forward pass successful!")
    print(f"Output shapes:")
    for target, output in outputs.items():
        print(f"  {target}: {output.shape}")
    
    # Test loss calculation
    print(f"\n Testing loss calculation...")
    
    # Create dummy labels
    labels = {}
    for target, num_classes in target_configs.items():
        labels[target] = torch.randint(0, 2, (batch_size, num_classes)).float()
    
    # Calculate class weights
    class_weights = {}
    for target, label_tensor in labels.items():
        weights = calculate_class_weights(label_tensor.numpy())
        class_weights[target] = weights
        print(f"  {target} class weights: min={weights.min():.2f}, max={weights.max():.2f}")
    
    # Create loss function
    loss_fn = MultiTargetLoss(
        target_configs=target_configs,
        class_weights=class_weights,
        focal_gamma=2.0
    )
    
    # Calculate loss
    total_loss, individual_losses = loss_fn(outputs, labels)
    
    print(f"\n Loss calculation successful!")
    print(f"Total loss: {total_loss:.4f}")
    print(f"Individual losses:")
    for target, loss in individual_losses.items():
        print(f"  {target}: {loss:.4f}")
    
    print(f"\n All model tests passed!")
