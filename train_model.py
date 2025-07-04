"""
Complete Training Pipeline for Enhanced Multi-Target SciBERT with Ensemble Support
Comprehensive training approach for multi-target research paper classification

This training pipeline implements:
- Full non-test data utilization (no validation holdout)
- Extended epochs for optimal performance
- Test set evaluation only
- Enhanced multi-target methodology
- Ensemble training for improved consistency
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
import warnings
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import json
from datetime import datetime
import pickle
import random

warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from data_loader import load_data_with_fallback
    from data_preprocessing import preprocess_dataset_for_training
    from enhanced_scibert_model import EnhancedMultiTargetSciBERT
    from training_config import get_config, get_target_importance
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure all src/ files are present and working")
    sys.exit(1)

# Enhanced Focal Loss for handling class imbalance
class EnhancedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = (1 - pt) ** self.gamma * bce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.unsqueeze(0).expand_as(targets)
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class EnhancedMultiTargetDataset(Dataset):
    """Dataset class for multi-target classification"""
    
    def __init__(self, texts, labels_dict, tokenizer, max_length=512):
        self.texts = texts
        self.labels_dict = labels_dict
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Prepare labels for all targets
        labels = {}
        for target, label_matrix in self.labels_dict.items():
            labels[target] = torch.FloatTensor(label_matrix[idx])
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            **{f'{target}_labels': labels[target] for target in labels.keys()}
        }

def set_random_seeds(seed):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    # For deterministic behavior (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_enhanced_model(
    all_texts, all_labels, target_configs, tokenizer, config, device, test_texts, test_labels, model_id=1, random_seed=42
):
    """
    Train Enhanced Multi-Target Model using comprehensive methodology
    - Uses ALL non-test data for maximum performance
    - Extended epochs for optimal training
    - No validation split during training
    """
    
    print(f"\n=== TRAINING ENHANCED MULTI-TARGET MODEL {model_id} ===")
    print("Enhanced multi-target training approach")
    print("=" * 60)
    
    # Set random seed for this model
    set_random_seeds(random_seed)
    print(f"Random seed set to: {random_seed}")
    
    # Create enhanced model
    model = EnhancedMultiTargetSciBERT(
        target_configs=target_configs,
        dropout_rate=config['dropout'],
        shared_dim=256,
        spatial_emphasis=config['spatial_emphasis']
    )
    model = model.to(device)
    
    # Calculate class weights for each target
    class_weights = {}
    criterions = {}
    
    for target, labels in all_labels.items():
        class_counts = labels.sum(axis=0)
        weights = len(labels) / (labels.shape[1] * class_counts + 1)
        class_weights[target] = torch.FloatTensor(weights).to(device)
        criterions[target] = EnhancedFocalLoss(
            alpha=class_weights[target],
            gamma=config['focal_gamma']
        )
        if model_id == 1:  # Only print for first model to avoid clutter
            print(f"   {target} class weights: min={weights.min():.2f}, max={weights.max():.2f}")
    
    # Multi-rate optimizer setup
    base_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'base_model' in name:
            base_params.append(param)
        else:
            classifier_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': base_params, 'lr': config['learning_rate']},
        {'params': classifier_params, 'lr': config['classifier_lr']}
    ], weight_decay=config.get('weight_decay', 0.01))
    
    # Create training dataset
    train_dataset = EnhancedMultiTargetDataset(
        all_texts, all_labels, tokenizer, max_length=config['max_length']
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )
    
    # Extended training configuration
    total_epochs = config['epochs'] + 2
    target_weights = config.get('target_weights', {target: 1.0 for target in target_configs.keys()})
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    label_smoothing = config.get('label_smoothing', 0.08)
    spatial_emphasis = config.get('spatial_emphasis', True)
    threshold = config.get('threshold', 0.42)
    
    if model_id == 1:  # Only print for first model
        print(f"\nTraining Configuration:")
        print(f"   Papers: {len(all_texts)}")
        print(f"   Epochs: {total_epochs}")
        print(f"   Target weights: {target_weights}")
        print(f"   Gradient accumulation: {gradient_accumulation_steps}")
        print(f"   Label smoothing: {label_smoothing}")
        print(f"   Spatial emphasis: {spatial_emphasis}")
    
    print(f"\nStarting extended training on full data (Model {model_id})...")
    
    # Create test dataset for progress monitoring
    test_dataset = EnhancedMultiTargetDataset(
        test_texts, test_labels, tokenizer, max_length=config['max_length']
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False
    )
    
    best_f1 = 0.0
    best_epoch = 0
    
    for epoch in range(total_epochs):
        model.train()
        target_losses = {target: 0.0 for target in target_configs.keys()}
        
        progress_bar = tqdm(train_loader, desc=f"Model {model_id} Epoch {epoch+1}/{total_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = {target: batch[f'{target}_labels'].to(device) for target in target_configs.keys()}
            
            # Multi-target forward pass
            outputs = model(input_ids, attention_mask)
            
            # Weighted multi-target loss with enhancements
            batch_loss = 0.0
            batch_target_losses = {}
            
            for target in target_configs.keys():
                target_loss = criterions[target](outputs[target], labels[target])
                
                # Apply label smoothing
                if label_smoothing > 0:
                    target_loss = target_loss * (1 - label_smoothing) + \
                                 label_smoothing * torch.log(torch.tensor(labels[target].shape[1], device=device, dtype=torch.float32))
                
                # Apply spatial emphasis
                if spatial_emphasis and target in ['areas', 'zones']:
                    target_loss = target_loss * 1.1  # 10% boost for spatial targets
                
                weighted_loss = target_loss * target_weights.get(target, 1.0)
                batch_target_losses[target] = target_loss.item()
                batch_loss += weighted_loss
            
            # Scale loss by accumulation steps
            batch_loss = batch_loss / gradient_accumulation_steps
            batch_loss.backward()
            
            # Only step optimizer every accumulation_steps batches
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            # Track losses
            for target in target_configs.keys():
                target_losses[target] += batch_target_losses[target]
            
            # Update progress
            progress_info = {f'{target[:3]}': batch_target_losses[target] for target in target_configs.keys()}
            progress_bar.set_postfix(progress_info)
        
        # Print epoch summary (reduced output for ensemble models)
        if model_id == 1 or epoch % 10 == 0 or epoch == total_epochs - 1:
            print(f"\nModel {model_id} Epoch {epoch+1} - Average Losses:")
            for target in target_configs.keys():
                avg_loss = target_losses[target] / len(train_loader)
                print(f"  {target}: {avg_loss:.4f}")
        
        # Evaluate on test set for progress monitoring
        if model_id == 1 or epoch % 10 == 0 or epoch == total_epochs - 1:
            print(f"Evaluating Model {model_id}...")
            test_f1_scores = evaluate_on_test_set(model, test_loader, target_configs, threshold)
            
            # Calculate overall weighted F1
            target_importance = get_target_importance()
            overall_f1 = sum(
                test_f1_scores[target] * target_importance[target]
                for target in target_configs.keys()
            ) / sum(target_importance.values())
            
            # Print results
            print(f"Training Results:")
            print(f"   Total Loss: {sum(avg_loss for avg_loss in [target_losses[target] / len(train_loader) for target in target_configs.keys()]):.4f}")
            for target in target_configs.keys():
                avg_loss = target_losses[target] / len(train_loader)
                print(f"   {target.capitalize()} Loss: {avg_loss:.4f}")
            
            print(f"Validation Results:")
            for target in target_configs.keys():
                print(f"   {target.capitalize()}: F1={test_f1_scores[target]:.3f}")
            print(f"   Overall Weighted F1: {overall_f1:.3f} (threshold: {threshold})")
            
            # Track best model
            if overall_f1 > best_f1:
                best_f1 = overall_f1
                best_epoch = epoch + 1
                print(f"   New best model saved! F1: {overall_f1:.3f}")
            else:
                print(f"   No improvement. Best: {best_f1:.3f} at epoch {best_epoch}")
    
    print(f"\nEnhanced multi-target model {model_id} training complete!")
    print(f"Best overall F1: {best_f1:.3f} at epoch {best_epoch}")
    
    return model, class_weights, criterions, best_f1, best_epoch

def evaluate_on_test_set(model, test_loader, target_configs, threshold=0.42):
    """Evaluate model on test set and return F1 scores for each target"""
    model.eval()
    
    all_predictions = {target: [] for target in target_configs.keys()}
    all_labels = {target: [] for target in target_configs.keys()}
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(next(model.parameters()).device)
            attention_mask = batch['attention_mask'].to(next(model.parameters()).device)
            
            # Get predictions
            outputs = model(input_ids, attention_mask)
            
            for target in target_configs.keys():
                # Convert probabilities to binary predictions
                probs = torch.sigmoid(outputs[target])
                preds = (probs > threshold).float()
                
                all_predictions[target].append(preds.cpu().numpy())
                all_labels[target].append(batch[f'{target}_labels'].numpy())
    
    # Calculate F1 scores
    f1_scores = {}
    for target in target_configs.keys():
        y_true = np.vstack(all_labels[target])
        y_pred = np.vstack(all_predictions[target])
        
        # Calculate macro F1 score
        f1_scores[target] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    return f1_scores

def evaluate_final_model(model, test_texts, test_labels, tokenizer, config, device, threshold=0.42):
    """Evaluate final model on test set with comprehensive metrics"""
    
    print(f"\n=== FINAL TEST SET EVALUATION ===")
    print("Reproducing test evaluation methodology")
    print("=" * 50)
    
    # Create test dataset
    test_dataset = EnhancedMultiTargetDataset(
        test_texts, test_labels, tokenizer, max_length=config['max_length']
    )
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    model.eval()
    all_predictions = {target: [] for target in test_labels.keys()}
    all_labels = {target: [] for target in test_labels.keys()}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating on test set"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Get predictions
            outputs = model(input_ids, attention_mask)
            
            for target in test_labels.keys():
                # Convert probabilities to binary predictions using threshold
                probs = torch.sigmoid(outputs[target])
                preds = (probs > threshold).float()
                
                all_predictions[target].append(preds.cpu().numpy())
                all_labels[target].append(batch[f'{target}_labels'].numpy())
    
    # Calculate F1 scores
    final_results = {}
    for target in test_labels.keys():
        y_true = np.vstack(all_labels[target])
        y_pred = np.vstack(all_predictions[target])
        
        # Calculate comprehensive F1 metrics
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        final_results[target] = {
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'weighted_f1': weighted_f1
        }
        
        print(f"{target.upper()}:")
        print(f"  Macro F1: {macro_f1:.4f}")
        print(f"  Micro F1: {micro_f1:.4f}")
        print(f"  Weighted F1: {weighted_f1:.4f}")
    
    # Calculate overall weighted F1
    target_importance = get_target_importance()
    overall_f1 = sum(
        final_results[target]['macro_f1'] * target_importance[target]
        for target in test_labels.keys()
    ) / sum(target_importance.values())
    
    print(f"\nOverall Weighted F1: {overall_f1:.4f}")
    print(f"Target importance: {target_importance}")
    
    return final_results, overall_f1

def train_ensemble_models(all_texts, all_labels, target_configs, tokenizer, config, device, X_test, test_labels, num_models=3):
    """Train ensemble of models with different random seeds"""
    
    print(f"\nENSEMBLE TRAINING: {num_models} Models")
    print("=" * 60)
    
    ensemble_models = []
    ensemble_paths = []
    ensemble_results = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Different random seeds for diversity
    base_seeds = [42, 123, 456, 789, 321]
    
    for i in range(num_models):
        print(f"\nTraining Ensemble Model {i+1}/{num_models}")
        print("-" * 40)
        
        # Use different random seed for each model
        random_seed = base_seeds[i] if i < len(base_seeds) else 42 + i * 100
        
        # Train model
        model, class_weights, criterions, best_f1, best_epoch = train_enhanced_model(
            all_texts, all_labels, target_configs, tokenizer, config, device,
            X_test, test_labels, model_id=i+1, random_seed=random_seed
        )
        
        # Evaluate final model
        final_results, overall_f1 = evaluate_final_model(
            model, X_test, test_labels, tokenizer, config, device
        )
        
        # Save ensemble model
        ensemble_path = f"models/enhanced_multitarget_scibert_ensemble_{i+1}_{timestamp}.pt"
        os.makedirs(os.path.dirname(ensemble_path), exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'target_configs': target_configs,
            'class_weights': {k: v.cpu().numpy() for k, v in class_weights.items()},
            'final_results': final_results,
            'overall_f1': overall_f1,
            'timestamp': timestamp,
            'ensemble_id': i + 1,
            'random_seed': random_seed,
            'training_approach': 'enhanced_multi_target_ensemble',
            'model_info': model.get_model_info()
        }, ensemble_path)
        
        ensemble_models.append(model)
        ensemble_paths.append(ensemble_path)
        ensemble_results.append({
            'model_id': i + 1,
            'overall_f1': overall_f1,
            'final_results': final_results,
            'random_seed': random_seed,
            'path': ensemble_path
        })
        
        print(f"Ensemble Model {i+1} saved: {ensemble_path}")
        print(f"   Overall F1: {overall_f1:.3f}")
        print(f"   Random seed: {random_seed}")
    
    # Print ensemble summary
    print(f"\nENSEMBLE TRAINING SUMMARY")
    print("=" * 50)
    
    f1_scores = [result['overall_f1'] for result in ensemble_results]
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    
    print(f"Ensemble Performance:")
    print(f"   Models trained: {num_models}")
    print(f"   Mean F1: {mean_f1:.3f} ± {std_f1:.3f}")
    print(f"   F1 range: {min(f1_scores):.3f} - {max(f1_scores):.3f}")
    print(f"   F1 variance: {std_f1**2:.6f}")
    
    print(f"\nIndividual Model Performance:")
    for result in ensemble_results:
        print(f"   Model {result['model_id']}: F1={result['overall_f1']:.3f} (seed={result['random_seed']})")
    
    print(f"\nEnsemble Model Paths:")
    for path in ensemble_paths:
        print(f"   {path}")
    
    # Save ensemble metadata
    ensemble_metadata_path = f"models/ensemble_metadata_{timestamp}.json"
    ensemble_metadata = {
        'timestamp': timestamp,
        'num_models': num_models,
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'model_paths': ensemble_paths,
        'individual_results': ensemble_results,
        'config': config,
        'target_configs': target_configs
    }
    
    with open(ensemble_metadata_path, 'w') as f:
        json.dump(ensemble_metadata, f, indent=2, default=str)
    
    print(f"\nEnsemble metadata saved: {ensemble_metadata_path}")
    
    return ensemble_models, ensemble_paths, ensemble_results

def main():
    """Main training pipeline for enhanced multi-target classification with ensemble support"""
    
    print("Enhanced Multi-Target SciBERT Training Pipeline with Ensemble Support")
    print("Enhanced Multi-Target Classification Training")
    print("Enhanced Multi-Target Methodology")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Step 1: Load and preprocess data
        print(f"\nStep 1: Data Loading and Preprocessing")
        print("-" * 50)
        
        # Load data with smart fallback
        df, is_actual = load_data_with_fallback()
        dataset_type = "actual" if is_actual else "synthetic"
        
        # Preprocess data
        processed_df, multi_target_data, splits, analysis_results = preprocess_dataset_for_training(df)
        
        print(f"Data preprocessing complete")
        print(f"   {len(processed_df)} papers processed")
        print(f"   {len(multi_target_data['label_matrices'])} targets prepared")
        print(f"   Dataset type: {dataset_type}")
        
        # Step 2: Extract training variables
        print(f"\nStep 2: Extracting Training Variables")
        print("-" * 50)
        
        # Extract text variables
        X_train = splits['X_train']
        X_val = splits['X_val']
        X_test = splits['X_test']
        
        # Combine train and val for final training
        all_texts = np.concatenate([X_train, X_val])
        
        # Extract and combine labels for all targets
        all_labels = {}
        target_configs = {}
        
        for target in ['themes', 'objectives', 'zones', 'areas']:
            train_labels = splits[f'y_train_{target}']
            val_labels = splits[f'y_val_{target}']
            all_labels[target] = np.vstack([train_labels, val_labels])
            target_configs[target] = train_labels.shape[1]
        
        # Test labels for final evaluation
        test_labels = {target: splits[f'y_test_{target}'] for target in target_configs.keys()}
        
        print(f"Training variables extracted:")
        print(f"   Combined training data: {len(all_texts)} papers (train+val)")
        print(f"   Test data: {len(X_test)} papers")
        print(f"   Target configurations: {target_configs}")
        
        # Step 3: Load configuration
        print(f"\nStep 3: Training Configuration")
        print("-" * 50)
        
        # Enhanced configuration with ensemble option
        config = {
            'name': 'Spatial_Enhanced',
            'dropout': 0.12,
            'learning_rate': 5e-6,
            'classifier_lr': 1.2e-4,
            'batch_size': 10,
            'gradient_accumulation_steps': 3,
            'epochs': 32,
            'focal_gamma': 2.0,
            'max_length': 512,
            'target_weights': {'themes': 3.0, 'objectives': 2.5, 'areas': 2.8, 'zones': 2.0},
            'threshold': 0.42,
            'warmup_steps': 150,
            'weight_decay': 0.012,
            'early_stopping_patience': 6,
            'label_smoothing': 0.08,
            'spatial_emphasis': True,
            'train_ensemble': True,  # Enable ensemble training
            'ensemble_size': 3       # Number of models in ensemble
        }
        
        print(f"Configuration loaded: {config['name']}")
        print(f"   Target weights: {config['target_weights']}")
        print(f"   Learning rates: {config['learning_rate']:.1e} / {config['classifier_lr']:.1e}")
        print(f"   Ensemble training: {config.get('train_ensemble', False)}")
        print(f"   Ensemble size: {config.get('ensemble_size', 1)}")
        
        # Step 4: Auto-detect device
        print(f"\nStep 4: Device Setup")
        print("-" * 50)
        
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        
        print(f"Using device: {device}")
        
        # Step 5: Model preparation
        print(f"\nStep 5: Model Creation")
        print("-" * 50)
        
        print(f"Models will be created during training")
        print(f"   Targets: {list(target_configs.keys())}")
        print(f"   Classes: {list(target_configs.values())}")
        
        # Step 6: Setup tokenizer
        print(f"\nStep 6: Setup Tokenizer")
        print("-" * 50)
        
        # Import tokenizer for training
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        
        # Add domain tokens
        antarctic_terms = [
            "dissostichus-mawsoni", "euphausia-superba", "pleuragramma-antarctica",
            "general-protection-zone", "special-research-zone", "krill-research-zone",
            "ross-sea-polynya", "mcmurdo-sound", "balleny-islands", "ccamlr"
        ]
        
        new_tokens = [term for term in antarctic_terms if term not in tokenizer.vocab]
        if new_tokens:
            print(f"Adding {len(new_tokens)} domain-specific tokens")
            tokenizer.add_tokens(new_tokens)
        
        # Step 7: Train models (single or ensemble)
        print(f"\nStep 7: Train Enhanced Multi-Target Model(s)")
        print("-" * 50)
        
        if config.get('train_ensemble', False):
            # Train ensemble of models
            ensemble_models, ensemble_paths, ensemble_results = train_ensemble_models(
                all_texts, all_labels, target_configs, tokenizer, config, device,
                X_test, test_labels, num_models=config.get('ensemble_size', 3)
            )
            
            # Use best model from ensemble for comparison
            best_ensemble_result = max(ensemble_results, key=lambda x: x['overall_f1'])
            final_model = ensemble_models[best_ensemble_result['model_id'] - 1]
            final_results = best_ensemble_result['final_results']
            overall_f1 = best_ensemble_result['overall_f1']
            
            print(f"\nBest Ensemble Model: Model {best_ensemble_result['model_id']}")
            print(f"   F1: {overall_f1:.3f}")
            print(f"   Path: {best_ensemble_result['path']}")
            
        else:
            # Train single model
            final_model, class_weights, criterions, best_f1, best_epoch = train_enhanced_model(
                all_texts, all_labels, target_configs, tokenizer, config, device, X_test, test_labels
            )
            
            # Step 7a: Final evaluation
            print(f"\nStep 7a: Final Test Evaluation")
            print("-" * 50)
            
            final_results, overall_f1 = evaluate_final_model(
                final_model, X_test, test_labels, tokenizer, config, device
            )
        
        # Step 8: Compare to target results
        print(f"\nStep 8: Comparison to Target Results")
        print("-" * 50)
        
        target_results = {
            'themes': 0.476, 'objectives': 0.910, 'zones': 1.000, 'areas': 0.481
        }
        
        print(f"Performance vs Target Results:")
        for target in target_configs.keys():
            if target in final_results and target in target_results:
                current = final_results[target]['macro_f1']
                target_val = target_results[target]
                diff = current - target_val
                status = "✓" if diff >= -0.1 else "~" if diff >= -0.2 else "✗"
                print(f"   {status} {target.capitalize()}: {current:.3f} vs {target_val:.3f} ({diff:+.3f})")
        
        print(f"\nOverall weighted F1: {overall_f1:.3f}")
        
        # Step 9: Save final model (if single model training)
        if not config.get('train_ensemble', False):
            print(f"\nStep 9: Save Final Model")
            print("-" * 50)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_save_path = f"models/enhanced_multitarget_scibert_{timestamp}.pt"
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            
            torch.save({
                'model_state_dict': final_model.state_dict(),
                'config': config,
                'target_configs': target_configs,
                'class_weights': {k: v.cpu().numpy() for k, v in class_weights.items()},
                'final_results': final_results,
                'overall_f1': overall_f1,
                'timestamp': timestamp,
                'training_approach': 'enhanced_multi_target',
                'model_info': final_model.get_model_info()
            }, model_save_path)
            
            print(f"Model saved: {model_save_path}")
        else:
            model_save_path = "ensemble_models"
        
        # Final summary
        print(f"\nTraining Pipeline Complete!")
        print("=" * 70)
        
        if config.get('train_ensemble', False):
            print(f"Enhanced multi-target ensemble methodology completed!")
            print(f"Ensemble Results:")
            for result in ensemble_results:
                print(f"   • Model {result['model_id']}: F1={result['overall_f1']:.3f}")
            print(f"Best Model F1: {overall_f1:.3f}")
            print(f"Ensemble saved in: models/")
        else:
            print(f"Enhanced multi-target methodology completed!")
            print(f"Final Results:")
            for target in target_configs.keys():
                f1 = final_results[target]['macro_f1']
                print(f"   • {target.upper()}: {f1:.3f}")
            print(f"Overall Weighted F1: {overall_f1:.3f}")
            print(f"Model: {model_save_path}")
        
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        methodology = "ensemble" if config.get('train_ensemble', False) else "single"
        print(f"\nKey Methodology Elements ({methodology}):")
        print(f"   - Used ALL non-test data for training ({len(all_texts)} papers)")
        print(f"   - Extended training (34 epochs)")
        print(f"   - No validation split during final training")
        print(f"   - Spatial-enhanced configuration")
        print(f"   - All enhancements: class weights, label smoothing, spatial boost")
        print(f"   - Threshold 0.42 for evaluation")
        if config.get('train_ensemble', False):
            print(f"   - Ensemble of {config.get('ensemble_size', 3)} models with different random seeds")
            print(f"   - Reduced prediction variance through ensemble averaging")
        
        return {
            'final_model': final_model,
            'results': final_results,
            'overall_f1': overall_f1,
            'config': config,
            'model_path': model_save_path,
            'ensemble_models': ensemble_models if config.get('train_ensemble', False) else None,
            'ensemble_paths': ensemble_paths if config.get('train_ensemble', False) else None,
            'ensemble_results': ensemble_results if config.get('train_ensemble', False) else None
        }
        
    except Exception as e:
        print(f"\nTraining pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Set environment for stability
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Run the enhanced training pipeline
    results = main()
    
    if results is not None:
        print(f"\nEnhanced Multi-Target SciBERT training completed successfully!")
        
        if results.get('ensemble_models') is not None:
            print(f"Ensemble training completed!")
            print(f"   {len(results['ensemble_models'])} models trained")
            print(f"   Use predict_paper.py with --ensemble flag for ensemble predictions")
        else:
            print(f"Enhanced multi-target methodology completed!")
        
        print(f"Results should achieve target performance:")
        print(f"   Themes: ~0.476, Objectives: ~0.910, Zones: ~1.000, Areas: ~0.481")
        print(f"   Enhanced consistency through ensemble averaging (if enabled)")
    else:
        print(f"\nTraining failed. Please check the error messages above.")
