"""
Training Configuration for Enhanced Multi-Target SciBERT
Optimized configuration for multi-target research paper classification

This configuration supports classification across:
- Research themes
- CCAMLR objectives  
- Management zones
- Monitoring areas
"""

from typing import Dict, Any, Optional
import torch

# Import experiment configuration
from experiment_config import get_active_targets, MODE

# OPTIMIZED CONFIGURATION: Spatial Enhanced
# This configuration has been optimized for multi-target classification performance

SPATIAL_ENHANCED_CONFIG = {
   # Configuration Metadata
   'name': 'Spatial_Enhanced',
   'description': 'Optimized configuration for multi-target classification',
   'selection_basis': 'Best performing from cross-validation analysis',
   'performance_targets': {
       'themes': 0.476,
       'objectives': 0.910,
       'zones': 1.000,
       'areas': 0.481,
       'overall_weighted_f1': 0.66
   },
   
   # Model Architecture
   'dropout': 0.12,
   'shared_dim': 256,
   'spatial_emphasis': True,
   'max_length': 512,
   
   # Training Hyperparameters
   'learning_rate': 5e-6,              # Base model learning rate
   'classifier_lr': 1.2e-4,            # Classifier learning rate
   'batch_size': 10,                   # Physical batch size
   'gradient_accumulation_steps': 3,    # Effective batch: 30
   'epochs': 32,                       # Base epochs
   'weight_decay': 0.012,              # L2 regularization
   'warmup_steps': 150,                # LR warmup
   'early_stopping_patience': 6,       # Early stopping
   
   # Loss Configuration
   'focal_gamma': 2.0,                 # Focal loss gamma
   'label_smoothing': 0.08,            # Label smoothing
   
   # Target Weights will be set dynamically based on mode
   'target_weights': {},  # Will be populated by get_target_weights()
   
   # Evaluation Configuration
   'threshold': 0.42,                  # Optimized threshold
   
   # Training methodology flags
   'use_all_non_test_data': True,      # Use train+val for final training
   'extended_epochs': True,            # Add 2 extra epochs (32+2=34)
   'no_validation_split': True,        # No validation during final training
   'enhanced_methodology': True        # Flag for enhanced approach
}

# PERFORMANCE TARGETS
PERFORMANCE_TARGETS = {
   'themes': {
       'macro_f1': 0.476,
       'description': 'Moderate performance on semantic complexity',
       'challenge': 'Diverse thematic content, class imbalance'
   },
   'objectives': {
       'macro_f1': 0.910,
       'description': 'Excellent performance on structured objectives',
       'challenge': 'Well-defined CCAMLR objectives'
   },
   'zones': {
       'macro_f1': 1.000,
       'description': 'Perfect performance due to GPZ dominance',
       'challenge': 'Universal GPZ coverage simplifies task'
   },
   'areas': {
       'macro_f1': 0.481,
       'description': 'Strong spatial classification performance',
       'challenge': 'Geographic naming variations, spatial complexity'
   },
   'overall_weighted_f1': 0.66,
   'note': 'These are performance targets for the enhanced system'
}

# TRAINING METHODOLOGY
TRAINING_METHODOLOGY = {
   'phase1_config_selection': {
       'method': '5-fold cross-validation',
       'purpose': 'Select best configuration',
       'configurations_tested': 4,
       'winner': 'Spatial_Enhanced'
   },
   'phase2_final_training': {
       'method': 'Enhanced Multi-Target Training',
       'data_usage': 'ALL non-test data (train + validation combined)',
       'validation_split': 'None (no holdout during final training)',
       'epochs': 'config_epochs + 2 (32 + 2 = 34)',
       'evaluation': 'Only on test set at the end',
       'key_principle': 'Use maximum available data for final training'
   }
}


def get_target_weights() -> Dict[str, float]:
   """Get target weights based on experiment mode"""
   active_targets = get_active_targets()
   
   if "zones" in active_targets and "areas" in active_targets:
       # Full mode - all 4 targets
       return {
           'themes': 3.0,        # High semantic complexity weight
           'objectives': 2.5,    # Policy alignment weight
           'areas': 2.8,         # Spatial emphasis weight
           'zones': 2.0          # Moderate zone weight
       }
   else:
       # Semantic mode - only themes and objectives
       return {
           'themes': 3.0,        # High semantic complexity weight
           'objectives': 2.5     # Policy alignment weight
       }


def get_target_importance() -> Dict[str, float]:
   """Get target importance weights for evaluation based on experiment mode"""
   active_targets = get_active_targets()
   
   if "zones" in active_targets and "areas" in active_targets:
       # Full mode - all 4 targets
       return {
           'themes': 0.35,      # Primary semantic classification
           'objectives': 0.25,  # Policy alignment importance
           'areas': 0.25,       # Spatial classification importance
           'zones': 0.15        # Lower complexity (GPZ dominance)
       }
   else:
       # Semantic mode - only themes and objectives
       return {
           'themes': 0.6,       # Higher weight for complex semantic task
           'objectives': 0.4    # Lower weight for structured task
       }


def get_config() -> Dict[str, Any]:
   """
   Get the optimized configuration for multi-target training
   
   Returns:
       Complete configuration dictionary
   """
   config = SPATIAL_ENHANCED_CONFIG.copy()
   # Dynamically set target weights based on mode
   config['target_weights'] = get_target_weights()
   # Add experiment mode to config
   config['experiment_mode'] = MODE
   return config


def get_performance_targets() -> Dict[str, Any]:
   """Get expected performance targets based on mode"""
   active_targets = get_active_targets()
   
   # Filter performance targets to only include active targets
   filtered_targets = {}
   for target in active_targets:
       if target in PERFORMANCE_TARGETS:
           filtered_targets[target] = PERFORMANCE_TARGETS[target]
   
   # Add overall targets
   if "zones" in active_targets and "areas" in active_targets:
       filtered_targets['overall_weighted_f1'] = 0.66
   else:
       # Semantic mode typically achieves higher F1 when not diluted by geographic tasks
       filtered_targets['overall_weighted_f1'] = 0.70
   
   filtered_targets['note'] = f'Performance targets for {MODE} mode'
   
   return filtered_targets


def get_training_methodology() -> Dict[str, Any]:
   """Get training methodology information"""
   methodology = TRAINING_METHODOLOGY.copy()
   methodology['experiment_mode'] = MODE
   return methodology


def validate_config(config: Dict[str, Any]) -> bool:
   """
   Validate that configuration contains required parameters
   
   Args:
       config: Configuration to validate
       
   Returns:
       True if valid, raises ValueError if not
   """
   
   # Check critical parameters
   required_params = {
       'name': 'Spatial_Enhanced',
       'learning_rate': 5e-6,
       'classifier_lr': 1.2e-4,
       'batch_size': 10,
       'gradient_accumulation_steps': 3,
       'epochs': 32,
       'threshold': 0.42,
       'spatial_emphasis': True
   }
   
   for param, expected_value in required_params.items():
       if config.get(param) != expected_value:
           raise ValueError(f"Parameter {param} should be {expected_value}, got {config.get(param)}")
   
   # Check target weights based on mode
   active_targets = get_active_targets()
   expected_weights = get_target_weights()
   actual_weights = config.get('target_weights', {})
   
   for target in active_targets:
       if target not in actual_weights:
           raise ValueError(f"Missing weight for active target: {target}")
       if actual_weights.get(target) != expected_weights.get(target):
           raise ValueError(f"Target weight for {target} should be {expected_weights.get(target)}, got {actual_weights.get(target)}")
   
   print(f"Configuration validation passed - all parameters correct for {MODE} mode")
   return True


def print_config_summary():
   """Print summary of training configuration"""
   
   config = get_config()
   targets = get_performance_targets()
   methodology = get_training_methodology()
   active_targets = get_active_targets()
   
   print(f"Training Configuration Summary: {config['name']} ({MODE.upper()} MODE)")
   print("=" * 60)
   
   print(f"Active Targets: {active_targets}")
   
   print(f"\nModel Architecture:")
   print(f"   Dropout: {config['dropout']}")
   print(f"   Shared dimension: {config['shared_dim']}")
   print(f"   Spatial emphasis: {config['spatial_emphasis']}")
   print(f"   Max length: {config['max_length']}")
   
   print(f"\nTraining Parameters:")
   print(f"   Learning rates: {config['learning_rate']:.1e} / {config['classifier_lr']:.1e}")
   print(f"   Batch size: {config['batch_size']} (effective: {config['batch_size'] * config['gradient_accumulation_steps']})")
   print(f"   Epochs: {config['epochs']} + 2 = 34 (extended training)")
   print(f"   Weight decay: {config['weight_decay']}")
   print(f"   Label smoothing: {config['label_smoothing']}")
   
   print(f"\nTarget Weights ({MODE} mode):")
   for target, weight in config['target_weights'].items():
       print(f"   {target}: {weight}")
   
   print(f"\nPerformance Targets ({MODE} mode):")
   for target in active_targets:
       if target in targets:
           f1 = targets[target]['macro_f1']
           desc = targets[target]['description']
           print(f"   {target.capitalize()}: {f1:.3f} - {desc}")
   if 'overall_weighted_f1' in targets:
       print(f"   Overall Weighted F1: {targets['overall_weighted_f1']:.2f}")
   
   print(f"\nTarget Importance Weights:")
   importance = get_target_importance()
   for target, weight in importance.items():
       print(f"   {target}: {weight}")
   
   print(f"\nTraining Methodology:")
   phase2 = methodology['phase2_final_training']
   print(f"   Method: {phase2['method']}")
   print(f"   Data usage: {phase2['data_usage']}")
   print(f"   Validation split: {phase2['validation_split']}")
   print(f"   Epochs: {phase2['epochs']}")
   print(f"   Evaluation: {phase2['evaluation']}")
   
   print(f"\nKey Success Factors:")
   print(f"   - Use ALL non-test data for final training")
   print(f"   - Extended epochs (34 total)")
   print(f"   - No validation holdout during final training")
   if MODE == "full":
       print(f"   - Spatial-enhanced target weights")
   else:
       print(f"   - Semantic-focused target weights")
   print(f"   - Threshold 0.42 for evaluation")


def create_training_config() -> Dict[str, Any]:
   """
   Create validated training configuration
   
   Returns:
       Configuration dict ready for training
   """
   
   # Start with base config
   config = get_config()
   
   # Validate configuration
   validate_config(config)
   
   print(f"Created training configuration for {MODE} mode")
   print(f"   Name: {config['name']}")
   print(f"   Target weights: {config['target_weights']}")
   print(f"   All parameters validated")
   
   return config


def demonstrate_config():
   """Demonstrate configuration setup"""
   
   print("Training Configuration Demo")
   print("=" * 40)
   
   try:
       # Create and validate config
       config = create_training_config()
       
       # Show detailed summary
       print_config_summary()
       
       # Show methodology
       methodology = get_training_methodology()
       print(f"\nEnhanced Training Approach:")
       print(f"   1. Use 5-fold CV to select best config (already done)")
       print(f"   2. Combine train + validation data")
       print(f"   3. Train final model on ALL non-test data")
       print(f"   4. Extended epochs (32 + 2 = 34)")
       print(f"   5. No validation split during final training")
       print(f"   6. Evaluate only on test set at the end")
       
       print(f"\nConfiguration ready for {MODE} mode!")
       print(f"   Active targets: {get_active_targets()}")
       print(f"   Use with train_model.py")
       
       return config
       
   except Exception as e:
       print(f"Configuration demo failed: {str(e)}")
       return None


# Compatibility functions for existing code
def get_original_config() -> Dict[str, Any]:
   """Compatibility wrapper for existing code"""
   return get_config()


if __name__ == "__main__":
   # Run demonstration
   demonstrate_config()
