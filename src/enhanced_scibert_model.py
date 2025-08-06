"""
Enhanced Multi-Target SciBERT Model
Multi-target classification architecture for Ross Sea research papers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

class EnhancedMultiTargetSciBERT(nn.Module):
    """
    Enhanced Multi-Target SciBERT for research paper classification
    """
    
    def __init__(
        self,
        target_configs: Optional[Dict[str, int]] = None,
        model_name: str = "allenai/scibert_scivocab_uncased",
        dropout_rate: float = 0.12,
        shared_dim: int = 256,
        spatial_emphasis: bool = True
    ):
        """
        Initialize Enhanced Multi-Target SciBERT
        
        Args:
            target_configs: Dictionary of {target_name: num_classes}
            model_name: SciBERT model name
            dropout_rate: Dropout rate
            shared_dim: Shared projection dimension
            spatial_emphasis: Spatial enhancement flag
        """
        super().__init__()
        
        # Default target configurations
        if target_configs is None:
            target_configs = {
                'themes': 27,      # Research themes
                'objectives': 9,   # CCAMLR objectives
                'zones': 3,        # Management zones (GPZ, SRZ, KRZ)
                'areas': 17        # Monitoring areas (top areas ≥5 papers)
            }
        
        self.target_configs = target_configs
        self.spatial_emphasis = spatial_emphasis
        self.shared_dim = shared_dim
        
        # Load SciBERT base model
        print(f"Loading SciBERT model: {model_name}")
        self.base_model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add domain-specific vocabulary
        self._extend_vocabulary()
        
        # Model architecture
        hidden_size = self.base_model.config.hidden_size  # 768
        self.shared_dropout = nn.Dropout(dropout_rate)
        
        # Shared projection layer
        self.shared_projection = nn.Linear(hidden_size, shared_dim)
        self.projection_dropout = nn.Dropout(dropout_rate)
        
        # Target-specific classification heads
        self.classifiers = nn.ModuleDict()
        for target_name, num_classes in target_configs.items():
            self.classifiers[target_name] = nn.Sequential(
                nn.Linear(shared_dim, shared_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(shared_dim // 2, num_classes)
            )
        
        # Initialize weights
        self._init_weights()
        
        print(f"Enhanced Multi-Target SciBERT initialized")
        print(f"   Targets: {list(target_configs.keys())}")
        print(f"   Classes: {list(target_configs.values())}")
        print(f"   Shared dimension: {hidden_size} → {shared_dim}")
        print(f"   Spatial emphasis: {spatial_emphasis}")
        
        # Calculate parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
    
    def _extend_vocabulary(self):
        """Add domain-specific tokens for Antarctic research"""
        
        # Antarctic research domain tokens
        antarctic_terms = [
            # === MARINE MAMMALS ===
            "weddell", "leptonychotes", "leptonychotes-weddellii",
            "leopard", "hydrurga", "hydrurga-leptonyx",
            "crabeater", "lobodon", "lobodon-carcinophaga",
            "ross-seal", "ommatophoca", "ommatophoca-rossii",
            "elephant-seal", "mirounga", "mirounga-leonina",
            "fur-seal", "arctocephalus", "arctocephalus-gazella",
            
            # === FISH ===
            "toothfish", "dissostichus", "dissostichus-mawsoni",
            "silverfish", "pleuragramma", "pleuragramma-antarctica",
            "icefish", "channichthyidae", "chaenocephalus",
            
            # === INVERTEBRATES ===
            "krill", "euphausia", "euphausia-superba",
            "crystal-krill", "euphausia-crystallorophias",
            
            # === BIRDS ===
            "adelie", "pygoscelis", "pygoscelis-adeliae",
            "emperor", "aptenodytes", "aptenodytes-forsteri",
            "chinstrap", "pygoscelis-antarctica",
            
            # === GEOGRAPHIC FEATURES ===
            "gpz", "srz", "krz", "ccamlr", "rsrmpa", "mpa",
            "general-protection-zone", "special-research-zone", "krill-research-zone",
            
            # === NAMED LOCATIONS ===
            "mcmurdo", "erebus", "terra-nova", "ross-sea",
            "balleny", "scott-island", "iselin-bank",
            "pennell-bank", "admiralty-range", "drygalski",
            "coulman-island", "cape-adare", "victoria-land",
            
            # === OCEANOGRAPHIC TERMS ===
            "circumpolar", "antarctic-circumpolar-current", "acc",
            "cdw", "circumpolar-deep-water", "mcdw",
            "thermocline", "halocline", "pycnocline",
            
            # === ICE TERMS ===
            "fast-ice", "pack-ice", "ice-shelf", "sea-ice",
            "ice-edge", "marginal-ice-zone", "miz",
            "platelet-ice", "anchor-ice", "sub-ice", "under-ice",
            
            # === ECOLOGICAL TERMS ===
            "pelagic", "benthic", "demersal", "epipelagic", "mesopelagic",
            "trophic", "food-web", "predator-prey", "top-predator",
            "biomass", "abundance", "biodiversity", "assemblage",
            "recruitment", "spawning", "nursery", "juvenile"
        ]
        
        # Check and add new tokens
        new_tokens = []
        for term in antarctic_terms:
            if term not in self.tokenizer.vocab:
                tokens = self.tokenizer.tokenize(term)
                if len(tokens) > 1:  # Multi-token, consider adding
                    new_tokens.append(term)
        
        if new_tokens:
            print(f"   Adding {len(new_tokens)} domain-specific tokens")
            self.tokenizer.add_tokens(new_tokens)
            self.base_model.resize_token_embeddings(len(self.tokenizer))
            print(f"   New vocabulary size: {len(self.tokenizer)}")
        else:
            print(f"   Domain vocabulary already complete")
    
    def _init_weights(self):
        """Initialize weights for new layers"""
        
        # Initialize shared projection
        nn.init.xavier_uniform_(self.shared_projection.weight)
        nn.init.zeros_(self.shared_projection.bias)
        
        # Initialize classification heads
        for target_name, classifier in self.classifiers.items():
            for module in classifier:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target: Optional[str] = None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the model
        
        Args:
            input_ids: Tokenized input
            attention_mask: Attention mask
            target: Specific target (if None, returns all)
            
        Returns:
            Predictions for target(s)
        """
        
        # Shared feature extraction through SciBERT
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use CLS token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.shared_dropout(pooled_output)
        
        # Shared projection
        shared_features = self.shared_projection(pooled_output)
        shared_features = self.projection_dropout(shared_features)
        
        # Target-specific predictions
        if target is not None:
            # Single target prediction
            return self.classifiers[target](shared_features)
        else:
            # All targets prediction
            predictions = {}
            for target_name in self.target_configs.keys():
                predictions[target_name] = self.classifiers[target_name](shared_features)
            return predictions
    
    def predict_paper(
        self,
        text: str,
        max_length: int = 512,
        threshold: float = 0.5
    ) -> Dict[str, Dict]:
        """
        Predict classifications for a paper
        
        Args:
            text: Enhanced combined text
            max_length: Max sequence length
            threshold: Classification threshold
            
        Returns:
            Predictions with confidence scores
        """
        
        self.eval()
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Move to device
        device = next(self.parameters()).device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
        
        # Process predictions
        predictions = {}
        for target, logits in outputs.items():
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            pred_indices = np.where(probs > threshold)[0]
            
            predictions[target] = {
                'predicted_indices': pred_indices.tolist(),
                'confidence_scores': probs.tolist(),
                'num_predictions': len(pred_indices),
                'threshold_used': threshold
            }
        
        return predictions
    
    def get_model_info(self) -> Dict:
        """Get comprehensive model information"""
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'Enhanced Multi-Target SciBERT',
            'model_version': 'v1.0',
            'base_model': self.base_model.config.name_or_path,
            'target_configs': self.target_configs,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': {
                'shared_dimension': self.shared_dim,
                'spatial_emphasis': self.spatial_emphasis,
                'vocabulary_size': len(self.tokenizer),
                'dropout_rate': 0.12
            }
        }


# Enhanced Focal Loss for imbalanced classification
class EnhancedFocalLoss(nn.Module):
    """Enhanced Focal Loss for handling class imbalance"""
    
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


def create_model(device: str = 'auto') -> EnhancedMultiTargetSciBERT:
    """
    Create Enhanced Multi-Target SciBERT model
    
    Args:
        device: Target device ('auto', 'cuda', 'mps', 'cpu')
        
    Returns:
        Initialized model
    """
    
    # Auto-detect device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Target configurations based on data analysis
    target_configs = {
        'themes': 27,      # Research themes
        'objectives': 9,   # CCAMLR objectives
        'zones': 3,        # Management zones
        'areas': 17        # Top monitoring areas
    }
    
    # Model configuration
    model_params = {
        'dropout_rate': 0.12,
        'shared_dim': 256,
        'spatial_emphasis': True
    }
    
    # Create model
    model = EnhancedMultiTargetSciBERT(
        target_configs=target_configs,
        **model_params
    )
    
    # Move to device
    model = model.to(device)
    
    return model


def test_model():
    """Test model functionality"""
    
    print("Testing Enhanced Multi-Target SciBERT")
    print("=" * 50)
    
    try:
        # Create model
        model = create_model(device='cpu')
        
        # Test with sample enhanced text
        sample_text = (
            "[TITLE] Krill population dynamics in Ross Sea "
            "[ABSTRACT] This study examines euphausia superba distribution "
            "[KEYWORDS] krill, population dynamics, Ross Sea "
            "[SPATIAL] Continental shelf slope "
            "[ZONES] General Protection Zone "
            "[AREAS] Ross Sea Polynya Western Ross Sea "
            "[OBJECTIVES] promote research conserve natural ecological structure"
        )
        
        # Test tokenization
        encoding = model.tokenizer(
            sample_text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        print(f"Tokenization Test:")
        print(f"   Input length: {len(sample_text)} chars")
        print(f"   Token count: {encoding['input_ids'].shape[1]}")
        print(f"   Vocabulary size: {len(model.tokenizer)}")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask']
            )
        
        print(f"\nMulti-Target Output Test:")
        for target, logits in outputs.items():
            probs = torch.sigmoid(logits)
            print(f"   {target}: {logits.shape} -> probs [{probs.min():.3f}, {probs.max():.3f}]")
        
        # Test prediction function
        predictions = model.predict_paper(sample_text, threshold=0.42)
        
        print(f"\nPrediction Test:")
        for target, pred_info in predictions.items():
            print(f"   {target}: {pred_info['num_predictions']} predictions")
        
        # Model info
        info = model.get_model_info()
        print(f"\nModel Info:")
        print(f"   Type: {info['model_type']}")
        print(f"   Version: {info['model_version']}")
        print(f"   Parameters: {info['total_parameters']:,}")
        
        print(f"\nModel test passed!")
        print(f"   All targets working: {list(model.target_configs.keys())}")
        print(f"   Domain vocabulary extended")
        print(f"   Ready for training")
        
        return model
        
    except Exception as e:
        print(f"Model test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("Enhanced Multi-Target SciBERT")
    print("=" * 60)
    
    # Test model
    model = test_model()
    
    if model is not None:
        print(f"\nModel ready for training!")
        print(f"   Use with train_model.py for multi-target classification")
        print(f"   Supports themes, objectives, zones, and areas classification")
    else:
        print(f"\nModel test failed")
