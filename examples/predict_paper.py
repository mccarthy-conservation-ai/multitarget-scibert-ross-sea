"""
Enhanced Paper Classification System for Multi-Target SciBERT with Ensemble Support
Compatible with enhanced training pipeline
Supports PDF upload, text extraction, and comprehensive paper analysis
WITH ANTARCTIC RESEARCH DETECTION - Ross Sea MPA Specific
Enhanced detection for geological/geophysical research papers
Rejects: Arctic papers, Antarctica travel books, non-Antarctic content
Includes ensemble prediction for improved consistency
Updated for 5-model ensemble with hierarchical voting
WITH EXPERIMENT CONFIG SUPPORT - Respects semantic/full mode settings
UPDATED FOR ENHANCED METADATA FORMAT - Uses same preprocessing as training
"""

import os
import sys
import torch
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import re
import glob

warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from enhanced_scibert_model import EnhancedMultiTargetSciBERT
    from data_preprocessing import EnhancedDataPreprocessor
    from training_config import get_config, get_target_importance
    from data_loader import load_data_with_fallback
    from transformers import AutoTokenizer
    # Import experiment configuration
    from experiment_config import get_active_targets, should_use_target, MODE
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure all src/ files are present and working")
    sys.exit(1)

# PDF processing imports
try:
    import pdfplumber
    import PyPDF2
    import pymupdf as fitz
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("PDF processing libraries not available. Install with:")
    print("   pip install pdfplumber PyPDF2 pymupdf")

# Define problematic classes that need stricter voting (from train_model.py)
PROBLEMATIC_CLASSES = {
    'themes': {
        'Exploitation effects on toothfish': 4,  # Require 4/5 votes
        'Climate-driven ocean circulation changes': 4,
        'Ocean acidification': 4,
        # Default for other theme classes
        'default': 3
    },
    'objectives': {
        'default': 3  # Standard majority for all objectives
    },
    'zones': {
        'default': 3
    },
    'areas': {
        'default': 3
    }
}


class PaperExtractor:
    """Extract structured information from research papers"""
    
    def extract_from_pdf(self, pdf_path: str) -> Dict[str, str]:
        """
        Extract text and metadata from PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with extracted paper information
        """
        
        if not PDF_AVAILABLE:
            print("PDF processing not available. Please install required libraries.")
            return {}
        
        print(f"Extracting text from PDF: {os.path.basename(pdf_path)}")
        
        extracted = {
            'title': '',
            'abstract': '',
            'keywords': '',
            'full_text': '',
            'authors': '',
            'year': '',
            'doi': '',
            'journal': '',
            'references': '',
            'methods': '',
            'results': '',
            'conclusion': ''
        }
        
        try:
            # Try pdfplumber first (best for layout preservation)
            with pdfplumber.open(pdf_path) as pdf:
                full_text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n"
                
                extracted['full_text'] = full_text
                print(f"   Extracted {len(full_text)} characters using pdfplumber")
                
        except Exception as e:
            print(f"   pdfplumber failed: {e}")
            
            try:
                # Fallback to PyPDF2
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    full_text = ""
                    
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            full_text += page_text + "\n"
                    
                    extracted['full_text'] = full_text
                    print(f"   Extracted {len(full_text)} characters using PyPDF2")
                    
            except Exception as e2:
                print(f"   PyPDF2 failed: {e2}")
                
                try:
                    # Final fallback to pymupdf
                    doc = fitz.open(pdf_path)
                    full_text = ""
                    
                    for page in doc:
                        page_text = page.get_text()
                        if page_text:
                            full_text += page_text + "\n"
                    
                    doc.close()
                    extracted['full_text'] = full_text
                    print(f"   Extracted {len(full_text)} characters using pymupdf")
                    
                except Exception as e3:
                    print(f"   All PDF extraction methods failed: {e3}")
                    return extracted
        
        # Parse structured information from text
        if extracted['full_text']:
            self._parse_paper_structure(extracted)
        
        return extracted
    
    def extract_from_text(self, text_content: str) -> Dict[str, str]:
        """
        Extract structured information from plain text
        
        Args:
            text_content: Full text content of paper
            
        Returns:
            Dictionary with extracted paper information
        """
        
        extracted = {
            'title': '',
            'abstract': '',
            'keywords': '',
            'full_text': text_content,
            'authors': '',
            'year': '',
            'doi': '',
            'journal': '',
            'references': '',
            'methods': '',
            'results': '',
            'conclusion': ''
        }
        
        self._parse_paper_structure(extracted)
        return extracted
    
    def _parse_paper_structure(self, extracted: Dict[str, str]):
        """Parse paper structure from full text"""
        
        text = extracted['full_text']
        lines = text.split('\n')
        
        # Extract title (usually first substantial line)
        if not extracted['title']:
            for line in lines[:10]:  # Check first 10 lines
                line = line.strip()
                if len(line) > 20 and not line.isupper() and '.' not in line[:20]:
                    extracted['title'] = line
                    break
        
        # Extract abstract
        abstract_patterns = [
            r'ABSTRACT\s*:?\s*(.*?)(?=\n\s*(?:KEYWORDS?|INTRODUCTION|1\.|Keywords:|Introduction))',
            r'Abstract\s*:?\s*(.*?)(?=\n\s*(?:Keywords?|Introduction|1\.))',
            r'SUMMARY\s*:?\s*(.*?)(?=\n\s*(?:KEYWORDS?|INTRODUCTION|1\.))'
        ]
        
        for pattern in abstract_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                abstract = match.group(1).strip()
                # Clean up abstract
                abstract = re.sub(r'\s+', ' ', abstract)
                if len(abstract) > 50:  # Reasonable abstract length
                    extracted['abstract'] = abstract
                    break
        
        # Extract keywords
        keyword_patterns = [
            r'KEYWORDS?\s*:?\s*(.*?)(?=\n\s*[A-Z])',
            r'Keywords?\s*:?\s*(.*?)(?=\n\s*[A-Z])',
            r'Key words?\s*:?\s*(.*?)(?=\n\s*[A-Z])'
        ]
        
        for pattern in keyword_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                keywords = match.group(1).strip()
                # Clean up keywords
                keywords = re.sub(r'\s+', ' ', keywords)
                if len(keywords) > 5:
                    extracted['keywords'] = keywords
                    break
        
        # Extract authors (patterns near the title)
        author_patterns = [
            r'(?:Authors?|By)\s*:?\s*(.*?)(?=\n)',
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]*\.?)*(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]*\.?)*)*)',
        ]
        
        first_few_lines = '\n'.join(lines[:15])
        for pattern in author_patterns:
            match = re.search(pattern, first_few_lines, re.MULTILINE)
            if match:
                authors = match.group(1).strip()
                if len(authors) > 5 and len(authors) < 200:
                    extracted['authors'] = authors
                    break
        
        # Extract year
        year_match = re.search(r'\b(19|20)\d{2}\b', text)
        if year_match:
            extracted['year'] = year_match.group(0)
        
        # Extract DOI
        doi_match = re.search(r'DOI\s*:?\s*(10\.\d+/[^\s]+)', text, re.IGNORECASE)
        if doi_match:
            extracted['doi'] = doi_match.group(1)
        
        print(f"   Parsed structure: Title: {bool(extracted['title'])}, "
              f"Abstract: {bool(extracted['abstract'])}, Keywords: {bool(extracted['keywords'])}")


class MetadataExtractor:
    """Extract metadata from paper text for enhanced preprocessing"""
    
    def __init__(self):
        # Common Antarctic species
        self.species_patterns = {
            'Dissostichus mawsoni': ['Dissostichus mawsoni', 'D. mawsoni', 'Antarctic toothfish'],
            'Dissostichus eleginoides': ['Dissostichus eleginoides', 'D. eleginoides', 'Patagonian toothfish'],
            'Euphausia superba': ['Euphausia superba', 'E. superba', 'Antarctic krill', 'krill'],
            'Pleuragramma antarctica': ['Pleuragramma antarctica', 'P. antarctica', 'Antarctic silverfish'],
            'Pygoscelis adeliae': ['Pygoscelis adeliae', 'P. adeliae', 'Adelie penguin'],
            'Aptenodytes forsteri': ['Aptenodytes forsteri', 'A. forsteri', 'Emperor penguin'],
            'Leptonychotes weddellii': ['Leptonychotes weddellii', 'L. weddellii', 'Weddell seal'],
            'Ommatophoca rossii': ['Ommatophoca rossii', 'O. rossii', 'Ross seal'],
            'Orcinus orca': ['Orcinus orca', 'O. orca', 'killer whale', 'orca'],
            'Balaenoptera musculus': ['Balaenoptera musculus', 'B. musculus', 'blue whale'],
        }
        
        # Research methods
        self.method_keywords = {
            'Acoustic Methods': ['acoustic', 'sonar', 'echo', 'hydroacoustic', 'echosounder'],
            'Tagging/Tracking': ['tag', 'tracking', 'telemetry', 'satellite tag', 'pop-up tag'],
            'Visual Census': ['visual census', 'visual survey', 'transect', 'observation'],
            'Genetic/Molecular': ['genetic', 'DNA', 'molecular', 'genomic', 'sequencing'],
            'Remote Sensing': ['remote sensing', 'satellite', 'imagery', 'MODIS', 'Sentinel'],
            'Oceanographic Sampling': ['CTD', 'oceanographic', 'water sampling', 'profiler'],
            'Modeling': ['model', 'simulation', 'forecast', 'projection', 'scenario'],
            'Chemical Analysis': ['chemical', 'pH', 'alkalinity', 'ocean acidification'],
            'Stock Assessment': ['stock assessment', 'biomass', 'population dynamics'],
        }
        
        # Temporal scales
        self.temporal_patterns = {
            'Multi-year': [r'\d{4}-\d{4}', r'long-term', r'decadal', r'inter-annual'],
            'Annual': [r'annual', r'yearly', r'year-round'],
            'Seasonal': [r'seasonal', r'summer', r'winter', r'spring', r'autumn'],
            'Short-term': [r'short-term', r'weekly', r'daily', r'cruise'],
        }
        
        # Named locations
        self.location_keywords = [
            'Ross Sea', 'McMurdo', 'Terra Nova', 'Balleny', 'Scott Base',
            'Cape Adare', 'Victoria Land', 'Ross Ice Shelf', 'McMurdo Sound',
            'Admiralty Range', 'Transantarctic Mountains', 'Ross Island',
            'Cape Crozier', 'Cape Bird', 'Cape Royds', 'Erebus Bay',
            'Drygalski Ice Tongue', 'Coulman Island', 'Cape Washington',
            'Ross Sea Polynya', 'Terra Nova Bay Polynya', 'McMurdo Ice Shelf'
        ]
        
    def extract_metadata_from_paper(self, paper_data: Dict[str, str]) -> Dict[str, str]:
        """Extract metadata from paper text to match training format"""
        
        # Combine all text fields for searching
        search_text = ' '.join([
            paper_data.get('title', ''),
            paper_data.get('abstract', ''),
            paper_data.get('keywords', ''),
            paper_data.get('full_text', '')[:10000] if 'full_text' in paper_data else ''
        ])
        
        # Make it case-insensitive for better matching
        search_text_lower = search_text.lower()
        
        # Extract species list
        found_species = []
        for species, patterns in self.species_patterns.items():
            if any(pattern.lower() in search_text_lower for pattern in patterns):
                found_species.append(species)
        
        # Extract methods
        found_methods = []
        for method, keywords in self.method_keywords.items():
            if any(keyword in search_text_lower for keyword in keywords):
                found_methods.append(method)
        
        # Extract temporal scale
        temporal_scale = ''
        for scale, patterns in self.temporal_patterns.items():
            for pattern in patterns:
                if re.search(pattern, search_text_lower):
                    temporal_scale = scale
                    break
            if temporal_scale:
                break
        
        # Extract locations
        found_locations = []
        for location in self.location_keywords:
            if location.lower() in search_text_lower:
                found_locations.append(location)
        
        # Extract research dates if present
        year_pattern = r'\b(19|20)\d{2}\b'
        years = re.findall(year_pattern, search_text)
        research_start_date = ''
        research_end_date = ''
        if len(years) >= 2:
            years_int = sorted([int(y + d) for y, d in re.findall(r'(19|20)(\d{2})', search_text)])
            if years_int:
                research_start_date = str(years_int[0])
                research_end_date = str(years_int[-1])
        
        # Determine spatial scale based on locations and text
        spatial_scale = ''
        if any(word in search_text_lower for word in ['circumpolar', 'southern ocean', 'antarctica-wide']):
            spatial_scale = 'Circumpolar'
        elif any(word in search_text_lower for word in ['regional', 'ross sea region', 'ross sea sector']):
            spatial_scale = 'Regional'
        elif any(word in search_text_lower for word in ['local', 'site', 'colony', 'bay']):
            spatial_scale = 'Local'
        
        # Return metadata in format expected by EnhancedDataPreprocessor
        metadata = {
            'Species_List': '; '.join(found_species),
            'Method_Tags': '; '.join(found_methods),
            'Named_Features': '; '.join(found_locations),
            'Temporal_Scale': temporal_scale,
            'Spatial_Scale': spatial_scale,
            'Research_start_date': research_start_date,
            'Research_end_date': research_end_date,
            'Data_Availability': '',  # Cannot determine from text
        }
        
        return metadata


class EnsembleClassifier:
    """Ensemble of multiple trained models for robust predictions with voting strategies"""
    
    def __init__(self, model_paths: List[str], device: str = 'auto'):
        """
        Initialize ensemble classifier
        
        Args:
            model_paths: List of paths to trained models
            device: Device to use ('auto', 'cuda', 'mps', 'cpu')
        """
        self.models = []
        self.configs = []
        self.model_paths = model_paths
        self.class_mappings = {}  # Will be set by PaperClassifier
        
        # Auto-detect device
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = device
        print(f"Loading ensemble of {len(model_paths)} models on {device}")
        
        # Load each model
        for i, model_path in enumerate(model_paths):
            try:
                model, config = self._load_single_model(model_path)
                self.models.append(model)
                self.configs.append(config)
                print(f"   Model {i+1}: {os.path.basename(model_path)} loaded")
            except Exception as e:
                print(f"   Failed to load model {i+1}: {e}")
                continue
        
        if not self.models:
            raise ValueError("No models could be loaded for ensemble")
        
        print(f"Ensemble loaded: {len(self.models)} models ready")
        
        # Setup tokenizer (use first model's config)
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        antarctic_terms = [
            "dissostichus-mawsoni", "euphausia-superba", "pleuragramma-antarctica",
            "general-protection-zone", "special-research-zone", "krill-research-zone",
            "ross-sea-polynya", "mcmurdo-sound", "balleny-islands", "ccamlr"
        ]
        new_tokens = [term for term in antarctic_terms if term not in self.tokenizer.vocab]
        if new_tokens:
            self.tokenizer.add_tokens(new_tokens)
        
        # Load problematic classes configuration
        self.PROBLEMATIC_CLASSES = PROBLEMATIC_CLASSES
    
    def _load_single_model(self, model_path: str) -> Tuple:
        """Load a single model and return model and config"""
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        config = checkpoint.get('config', {})
        target_configs = checkpoint.get('target_configs', {
            'themes': 27, 'objectives': 9, 'zones': 3, 'areas': 17
        })
        
        # Create model
        model = EnhancedMultiTargetSciBERT(
            target_configs=target_configs,
            dropout_rate=config.get('dropout', 0.12),
            shared_dim=256,
            spatial_emphasis=config.get('spatial_emphasis', True)
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model, config
    
    def predict_ensemble(self, enhanced_text: str, threshold: float = 0.42, voting_strategy: str = 'hierarchical') -> Dict:
        """
        Make ensemble predictions using specified voting strategy
        
        Args:
            enhanced_text: Text to classify
            threshold: Classification threshold
            voting_strategy: 'averaging', 'majority', or 'hierarchical'
            
        Returns:
            Predictions based on voting strategy
        """
        
        # Tokenize text
        encoding = self.tokenizer(
            enhanced_text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Collect predictions from all models
        all_probabilities = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
                
                model_probs = {}
                for target in outputs.keys():
                    probs = torch.sigmoid(outputs[target])
                    model_probs[target] = probs
                    
                all_probabilities.append(model_probs)
        
        # Apply voting strategy
        if voting_strategy == 'hierarchical':
            return self._hierarchical_voting(all_probabilities, threshold)
        elif voting_strategy == 'majority':
            return self._majority_voting(all_probabilities, threshold)
        else:
            return self._average_voting(all_probabilities, threshold)
    
    def _hierarchical_voting(self, all_probabilities: List[Dict], threshold: float) -> Dict:
        """Apply hierarchical voting with class-specific thresholds"""
        
        final_predictions = {}
        
        for target in all_probabilities[0].keys():
            # Stack predictions from all models
            target_probs = torch.stack([p[target] for p in all_probabilities])
            
            # Get class names for this target
            class_names = self.class_mappings.get(target, [])
            
            # Apply class-specific voting
            class_predictions = []
            class_confidences = []
            
            for class_idx in range(target_probs.shape[2]):
                # Get class name
                if class_idx < len(class_names):
                    class_name = class_names[class_idx]
                else:
                    class_name = f"{target}_{class_idx}"
                
                # Determine required votes for this class
                if target in self.PROBLEMATIC_CLASSES:
                    if class_name in self.PROBLEMATIC_CLASSES[target]:
                        required_votes = self.PROBLEMATIC_CLASSES[target][class_name]
                    else:
                        required_votes = self.PROBLEMATIC_CLASSES[target].get('default', 3)
                else:
                    required_votes = 3  # Default majority
                
                # Count votes (models predicting > threshold)
                votes = (target_probs[:, 0, class_idx] > threshold).sum().item()
                
                # Average confidence
                avg_confidence = target_probs[:, 0, class_idx].mean().item()
                std_confidence = target_probs[:, 0, class_idx].std().item()
                
                # Prediction based on voting
                predicted = votes >= required_votes
                
                class_predictions.append(predicted)
                class_confidences.append({
                    'confidence': avg_confidence,
                    'std': std_confidence,
                    'votes': votes,
                    'required_votes': required_votes,
                    'num_models': len(all_probabilities)
                })
            
            final_predictions[target] = {
                'predictions': class_predictions,
                'confidences': class_confidences,
                'voting_method': 'hierarchical',
                'num_models': len(all_probabilities)
            }
        
        return final_predictions
    
    def _majority_voting(self, all_probabilities: List[Dict], threshold: float) -> Dict:
        """Apply simple majority voting"""
        
        final_predictions = {}
        
        for target in all_probabilities[0].keys():
            # Stack predictions from all models
            target_probs = torch.stack([p[target] for p in all_probabilities])
            
            # Count votes for each class
            votes = (target_probs > threshold).sum(dim=0)[0]  # Sum across models
            
            # Majority decision
            predictions = (votes >= (len(all_probabilities) + 1) // 2)  # Majority
            
            # Average confidence
            avg_probs = target_probs.mean(dim=0)[0]
            std_probs = target_probs.std(dim=0)[0]
            
            final_predictions[target] = {
                'predictions': predictions.cpu().numpy(),
                'probs': avg_probs.cpu().numpy(),
                'std': std_probs.cpu().numpy(),
                'votes': votes.cpu().numpy(),
                'voting_method': 'majority',
                'num_models': len(all_probabilities)
            }
        
        return final_predictions
    
    def _average_voting(self, all_probabilities: List[Dict], threshold: float) -> Dict:
        """Original averaging method (backward compatibility)"""
        
        averaged_predictions = {}
        
        for target in all_probabilities[0].keys():
            # Stack predictions and take mean
            stacked_predictions = torch.stack([p[target] for p in all_probabilities], dim=0)
            averaged_probs = stacked_predictions.mean(dim=0)[0]
            std_probs = stacked_predictions.std(dim=0)[0]
            
            averaged_predictions[target] = {
                'probs': averaged_probs.cpu().numpy(),
                'std': std_probs.cpu().numpy(),
                'num_models': len(all_probabilities),
                'voting_method': 'averaging'
            }
        
        return averaged_predictions
    
    def get_ensemble_info(self) -> Dict:
        """Get information about the ensemble"""
        return {
            'num_models': len(self.models),
            'model_paths': self.model_paths,
            'device': self.device,
            'voting_strategies': ['averaging', 'majority', 'hierarchical'],
            'problematic_classes': self.PROBLEMATIC_CLASSES
        }


class PaperClassifier:
    """
    Enhanced paper classification system for Ross Sea Antarctic Research
    WITH ANTARCTIC RESEARCH DETECTION - Rejects Arctic, travel books, non-research
    Enhanced detection for geological/geophysical research papers
    Supports ensemble predictions for improved consistency
    Updated for 5-model ensemble with hierarchical voting
    WITH EXPERIMENT CONFIG SUPPORT - Respects semantic/full mode settings
    USES SAME PREPROCESSING AS TRAINING - EnhancedDataPreprocessor
    """
    
    def __init__(self, model_path: str = None, ensemble_mode: bool = False,
                 ensemble_size: int = 3, voting_strategy: str = 'hierarchical'):
        """
        Initialize the classifier
        
        Args:
            model_path: Path to specific model (if None, auto-detect)
            ensemble_mode: Whether to use ensemble of models
            ensemble_size: Number of models to use in ensemble
            voting_strategy: Voting strategy for ensemble ('averaging', 'majority', 'hierarchical')
        """
        
        self.ensemble_mode = ensemble_mode
        self.ensemble_size = ensemble_size
        self.voting_strategy = voting_strategy
        self.model = None
        self.ensemble = None
        self.tokenizer = None
        self.config = None
        self.target_configs = None
        self.extractor = PaperExtractor()
        self.metadata_extractor = MetadataExtractor()
        
        # Initialize the same preprocessor used in training
        self.data_preprocessor = EnhancedDataPreprocessor()
        
        # Get active targets from experiment config
        self.active_targets = get_active_targets()
        
        # Load class mappings for classification output
        self.class_mappings = {
            'themes': [
                'Bioregionalisation & biodiversity mapping',  # Index 0 (Theme 1)
                'Dependence on coastal habitats',  # Index 1 (Theme 10)
                'Toothfish population structure & roles',  # Index 2 (Theme 11)
                'Silverfish spawning & nursery factors',  # Index 3 (Theme 12)
                'Trophic productivity at ice edges',  # Index 4 (Theme 13)
                'Toothfish recruitment variation',  # Index 5 (Theme 14)
                'Under-ice & ice shelf habitat importance',  # Index 6 (Theme 15)
                'Krill swarming & top-down effects',  # Index 7 (Theme 16)
                'Seal & penguin foraging movements',  # Index 8 (Theme 17)
                'Demersal fish community ecology',  # Index 9 (Theme 18)
                'Exploitation effects on toothfish',  # Index 10 (Theme 19)
                'Physical & biological habitat changes',  # Index 11 (Theme 2)
                'Toothfish spawning on Ross slope',  # Index 12 (Theme 20)
                'Benthic community structure & function',  # Index 13 (Theme 21)
                'Krill role in demersal slope environment',  # Index 14 (Theme 22)
                'Toothfish spawning migrations N. Ross',  # Index 15 (Theme 28)
                'Toothfish spawning behaviour/dynamics',  # Index 16 (Theme 29)
                'Functional ecology processes',  # Index 17 (Theme 3)
                'Toothfish recruitment factors',  # Index 18 (Theme 30)
                'Toothfish stocks on seamounts & islands',  # Index 19 (Theme 34)
                'Toothfish distribution & biomass NW Ross',  # Index 20 (Theme 36)
                'Evolutionary biology processes',  # Index 21 (Theme 4)
                'Toothfish stock assessment methods',  # Index 22 (Theme 5)
                'Krill & silverfish population trends',  # Index 23 (Theme 6)
                'Prey availability effects on predators',  # Index 24 (Theme 7)
                'Toothfish & predator distributions',  # Index 25 (Theme 8)
                'Toothfish vertical & seasonal distribution'  # Index 26 (Theme 9)
            ],
            'objectives': [
                'conserve natural ecological structure',  # Index 0 (Objective i)
                'promote research',  # Index 1 (Objective iii)
                'representativeness of benthic and pelagic environments',  # Index 2 (Objective iv)
                'D. mawsoni life cycle areas',  # Index 3 (Objective ix)
                'large-scale ecosystem processes/areas',  # Index 4 (Objective v)
                'trophically dominant pelagic prey species',  # Index 5 (Objective vi)
                'key top predator foraging distributions',  # Index 6 (Objective vii)
                'coastal/localized areas of particular ecosystem importance',  # Index 7 (Objective viii)
                'rare or vulnerable benthic habitats'  # Index 8 (Objective x)
            ],
            'zones': [
                'General Protection Zone',  # Index 0 (GPZ)
                'Krill Research Zone',  # Index 1 (KRZ)
                'Special Research Zone'  # Index 2 (SRZ)
            ],
            'areas': [
                'All RSRMPA',  # Index 0
                'Cape Crozier',  # Index 1
                'Central Ross Sea',  # Index 2
                'Eastern Ross Sea',  # Index 3
                'Erebus Bay',  # Index 4
                'Marginal Ice Zone',  # Index 5
                'McMurdo Sound',  # Index 6
                'Northern Ross Sea',  # Index 7
                'Ross Ice Shelf',  # Index 8
                'Ross Island',  # Index 9
                'Ross Sea Continental Shelf',  # Index 10
                'Ross Sea Polynya',  # Index 11
                'Southern Ross Sea',  # Index 12
                'Southwestern Ross Sea',  # Index 13
                'Terra Nova Bay',  # Index 14
                'Terra Nova Bay Polynya',  # Index 15
                'Western Ross Sea'  # Index 16
            ]
        }
        
        print("Enhanced Multi-Target SciBERT Paper Classifier")
        print("ROSS SEA ANTARCTIC RESEARCH DETECTION")
        print("Enhanced geological/geophysical research detection")
        print(f"EXPERIMENT MODE: {MODE.upper()}")
        print(f"Active targets: {self.active_targets}")
        print("Using EnhancedDataPreprocessor (same as training)")
        if ensemble_mode:
            print(f"ENSEMBLE MODE: Using up to {ensemble_size} models for robust predictions")
            print(f"Voting Strategy: {voting_strategy}")
        print("=" * 60)
        
        self._initialize_models(model_path)
    
    def _initialize_models(self, model_path: str = None):
        """Initialize models (single or ensemble)"""
        
        if self.ensemble_mode:
            # Auto-detect ensemble models
            model_dir = 'models'
            if os.path.exists(model_dir):
                # Look for MODE-SPECIFIC ensemble models
                if MODE == "semantic":
                    ensemble_files = sorted(glob.glob(os.path.join(model_dir, '*semantic_ensemble*.pt')))
                    print(f"Looking for semantic models: found {len(ensemble_files)}")
                else:
                    ensemble_files = sorted(glob.glob(os.path.join(model_dir, '*full_ensemble*.pt')))
                    print(f"Looking for full models: found {len(ensemble_files)}")
                
                if len(ensemble_files) >= 2:
                    # Prefer 5 models if available
                    if len(ensemble_files) >= 5 and self.ensemble_size == 3:
                        print("Found 5 ensemble models - upgrading to 5-model ensemble for better accuracy")
                        self.ensemble_size = 5
                    
                    # Use ensemble models
                    model_paths = ensemble_files[:self.ensemble_size]
                    self.ensemble = EnsembleClassifier(model_paths)
                    self.ensemble.class_mappings = self.class_mappings  # Pass mappings
                    self.config = self.ensemble.configs[0]  # Use first model's config
                    self.target_configs = self.ensemble.models[0].target_configs
                    print(f"Ensemble mode activated: {len(model_paths)} models loaded")
                    print(f"Using {self.voting_strategy} voting strategy")
                    print(f"Models are from {MODE} training")
                    return
                else:
                    # Fallback to regular models based on mode
                    if MODE == "semantic":
                        pattern = '*semantic*.pt'
                    else:
                        pattern = '*full*.pt'
                    regular_files = sorted(glob.glob(os.path.join(model_dir, pattern)))
                    
                    # If still no mode-specific files, try any .pt files
                    if len(regular_files) < 2:
                        regular_files = sorted(glob.glob(os.path.join(model_dir, '*.pt')))
                    
                    if len(regular_files) >= 2:
                        model_paths = regular_files[-self.ensemble_size:]  # Use most recent
                        self.ensemble = EnsembleClassifier(model_paths)
                        self.ensemble.class_mappings = self.class_mappings
                        self.config = self.ensemble.configs[0]
                        self.target_configs = self.ensemble.models[0].target_configs
                        print(f"Ensemble mode: Using {len(model_paths)} regular models")
                        return
            
            print("Warning: Not enough models for ensemble, falling back to single model")
            self.ensemble_mode = False
        
        # Single model mode
        self._load_single_model(model_path)
    
    def _is_antarctic_domain(self, paper_data: Dict[str, str]) -> Tuple[bool, str, float]:
        """
        Check if paper is in Antarctic/marine domain before classification
        (Basic domain filter - used by research paper filter)
        """
        
        # Combine all text for analysis
        full_text = f"{paper_data.get('title', '')} {paper_data.get('abstract', '')} {paper_data.get('keywords', '')} {paper_data.get('full_text', '')}"
        full_text_lower = full_text.lower()
        
        # Core Antarctic keywords (high confidence indicators)
        antarctic_core_keywords = {
            'antarctic', 'antarctica', 'ross sea', 'southern ocean', 'antarctic peninsula',
            'mcmurdo', 'weddell sea', 'terra nova bay', 'ross ice shelf', 'antarctic circumpolar',
            'ccamlr', 'toothfish', 'dissostichus mawsoni', 'antarctic krill', 'euphausia superba',
            'adelie penguin', 'emperor penguin', 'chinstrap penguin', 'leopard seal', 'weddell seal',
            'pleuragramma antarctica', 'silverfish antarctic', 'balleny islands', 'rsrmpa',
            'ross sea marine protected area', 'mpa', 'erebus bay', 'mcmurdo sound'
        }
        
        # Marine/polar keywords (moderate confidence)
        marine_polar_keywords = {
            'marine protected area', 'marine conservation', 'fisheries', 'krill', 'penguin',
            'seal', 'whale', 'polar', 'ice shelf', 'polynya', 'sea ice', 'ice edge',
            'southern hemisphere', 'subantarctic', 'circumpolar', 'pelagic', 'benthic',
            'oceanographic', 'marine ecosystem', 'polar research', 'conservation biology',
            'pack ice', 'fast ice', 'marginal ice zone', 'ice-associated', 'polar waters'
        }
        
        # Conflicting geography keywords (exclusion indicators)
        conflicting_geography = {
            'africa', 'african', 'kenya', 'nigeria', 'sahara', 'serengeti', 'kilimanjaro',
            'asia', 'asian', 'china', 'india', 'himalaya', 'mekong', 'yangtze',
            'europe', 'european', 'mediterranean', 'north sea', 'baltic',
            'north america', 'america', 'california', 'texas', 'canada',
            'south america', 'amazon', 'brazil', 'andes', 'patagonia',
            'australia', 'australian', 'great barrier reef', 'tasmania',
            'arctic', 'greenland', 'norway', 'iceland', 'svalbard', 'finland',
            'tropical', 'equatorial', 'caribbean', 'pacific northwest', 'atlantic',
            'indian ocean', 'pacific ocean', 'alaska'
        }
        
        # Non-marine domains (exclusion indicators)
        non_marine_domains = {
            'agriculture', 'farming', 'crop', 'livestock', 'drone', 'uav', 'unmanned aerial',
            'terrestrial', 'forest', 'desert', 'mountain', 'urban', 'city', 'rural',
            'medicine', 'medical', 'hospital', 'clinical', 'patient', 'drug', 'healthcare',
            'technology', 'computer science', 'artificial intelligence', 'machine learning',
            'engineering', 'manufacturing', 'industrial', 'automotive', 'aerospace',
            'economics', 'finance', 'business', 'marketing', 'sociology', 'psychology',
            'education', 'pedagogy', 'literature', 'history', 'philosophy', 'smallholder'
        }
        
        # Calculate scores
        antarctic_score = sum(1 for keyword in antarctic_core_keywords if keyword in full_text_lower)
        marine_score = sum(1 for keyword in marine_polar_keywords if keyword in full_text_lower)
        conflict_score = sum(1 for keyword in conflicting_geography if keyword in full_text_lower)
        non_marine_score = sum(1 for keyword in non_marine_domains if keyword in full_text_lower)
        
        # Extra weight for title/abstract matches
        title_text = paper_data.get('title', '').lower()
        abstract_text = paper_data.get('abstract', '').lower()
        
        title_antarctic = sum(1 for keyword in antarctic_core_keywords if keyword in title_text) * 2
        abstract_antarctic = sum(1 for keyword in antarctic_core_keywords if keyword in abstract_text) * 1.5
        
        # Weighted domain score
        domain_score = (
            (antarctic_score + title_antarctic + abstract_antarctic) * 3.0 +
            marine_score * 1.0 -
            conflict_score * 2.0 -
            non_marine_score * 1.5
        )
        
        # Normalize to 0-1 scale
        max_possible_positive = (len(antarctic_core_keywords) * 3.0) + (len(marine_polar_keywords) * 1.0)
        confidence = max(0, min(1, domain_score / max_possible_positive))
        
        # Decision logic
        total_antarctic = antarctic_score + title_antarctic + abstract_antarctic
        
        if total_antarctic >= 3:  # Strong Antarctic indicators
            return True, f"Strong Antarctic indicators found (score: {total_antarctic})", confidence
        elif total_antarctic >= 1 and conflict_score == 0 and domain_score >= 2:
            return True, f"Antarctic domain with supporting terms (score: {domain_score:.1f})", confidence
        elif conflict_score >= 3:  # Strong geographic conflicts
            return False, f"Conflicting geography detected: {conflict_score} non-Antarctic regions", confidence
        elif non_marine_score >= 4:  # Strong non-marine domain
            return False, f"Non-marine domain detected: {non_marine_score} non-marine terms", confidence
        elif domain_score < 0:  # Negative score
            return False, f"Negative domain relevance (score: {domain_score:.1f})", confidence
        elif domain_score < 1 and total_antarctic == 0:  # Very low domain relevance
            return False, f"No Antarctic indicators found (score: {domain_score:.1f})", confidence
        else:
            # Conservative threshold for borderline cases
            is_antarctic = domain_score >= 2.0
            return is_antarctic, f"Borderline case (score: {domain_score:.1f})", confidence
    
    def _is_antarctic_research_paper(self, paper_data: Dict[str, str]) -> Tuple[bool, str, float]:
        """
        Enhanced Antarctic research detection - handles geological papers better
        Specifically improved for geological/geophysical research papers
        """
        
        # First run existing domain check
        is_antarctic, antarctic_reason, antarctic_confidence = self._is_antarctic_domain(paper_data)
        
        if not is_antarctic:
            return False, antarctic_reason, antarctic_confidence
        
        # Enhanced research paper detection
        full_text = f"{paper_data.get('title', '')} {paper_data.get('abstract', '')} {paper_data.get('keywords', '')} {paper_data.get('full_text', '')}"
        full_text_lower = full_text.lower()
        
        # Research indicators - EXPANDED for geological research
        research_indicators = {
            # Traditional research terms
            'abstract', 'methods', 'methodology', 'results', 'discussion', 'conclusion',
            'data', 'analysis', 'study', 'research', 'investigation', 'experiment',
            'observation', 'measurement', 'statistical', 'significant', 'correlation',
            'hypothesis', 'model', 'population', 'sample', 'distribution', 'abundance',
            'biomass', 'density', 'survey', 'monitoring', 'assessment', 'evaluation',
            'conservation', 'management', 'policy', 'ecosystem', 'species', 'habitat',
            'peer review', 'journal', 'publication', 'citation', 'doi',
            
            # GEOLOGICAL/GEOPHYSICAL research terms
            'seismic', 'geophysical', 'acoustic', 'velocity', 'reflector',
            'bathymetric', 'topographic', 'geological', 'geophysics',
            'multichannel seismic', 'seismic profiles', 'morpho-bathymetric',
            'acoustic impedance', 'velocity analysis', 'seismic data',
            'bottom simulating reflector', 'mud volcanoes', 'gas hydrate',
            'seafloor geology', 'marine geology', 'sedimentary', 'stratigraphy',
            'tectonic', 'structural', 'geophysical evidence', 'acoustic anomalies',
            'magnetic', 'gravity', 'tomography', 'subsurface'
        }
        
        # More conservative non-research list
        non_research_indicators = {
            'personal journey', 'expedition memoir', 'personal account', 'travel diary',
            'adventure story', 'coffee table book', 'photography book', 'photo essay',
            'documentary film', 'travel narrative', 'expedition story',
            'tourist guide', 'visitor guide', 'travel guide', 'guidebook',
            'beginner guide', 'popular science book', 'illustrated book',
            'biography', 'autobiography', 'cruise experience', 'expedition cruise'
        }
        
        # Count indicators
        research_score = sum(1 for indicator in research_indicators if indicator in full_text_lower)
        non_research_score = sum(1 for indicator in non_research_indicators if indicator in full_text_lower)
        
        # Check for academic structure
        has_abstract = bool(paper_data.get('abstract', '').strip())
        has_keywords = bool(paper_data.get('keywords', '').strip())
        has_doi = bool(paper_data.get('doi', '').strip())
        academic_structure_score = sum([has_abstract, has_keywords, has_doi])
        
        # Enhanced: Check for geological research patterns
        geological_patterns = [
            'mud volcanoes', 'seismic profiles', 'acoustic impedance',
            'bottom simulating reflector', 'multichannel seismic',
            'morpho-bathymetric', 'geophysical evidence', 'seafloor structures'
        ]
        geological_score = sum(1 for pattern in geological_patterns if pattern in full_text_lower)
        
        # Calculate total research confidence
        total_research_indicators = research_score + academic_structure_score + geological_score
        research_confidence = total_research_indicators / (total_research_indicators + non_research_score + 1)
        
        # Relaxed decision logic for geological papers
        if non_research_score >= 5:  # Travel/popular content
            return False, f"Antarctica content but not research paper (travel/popular: {non_research_score} indicators)", research_confidence
        elif geological_score >= 2:  # Strong geological research
            return True, f"Antarctic geological research confirmed (geological: {geological_score}, research: {research_score})", research_confidence
        elif research_score >= 2 or academic_structure_score >= 2:
            return True, f"Antarctic research paper confirmed (research: {research_score}, structure: {academic_structure_score})", research_confidence
        elif total_research_indicators >= 3 and non_research_score <= 2:
            return True, f"Antarctic research paper (total indicators: {total_research_indicators})", research_confidence
        else:
            # More permissive for geological content
            return True, f"Antarctica research content accepted (research: {research_score})", research_confidence
    
    def _load_single_model(self, model_path: str = None):
        """Load single model"""
        
        # Auto-detect model if not specified
        if model_path is None:
            model_dir = 'models'
            if os.path.exists(model_dir):
                # Look for mode-specific models first
                if MODE == "semantic":
                    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt') and 'semantic' in f]
                else:
                    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt') and 'full' in f]
                
                # Fallback to any models if no mode-specific ones found
                if not model_files:
                    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
                    
                if model_files:
                    # Get most recent model
                    model_files.sort(reverse=True)
                    model_path = os.path.join(model_dir, model_files[0])
                    print(f"Auto-detected model: {model_path}")
                    print(f"Model is from {MODE} training" if MODE in model_path else "Model mode unclear")
                else:
                    model_path = 'models/enhanced_multitarget_scibert.pt'
            else:
                model_path = 'models/enhanced_multitarget_scibert.pt'
        
        self.model_path = model_path
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and configuration"""
        
        if not os.path.exists(self.model_path):
            print(f"Model file not found: {self.model_path}")
            print("Please run examples/train_model.py first to train a model")
            sys.exit(1)
            
        print(f"Loading trained model from: {self.model_path}")
        
        try:
            # Auto-detect device
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
            
            print(f"Using device: {device}")
            
            # Load checkpoint (fix for PyTorch 2.6 weights_only default change)
            checkpoint = torch.load(self.model_path, map_location=device, weights_only=False)
            
            # Extract configuration (compatible with updated training format)
            self.config = checkpoint.get('config', get_config())
            self.target_configs = checkpoint.get('target_configs', {
                'themes': 27, 'objectives': 9, 'zones': 3, 'areas': 17
            })
            
            # Create model with updated training format
            self.model = EnhancedMultiTargetSciBERT(
                target_configs=self.target_configs,
                dropout_rate=self.config.get('dropout', 0.12),
                shared_dim=256,  # Fixed in updated training
                spatial_emphasis=self.config.get('spatial_emphasis', True)
            )
            
            # Load weights and setup
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(device)
            self.model.eval()
            
            # Setup tokenizer (compatible with training)
            self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
            
            # Add domain tokens (same as training)
            antarctic_terms = [
                "dissostichus-mawsoni", "euphausia-superba", "pleuragramma-antarctica",
                "general-protection-zone", "special-research-zone", "krill-research-zone",
                "ross-sea-polynya", "mcmurdo-sound", "balleny-islands", "ccamlr"
            ]
            
            new_tokens = [term for term in antarctic_terms if term not in self.tokenizer.vocab]
            if new_tokens:
                self.tokenizer.add_tokens(new_tokens)
                # Note: Model embedding resize should match training
            
            # Update class mappings if available
            if 'class_mappings' in checkpoint:
                self.class_mappings.update(checkpoint['class_mappings'])
            
            print(f"Model loaded successfully")
            print(f"   Targets: {list(self.target_configs.keys())}")
            print(f"   Classes: {list(self.target_configs.values())}")
            print(f"   Configuration: {self.config.get('name', 'Unknown')}")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def _create_enhanced_text_from_paper(self, paper_data: Dict[str, str]) -> str:
        """
        Create enhanced text using the same preprocessor as training
        This ensures the format matches exactly what the model expects
        """
        
        # Extract metadata from paper if not already present
        metadata = self.metadata_extractor.extract_metadata_from_paper(paper_data)
        
        # Create a pandas Series with all required fields for the preprocessor
        # Use the same field names as in the training data
        paper_series = pd.Series({
            'Title': paper_data.get('title', ''),
            'Abstract': paper_data.get('abstract', ''),
            'Keywords': paper_data.get('keywords', ''),
            'Species_List': metadata.get('Species_List', ''),
            'Method_Tags': metadata.get('Method_Tags', ''),
            'Named_Features': metadata.get('Named_Features', ''),
            'Temporal_Scale': metadata.get('Temporal_Scale', ''),
            'Spatial_Scale': metadata.get('Spatial_Scale', ''),
            'Research_start_date': metadata.get('Research_start_date', ''),
            'Research_end_date': metadata.get('Research_end_date', ''),
            'RMP_Spatial_Themes': '',  # Not available from PDF
            'Management_zones': '',  # Not available from PDF
            'Monitoring_areas': '',  # Not available from PDF
            'CCAMLR_Objectives_Text': '',  # Not available from PDF
            'Data_Availability': metadata.get('Data_Availability', '')
        })
        
        # Use the same preprocessor method as training
        enhanced_text = self.data_preprocessor.create_enhanced_combined_text(paper_series)
        
        # Log what was included
        print(f"\nEnhanced text created with sections:")
        sections = ['[TITLE]', '[ABSTRACT]', '[KEYWORDS]', '[SPECIES]', '[METHODS]',
                   '[LOCATIONS]', '[TEMPORAL]', '[SPATIAL]', '[ZONES]', '[AREAS]', '[OBJECTIVES]']
        for section in sections:
            if section in enhanced_text:
                start = enhanced_text.find(section)
                end = enhanced_text.find('[', start + 1)
                if end == -1:
                    end = min(start + 100, len(enhanced_text))
                content = enhanced_text[start:end].strip()
                if len(content) > len(section) + 2:  # Has content beyond the section marker
                    print(f"   ✓ {section} - has content")
                else:
                    print(f"   - {section} - empty")
        
        return enhanced_text
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Convert numerical confidence to descriptive level"""
        if confidence >= 0.8:
            return "Very High"
        elif confidence >= 0.6:
            return "High"
        elif confidence >= 0.4:
            return "Moderate"
        elif confidence >= 0.2:
            return "Low"
        else:
            return "Very Low"
    
    def classify_paper_file(self, file_path: str) -> Dict:
        """
        Classify a paper from file (PDF or text)
        
        Args:
            file_path: Path to paper file
            
        Returns:
            Classification results
        """
        
        print(f"\nProcessing file: {os.path.basename(file_path)}")
        
        # Extract paper content
        if file_path.lower().endswith('.pdf'):
            if not PDF_AVAILABLE:
                print("PDF processing not available")
                return None
            paper_data = self.extractor.extract_from_pdf(file_path)
        else:
            # Assume text file
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                paper_data = self.extractor.extract_from_text(content)
            except Exception as e:
                print(f"Error reading file: {e}")
                return None
        
        if not paper_data or not paper_data.get('full_text'):
            print("Could not extract content from file")
            return None
        
        return self.classify_paper_data(paper_data, file_path)
    
    def classify_paper_data(self, paper_data: Dict[str, str], source: str = "manual") -> Dict:
        """
        Classify paper from extracted data with Antarctic research filtering
        
        Args:
            paper_data: Dictionary with paper information
            source: Source description
            
        Returns:
            Detailed classification results or rejection
        """
        
        print(f"\nClassifying paper content...")
        print("-" * 30)
        
        # STEP 1: Antarctic Research Detection
        is_antarctic_research, reason, confidence = self._is_antarctic_research_paper(paper_data)
        
        print(f"Antarctic Research Detection:")
        print(f"   Ross Sea Research: {'YES' if is_antarctic_research else 'NO'}")
        print(f"   Reason: {reason}")
        print(f"   Confidence: {confidence:.3f}")
        
        if not is_antarctic_research:
            # Return rejection response instead of classification
            return {
                'source': source,
                'timestamp': datetime.now().isoformat(),
                'paper_info': {
                    'title': paper_data.get('title', 'Unknown'),
                    'authors': paper_data.get('authors', 'Unknown'),
                    'year': paper_data.get('year', 'Unknown'),
                    'doi': paper_data.get('doi', ''),
                    'abstract': paper_data.get('abstract', '')[:200] + "..." if len(paper_data.get('abstract', '')) > 200 else paper_data.get('abstract', ''),
                    'keywords': paper_data.get('keywords', ''),
                    'content_length': len(paper_data.get('full_text', ''))
                },
                'domain_check': {
                    'is_antarctic_research': False,
                    'reason': reason,
                    'confidence': confidence,
                    'status': 'REJECTED - NOT ROSS SEA RESEARCH'
                },
                'classifications': None,
                'model_info': {
                    'model_version': 'Enhanced Multi-Target SciBERT - Ross Sea MPA',
                    'configuration': self.config.get('name', 'Enhanced') if self.config else 'Unknown',
                    'note': 'Paper rejected before classification - not Antarctic research',
                    'ensemble_mode': self.ensemble_mode,
                    'voting_strategy': self.voting_strategy if self.ensemble_mode else None,
                    'experiment_mode': MODE,
                    'active_targets': self.active_targets
                }
            }
        
        # STEP 2: Proceed with normal classification for Antarctic research papers
        print(f"Antarctic research check passed - proceeding with classification...")
        
        try:
            # Create enhanced text using the same preprocessor as training
            enhanced_text = self._create_enhanced_text_from_paper(paper_data)
            print(f"Enhanced text created ({len(enhanced_text)} characters)")
            
            # Get predictions (ensemble or single model)
            if self.ensemble_mode and self.ensemble:
                # Use ensemble prediction with voting strategy
                threshold = self.config.get('threshold', 0.42)
                ensemble_predictions = self.ensemble.predict_ensemble(
                    enhanced_text, threshold, self.voting_strategy
                )
                detailed_results = self._process_ensemble_predictions(ensemble_predictions, threshold)
                model_info_extra = {
                    'ensemble_info': self.ensemble.get_ensemble_info(),
                    'voting_strategy': self.voting_strategy,
                    'prediction_variance': self._calculate_prediction_variance(ensemble_predictions)
                }
            else:
                # Use single model prediction
                detailed_results = self._process_single_model_predictions(enhanced_text)
                model_info_extra = {}
            
            # Filter results to only include active targets
            filtered_results = {}
            for target in self.active_targets:
                if target in detailed_results:
                    filtered_results[target] = detailed_results[target]
            
            # Extract metadata that was found
            metadata = self.metadata_extractor.extract_metadata_from_paper(paper_data)
            
            # Compile final results
            results = {
                'source': source,
                'timestamp': datetime.now().isoformat(),
                'paper_info': {
                    'title': paper_data.get('title', 'Unknown'),
                    'authors': paper_data.get('authors', 'Unknown'),
                    'year': paper_data.get('year', 'Unknown'),
                    'doi': paper_data.get('doi', ''),
                    'abstract': paper_data.get('abstract', '')[:500] + "..." if len(paper_data.get('abstract', '')) > 500 else paper_data.get('abstract', ''),
                    'keywords': paper_data.get('keywords', ''),
                    'content_length': len(paper_data.get('full_text', '')),
                    'extracted_metadata': {
                        'species': metadata.get('Species_List', ''),
                        'methods': metadata.get('Method_Tags', ''),
                        'locations': metadata.get('Named_Features', ''),
                        'temporal_scale': metadata.get('Temporal_Scale', ''),
                        'spatial_scale': metadata.get('Spatial_Scale', '')
                    }
                },
                'domain_check': {
                    'is_antarctic_research': True,
                    'reason': reason,
                    'confidence': confidence,
                    'status': 'ACCEPTED - ROSS SEA RESEARCH'
                },
                'classifications': filtered_results,
                'model_info': {
                    'model_version': 'Enhanced Multi-Target SciBERT - Ross Sea MPA',
                    'configuration': self.config.get('name', 'Enhanced') if self.config else 'Unknown',
                    'targets': list(self.target_configs.keys()) if self.target_configs else [],
                    'active_targets': self.active_targets,
                    'classes_per_target': dict(self.target_configs) if self.target_configs else {},
                    'threshold': self.config.get('threshold', 0.42) if self.config else 0.42,
                    'ensemble_mode': self.ensemble_mode,
                    'experiment_mode': MODE,
                    'preprocessing': 'EnhancedDataPreprocessor (same as training)',
                    **model_info_extra
                }
            }
            
            print(f"Classification complete!")
            return results
            
        except Exception as e:
            print(f"Classification failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _process_ensemble_predictions(self, ensemble_predictions: Dict, threshold: float) -> Dict:
        """Process ensemble predictions with voting information"""
        
        detailed_results = {}
        
        for target, pred_data in ensemble_predictions.items():
            # Only process active targets
            if not should_use_target(target):
                continue
                
            predictions = []
            
            if 'voting_method' in pred_data:  # New voting format
                # Process hierarchical/majority voting results
                if pred_data['voting_method'] in ['hierarchical', 'majority']:
                    for i, (is_predicted, confidence_data) in enumerate(
                        zip(pred_data['predictions'], pred_data['confidences'])
                    ):
                        if is_predicted:
                            class_names = self.class_mappings.get(target, [])
                            
                            if i < len(class_names):
                                class_name = class_names[i]
                                description = class_name
                            else:
                                class_name = f"{target}_{i}"
                                description = class_name
                            
                            predictions.append({
                                'class_index': i,
                                'class_name': class_name,
                                'description': description,
                                'confidence': confidence_data['confidence'],
                                'uncertainty': confidence_data.get('std', 0.0),
                                'votes': f"{confidence_data['votes']}/{confidence_data['num_models']}",
                                'required_votes': confidence_data.get('required_votes', 'N/A'),
                                'confidence_level': self._get_confidence_level(confidence_data['confidence'])
                            })
                else:
                    # Process averaging results
                    probs = pred_data['probs']
                    std = pred_data['std']
                    
                    for i, (score, uncertainty) in enumerate(zip(probs, std)):
                        if score > threshold:
                            class_names = self.class_mappings.get(target, [])
                            
                            if i < len(class_names):
                                class_name = class_names[i]
                                description = class_name
                            else:
                                class_name = f"{target}_{i}"
                                description = class_name
                            
                            predictions.append({
                                'class_index': i,
                                'class_name': class_name,
                                'description': description,
                                'confidence': float(score),
                                'uncertainty': float(uncertainty),
                                'confidence_level': self._get_confidence_level(score)
                            })
            else:
                # Original averaging format (backward compatibility)
                probs = pred_data['probs']
                std = pred_data['std']
                
                for i, (score, uncertainty) in enumerate(zip(probs, std)):
                    if score > threshold:
                        class_names = self.class_mappings.get(target, [])
                        
                        if i < len(class_names):
                            class_name = class_names[i]
                            description = class_name
                        else:
                            class_name = f"{target}_{i}"
                            description = class_name
                        
                        predictions.append({
                            'class_index': i,
                            'class_name': class_name,
                            'description': description,
                            'confidence': float(score),
                            'uncertainty': float(uncertainty),
                            'confidence_level': self._get_confidence_level(score)
                        })
            
            # Sort by confidence
            predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            detailed_results[target] = {
                'predictions': predictions,
                'total_predictions': len(predictions),
                'threshold_used': threshold,
                'voting_method': pred_data.get('voting_method', 'averaging'),
                'ensemble_stats': {
                    'num_models': pred_data.get('num_models', 1),
                    'mean_uncertainty': float(np.mean(pred_data.get('std', [0]))) if 'std' in pred_data else 0.0,
                    'max_uncertainty': float(np.max(pred_data.get('std', [0]))) if 'std' in pred_data else 0.0
                }
            }
        
        return detailed_results
    
    def _process_single_model_predictions(self, enhanced_text: str) -> Dict:
        """Process single model predictions"""
        
        # Tokenize text (same as training)
        encoding = self.tokenizer(
            enhanced_text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        # Move to device
        device = next(self.model.parameters()).device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Get model predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
        
        # Process predictions with optimal thresholds
        threshold = self.config.get('threshold', 0.42)
        detailed_results = {}
        
        for target in self.target_configs.keys():
            # Only process active targets
            if not should_use_target(target):
                continue
                
            if target in outputs:
                # Get probabilities
                probs = torch.sigmoid(outputs[target]).cpu().numpy()[0]
                
                # Apply threshold
                predictions = []
                for i, score in enumerate(probs):
                    if score > threshold:
                        # Get class name from mappings
                        class_names = self.class_mappings.get(target, [])
                        
                        if i < len(class_names):
                            class_name = class_names[i]
                            description = class_name
                            print(f"   DEBUG: Index {i} → {description}")
                        else:
                            class_name = f"{target}_{i}"
                            description = class_name
                            print(f"   DEBUG: Using fallback: {class_name}")
                        
                        predictions.append({
                            'class_index': i,
                            'class_name': class_name,
                            'description': description,
                            'confidence': float(score),
                            'confidence_level': self._get_confidence_level(score)
                        })
                
                # Sort by confidence
                predictions.sort(key=lambda x: x['confidence'], reverse=True)
                
                detailed_results[target] = {
                    'predictions': predictions,
                    'total_predictions': len(predictions),
                    'threshold_used': threshold,
                    'voting_method': 'single_model',
                    'all_scores': [float(s) for s in probs]
                }
        
        return detailed_results
    
    def _calculate_prediction_variance(self, ensemble_predictions: Dict) -> Dict:
        """Calculate prediction variance statistics"""
        
        variance_stats = {}
        for target, pred_data in ensemble_predictions.items():
            # Only process active targets
            if not should_use_target(target):
                continue
                
            if 'std' in pred_data:
                std = pred_data['std']
                variance_stats[target] = {
                    'mean_variance': float(np.mean(std**2)),
                    'max_variance': float(np.max(std**2)),
                    'total_uncertainty': float(np.sum(std**2))
                }
            else:
                # For voting methods, calculate based on vote distribution
                if 'confidences' in pred_data:
                    confidences = [c['confidence'] for c in pred_data['confidences']]
                    variance_stats[target] = {
                        'mean_confidence': float(np.mean(confidences)),
                        'std_confidence': float(np.std(confidences)),
                        'voting_method': pred_data.get('voting_method', 'unknown')
                    }
        
        return variance_stats
    
    def print_detailed_results(self, results: Dict):
        """Print comprehensive classification results with Antarctic research checking"""
        
        if not results:
            print("No results to display")
            return
        
        print(f"\nENHANCED MULTI-TARGET CLASSIFICATION RESULTS")
        print("=" * 60)
        
        # Paper information
        paper_info = results['paper_info']
        print(f"Paper: {paper_info['title']}")
        if paper_info.get('authors') and paper_info['authors'] != 'Unknown':
            print(f"Authors: {paper_info['authors']}")
        if paper_info.get('year') and paper_info['year'] != 'Unknown':
            print(f"Year: {paper_info['year']}")
        if paper_info.get('doi'):
            print(f"DOI: {paper_info['doi']}")
        if paper_info.get('keywords'):
            print(f"Keywords: {paper_info['keywords']}")
        
        print(f"Content: {paper_info['content_length']:,} characters")
        print(f"Classified: {results['timestamp']}")
        
        # Show extracted metadata
        if 'extracted_metadata' in paper_info:
            metadata = paper_info['extracted_metadata']
            if any(metadata.values()):
                print(f"\nExtracted Metadata:")
                print("=" * 30)
                for key, value in metadata.items():
                    if value:
                        print(f"   {key.replace('_', ' ').title()}: {value}")
        
        # Domain check results
        if 'domain_check' in results:
            domain_check = results['domain_check']
            print(f"\nROSS SEA RESEARCH VERIFICATION:")
            print("=" * 35)
            
            is_research = domain_check['is_antarctic_research']
            status_icon = "✓" if is_research else "✗"
            print(f"{status_icon} Status: {domain_check['status']}")
            print(f"Reason: {domain_check['reason']}")
            print(f"Confidence: {domain_check['confidence']:.3f}")
            
            if not is_research:
                print(f"\nCLASSIFICATION REJECTED")
                print("=" * 30)
                print("This paper was identified as outside the Ross Sea Antarctic research domain.")
                print("The model is specifically designed for Ross Sea MPA conservation research.")
                print("\nSuggestions:")
                print("   • Verify paper content relates to Antarctic/Southern Ocean research")
                print("   • Check for Antarctic geographic locations or species")
                print("   • Ensure paper is a research publication (not travel/popular content)")
                print("   • Arctic research papers are excluded (different ecosystem)")
                return
        
        # Classification results (only if domain check passed)
        if results.get('classifications'):
            classifications = results['classifications']
            
            print(f"\nCLASSIFICATION PREDICTIONS:")
            print("=" * 50)
            
            for target, target_results in classifications.items():
                print(f"\n{target.upper()} ({target_results['total_predictions']} predictions)")
                print(f"   Threshold: {target_results['threshold_used']:.2f}")
                
                # Show voting method
                if 'voting_method' in target_results:
                    print(f"   Voting Method: {target_results['voting_method']}")
                
                # Show ensemble info if available
                if 'ensemble_stats' in target_results:
                    stats = target_results['ensemble_stats']
                    print(f"   Ensemble: {stats['num_models']} models")
                    if stats.get('mean_uncertainty', 0) > 0:
                        print(f"   Avg uncertainty: {stats['mean_uncertainty']:.3f}")
                
                print(f"   " + "-" * 40)
                
                if target_results['predictions']:
                    for i, pred in enumerate(target_results['predictions'], 1):
                        confidence = pred['confidence']
                        level = pred['confidence_level']
                        description = pred['description']
                        
                        # Build info string
                        info_parts = [f"Confidence: {confidence:.3f}"]
                        
                        if 'uncertainty' in pred and pred['uncertainty'] > 0:
                            info_parts.append(f"± {pred['uncertainty']:.3f}")
                        
                        if 'votes' in pred:
                            info_parts.append(f"Votes: {pred['votes']}")
                            if pred.get('required_votes') != 'N/A':
                                info_parts.append(f"(required: {pred['required_votes']})")
                        
                        info_parts.append(f"({level})")
                        
                        print(f"   {i:2d}. {description}")
                        print(f"       {' '.join(info_parts)}")
                        
                        if i >= 10:  # Limit display
                            remaining = len(target_results['predictions']) - 10
                            if remaining > 0:
                                print(f"       ... and {remaining} more predictions")
                            break
                else:
                    print(f"   No predictions above threshold ({target_results['threshold_used']:.2f})")
        
        # Model information
        model_info = results['model_info']
        print(f"\nMODEL INFORMATION:")
        print("=" * 30)
        print(f"Model: {model_info['model_version']}")
        print(f"Configuration: {model_info.get('configuration', 'Unknown')}")
        print(f"Experiment Mode: {model_info.get('experiment_mode', 'Unknown').upper()}")
        print(f"Preprocessing: {model_info.get('preprocessing', 'Unknown')}")
        
        if 'active_targets' in model_info:
            print(f"Active Targets: {', '.join(model_info['active_targets'])}")
        
        if model_info.get('ensemble_mode'):
            print(f"Mode: Ensemble Prediction")
            if 'ensemble_info' in model_info:
                ensemble_info = model_info['ensemble_info']
                print(f"Ensemble: {ensemble_info['num_models']} models")
                if 'voting_strategies' in ensemble_info:
                    print(f"Available strategies: {', '.join(ensemble_info['voting_strategies'])}")
            if 'voting_strategy' in model_info:
                print(f"Voting Strategy: {model_info['voting_strategy']}")
            
            # Show problematic classes info if using hierarchical voting
            if model_info.get('voting_strategy') == 'hierarchical' and 'ensemble_info' in model_info:
                if 'problematic_classes' in model_info['ensemble_info']:
                    print(f"\nClass-Specific Voting Rules:")
                    prob_classes = model_info['ensemble_info']['problematic_classes']
                    for target, rules in prob_classes.items():
                        if any(k != 'default' for k in rules.keys()):
                            print(f"   {target}:")
                            for class_name, required_votes in rules.items():
                                if class_name != 'default':
                                    print(f"      - {class_name}: {required_votes}/5 votes required")
        else:
            print(f"Mode: Single Model")
        
        if 'targets' in model_info:
            print(f"All Targets: {', '.join(model_info['targets'])}")
        if 'classes_per_target' in model_info:
            print(f"Classes: {model_info['classes_per_target']}")
        if 'threshold' in model_info:
            print(f"Threshold: {model_info['threshold']}")
        if 'note' in model_info:
            print(f"Note: {model_info['note']}")
        
        # Show prediction variance if available
        if 'prediction_variance' in model_info:
            print(f"\nPrediction Variance Analysis:")
            for target, variance_info in model_info['prediction_variance'].items():
                if 'voting_method' in variance_info:
                    print(f"   {target}: voting-based confidence")
                else:
                    print(f"   {target}: mean variance = {variance_info.get('mean_variance', 0):.4f}")


def main():
    """Main function with enhanced command line interface"""
    
    parser = argparse.ArgumentParser(
        description="Enhanced Multi-Target SciBERT Paper Classifier - Ross Sea Antarctic Research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classify PDF file with single model
  python examples/predict_paper.py --pdf paper.pdf
  
  # Classify with 5-model ensemble (if available) using hierarchical voting
  python examples/predict_paper.py --pdf paper.pdf --ensemble
  
  # Specify voting strategy
  python examples/predict_paper.py --pdf paper.pdf --ensemble --voting hierarchical
  
  # Use specific number of models
  python examples/predict_paper.py --pdf paper.pdf --ensemble --ensemble-size 5
  
  # Classify text file
  python examples/predict_paper.py --file paper.txt
  
  # Manual input (interactive)
  python examples/predict_paper.py --manual
  
  # Command line input
  python examples/predict_paper.py --title "Title" --abstract "Abstract..."
  
  # Save detailed results
  python examples/predict_paper.py --pdf paper.pdf --output results.json

Note: This classifier is specifically designed for Ross Sea Antarctic research papers.
It will reject: Arctic papers, Antarctica travel books, and non-Antarctic content.
Enhanced detection for geological/geophysical research papers.

The classifier uses the same EnhancedDataPreprocessor as training, ensuring
consistent text formatting with metadata sections like [SPECIES], [METHODS], etc.

Ensemble mode with 5 models improves prediction consistency and reduces false positives.
Hierarchical voting applies stricter thresholds to problematic classes like "toothfish".

Experiment Mode: The classifier respects the MODE setting in experiment_config.py
- MODE="semantic": Only classifies themes and objectives
- MODE="full": Classifies all targets (themes, objectives, zones, areas)
        """
    )
    
    parser.add_argument('--model', type=str, help='Path to trained model file (auto-detected if not specified)')
    parser.add_argument('--ensemble', action='store_true', help='Use ensemble of models for robust predictions')
    parser.add_argument('--ensemble-size', type=int, default=3, help='Number of models to use in ensemble (default: 3, recommended: 5)')
    parser.add_argument('--voting', type=str, default='hierarchical',
                       choices=['averaging', 'majority', 'hierarchical'],
                       help='Voting strategy for ensemble (default: hierarchical)')
    parser.add_argument('--pdf', type=str, help='Path to PDF file to classify')
    parser.add_argument('--file', type=str, help='Path to text file to classify')
    parser.add_argument('--title', type=str, help='Paper title (manual input)')
    parser.add_argument('--abstract', type=str, help='Paper abstract (manual input)')
    parser.add_argument('--keywords', type=str, help='Paper keywords (manual input)')
    parser.add_argument('--manual', action='store_true', help='Interactive manual input mode')
    parser.add_argument('--output', type=str, help='Save detailed results to JSON file')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    
    args = parser.parse_args()
    
    # Initialize classifier
    try:
        classifier = PaperClassifier(
            model_path=args.model,
            ensemble_mode=args.ensemble,
            ensemble_size=args.ensemble_size,
            voting_strategy=args.voting
        )
    except Exception as e:
        print(f"Failed to initialize classifier: {str(e)}")
        return
    
    results = None
    
    # Determine input mode
    if args.pdf:
        # PDF mode
        if not PDF_AVAILABLE:
            print("PDF processing not available. Please install: pip install pdfplumber PyPDF2 pymupdf")
            return
        results = classifier.classify_paper_file(args.pdf)
        
    elif args.file:
        # Text file mode
        results = classifier.classify_paper_file(args.file)
        
    elif args.title and args.abstract:
        # Manual command line mode
        paper_data = {
            'title': args.title,
            'abstract': args.abstract,
            'keywords': args.keywords or '',
            'full_text': f"{args.title} {args.abstract}"
        }
        results = classifier.classify_paper_data(paper_data, "command_line")
        
    elif args.manual:
        # Interactive manual mode
        print(f"\nManual Paper Input Mode")
        print("=" * 40)
        
        title = input("Paper Title: ").strip()
        print("Abstract (press Enter twice when done):")
        abstract_lines = []
        while True:
            line = input()
            if line == "":
                break
            abstract_lines.append(line)
        abstract = " ".join(abstract_lines).strip()
        keywords = input("Keywords (optional): ").strip()
        
        if title and abstract:
            paper_data = {
                'title': title,
                'abstract': abstract,
                'keywords': keywords,
                'full_text': f"{title} {abstract}"
            }
            results = classifier.classify_paper_data(paper_data, "manual_input")
        else:
            print("Title and abstract are required")
            return
            
    else:
        # Show help
        parser.print_help()
        return
    
    # Display results
    if results:
        if not args.quiet:
            classifier.print_detailed_results(results)
        
        # Save to file if requested
        if args.output:
            try:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"\nDetailed results saved to: {args.output}")
            except Exception as e:
                print(f"Error saving results: {str(e)}")
    else:
        print(f"Classification failed")


if __name__ == "__main__":
    main()
