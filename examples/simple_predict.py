# simple_predict.py
"""
Simple Paper Classification System for SciBERT with Ross Sea Research Detection
Compatible with the new simple training pipeline
Preserves all Antarctic research detection and filtering logic
"""

import os
import sys
import torch
import json
import argparse
import warnings
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import re
import glob

warnings.filterwarnings('ignore')

# Import our simple modules
from simple_preprocessing import SimplePreprocessor, get_class_descriptions
from simple_model import SimpleMultiTargetModel, load_model

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


class PaperExtractor:
    """Extract structured information from research papers"""
    
    def extract_from_pdf(self, pdf_path: str) -> Dict[str, str]:
        """Extract text and metadata from PDF file"""
        
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
            'doi': ''
        }
        
        try:
            # Try pdfplumber first
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
                return extracted
        
        # Parse structured information from text
        if extracted['full_text']:
            self._parse_paper_structure(extracted)
        
        return extracted
    
    def extract_from_text(self, text_content: str) -> Dict[str, str]:
        """Extract structured information from plain text"""
        
        extracted = {
            'title': '',
            'abstract': '',
            'keywords': '',
            'full_text': text_content,
            'authors': '',
            'year': '',
            'doi': ''
        }
        
        self._parse_paper_structure(extracted)
        return extracted
    
    def _parse_paper_structure(self, extracted: Dict[str, str]):
        """Parse paper structure from full text"""
        
        text = extracted['full_text']
        lines = text.split('\n')
        
        # Extract title (usually first substantial line)
        if not extracted['title']:
            for line in lines[:10]:
                line = line.strip()
                if len(line) > 20 and not line.isupper() and '.' not in line[:20]:
                    extracted['title'] = line
                    break
        
        # Extract abstract
        abstract_patterns = [
            r'ABSTRACT\s*:?\s*(.*?)(?=\n\s*(?:KEYWORDS?|INTRODUCTION|1\.|Keywords:|Introduction))',
            r'Abstract\s*:?\s*(.*?)(?=\n\s*(?:Keywords?|Introduction|1\.))',
        ]
        
        for pattern in abstract_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                abstract = match.group(1).strip()
                abstract = re.sub(r'\s+', ' ', abstract)
                if len(abstract) > 50:
                    extracted['abstract'] = abstract
                    break
        
        # Extract keywords
        keyword_patterns = [
            r'KEYWORDS?\s*:?\s*(.*?)(?=\n\s*[A-Z])',
            r'Keywords?\s*:?\s*(.*?)(?=\n\s*[A-Z])',
        ]
        
        for pattern in keyword_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                keywords = match.group(1).strip()
                keywords = re.sub(r'\s+', ' ', keywords)
                if len(keywords) > 5:
                    extracted['keywords'] = keywords
                    break
        
        # Extract year
        year_match = re.search(r'\b(19|20)\d{2}\b', text)
        if year_match:
            extracted['year'] = year_match.group(0)
        
        # Extract DOI
        doi_match = re.search(r'DOI\s*:?\s*(10\.\d+/[^\s]+)', text, re.IGNORECASE)
        if doi_match:
            extracted['doi'] = doi_match.group(1)


class PaperClassifier:
    """
    Paper classification system for Ross Sea Antarctic Research
    WITH ANTARCTIC RESEARCH DETECTION - Rejects Arctic, travel books, non-research
    Enhanced detection for geological/geophysical research papers
    """
    
    def __init__(self, model_path: str = None):
        """Initialize the classifier"""
        
        self.model = None
        self.tokenizer = None
        self.config = None
        self.target_configs = None
        self.class_descriptions = None
        self.extractor = PaperExtractor()
        
        print("Ross Sea Paper Classifier")
        print("ANTARCTIC RESEARCH DETECTION ENABLED")
        print("=" * 60)
        
        self._load_model(model_path)
    
    def _load_model(self, model_path: str = None):
        """Load the trained model"""
        
        # Auto-detect model if not specified
        if model_path is None:
            model_dir = 'models'
            if os.path.exists(model_dir):
                # Look for SciBERT models first
                model_files = [f for f in os.listdir(model_dir)
                             if f.startswith('scibert_ross_sea_') and f.endswith('.pt')]
                if model_files:
                    model_files.sort(reverse=True)
                    model_path = os.path.join(model_dir, model_files[0])
                    print(f"Auto-detected model: {model_path}")
        
        if not model_path or not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            print("Please run bert_vs_scibert.py first to train a model")
            sys.exit(1)
        
        print(f"Loading model from: {model_path}")
        
        # Auto-detect device
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        
        print(f"Using device: {device}")
        
        # Load model
        self.model, self.config, tokenizer_name = load_model(model_path, device)
        self.target_configs = self.config['target_configs']
        
        # Load tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Load preprocessor and get class descriptions
        preprocessor = SimplePreprocessor()
        
        # Create dummy info for class descriptions
        info = {
            'class_names': {
                'themes': list(range(1, self.target_configs['themes'] + 1)),
                'objectives': ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix'],
                'zones': ['GPZ', 'KRZ', 'SRZ'],
                'areas': [f'Area_{i}' for i in range(self.target_configs['areas'])]
            },
            'mappings': {
                'theme_mapping': preprocessor.theme_mapping,
                'objectives_mapping': preprocessor.ccamlr_objectives_mapping,
                'zone_mapping': preprocessor.zone_mapping,
                'areas_mapping': preprocessor.monitoring_areas_mapping
            }
        }
        
        self.class_descriptions = get_class_descriptions(info)
        
        print(f"Model loaded successfully")
        print(f"   Targets: {list(self.target_configs.keys())}")
        print(f"   Classes: {list(self.target_configs.values())}")
    
    def _is_antarctic_domain(self, paper_data: Dict[str, str]) -> Tuple[bool, str, float]:
        """Check if paper is in Antarctic/marine domain"""
        
        # Combine all text for analysis
        full_text = f"{paper_data.get('title', '')} {paper_data.get('abstract', '')} {paper_data.get('keywords', '')} {paper_data.get('full_text', '')}"
        full_text_lower = full_text.lower()
        
        # Core Antarctic keywords
        antarctic_core_keywords = {
            'antarctic', 'antarctica', 'ross sea', 'southern ocean', 'antarctic peninsula',
            'mcmurdo', 'weddell sea', 'terra nova bay', 'ross ice shelf', 'antarctic circumpolar',
            'ccamlr', 'toothfish', 'dissostichus mawsoni', 'antarctic krill', 'euphausia superba',
            'adelie penguin', 'emperor penguin', 'chinstrap penguin', 'leopard seal', 'weddell seal',
            'pleuragramma antarctica', 'silverfish antarctic', 'balleny islands', 'rsrmpa',
            'ross sea marine protected area', 'mpa', 'erebus bay', 'mcmurdo sound'
        }
        
        # Marine/polar keywords
        marine_polar_keywords = {
            'marine protected area', 'marine conservation', 'fisheries', 'krill', 'penguin',
            'seal', 'whale', 'polar', 'ice shelf', 'polynya', 'sea ice', 'ice edge',
            'southern hemisphere', 'subantarctic', 'circumpolar', 'pelagic', 'benthic',
            'oceanographic', 'marine ecosystem', 'polar research', 'conservation biology'
        }
        
        # Conflicting geography keywords
        conflicting_geography = {
            'arctic', 'greenland', 'norway', 'iceland', 'svalbard', 'finland',
            'alaska', 'canada', 'russia', 'north pole', 'north atlantic',
            'tropical', 'equatorial', 'caribbean', 'mediterranean', 'baltic'
        }
        
        # Calculate scores
        antarctic_score = sum(1 for keyword in antarctic_core_keywords if keyword in full_text_lower)
        marine_score = sum(1 for keyword in marine_polar_keywords if keyword in full_text_lower)
        conflict_score = sum(1 for keyword in conflicting_geography if keyword in full_text_lower)
        
        # Extra weight for title/abstract
        title_text = paper_data.get('title', '').lower()
        abstract_text = paper_data.get('abstract', '').lower()
        
        title_antarctic = sum(2 for keyword in antarctic_core_keywords if keyword in title_text)
        abstract_antarctic = sum(1.5 for keyword in antarctic_core_keywords if keyword in abstract_text)
        
        # Domain score
        domain_score = (
            (antarctic_score + title_antarctic + abstract_antarctic) * 3.0 +
            marine_score * 1.0 -
            conflict_score * 2.0
        )
        
        # Normalize confidence
        confidence = max(0, min(1, domain_score / 20))
        
        # Decision
        total_antarctic = antarctic_score + title_antarctic + abstract_antarctic
        
        if total_antarctic >= 3:
            return True, f"Strong Antarctic indicators found (score: {total_antarctic})", confidence
        elif conflict_score >= 3:
            return False, f"Conflicting geography detected: {conflict_score} non-Antarctic regions", confidence
        elif domain_score < 1:
            return False, f"No Antarctic indicators found (score: {domain_score:.1f})", confidence
        else:
            is_antarctic = domain_score >= 2.0
            return is_antarctic, f"Domain score: {domain_score:.1f}", confidence
    
    def _is_antarctic_research_paper(self, paper_data: Dict[str, str]) -> Tuple[bool, str, float]:
        """Enhanced Antarctic research detection"""
        
        # First check domain
        is_antarctic, antarctic_reason, antarctic_confidence = self._is_antarctic_domain(paper_data)
        
        if not is_antarctic:
            return False, antarctic_reason, antarctic_confidence
        
        # Check if it's actual research
        full_text = f"{paper_data.get('title', '')} {paper_data.get('abstract', '')} {paper_data.get('keywords', '')} {paper_data.get('full_text', '')}"
        full_text_lower = full_text.lower()
        
        # Research indicators
        research_indicators = {
            'abstract', 'methods', 'methodology', 'results', 'discussion', 'conclusion',
            'data', 'analysis', 'study', 'research', 'investigation', 'experiment',
            'observation', 'measurement', 'statistical', 'significant', 'correlation',
            'hypothesis', 'model', 'population', 'sample', 'distribution', 'abundance',
            'peer review', 'journal', 'publication', 'citation', 'doi',
            # Geological/geophysical terms
            'seismic', 'geophysical', 'acoustic', 'velocity', 'bathymetric',
            'geological', 'geophysics', 'sedimentary', 'stratigraphy', 'tectonic'
        }
        
        # Non-research indicators
        non_research_indicators = {
            'personal journey', 'expedition memoir', 'travel diary', 'adventure story',
            'coffee table book', 'photography book', 'tourist guide', 'travel guide'
        }
        
        research_score = sum(1 for indicator in research_indicators if indicator in full_text_lower)
        non_research_score = sum(1 for indicator in non_research_indicators if indicator in full_text_lower)
        
        # Check structure
        has_abstract = bool(paper_data.get('abstract', '').strip())
        has_keywords = bool(paper_data.get('keywords', '').strip())
        
        research_confidence = research_score / (research_score + non_research_score + 1)
        
        if non_research_score >= 3:
            return False, f"Non-research content detected ({non_research_score} indicators)", research_confidence
        elif research_score >= 3 or (has_abstract and research_score >= 2):
            return True, f"Research paper confirmed ({research_score} indicators)", research_confidence
        else:
            return True, f"Research content accepted", research_confidence
    
    def classify_paper_file(self, file_path: str) -> Dict:
        """Classify a paper from file"""
        
        print(f"\nProcessing file: {os.path.basename(file_path)}")
        
        # Extract content
        if file_path.lower().endswith('.pdf'):
            if not PDF_AVAILABLE:
                print("PDF processing not available")
                return None
            paper_data = self.extractor.extract_from_pdf(file_path)
        else:
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
        """Classify paper from extracted data"""
        
        print(f"\nClassifying paper...")
        print("-" * 30)
        
        # Check if it's Antarctic research
        is_antarctic_research, reason, confidence = self._is_antarctic_research_paper(paper_data)
        
        print(f"Antarctic Research Detection:")
        print(f"   Ross Sea Research: {'YES' if is_antarctic_research else 'NO'}")
        print(f"   Reason: {reason}")
        print(f"   Confidence: {confidence:.3f}")
        
        if not is_antarctic_research:
            return {
                'source': source,
                'timestamp': datetime.now().isoformat(),
                'paper_info': {
                    'title': paper_data.get('title', 'Unknown'),
                    'year': paper_data.get('year', 'Unknown')
                },
                'domain_check': {
                    'is_antarctic_research': False,
                    'reason': reason,
                    'confidence': confidence,
                    'status': 'REJECTED - NOT ROSS SEA RESEARCH'
                },
                'classifications': None
            }
        
        # Create simple text for classification
        preprocessor = SimplePreprocessor()
        text = preprocessor.create_simple_text(pd.Series(paper_data))
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        # Move to device
        device = next(self.model.parameters()).device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
        
        # Process predictions
        threshold = 0.5
        results = {}
        
        for target in self.target_configs.keys():
            probs = torch.sigmoid(outputs[target]).cpu().numpy()[0]
            predictions = []
            
            for i, score in enumerate(probs):
                if score > threshold:
                    # Get class description
                    if i in self.class_descriptions.get(target, {}):
                        description = self.class_descriptions[target][i]
                    else:
                        description = f"{target}_{i}"
                    
                    predictions.append({
                        'class_index': i,
                        'description': description,
                        'confidence': float(score),
                        'confidence_level': self._get_confidence_level(score)
                    })
            
            predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            results[target] = {
                'predictions': predictions,
                'total_predictions': len(predictions),
                'threshold_used': threshold
            }
        
        return {
            'source': source,
            'timestamp': datetime.now().isoformat(),
            'paper_info': {
                'title': paper_data.get('title', 'Unknown'),
                'authors': paper_data.get('authors', 'Unknown'),
                'year': paper_data.get('year', 'Unknown'),
                'doi': paper_data.get('doi', ''),
                'abstract': paper_data.get('abstract', '')[:500] + "..." if len(paper_data.get('abstract', '')) > 500 else paper_data.get('abstract', ''),
                'keywords': paper_data.get('keywords', '')
            },
            'domain_check': {
                'is_antarctic_research': True,
                'reason': reason,
                'confidence': confidence,
                'status': 'ACCEPTED - ROSS SEA RESEARCH'
            },
            'classifications': results,
            'model_info': {
                'model_type': 'SciBERT',
                'model_path': self.config.get('base_model', 'Unknown')
            }
        }
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Convert confidence to level"""
        if confidence >= 0.8:
            return "Very High"
        elif confidence >= 0.6:
            return "High"
        elif confidence >= 0.4:
            return "Moderate"
        else:
            return "Low"
    
    def print_results(self, results: Dict):
        """Print classification results"""
        
        if not results:
            print("No results to display")
            return
        
        print(f"\nCLASSIFICATION RESULTS")
        print("=" * 60)
        
        # Paper info
        paper_info = results['paper_info']
        print(f"Paper: {paper_info['title']}")
        if paper_info.get('year') != 'Unknown':
            print(f"Year: {paper_info['year']}")
        
        # Domain check
        domain = results['domain_check']
        print(f"\nDomain Check: {domain['status']}")
        
        if not domain['is_antarctic_research']:
            print("Paper rejected - outside Ross Sea Antarctic research domain")
            return
        
        # Classifications
        if results.get('classifications'):
            print(f"\nClassifications:")
            
            for target, target_results in results['classifications'].items():
                print(f"\n{target.upper()} ({target_results['total_predictions']} predictions)")
                
                if target_results['predictions']:
                    for i, pred in enumerate(target_results['predictions'][:5], 1):
                        print(f"   {i}. {pred['description']}")
                        print(f"      Confidence: {pred['confidence']:.3f} ({pred['confidence_level']})")
                else:
                    print(f"   No predictions above threshold")


def main():
    """Main function with command line interface"""
    
    parser = argparse.ArgumentParser(
        description="Ross Sea Paper Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classify PDF file
  python simple_predict.py --pdf paper.pdf
  
  # Classify text file
  python simple_predict.py --file paper.txt
  
  # Manual input
  python simple_predict.py --manual
  
  # Use specific model
  python simple_predict.py --model models/scibert_ross_sea_20240101.pt --pdf paper.pdf

Note: This classifier only accepts Ross Sea Antarctic research papers.
        """
    )
    
    parser.add_argument('--model', type=str, help='Path to model file')
    parser.add_argument('--pdf', type=str, help='Path to PDF file')
    parser.add_argument('--file', type=str, help='Path to text file')
    parser.add_argument('--title', type=str, help='Paper title')
    parser.add_argument('--abstract', type=str, help='Paper abstract')
    parser.add_argument('--keywords', type=str, help='Paper keywords')
    parser.add_argument('--manual', action='store_true', help='Manual input mode')
    parser.add_argument('--output', type=str, help='Save results to JSON')
    
    args = parser.parse_args()
    
    # Initialize classifier
    try:
        classifier = PaperClassifier(model_path=args.model)
    except Exception as e:
        print(f"Failed to initialize classifier: {str(e)}")
        return
    
    results = None
    
    # Process input
    if args.pdf:
        results = classifier.classify_paper_file(args.pdf)
    elif args.file:
        results = classifier.classify_paper_file(args.file)
    elif args.title and args.abstract:
        paper_data = {
            'title': args.title,
            'abstract': args.abstract,
            'keywords': args.keywords or ''
        }
        results = classifier.classify_paper_data(paper_data, "command_line")
    elif args.manual:
        print(f"\nManual Input Mode")
        print("=" * 40)
        
        title = input("Title: ").strip()
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
                'keywords': keywords
            }
            results = classifier.classify_paper_data(paper_data, "manual_input")
    else:
        parser.print_help()
        return
    
    # Display results
    if results:
        classifier.print_results(results)
        
        # Save if requested
        if args.output:
            try:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\nResults saved to: {args.output}")
            except Exception as e:
                print(f"Error saving results: {str(e)}")


if __name__ == "__main__":
    main()
