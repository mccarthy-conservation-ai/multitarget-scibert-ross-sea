# Multi-Target SciBERT for Ross Sea Conservation Research

This repository contains the implementation of our enhanced multi-target classification framework for conservation research literature, as described in "Where Are the Research Gaps? AI-Assisted Multi-Target Classification for Research-Policy Alignment in Conservation Science."

## Quick Start

```bash
# Clone the repository
git clone https://github.com/mccarthy-conservation-ai/multitarget-scibert-ross-sea.git
cd multitarget-scibert-ross-sea

# Install dependencies
pip install -r requirements.txt

# Run with synthetic data (works immediately)
python train_model.py

# Classify a single paper
python predict_paper.py
```

## Quick Classification Example

```bash
# Classify a paper from PDF
python predict_paper.py --pdf "path/to/your/paper.pdf"

# Classify with ensemble (recommended for production)
python predict_paper.py --pdf "path/to/your/paper.pdf" --ensemble

# Classify with manual input
python predict_paper.py --title "Your Paper Title" --abstract "Your abstract text..."

# Interactive mode
python predict_paper.py --manual

# Ensemble with custom size
python predict_paper.py --pdf "path/to/your/paper.pdf" --ensemble --ensemble-size 5
```

Results show classification across all four dimensions:
- **Research themes** (with confidence scores)
- **CCAMLR objectives** alignment  
- **Management zones**
- **Monitoring areas**

## Overview

The Enhanced Multi-Target SciBERT framework simultaneously classifies research papers across four conservation dimensions:

- **Research Themes** (38 classes): Semantic classification of research focus areas
- **CCAMLR Objectives** (11 classes): Policy alignment with conservation goals
- **Management Zones** (3 classes): Spatial classification into MPA zones
- **Monitoring Areas** (18 classes): Geographic classification into standardized locations

## Dataset Options

### Option 1: Synthetic Dataset (Default)
The repository includes a synthetic dataset (30 papers) for immediate testing and demonstration. This allows you to:

- Test the model architecture and verify installation
- Understand the classification framework and data format
- Reproduce the methodology without data access barriers
- Adapt the approach to your own data

**Note**: This synthetic dataset is for code testing only. All performance results in the paper are based on the actual 295-paper expert-curated dataset.

### Option 2: Actual Dataset (Recommended for Research)
To use the actual Ross Sea MPA research dataset (295 papers):

1. **Request access** from Brooks & Ainley (2022): "A Summary of United States Research and Monitoring in Support of the Ross Sea Region Marine Protected Area"
2. **Place the file** `rosssea_research_dataset.xlsx` in the root directory
3. **Re-run the training** - the system will automatically detect and use the actual dataset

The actual dataset provides:

- Real research papers from 2010-2021
- Expert-validated classifications
- Comprehensive coverage of Ross Sea research themes
- Authentic policy alignment annotations

## Model Configurations

We tested four enhanced configurations:

1. **Spatial Enhanced** (Recommended): Optimized for geographic classification with spatial emphasis
2. **Gradient Accumulation Enhanced**: Improved training stability through gradient accumulation
3. **Stability Focused**: Conservative approach with high regularization
4. **Production Ready**: Balanced configuration for deployment

## Performance Expectations

### With Actual Dataset:
- **Themes**: Macro F1 ≈ 0.48 (semantic complexity)
- **CCAMLR Objectives**: Macro F1 ≈ 0.91 (structured policy targets)
- **Management Zones**: Macro F1 ≈ 1.00 (universal GPZ detection)
- **Monitoring Areas**: Macro F1 ≈ 0.48 (spatial complexity)

### Processing Performance:
- **Classification time**: ~0.8 seconds per paper
- **Expert agreement**: 89% across all classification dimensions
- **Deployment ready**: Tested on 30+ independent papers

### With Synthetic Dataset:
The synthetic dataset (30 papers) is for code testing and demonstration only. Performance will be different due to the small size and synthetic nature of the data. **Do not use synthetic results for performance benchmarking** - all published results are based on the actual 295-paper dataset.

## Key Features

- **Multi-Target Learning**: Simultaneous classification across four dimensions
- **Spatial Enhancement**: Specialized processing for geographic classification
- **Enhanced Text Processing**: Structured preprocessing with spatial context
- **Configuration Testing**: Systematic evaluation of training approaches
- **Comprehensive Evaluation**: Cross-validation with statistical validation

## Architecture

The framework combines:

- **SciBERT Base**: Pre-trained scientific language model
- **Shared Projection Layer**: Common representations across targets
- **Target-Specific Heads**: Specialized classification for each dimension
- **Enhanced Focal Loss**: Handles severe class imbalance
- **Gradient Accumulation**: Effective larger batch sizes

## Usage Examples

### Training a Model
```python
from src.data_loader import load_data_with_fallback
from src.enhanced_scibert_model import EnhancedMultiTargetSciBERT
from src.training_config import get_original_config

# Load data (automatically detects actual vs synthetic)
df, is_actual = load_data_with_fallback()

# Use recommended configuration
config = get_original_config()  # Spatial Enhanced

# Train model
from examples.train_model import main
results = main()
```

### Classifying a Paper
```python
from examples.predict_paper import PaperClassifier

# Initialize classifier
classifier = PaperClassifier()

# Classify from file
results = classifier.classify_paper_file("path/to/paper.pdf")

# Classify from text
paper_data = {
    'title': "Krill Population Dynamics in Ross Sea",
    'abstract': "Study of krill distribution patterns...",
    'keywords': "krill,ross sea,population"
}
results = classifier.classify_paper_data(paper_data)

# Display results
classifier.print_detailed_results(results)
```

## Repository Structure

```
multitarget-scibert-ross-sea/
├── data/                   # Dataset and specifications
│   ├── rosssea_research_dataset.xlsx  # Requires permission from Brooks & Ainley (2022)
│   └── synthetic_dataset.csv          # Synthetic data for testing
├── examples/               # Usage examples
│   ├── debug_class_mappings.py        # Debug utilities
│   ├── predict_paper.py               # Classification script
│   └── train_model.py                 # Training script
├── src/                    # Core implementation
│   ├── data_loader.py                 # Data loading utilities
│   ├── data_preprocessing.py          # Text preprocessing
│   ├── enhanced_scibert_model.py      # Multi-target model architecture
│   └── training_config.py             # Training configurations
├── tests/                  # Test files
│   └── testpaper1.pdf                 # Sample paper for testing
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.35.0
- scikit-learn
- pandas
- numpy
- scipy

For PDF processing (optional):
- pdfplumber
- PyPDF2
- pymupdf

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{mccarthy2024research,
  title={Where Are the Research Gaps? AI-Assisted Multi-Target Classification for Research-Policy Alignment in Conservation Science},
  author={McCarthy, Chris and Titmus, Andrew and Sternberg, Troy and Shaney, Kyle and Brooks, Cassandra},
  journal={Ecological Informatics},
  note={Under Review},
  year={2024}
}
```

## Data Access

The training dataset requires permission from the original authors. Please contact:

- Brooks, C.M. & Ainley, D.G. (2022). "A Summary of United States Research and Monitoring in Support of the Ross Sea Region Marine Protected Area." *Diversity* 14(6), 447.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions about the methodology or implementation:

- Open an issue in this repository
- Contact the authors through the paper's corresponding author

## Transferability

While demonstrated on Ross Sea MPA research, this framework is designed for transferability to other conservation domains. The modular architecture separates domain-specific components (conservation objectives, spatial zones) from core methodology, facilitating adaptation to:

- Terrestrial protected areas
- Marine conservation networks
- Ecosystem-based management contexts
- Other research-policy alignment applications

## Model Performance

The framework achieves:
- **64.9% improvement** over SciBERT baseline
- **89% expert validation** agreement
- **Statistical significance** (p < 0.001) across all targets
- **Robust cross-validation** performance

For detailed performance metrics and validation results, see the paper.
