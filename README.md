# Multi-Target SciBERT for Ross Sea Conservation Research

Implementation of an enhanced multi-target classification framework for conservation research literature, as described in "Where Are the Research Gaps? AI-Assisted Multi-Target Classification for Research-Policy Alignment in Conservation Science."

## Quick Start

```bash
# Clone the repository
git clone https://github.com/mccarthy-conservation-ai/multitarget-scibert-ross-sea.git
cd multitarget-scibert-ross-sea

# Install dependencies
pip install -r requirements.txt

# Train the model
python examples/train_model.py

# Classify a paper
python examples/predict_paper.py --pdf paper.pdf
```

## Overview

The Enhanced Multi-Target SciBERT framework simultaneously classifies research papers across two key conservation dimensions:

- **Research Themes** (27 classes): Research focus areas from the RSRMPA Research and Monitoring Plan
- **CCAMLR Objectives** (9 classes): Conservation objectives from CCAMLR Conservation Measure 91-05

## Dataset

### Using the Provided Dataset
This repository includes the Ross Sea research dataset (`rosssea_research_dataset.xlsx`) with 295 expert-annotated papers from Brooks & Ainley (2022). The dataset contains:

- Research papers from 2010-2021
- Expert-validated classifications
- Comprehensive coverage of Ross Sea research themes
- Policy alignment annotations

### Dataset Citation
Brooks, C.M. & Ainley, D.G. (2022). "A Summary of United States Research and Monitoring in Support of the Ross Sea Region Marine Protected Area." *Diversity* 14(6), 447.

## Model Architecture

The framework combines:

- **SciBERT Base**: Pre-trained scientific language model
- **Shared Projection Layer**: Common representations across targets
- **Target-Specific Heads**: Specialized classification for each dimension
- **Enhanced Focal Loss**: Handles severe class imbalance
- **Gradient Accumulation**: Enables effective larger batch sizes

## Performance

With the full dataset (295 papers):
- **Research Themes**: Macro F1 = 0.58
- **CCAMLR Objectives**: Macro F1 = 0.88
- **Overall Weighted F1**: 0.70
- **Expert Agreement**: 78% (Jaccard similarity)

### BERT vs SciBERT Comparison
- SciBERT achieves 35% improvement over BERT baseline
- Statistical significance: p < 0.01
- Large effect size (Cohen's d = 2.0)

## Usage

### Training a Model

```python
# Basic training
python examples/train_model.py

# Training creates an ensemble of 5 models for improved predictions
```

### Classifying Papers

```bash
# Classify a PDF
python examples/predict_paper.py --pdf paper.pdf

# Use ensemble prediction (recommended)
python examples/predict_paper.py --pdf paper.pdf --ensemble

# Interactive mode
python examples/predict_paper.py --manual

# From command line
python examples/predict_paper.py --title "Title" --abstract "Abstract text..."
```

### BERT vs SciBERT Comparison

```bash
# Run the comparison experiment
python examples/bert_vs_scibert.py
```

## Repository Structure

```
multitarget-scibert-ross-sea/
├── src/                    # Core implementation
│   ├── data_loader.py
│   ├── data_preprocessing.py
│   ├── enhanced_scibert_model.py
│   ├── training_config.py
│   ├── experiment_config.py
│   ├── simple_model.py
│   ├── simple_preprocessing.py
│   └── simple_data_loader.py
├── examples/               # Example scripts
│   ├── train_model.py
│   ├── predict_paper.py
│   ├── bert_vs_scibert.py
│   ├── simple_predict.py
│   └── debug_class_mappings.py
├── data/                   # Data directory
├── models/                 # Saved models
├── test_papers/            # Sample papers for testing
│   ├── testpaper1.pdf
│   ├── testpaper2.pdf
│   ├── testpaper3.pdf
│   └── testpaper4.pdf
├── rosssea_research_dataset.xlsx
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
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

## Key Features

- **Multi-Target Learning**: Simultaneous classification across research themes and policy objectives
- **Domain-Specific Pretraining**: SciBERT provides superior performance for scientific text
- **Enhanced Text Processing**: Includes species, methods, and spatial metadata
- **Ensemble Predictions**: 5-model ensemble with voting strategies
- **Ross Sea Domain Detection**: Automatically filters non-Antarctic research

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{mccarthy2025research,
  title={Where Are the Research Gaps? AI-Assisted Multi-Target Classification for Research-Policy Alignment in Conservation Science},
  author={McCarthy, Chris and Brooks, Cassandra and Sternberg, Troy and Shaney, Kyle and Hoshino, Buho},
  journal={Ecological Informatics},
  year={2025},
  note={Under Review}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Transferability

While demonstrated on Ross Sea MPA research, this framework is designed for transferability to other conservation domains. The modular architecture separates domain-specific components from core methodology, facilitating adaptation to:

- Terrestrial protected areas
- Marine conservation networks
- Ecosystem-based management contexts
- Other research-policy alignment applications

## Support

For questions about the methodology or implementation:
- Open an issue in this repository
- Contact the authors through the paper's corresponding author
