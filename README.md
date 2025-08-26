# GeoNeXt: Efficient Landslide Mapping using ConvNeXt V2 and PSA-ASPP

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **GeoNeXt: Efficient landslide mapping using a pre-trained ConvNeXt V2 encoder with a PSA-ASPP decoder**
> 
> *Rodrigo Uribe-Ventura, Willem Viveen, Ferdinand Pineda-Ancco, César Beltrán-Castallon*

## Overview

GeoNeXt is a novel deep learning architecture for landslide detection and mapping that combines a pre-trained ConvNeXt V2 encoder with a decoder integrating Pyramid Squeeze Attention (PSA) and Atrous Spatial Pyramid Pooling (ASPP). The model achieves state-of-the-art performance with 10× fewer parameters than transformer-based approaches while maintaining superior accuracy across diverse geographical regions.

### Key Features

- **High Performance**: F1 scores of 94.25%, 86.43%, and 92.27% on Bijie, Landslide4Sense, and GVLM datasets
- **Computational Efficiency**: 32.2M parameters vs 312.5M for SAM-based methods
- **Zero-shot Transferability**: Strong cross-domain performance without retraining
- **Multi-scale Detection**: Handles landslides from small flows to large-scale events

## Architecture

GeoNeXt follows a U-Net-based encoder-decoder structure:

- **Encoder**: Pre-trained ConvNeXt V2-Tiny with domain adaptation
- **Decoder**: Novel PSA-ASPP design for multi-scale feature aggregation
- **Loss Function**: Composite loss combining BCE, uncertainty-weighted Dice, and focal-Tversky

## Installation

### Setup

```bash
# Clone the repository
git clone https://github.com/ruribev/GeoNeXt.git
cd GeoNeXt

# Install dependencies
pip install -r requirements.txt
```

### Download Required Resources

Download the following required folders from Google Drive:

- **Pre-trained Weights**: [Download pretrained folder](https://drive.google.com/file/d/1uz7Y_2zpBiR54c_p_DQm1QDwql_Lzrh4/view?usp=sharing)
- **Trained Models**: [Download experiments folder](https://drive.google.com/file/d/11N16l2eXEn0YWJ3S1pFC-kSAAcn9HlBN/view?usp=drive_link)
- **Datasets**: [Download dataset folder](https://drive.google.com/file/d/1lHKEfCMBnR93dGvuT4_1m4XPWRJc5Psz/view?usp=drive_link)

Extract these folders in the root directory of the project to maintain the proper structure.

## Quick Start

### Training

```python
from utils.experiments import TrainingExperiment, run_experiment

# Configure training experiment
config = {
    'dataset': 'BJL',  # or 'L4S', 'GVLM', 'CAS'
    'model_variant': 'tiny',
    'gpu_id': 0,
    'batch_size': 8,
    'epochs': 60,
    'lr': 3.5e-3,
    'convnextv2_pretrained_weights': 'pretrained/convnextv2_tiny_22k_224_ema.pt',
    'name_suffix': 'your_experiment_name'
}

# Run training
experiment = TrainingExperiment(config)
run_experiment(experiment)
```

### Benchmarking

```python
from utils.benchmarks import Benchmark

# Initialize benchmark
benchmark = Benchmark(
    dataset='BJL',
    gpu_id=0,  
    model_variant='tiny',
    model_weight_path="experiments/GeoNeXt_BJL/best_model.pth.tar",
    batch_size=2
)

# Run evaluation
results = benchmark.run_benchmark()
benchmark.show_metrics()
benchmark.show_examples(num_examples=5, min_mask_percentage=0.1)
```

### Jupyter Notebooks

We provide ready-to-use Jupyter notebooks:

- `train.ipynb`: Complete training pipeline example
- `benchmark.ipynb`: Evaluation of main datasets

### Data Preparation

1. Download the complete dataset folder from [this link](https://drive.google.com/file/d/1lHKEfCMBnR93dGvuT4_1m4XPWRJc5Psz/view?usp=drive_link)
2. Extract and place in the project root directory
3. The structure should be:
```
dataset/
├── BJL/
├── L4S/
├── GVLM/
└── CAS/
```

## Project Structure

```
GeoNeXt/
├── models/
│   ├── __init__.py
│   ├── GeoNeXt.py          # Main model architecture
│   └── ConvNeXtV2.py       # ConvNeXt V2 backbone
├── utils/
│   ├── __init__.py
│   ├── train.py            # Training utilities
│   ├── evaluator.py        # Evaluation metrics
│   ├── benchmarks.py       # Benchmarking tools
│   ├── experiments.py      # Experiment management
│   ├── augmentations.py    # Data augmentation
│   ├── losses.py           # Loss functions
│   ├── ema.py              # Exponential moving average
│   └── dataset.py          # Dataset utilities
├── dataset/                # Dataset directory (download from link above)
├── pretrained/             # Pre-trained weights (download from link above)
├── experiments/            # Training outputs (download from link above)
├── train.ipynb            # Training notebook
├── benchmark.ipynb        # Evaluation notebook
├── requirements.txt       # Dependencies
├── LICENSE                # MIT License
└── README.md             # This file
```

## Pre-trained Models & Weights

### Available Model Weights

Pre-trained models and weights are available in the downloads section:

- **ConvNeXt V2 Tiny Backbone**: ImageNet-22K pre-trained weights (32.2M parameters)
  - File: `convnextv2_tiny_22k_224_ema.pt`
  - Location: pretrained folder
  
- **GeoNeXt Complete Models**: Fine-tuned models for each dataset
  - **BJL Dataset**: `GeoNeXt_BJL/best_model.pth.tar` (F1: 94.25%)
  - **L4S Dataset**: `GeoNeXt_L4S/best_model.pth.tar` (F1: 86.43%)
  - **GVLM Dataset**: `GeoNeXt_GVLM/best_model.pth.tar` (F1: 92.27%)
  - **CAS Domain-adapted**: `GeoNeXt_CAS/encoder_weights.pth` 
  - Location: experiments folder

Download the [pretrained folder](https://drive.google.com/file/d/1uz7Y_2zpBiR54c_p_DQm1QDwql_Lzrh4/view?usp=sharing) and [experiments folder](https://drive.google.com/file/d/11N16l2eXEn0YWJ3S1pFC-kSAAcn9HlBN/view?usp=drive_link) to access all pre-trained models and weights.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use GeoNeXt in your research, please cite our work:

```bibtex
@article{uribe2025geonext,
  title={GeoNeXt: Efficient landslide mapping using a pre-trained ConvNeXt V2 encoder with a PSA-ASPP decoder},
  author={Uribe-Ventura, Rodrigo and Viveen, Willem and Pineda-Ancco, Ferdinand and Beltrán-Castallon, César},
  journal={Under Review},
  year={2025}
}
```

*Paper currently under review. BibTeX entry will be updated upon publication.* 

## Contact

- **Corresponding Author**: Rodrigo Uribe-Ventura (a20234215@pucp.edu.pe)
- **Institution**: Pontificia Universidad Católica del Perú
- **GitHub**: [@ruribev](https://github.com/ruribev)

---

For detailed methodology and experimental results, please refer to our paper: *"GeoNeXt: Efficient landslide mapping using a pre-trained ConvNeXt V2 encoder with a PSA-ASPP decoder"*