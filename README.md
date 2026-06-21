# Semantic Segmentation of Remote Sensing Images using Multi-Fusion Network

A comprehensive PyTorch implementation for semantic segmentation of remote sensing images using a multi-fusion deep learning network. This project combines multiple input modalities (RGB, multispectral, SAR) through intelligent fusion mechanisms.

## Features
- 🛰️ Multi-modal Fusion (RGB, Multispectral, SAR)
- 🔄 Advanced Fusion Strategies (Early, Late, Deep)
- 📊 State-of-the-art Architecture with Attention
- 🎯 Comprehensive Metrics
- 📈 Professional Training Pipeline
- 🚀 Inference Ready

## Installation

```bash
git clone https://github.com/Benny-Web-Code/semantic-segmentation-remote-sensing.git
cd semantic-segmentation-remote-sensing
pip install -r requirements.txt
```

## Quick Start

```bash
# Train
python scripts/train.py --config config.yaml --epochs 100

# Evaluate
python scripts/evaluate.py --checkpoint results/checkpoints/best_model.pth

# Predict
python scripts/predict.py --checkpoint results/checkpoints/best_model.pth --rgb-path image.png --ms-path ms.png --sar-path sar.png
```

## Project Structure

```
├── src/
│   ├── data/              # Data loading
│   ├── models/            # Model architectures
│   ├── training/          # Training pipeline
│   ├── inference/         # Prediction
│   └── utils/             # Utilities
├── scripts/               # Executable scripts
├── tests/                 # Unit tests
├── config.yaml            # Configuration
└── requirements.txt       # Dependencies
```

## License

MIT License
