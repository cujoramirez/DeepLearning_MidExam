# Deep Learning Midterm Exam - COMP6826001

**Student:** Gading Aditya Perdana (2702268725)  
**Course:** Deep Learning - COMP6826001  
**Institution:** Binus University

## Overview

This repository contains the complete implementation and report for the Deep Learning midterm examination, consisting of two independent case studies:

1. **Case 1: Occupant-Aware Room Climate Forecasting** - LSTM-based temperature prediction using IoT sensor data
2. **Case 2: Pine Tree DCGAN and Fake-Original Classification** - Generative adversarial network for CIFAR-100 pine tree synthesis and fake detection

## Repository Structure

```
deeplearning/
├── case1.py                    # Case 1: LSTM temperature forecasting
├── case2.py                    # Case 2: DCGAN + classifier pipeline
├── report.txt                  # Comprehensive LaTeX report
├── exam.txt                    # Assignment specifications
├── problem1_outputs/           # Case 1 results
│   ├── baseline_lstm_model.pt
│   ├── final/
│   │   ├── final_model.pt
│   │   ├── test_metrics.json
│   │   ├── final_loss.png
│   │   ├── scatter_predictions.png
│   │   └── residual_histogram.png
│   └── tuning/
│       └── tuning_results.csv
├── problem2_outputs/           # Case 2 results
│   ├── classifier_model.pt
│   ├── test_metrics.txt
│   ├── checkpoints/
│   │   └── final_model.pt
│   ├── figures/
│   │   ├── training_metrics.png
│   │   ├── real_vs_fake.png
│   │   ├── confusion_matrix.png
│   │   ├── samples_epoch_*.png  # 300 epoch snapshots
│   │   └── gan_training_analysis.txt
│   ├── generated/              # 500 synthetic images (.npy format)
│   └── generated_png/          # PNG versions of generated images
└── Room-Climate-Datasets/      # Input data (not included)
```

## Case 1: Room Climate Forecasting

### Approach
- **Model:** LSTM-based sequence regressor with 48-timestep sliding windows
- **Features:** Temperature, humidity, light sensors + first-order deltas (8 features total)
- **Architecture:** 256 hidden units, 2 layers, dropout 0.25, batch size 96
- **Training:** 3-phase pipeline (baseline → hyperparameter tuning → final training)

### Results
| Metric | Value |
|--------|-------|
| RMSE   | 0.1074°C |
| MAE    | 0.0869°C |
| R²     | 0.1709 |

See `problem1_outputs/final/test_metrics.json` for full results.

## Case 2: Pine Tree DCGAN

### Approach
- **Generator:** 4-layer transposed CNN (latent dim 100 → 32×32×3)
- **Discriminator:** 4-layer CNN with spectral normalization
- **Training:** 3000 epochs, batch size 225, BCE loss, AMP enabled
- **Classifier:** ResNet18 (pretrained) for fake/real discrimination

### Results
**GAN Training:**
- 3000 epochs completed in ~5.5 hours on RTX 3060
- 300 checkpoint visualizations documenting convergence
- Generator evolved from noise to recognizable pine tree structures

**Fake Detection Classifier:**
| Metric     | Value  |
|------------|--------|
| Accuracy   | 83.33% |
| Precision  | 78.41% |
| Recall     | 92.00% |
| F1 Score   | 84.66% |

## Requirements

```bash
# Core dependencies
torch>=2.0.0
torchvision
numpy
pandas
scikit-learn
matplotlib
seaborn
tqdm

# Optional
jupyter  # for exploring notebooks
```

## Usage

### Case 1: Temperature Forecasting
```bash
python case1.py --epochs 60 --tune-epochs 40 --final-epochs 120 --seq-len 48
```

### Case 2: DCGAN Training
```bash
python case2.py --epochs 3000 --comparison-count 10
```

### Converting Generated Images to PNG
The generated images are saved in `.npy` format by default. To convert them to PNG:
```bash
python npy2png.py
```
This reads from `problem2_outputs/generated/` and outputs to `problem2_outputs/generated_png/`.

### Quick Test (Lightweight)
```bash
python case1.py --quick
python case2.py --epochs 100
```

## Report

The comprehensive scientific report (`report.txt`) includes:
- Mathematical formulations of LSTM gates and GAN objectives
- Detailed methodology and preprocessing pipelines
- Hyperparameter tuning results
- Quantitative performance metrics
- Training dynamics analysis (3000-epoch GAN convergence)
- Discussion of mode collapse, discriminator advantage, and classifier bias
- Future work recommendations

**To compile:** Rename `report.txt` to `report.tex` and use any LaTeX compiler or upload to Overleaf.

## Key Findings

### Case 1
- Delta features improved R² from 0.1436 to 0.1709 (19% relative gain)
- Longer context windows (48 vs 32 timesteps) captured more temporal variance
- LSTM with 256 hidden units achieved best validation performance

### Case 2
- DCGAN exhibited 4 distinct training phases over 3000 epochs
- Mode collapse occurred at epoch 42, recovered by epoch 50
- Generator loss increased (+0.79) while visual quality improved
- Classifier showed high recall (92%) but lower precision (78%) due to GAN artifacts

## Hardware

- **GPU:** NVIDIA GeForce RTX 3060 Laptop (6.4 GB VRAM)
- **Framework:** PyTorch 2.x with automatic mixed precision (AMP)
- **Training Time:** ~5.5 hours for DCGAN, ~2 hours for LSTM experiments

## License

Academic project - COMP6826001 Deep Learning Midterm Examination

## Author

**Gading Aditya Perdana**  
Student ID: 2702268725  
Email: 2702268725@student.binus.ac.id  
Institution: Binus University

---

**Note:** Dataset files (`Room-Climate-Datasets/`, `cifar100-cache/`) are not included in this repository due to size constraints. CIFAR-100 will auto-download on first run. Room climate data should be placed in the `Room-Climate-Datasets/datasets-location_C/` directory.
