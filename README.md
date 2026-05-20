# OliveNet: Hyperspectral-CNN for Rapid Olivine Authentication
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![IEEE](https://img.shields.io/badge/IEEE-ICEACE2025-red)

![OliveNet hyperspectral workflow](olivenet-workflow-cover.png)

This repository converts the paper **"Research on a Rapid Gemstone Authenticity Identification Method Based on Hyperspectral Imaging and Convolutional Neural Network"** into a compact, runnable Python project for olivine authenticity identification.

The project focuses on a complete paper-to-code workflow: calibrated hyperspectral cube preprocessing, PCA spectral compression, lightweight CNN training, evaluation, and single-sample inference for three classes: `natural`, `synthetic`, and `dyed`.

## Technical Workflow

![Detailed OliveNet technical workflow](olivenet-technical-workflow.png)
*Click to enlarge — or see the [full-size image](olivenet-technical-workflow.png)*

The complete runnable implementation is contained in [`olivenet.py`](olivenet.py), including preprocessing, PCA projection, OliveNet model definition, training, evaluation, synthetic demo data generation, and single-cube inference.

## Highlights

- Implements the paper-inspired **OliveNet** CNN in a single readable Python file.
- Supports hyperspectral cubes stored as `.npy` arrays with shape `(height, width, bands)`.
- Includes white/dark reflectance correction: `R = (I - D) / (W - D)`.
- Uses PCA to retain `>=95%` cumulative variance, typically reducing 224 bands to `8-12` components.
- Provides sample-level `70% / 15% / 15%` train, validation, and test splitting.
- Exports accuracy, confusion matrix, PCA component count, model parameter count, and training history.

## Performance

| Category | Accuracy | F1 |
|---|---|---|
| Natural Olivine | 99.2% | 0.99 |
| Synthetic Olivine | 98.5% | 0.98 |
| Dyed Olivine | 96.8% | 0.97 |
| **Overall** | **98.3%** | **0.98** |

- +6.9% accuracy vs SVM baseline
- 78% fewer parameters than ResNet18
- 0.45s inference per sample
- ≤2.1% accuracy fluctuation under real-world perturbations

## Repository Contents

```text
README.md
olivenet.py
olivenet-workflow-cover.png
olivenet-technical-workflow.png
```

## Paper-To-Code Mapping

| Paper method | Implementation |
| --- | --- |
| Hyperspectral cube input, 400-1000 nm, 224 bands | `olivenet.py` data loading contract |
| White/dark reflectance correction | `reflectance_correction()` |
| Robust cube normalization | `normalize_cube()` |
| PCA spectral compression | `PcaProjector` |
| Spectral/spatial augmentation | `augment_cube()` |
| OliveNet CNN classifier | `OliveNet` |
| Adam + cross-entropy + early stopping | `train_model()` |
| Single-cube class prediction | `predict()` |

## Quick Start

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scikit-learn torch tqdm pyyaml
```

Run the built-in synthetic demo:

```bash
python olivenet.py demo-data --output demo_data --samples-per-class 12
python olivenet.py train --data-root demo_data --output-dir demo_outputs
python olivenet.py predict \
  --checkpoint demo_outputs/best_model.pt \
  --cube demo_data/natural/natural_000.npy
```

The demo data is synthetic and exists only to verify that the full pipeline runs end to end. Scientific reporting should use calibrated real hyperspectral olivine cubes with verified sample provenance.

## Validation Scope

This repository provides a reproducible engineering implementation of the paper-inspired OliveNet workflow. The built-in demo dataset is synthetic and is intended only to test the code path from data generation to inference.

Reported scientific performance should be evaluated with calibrated real hyperspectral cubes, verified gemstone labels, sample-level train/validation/test splitting, and external validation. The original paper's reported accuracy is treated as a reference target, not as a claim reproduced by this repository.

## Real Data Format

Prepare real data as:

```text
your_data/
  natural/
    sample_001.npy
  synthetic/
    sample_001.npy
  dyed/
    sample_001.npy
  calibration/
    white.npy
    dark.npy
```

Each sample should be a NumPy cube:

```python
cube.shape == (height, width, bands)
```

For the paper setup, the expected acquisition range is:

- spectral range: `400-1000 nm`
- spectral resolution: about `3 nm`
- original bands: `224`
- classes: `natural`, `synthetic`, `dyed`
- target single-sample inference time: about `0.45 s`

## Notes

This project is a reproducible engineering implementation of the research workflow, not a certified gemological instrument. Real deployment requires calibrated acquisition hardware, controlled illumination, verified labels, sample-level data splitting, and external validation.

## Citation
If you use this code, please cite the original paper:

```bibtex
@INPROCEEDINGS{11439579,
  author={Yu, Jiarui},
  booktitle={2025 IEEE 3rd International Conference on Electrical, Automation and Computer Engineering (ICEACE)}, 
  title={Research on a Rapid Gemstone Authenticity Identification Method Based on Hyperspectral Imaging and Convolutional Neural Network}, 
  year={2025},
  volume={},
  number={},
  pages={932-936},
  keywords={Accuracy;Fluctuations;Transfer learning;Lighting;Feature extraction;Convolutional neural networks;Object recognition;Contamination;Hyperspectral imaging;Principal component analysis;Hyper-spectral imaging;Convolutional Neural Network;Identification of olivine authenticity;Rapid identification;Lightweight model},
  doi={10.1109/ICEACE67491.2025.11439579}}
