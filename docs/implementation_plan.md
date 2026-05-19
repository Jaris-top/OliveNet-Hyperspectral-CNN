# Implementation Plan

## Goal

Convert the hyperspectral olivine authenticity identification paper into a GitHub-ready project that can be run, inspected, extended, and eventually connected to real acquisition hardware.

## Method Implemented

The paper pipeline is implemented as:

1. **Data acquisition contract**: hyperspectral cubes are saved as `.npy` arrays with shape `(H, W, B)`.
2. **Reflectance correction**: optional dark/white calibration converts raw intensity into reflectance.
3. **Preprocessing**: bad values are clipped, cubes are normalized, resized/cropped, and reduced with PCA.
4. **Model**: OliveNet uses a lightweight first-stage spectral-spatial convolution followed by 2D convolutional blocks and global average pooling.
5. **Training**: Adam optimizer, cross-entropy loss, StepLR decay, early stopping, best checkpoint export.
6. **Inference**: a single cube is preprocessed consistently and classified into natural/synthetic/dyed.

## Real Data Checklist

- Confirm camera export format and convert ENVI/RAW/TIFF stacks to `(H, W, B)` `.npy`.
- Save white reference and optional dark current cubes under `data/raw/calibration/`.
- Verify each class label with gemological certification.
- Keep train/val/test splits sample-level, not capture-level, to avoid leakage from repeated scans.
- Record acquisition metadata: camera, illumination, integration time, wavelength range, and spatial resolution.

## GitHub Presentation Checklist

- Add representative screenshots of spectral curves and PCA components once real data is available.
- Add a small public sample dataset or a download script.
- Add trained checkpoint metadata, not just the `.pt` file.
- Add a results table comparing OliveNet, SVM, VGG16, and ResNet18 after real experiments.
- Add a model card explaining intended use and limitations.

