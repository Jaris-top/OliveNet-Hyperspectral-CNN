# Data Format

## Cube Format

The training code expects each hyperspectral sample as a `.npy` file:

```python
cube.shape == (height, width, bands)
cube.dtype in [float32, float64, uint16]
```

For the paper setup, the expected raw cube is approximately:

```text
height = 512
width  = 512
bands  = 224
range  = 400-1000 nm
resolution = 3 nm
```

## Class Folders

```text
data/raw/
  natural/
  synthetic/
  dyed/
```

The folder name is the class label. Each file is treated as one sample. If one gemstone is scanned multiple times, keep repeated scans in the same split to avoid leakage.

## Calibration

Optional calibration files:

```text
data/raw/calibration/white.npy
data/raw/calibration/dark.npy
```

Reflectance is computed as:

```text
R = (sample - dark) / (white - dark)
```

If `dark.npy` is unavailable, the code falls back to:

```text
R = sample / white
```

## Converting Camera Exports

Many hyperspectral cameras export ENVI `.hdr/.raw`, TIFF stacks, or vendor-specific binaries. Convert those files to `(H, W, B)` NumPy arrays before training. Keep the wavelength metadata in a sidecar file such as:

```text
sample_001.npy
sample_001_wavelengths.csv
```

The current model does not require wavelength metadata at runtime, but keeping it is important for auditing absorption peaks and comparing experiments.

