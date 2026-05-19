from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


CLASSES = ["natural", "synthetic", "dyed"]


def spectral_signature(label: str, wavelengths: np.ndarray) -> np.ndarray:
    baseline = 0.45 + 0.05 * np.sin(wavelengths / 55)
    if label == "natural":
        absorption = 0.18 * np.exp(-0.5 * ((wavelengths - 550) / 28) ** 2)
        return baseline - absorption
    if label == "synthetic":
        absorption = 0.08 * np.exp(-0.5 * ((wavelengths - 545) / 18) ** 2)
        return baseline - absorption + 0.03
    absorption = 0.12 * np.exp(-0.5 * ((wavelengths - 550) / 25) ** 2)
    dye = 0.16 * np.exp(-0.5 * ((wavelengths - 740) / 35) ** 2)
    return baseline - absorption - dye


def make_cube(label: str, size: int, bands: int, rng: np.random.Generator) -> np.ndarray:
    wavelengths = np.linspace(400, 1000, bands)
    signature = spectral_signature(label, wavelengths)
    yy, xx = np.mgrid[:size, :size]
    texture = 1.0 + 0.08 * np.sin(xx / 7) + 0.05 * np.cos(yy / 9)
    if label == "natural":
        texture += 0.04 * np.sin((xx + yy) / 11)
    elif label == "synthetic":
        texture += 0.02 * np.cos(xx / 5)
    else:
        texture += 0.07 * ((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (size**2)
    cube = texture[..., None] * signature[None, None, :]
    cube += rng.normal(0, 0.015, size=cube.shape)
    return np.clip(cube, 0, 1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/demo")
    parser.add_argument("--samples-per-class", type=int, default=20)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--bands", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    root = Path(args.output)
    for label in CLASSES:
        folder = root / label
        folder.mkdir(parents=True, exist_ok=True)
        for idx in range(args.samples_per_class):
            cube = make_cube(label, args.size, args.bands, rng)
            np.save(folder / f"{label}_{idx:03d}.npy", cube)
    print(f"Wrote demo dataset to {root}")


if __name__ == "__main__":
    main()

