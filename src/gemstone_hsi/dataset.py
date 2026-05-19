from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import center_crop_or_pad, normalize_cube, reflectance_correction


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int


def discover_samples(root: str | Path, classes: list[str]) -> list[Sample]:
    root = Path(root)
    samples: list[Sample] = []
    for label, cls in enumerate(classes):
        for path in sorted((root / cls).glob("*.npy")):
            samples.append(Sample(path=path, label=label))
    if not samples:
        raise FileNotFoundError(f"No .npy cubes found under {root}")
    return samples


def split_samples(
    samples: list[Sample],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    rng = random.Random(seed)
    by_label: dict[int, list[Sample]] = {}
    for sample in samples:
        by_label.setdefault(sample.label, []).append(sample)

    train: list[Sample] = []
    val: list[Sample] = []
    test: list[Sample] = []
    for group in by_label.values():
        rng.shuffle(group)
        n = len(group)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio)) if n - n_train > 1 else 0
        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val :])
    return train, val, test


class HyperspectralDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        image_size: int,
        projector=None,
        white: np.ndarray | None = None,
        dark: np.ndarray | None = None,
        augment: bool = False,
    ) -> None:
        self.samples = samples
        self.image_size = image_size
        self.projector = projector
        self.white = white
        self.dark = dark
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        cube = np.load(sample.path)
        cube = reflectance_correction(cube, self.white, self.dark)
        cube = normalize_cube(cube)
        if self.projector is not None:
            cube = self.projector.transform(cube)
        cube = center_crop_or_pad(cube, self.image_size)
        if self.augment:
            cube = augment_cube(cube)
        tensor = torch.from_numpy(cube.transpose(2, 0, 1)).float()
        return tensor, torch.tensor(sample.label, dtype=torch.long)


def augment_cube(cube: np.ndarray) -> np.ndarray:
    """Apply lightweight spectral/spatial perturbations from the paper."""
    if random.random() < 0.5:
        shift = random.choice([-1, 1])
        cube = np.roll(cube, shift=shift, axis=2)
    if random.random() < 0.5:
        cube = np.rot90(cube, k=random.randint(0, 3), axes=(0, 1)).copy()
    if random.random() < 0.5:
        brightness = random.uniform(0.9, 1.1)
        cube = np.clip(cube * brightness, 0.0, 1.5)
    return cube.astype(np.float32)


def load_optional_calibration(root: str | Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    cal = Path(root) / "calibration"
    white_path = cal / "white.npy"
    dark_path = cal / "dark.npy"
    white = np.load(white_path) if white_path.exists() else None
    dark = np.load(dark_path) if dark_path.exists() else None
    return white, dark

