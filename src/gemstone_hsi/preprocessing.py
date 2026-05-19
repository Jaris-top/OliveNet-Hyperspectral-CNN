from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


EPS = 1e-8


@dataclass
class PcaProjector:
    """Small wrapper around sklearn PCA for hyperspectral cubes."""

    n_components: int | float = 0.95
    random_state: int = 42

    def __post_init__(self) -> None:
        self.model = PCA(n_components=self.n_components, svd_solver="full", random_state=self.random_state)

    def fit(self, cubes: list[np.ndarray]) -> "PcaProjector":
        pixels = [cube.reshape(-1, cube.shape[-1]) for cube in cubes]
        self.model.fit(np.concatenate(pixels, axis=0))
        return self

    def transform(self, cube: np.ndarray) -> np.ndarray:
        h, w, _ = cube.shape
        reduced = self.model.transform(cube.reshape(-1, cube.shape[-1]))
        return reduced.reshape(h, w, -1).astype(np.float32)

    @property
    def components_count(self) -> int:
        return int(self.model.n_components_)


def reflectance_correction(
    cube: np.ndarray,
    white: np.ndarray | None = None,
    dark: np.ndarray | None = None,
) -> np.ndarray:
    """Convert raw intensity to reflectance with optional white/dark references."""
    cube = cube.astype(np.float32)
    if white is None:
        return cube

    white = white.astype(np.float32)
    if dark is None:
        corrected = cube / (white + EPS)
    else:
        dark = dark.astype(np.float32)
        corrected = (cube - dark) / (white - dark + EPS)
    return np.clip(corrected, 0.0, 1.5).astype(np.float32)


def normalize_cube(cube: np.ndarray) -> np.ndarray:
    """Robust min-max normalization per cube."""
    cube = np.nan_to_num(cube.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(cube, [1, 99])
    cube = np.clip(cube, lo, hi)
    return ((cube - lo) / (hi - lo + EPS)).astype(np.float32)


def center_crop_or_pad(cube: np.ndarray, size: int) -> np.ndarray:
    """Center crop or zero-pad a cube to a square spatial size."""
    h, w, bands = cube.shape
    out = np.zeros((size, size, bands), dtype=np.float32)

    crop_h = min(h, size)
    crop_w = min(w, size)
    src_y = max((h - crop_h) // 2, 0)
    src_x = max((w - crop_w) // 2, 0)
    dst_y = max((size - crop_h) // 2, 0)
    dst_x = max((size - crop_w) // 2, 0)

    out[dst_y : dst_y + crop_h, dst_x : dst_x + crop_w] = cube[
        src_y : src_y + crop_h,
        src_x : src_x + crop_w,
    ]
    return out

