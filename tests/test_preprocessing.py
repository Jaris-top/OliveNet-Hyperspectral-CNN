from __future__ import annotations

import numpy as np

from gemstone_hsi.preprocessing import center_crop_or_pad, normalize_cube, reflectance_correction


def test_reflectance_correction_with_white_and_dark() -> None:
    cube = np.ones((4, 4, 3), dtype=np.float32) * 5
    white = np.ones((4, 4, 3), dtype=np.float32) * 10
    dark = np.ones((4, 4, 3), dtype=np.float32) * 2
    corrected = reflectance_correction(cube, white, dark)
    assert corrected.shape == cube.shape
    assert np.allclose(corrected, 0.375)


def test_center_crop_or_pad() -> None:
    cube = np.ones((2, 3, 4), dtype=np.float32)
    resized = center_crop_or_pad(cube, 5)
    assert resized.shape == (5, 5, 4)
    assert resized.sum() == cube.sum()


def test_normalize_cube_handles_nan() -> None:
    cube = np.array([[[0.0, np.nan], [2.0, 4.0]]], dtype=np.float32)
    normalized = normalize_cube(cube)
    assert np.isfinite(normalized).all()
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0

