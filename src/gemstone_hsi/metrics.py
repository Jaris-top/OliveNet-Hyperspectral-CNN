from __future__ import annotations

import numpy as np


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    return float((true == pred).mean()) if len(true) else 0.0


def confusion_matrix(y_true: list[int], y_pred: list[int], num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[true, pred] += 1
    return matrix

