from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config
from .dataset import HyperspectralDataset, discover_samples, load_optional_calibration, split_samples
from .metrics import accuracy, confusion_matrix
from .model import OliveNet, count_parameters
from .preprocessing import PcaProjector, normalize_cube, reflectance_correction


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fit_projector(samples, root: str, n_components: int | float, seed: int):
    white, dark = load_optional_calibration(root)
    cubes = []
    for sample in samples:
        cube = np.load(sample.path)
        cube = reflectance_correction(cube, white, dark)
        cubes.append(normalize_cube(cube))
    return PcaProjector(n_components=n_components, random_state=seed).fit(cubes)


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []

    for x, y in tqdm(loader, leave=False):
        x = x.to(device)
        y = y.to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        losses.append(float(loss.item()))
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": accuracy(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    classes = cfg["classes"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = discover_samples(data_cfg["root"], classes)
    train_samples, val_samples, test_samples = split_samples(
        samples,
        train_ratio=float(data_cfg["split"]["train"]),
        val_ratio=float(data_cfg["split"]["val"]),
        seed=int(cfg.get("seed", 42)),
    )

    n_components: int | float
    if data_cfg.get("pca_variance"):
        n_components = float(data_cfg["pca_variance"])
    else:
        n_components = int(data_cfg.get("pca_components", 10))
    projector = fit_projector(train_samples, data_cfg["root"], n_components, int(cfg.get("seed", 42)))
    white, dark = load_optional_calibration(data_cfg["root"])

    datasets = {
        "train": HyperspectralDataset(
            train_samples,
            image_size=int(data_cfg["image_size"]),
            projector=projector,
            white=white,
            dark=dark,
            augment=bool(data_cfg.get("augment", False)),
        ),
        "val": HyperspectralDataset(
            val_samples,
            image_size=int(data_cfg["image_size"]),
            projector=projector,
            white=white,
            dark=dark,
        ),
        "test": HyperspectralDataset(
            test_samples,
            image_size=int(data_cfg["image_size"]),
            projector=projector,
            white=white,
            dark=dark,
        ),
    }
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=(name == "train"),
            num_workers=0,
        )
        for name, dataset in datasets.items()
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OliveNet(
        input_channels=projector.components_count,
        num_classes=len(classes),
        dropout=float(cfg["model"].get("dropout", 0.5)),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(train_cfg.get("lr_decay_step", 5)),
        gamma=float(train_cfg.get("lr_decay_gamma", 0.9)),
    )

    best_val = -1.0
    stale_epochs = 0
    history = []
    best_path = output_dir / "best_model.pt"

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        train_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_metrics = run_epoch(model, loaders["val"], criterion, device)
        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))

        if val_metrics["accuracy"] > best_val:
            best_val = val_metrics["accuracy"]
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": classes,
                    "image_size": int(data_cfg["image_size"]),
                    "input_channels": projector.components_count,
                    "pca": projector.model,
                    "config": cfg,
                },
                best_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= int(train_cfg.get("early_stopping_patience", 5)):
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = run_epoch(model, loaders["test"], criterion, device)
    result = {
        "best_val_accuracy": best_val,
        "test_accuracy": test_metrics["accuracy"],
        "confusion_matrix": confusion_matrix(
            test_metrics["y_true"],
            test_metrics["y_pred"],
            num_classes=len(classes),
        ).tolist(),
        "parameters": count_parameters(model),
        "pca_components": projector.components_count,
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
