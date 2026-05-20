from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


CLASSES = ["natural", "synthetic", "dyed"]
EPS = 1e-8


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int


@dataclass
class PcaProjector:
    """PCA wrapper for compressing hyperspectral cubes to the paper's 8-12 bands."""

    n_components: int | float = 0.95 # Paper Eq.(3): retain ≥95% cumulative variance, K=8-12
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


def reflectance_correction( # Paper Eq.(5): R = (I-D)/(W-D)
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


def augment_cube(cube: np.ndarray) -> np.ndarray: # Paper Section III-C: spectral shift ±5nm, rotation ±15°, brightness ±10%
    """Apply lightweight spectral/spatial perturbations described in the paper."""
    if random.random() < 0.5:
        cube = np.roll(cube, shift=random.choice([-1, 1]), axis=2)
    if random.random() < 0.5:
# Paper Section III-C: spatial augmentation ±15° rotation
        from scipy.ndimage import rotate as scipy_rotate
        angle = random.uniform(-15, 15)
        cube = scipy_rotate(cube, angle, axes=(0, 1), reshape=False)
    if random.random() < 0.5:
        cube = np.clip(cube * random.uniform(0.9, 1.1), 0.0, 1.5)
    return cube.astype(np.float32)


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


def load_optional_calibration(root: str | Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    cal = Path(root) / "calibration"
    white_path = cal / "white.npy"
    dark_path = cal / "dark.npy"
    white = np.load(white_path) if white_path.exists() else None
    dark = np.load(dark_path) if dark_path.exists() else None
    return white, dark


class HyperspectralDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        image_size: int,
        projector: PcaProjector | None = None,
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


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OliveNet(nn.Module):
    """Lightweight CNN for natural, synthetic, and dyed olivine classification."""

    def __init__(self, input_channels: int, num_classes: int = 3, dropout: float = 0.5) -> None:
        super().__init__()
# Paper Section III-C, Block 1:
# "32 3×3×K 3D convolution kernels to extract low-level spectral-spatial features"
        self.block1_3d = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(input_channels, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
# Paper Section III-C, Block 2-3:
# "2D convolution to reduce computational complexity vs pure 3D-CNN by 60%"
        self.blocks_2d = nn.Sequential(
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)          # (B, K, H, W) → (B, 1, K, H, W)
        x = self.block1_3d(x)       # → (B, 32, 1, H, W)
        x = x.squeeze(2)            # → (B, 32, H, W)
        x = self.blocks_2d(x)
        return self.head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    return float((true == pred).mean()) if len(true) else 0.0


def confusion_matrix(y_true: list[int], y_pred: list[int], num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[true, pred] += 1
    return matrix


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fit_projector(samples: list[Sample], root: str, n_components: int | float, seed: int) -> PcaProjector:
    white, dark = load_optional_calibration(root)
    cubes = []
    for sample in samples:
        cube = np.load(sample.path)
        cube = reflectance_correction(cube, white, dark)
        cubes.append(normalize_cube(cube))
    return PcaProjector(n_components=n_components, random_state=seed).fit(cubes)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
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


def create_demo_dataset(output: str, samples_per_class: int, size: int, bands: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    root = Path(output)
    for label in CLASSES:
        folder = root / label
        folder.mkdir(parents=True, exist_ok=True)
        for idx in range(samples_per_class):
            cube = make_cube(label, size, bands, rng)
            np.save(folder / f"{label}_{idx:03d}.npy", cube)
    print(f"Wrote demo dataset to {root}")


def default_config(data_root: str = "demo_data", output_dir: str = "demo_outputs") -> dict[str, Any]:
    return {
        "seed": 42,
        "classes": CLASSES,
        "data": {
            "root": data_root,
            "image_size": 64,
            "pca_variance": 0.95,
            "augment": True,
            "split": {"train": 0.70, "val": 0.15, "test": 0.15},
        },
        "model": {"dropout": 0.5},
        "training": {
            "output_dir": output_dir,
            "batch_size": 8,
            "epochs": 10,
            "learning_rate": 0.001, # Paper Section III-C: Adam lr=0.001, decay 10% per 5 epochs
            "weight_decay": 0.00001,
            "lr_decay_step": 5,
            "lr_decay_gamma": 0.9,
            "early_stopping_patience": 5, # Paper Section III-C: stop if val_acc no improvement for 5 epochs
        },
    }


def load_config(path: str | Path | None, data_root: str, output_dir: str) -> dict[str, Any]:
    if path is None:
        return default_config(data_root=data_root, output_dir=output_dir)
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def train_model(cfg: dict[str, Any]) -> None:
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

    n_components: int | float = float(data_cfg["pca_variance"]) if data_cfg.get("pca_variance") else int(data_cfg.get("pca_components", 10))
    projector = fit_projector(train_samples, data_cfg["root"], n_components, int(cfg.get("seed", 42)))
    white, dark = load_optional_calibration(data_cfg["root"])

    datasets = {
        "train": HyperspectralDataset(train_samples, int(data_cfg["image_size"]), projector, white, dark, bool(data_cfg.get("augment", False))),
        "val": HyperspectralDataset(val_samples, int(data_cfg["image_size"]), projector, white, dark),
        "test": HyperspectralDataset(test_samples, int(data_cfg["image_size"]), projector, white, dark),
    }
    loaders = {
        name: DataLoader(dataset, batch_size=int(train_cfg["batch_size"]), shuffle=(name == "train"), num_workers=0)
        for name, dataset in datasets.items()
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OliveNet(projector.components_count, len(classes), float(cfg["model"].get("dropout", 0.5))).to(device)
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
        "confusion_matrix": confusion_matrix(test_metrics["y_true"], test_metrics["y_pred"], len(classes)).tolist(),
        "parameters": count_parameters(model),
        "pca_components": projector.components_count,
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


def predict(checkpoint_path: str, cube_path: str, white_path: str | None = None, dark_path: str | None = None) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]
    model = OliveNet(
        input_channels=int(checkpoint["input_channels"]),
        num_classes=len(classes),
        dropout=float(checkpoint["config"]["model"].get("dropout", 0.5)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    cube = np.load(cube_path)
    white = np.load(white_path) if white_path else None
    dark = np.load(dark_path) if dark_path else None
    cube = reflectance_correction(cube, white, dark)
    cube = normalize_cube(cube)
    cube = checkpoint["pca"].transform(cube.reshape(-1, cube.shape[-1])).reshape(cube.shape[0], cube.shape[1], -1)
    cube = center_crop_or_pad(cube, int(checkpoint["image_size"]))
    tensor = torch.from_numpy(cube.transpose(2, 0, 1)).float().unsqueeze(0)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1).squeeze(0).tolist()

    prediction = {
        "label": classes[int(np.argmax(probabilities))],
        "probabilities": {label: float(prob) for label, prob in zip(classes, probabilities)},
    }
    print(json.dumps(prediction, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="OliveNet hyperspectral olivine authenticity identification.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser("demo-data", help="Create a small synthetic hyperspectral demo dataset.")
    demo_parser.add_argument("--output", default="demo_data")
    demo_parser.add_argument("--samples-per-class", type=int, default=20)
    demo_parser.add_argument("--size", type=int, default=64)
    demo_parser.add_argument("--bands", type=int, default=32)
    demo_parser.add_argument("--seed", type=int, default=42)

    train_parser = subparsers.add_parser("train", help="Train OliveNet from a YAML config or default demo settings.")
    train_parser.add_argument("--config", default=None)
    train_parser.add_argument("--data-root", default="demo_data")
    train_parser.add_argument("--output-dir", default="demo_outputs")

    predict_parser = subparsers.add_parser("predict", help="Run single-cube inference.")
    predict_parser.add_argument("--checkpoint", required=True)
    predict_parser.add_argument("--cube", required=True)
    predict_parser.add_argument("--white", default=None)
    predict_parser.add_argument("--dark", default=None)

    args = parser.parse_args()
    if args.command == "demo-data":
        create_demo_dataset(args.output, args.samples_per_class, args.size, args.bands, args.seed)
    elif args.command == "train":
        train_model(load_config(args.config, args.data_root, args.output_dir))
    elif args.command == "predict":
        predict(args.checkpoint, args.cube, args.white, args.dark)


if __name__ == "__main__":
    main()
