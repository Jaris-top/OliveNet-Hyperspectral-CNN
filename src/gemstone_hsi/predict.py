from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .model import OliveNet
from .preprocessing import center_crop_or_pad, normalize_cube, reflectance_correction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cube", required=True)
    parser.add_argument("--white", default=None, help="Optional white reference .npy file")
    parser.add_argument("--dark", default=None, help="Optional dark current .npy file")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]
    model = OliveNet(
        input_channels=int(checkpoint["input_channels"]),
        num_classes=len(classes),
        dropout=float(checkpoint["config"]["model"].get("dropout", 0.5)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    cube = np.load(args.cube)
    white = np.load(args.white) if args.white else None
    dark = np.load(args.dark) if args.dark else None
    cube = reflectance_correction(cube, white, dark)
    cube = normalize_cube(cube)
    cube = checkpoint["pca"].transform(cube.reshape(-1, cube.shape[-1])).reshape(
        cube.shape[0],
        cube.shape[1],
        -1,
    )
    cube = center_crop_or_pad(cube, int(checkpoint["image_size"]))
    tensor = torch.from_numpy(cube.transpose(2, 0, 1)).float().unsqueeze(0)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1).squeeze(0).tolist()

    prediction = {
        "label": classes[int(np.argmax(probabilities))],
        "probabilities": {label: float(prob) for label, prob in zip(classes, probabilities)},
    }
    print(json.dumps(prediction, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
