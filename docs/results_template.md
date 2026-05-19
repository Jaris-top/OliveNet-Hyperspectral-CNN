# Results Template

Use this file as the GitHub results page once real data experiments are run.

## Dataset

| Class | Samples | Notes |
| --- | ---: | --- |
| Natural olivine | 500 | Myanmar/Pakistan samples in the paper setup |
| Synthetic olivine | 500 | Flame fusion and hydrothermal synthesis |
| Dyed olivine | 300 | Iron-salt dyed low-quality natural olivine |

## Training Setup

| Item | Value |
| --- | --- |
| Spectral range | 400-1000 nm |
| Original bands | 224 |
| PCA components | 8-12 or cumulative variance >= 95% |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Epochs | 50 max |
| Early stopping | patience 5 |

## Classification Performance

| Class | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Natural | TBD | TBD | TBD | TBD |
| Synthetic | TBD | TBD | TBD | TBD |
| Dyed | TBD | TBD | TBD | TBD |
| Overall | TBD | TBD | TBD | TBD |

## Efficiency

| Model | Parameters | Inference time/sample | Accuracy |
| --- | ---: | ---: | ---: |
| Hyperspectral + SVM | TBD | TBD | TBD |
| VGG16 | TBD | TBD | TBD |
| ResNet18 | TBD | TBD | TBD |
| OliveNet | TBD | TBD | TBD |

## Notes

- Report hardware, PyTorch version, and whether inference used CPU or GPU.
- Report whether timing includes preprocessing/PCA.
- Add a confusion matrix and class-level failure examples.

