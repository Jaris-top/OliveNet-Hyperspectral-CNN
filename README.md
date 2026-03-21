# OliveNet: Deep Learning for Mineralogical Characterisation

![Python](https://img.shields.io/badge/Python-3.8-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10-red.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)
![Publication](https://img.shields.io/badge/Publication-IEEE_ICEACE_2025-orange.svg)

<p align="center">
  <img src="cover.png" alt="Hyperspectral Scanning Concept" width="800">
</p>

## 📌 Project Overview
This repository contains the core methodology and conceptual framework for **OliveNet**, a lightweight Convolutional Neural Network (CNN) designed for the rapid authenticity identification of olivine (peridot) using hyperspectral imaging (400-1000nm). 

## 🔬 Dataset & PCA Dimensionality Reduction
* **Dataset:** 1,300 olivine samples (500 Natural, 500 Synthetic, 300 Dyed).
* **Dimensionality Reduction:** Applied Principal Component Analysis (PCA) to reduce 224 spectral bands down to 8-12 principal components, retaining >=95% of data variance while filtering noise.

## ⚙️ Model Architecture (OliveNet)
To meet the requirements of on-site rapid identification, OliveNet was designed with a focus on efficiency and parameter reduction:
* **Hybrid Convolution:** Utilised 3D convolutions for initial spectral-spatial feature extraction, followed by 2D convolutions to significantly reduce computational load.
* **Optimization:** Applied L1 regularization for sparsity and 16-bit floating-point weight quantization.

## 🚀 Performance Highlights
* **Accuracy:** 98.3% overall classification accuracy (99.2% for natural olivine).
* **Inference Speed:** 0.45 seconds per sample.
* **Size:** 1.2M parameters (78% smaller than ResNet18 and 3X faster).

## 🏆 Academic & Professional Outcomes
* **Publication:** Accepted by 2025 IEEE 3rd International Conference on Electrical, Automation and Computer Engineering (ICEACE 2025). IEEE.
* **Intellectual Property:** Registered National Computer Software Copyright (Registration No. 2025SR0655319).
