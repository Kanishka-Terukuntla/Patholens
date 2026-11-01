# PathoLens AI

### An Explainable and Uncertainty-Aware Deep Learning Framework
### for Multi-Class Colorectal Histopathology Classification

## Overview

PathoLens AI is a medical AI research project that classifies
colorectal histopathology images into nine tissue categories.

The system combines deep-learning classification,
Grad-CAM-based visual explanations, and uncertainty
estimation to identify predictions that may require
human review.

## Dataset

PathMNIST from the MedMNIST collection.

Official source:
https://medmnist.com/

## Models

- Custom CNN
- ResNet-18
- EfficientNet-B0

## Explainable AI

- Grad-CAM

## Uncertainty Estimation

- Predictive entropy
- Temperature scaling
- Selective prediction
- Expected Calibration Error

## Technology Stack

- Python
- PyTorch
- Torchvision
- MedMNIST
- Scikit-learn
- Streamlit

## Project Status

Phase 1: Dataset acquisition and exploration.

## Disclaimer

This project is an educational research prototype.
It is not a clinically validated diagnostic system.