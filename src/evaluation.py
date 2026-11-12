"""Load the saved, descriptive experiment results packaged with the app."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

def experiment_results():
    return pd.read_csv(REPORTS / "test_experiment_summary.csv")

def per_class_results(experiment="E2"):
    data = pd.read_csv(REPORTS / "test_per_class_metrics.csv")
    return data[(data.experiment_id == experiment) & (data.split == "test")].copy()

def confusion_results(experiment="E2"):
    data = pd.read_csv(REPORTS / "test_confusion_matrices_long.csv")
    return data[data.experiment_id == experiment].copy()

def validation_results():
    return pd.read_csv(REPORTS / "validation_experiment_comparison.csv")

def training_results():
    return pd.read_csv(REPORTS / "epoch_metrics.csv")

def training_class_counts():
    labels = np.load(ROOT / "data" / "train" / "train_labels.npy", mmap_mode="r").reshape(-1)
    return np.bincount(labels.astype(int), minlength=9)
