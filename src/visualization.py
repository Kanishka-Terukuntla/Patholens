"""Display and chart helpers."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

BG = "#080D19"
PANEL = "#111A2D"
TEXT = "#DCE5F3"
GRID = "#2B3A55"
TEAL = "#62D5C4"
VIOLET = "#8B82F6"

def _dark_axes(fig, ax):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT)
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color(GRID)

def probability_figure(names, probabilities):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    _dark_axes(fig, ax)
    values = np.asarray(probabilities)
    order = np.argsort(values)
    ax.barh(np.asarray(names)[order], values[order], color=TEAL, height=.68)
    ax.set_xlim(0, 1); ax.set_xlabel("Model probability")
    ax.grid(axis="x", color=GRID, alpha=.45); ax.set_axisbelow(True)
    fig.tight_layout()
    return fig

def metric_chart(frame, metric, label):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    _dark_axes(fig, ax)
    colors={"E1":"#8492B0","E2":TEAL,"E3":"#596681","E4":"#F0B567"}
    ax.bar(frame["experiment_id"], frame[metric], color=[colors.get(x,"#35B7AA") for x in frame["experiment_id"]])
    ax.set_ylim(0, 1); ax.set_ylabel(label); ax.grid(axis="y", color=GRID, alpha=.45); ax.set_axisbelow(True)
    fig.tight_layout(); return fig

def class_distribution_figure(counts, names):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    _dark_axes(fig, ax)
    y=np.arange(len(names))
    ax.barh(y,counts,color=TEAL)
    ax.set_yticks(y,names); ax.set_xlabel("Training images")
    ax.grid(axis="x",color=GRID,alpha=.45); ax.set_axisbelow(True)
    fig.tight_layout(); return fig

def validation_test_figure(validation, test):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    _dark_axes(fig, ax)
    merged = validation[["experiment_id", "val_select_macro_f1"]].merge(
        test[["experiment_id", "test_macro_f1"]], on="experiment_id", how="inner"
    ).set_index("experiment_id").reindex(["E1", "E2", "E3", "E4"])
    x = np.arange(len(merged.index))
    ax.bar(x - .18, merged["val_select_macro_f1"], width=.36, color="#718A98", label="Validation selection")
    ax.bar(x + .18, merged["test_macro_f1"], width=.36, color="#35B7AA", label="Test comparison")
    ax.set_xticks(x, merged.index); ax.set_ylim(0, 1); ax.set_ylabel("Macro-F1")
    ax.grid(axis="y", color=GRID, alpha=.45); ax.set_axisbelow(True)
    legend = ax.legend(frameon=False)
    for label in legend.get_texts(): label.set_color(TEXT)
    fig.tight_layout(); return fig

def training_curves_figure(frame, experiment="E2"):
    rows = frame[frame.experiment_id == experiment].sort_values("epoch")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax in axes: _dark_axes(fig, ax)
    axes[0].plot(rows["epoch"], rows["train_accuracy"], color="#8795B2", label="Train", linewidth=2.2)
    axes[0].plot(rows["epoch"], rows["val_accuracy"], color=TEAL, label="Validation", linewidth=2.2)
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("Epoch"); axes[0].set_ylim(0, 1.02)
    axes[1].plot(rows["epoch"], rows["train_macro_f1"], color="#8795B2", label="Train", linewidth=2.2)
    axes[1].plot(rows["epoch"], rows["val_macro_f1"], color=TEAL, label="Validation", linewidth=2.2)
    axes[1].set_title("Macro-F1"); axes[1].set_xlabel("Epoch"); axes[1].set_ylim(0, 1.02)
    for ax in axes:
        ax.grid(color=GRID, alpha=.4); ax.set_axisbelow(True)
        legend = ax.legend(frameon=False)
        for label in legend.get_texts(): label.set_color(TEXT)
    fig.tight_layout(); return fig

def confusion_figure(frame):
    matrix = np.zeros((9, 9), dtype=int)
    for row in frame.itertuples(index=False):
        matrix[int(row.true_class_id), int(row.pred_class_id)] = int(row.count)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    cmap = LinearSegmentedColormap.from_list(
        "patholens_midnight", ["#354C72", "#6578B4", VIOLET, TEAL, "#FFE08A"]
    )
    cmap.set_bad(PANEL)
    visible_counts = np.ma.masked_equal(matrix, 0)
    im = ax.imshow(
        visible_counts,
        cmap=cmap,
        norm=PowerNorm(gamma=0.45, vmin=0, vmax=max(1, int(matrix.max()))),
        interpolation="nearest",
        aspect="equal",
    )
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
    ax.set_xticks(range(9)); ax.set_yticks(range(9))
    ax.tick_params(axis="x", labelrotation=0)
    ax.set_xlabel("Predicted class", labelpad=10); ax.set_ylabel("True class", labelpad=10)
    colorbar=fig.colorbar(im, ax=ax); colorbar.ax.tick_params(colors=TEXT); colorbar.set_label("Image count", color=TEXT)
    fig.tight_layout(); return fig
