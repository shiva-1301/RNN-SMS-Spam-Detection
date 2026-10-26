"""Pick the best experiment by VALIDATION performance, evaluate it on the test set
and save the accuracy / loss / confusion-matrix graphs to outputs/.

Usage:  python src/evaluate.py
"""
import json
import os
import shutil

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import tensorflow as tf
from sklearn.metrics import classification_report

from config import (BEST_CONFIG_FILE, BEST_HISTORY_FILE, BEST_MODEL_FILE,
                    EXPERIMENTS_DIR, OUTPUTS_DIR, RESULTS_FILE, TEST_METRICS_FILE)
from data_prep import prepare_data
from metrics import compute_metrics

TRAIN_COLOR, VAL_COLOR = "#2a78d6", "#eb6834"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def select_best(results):
    """Highest validation spam-F1; ties broken by validation accuracy, then lower val loss."""
    return max(results.values(), key=lambda r: (round(r["val"]["f1"], 4),
                                                round(r["val"]["accuracy"], 4),
                                                -r["min_val_loss"]))


def _style(ax, title, ylabel):
    ax.set_title(title, fontsize=13, color=TEXT, pad=10)
    ax.set_xlabel("Epoch", color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ae")
    ax.tick_params(colors=MUTED)


def plot_curve(history, key, ylabel, title, path, best_epoch=None):
    epochs = range(1, len(history[key]) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    ax.plot(epochs, history[key], color=TRAIN_COLOR, linewidth=2, marker="o", markersize=4,
            label=f"Training {ylabel.lower()}")
    ax.plot(epochs, history[f"val_{key}"], color=VAL_COLOR, linewidth=2, marker="o",
            markersize=4, label=f"Validation {ylabel.lower()}")
    if best_epoch:
        ax.axvline(best_epoch, color="#8a8984", linestyle="--", linewidth=1)
        ax.annotate(f"restored weights (epoch {best_epoch})", xy=(best_epoch, 0.5),
                    xycoords=("data", "axes fraction"), xytext=(4, 0),
                    textcoords="offset points", fontsize=8, color=MUTED, rotation=90,
                    va="center")
    ax.set_xticks(list(epochs))
    _style(ax, title, ylabel)
    ax.legend(frameon=False, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def plot_confusion(cm, path):
    cmap = LinearSegmentedColormap.from_list("blue_seq", ["#f2f6fc", "#2a78d6", "#123a6b"])
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=150)
    im = ax.imshow(cm, cmap=cmap)
    labels = ["Ham", "Spam"]
    ax.set_xticks([0, 1], labels=[f"Predicted {l}" for l in labels])
    ax.set_yticks([0, 1], labels=[f"Actual {l}" for l in labels])
    vmax = max(max(row) for row in cm)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", fontsize=16,
                    color="white" if cm[i][j] > vmax * 0.45 else TEXT)
    ax.set_title("Confusion Matrix (Test Set)", fontsize=13, color=TEXT, pad=10)
    ax.tick_params(colors=TEXT, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main():
    results = json.loads(RESULTS_FILE.read_text())
    best = select_best(results)
    name = best["name"]
    print(f"Best experiment by validation: {name} (val f1={best['val']['f1']:.4f}, "
          f"val acc={best['val']['accuracy']:.4f})")

    shutil.copy(EXPERIMENTS_DIR / f"{name}.keras", BEST_MODEL_FILE)
    shutil.copy(EXPERIMENTS_DIR / f"{name}_history.json", BEST_HISTORY_FILE)
    BEST_CONFIG_FILE.write_text(json.dumps(
        {"name": name, "config": best["config"], "class_weight": best["class_weight"],
         "epochs_run": best["epochs_run"], "weights_from_epoch": best["weights_from_epoch"],
         "params": best["params"]}, indent=2))

    data = prepare_data()
    X_test, y_test = data["test"]
    model = tf.keras.models.load_model(BEST_MODEL_FILE)
    probs = model.predict(X_test, verbose=0).ravel()
    metrics = compute_metrics(y_test, probs)
    metrics.update({"model": name, "test_size": int(len(y_test)),
                    "test_spam": int(y_test.sum()), "val": best["val"], "train": best["train"]})
    TEST_METRICS_FILE.write_text(json.dumps(metrics, indent=2))
    print(classification_report(y_test, (probs >= 0.5).astype(int),
                                target_names=["ham", "spam"], digits=4))
    print(json.dumps({k: metrics[k] for k in ("accuracy", "precision", "recall", "f1",
                                              "confusion_matrix")}, indent=2))

    history = json.loads(BEST_HISTORY_FILE.read_text())
    restored = best["weights_from_epoch"] if best["weights_from_epoch"] != best["epochs_run"] else None
    OUTPUTS_DIR.mkdir(exist_ok=True)
    plot_curve(history, "accuracy", "Accuracy", "Training vs Validation Accuracy",
               OUTPUTS_DIR / "accuracy_graph.png", restored)
    plot_curve(history, "loss", "Loss", "Training vs Validation Loss",
               OUTPUTS_DIR / "loss_graph.png", restored)
    plot_confusion(metrics["confusion_matrix"], OUTPUTS_DIR / "confusion_matrix.png")
    print("Saved graphs to", OUTPUTS_DIR)


if __name__ == "__main__":
    main()
