"""Train one experiment configuration and log its validation and test metrics.

Usage:  python src/train.py --exp A1_baseline
"""
import argparse
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from config import EXPERIMENTS_DIR, RESULTS_FILE, SEED
from data_prep import prepare_data
from metrics import compute_metrics
from model import build_model

# Each attempt of the self-evaluation loop is one entry here (see NOTES.md).
EXPERIMENTS = {
    "A1_baseline": dict(
        model=dict(rnn_type="simplernn", embed_dim=64, rnn_units=64, dense_units=32,
                   dropout=0.0, learning_rate=1e-3),
        epochs=15, batch_size=32, class_weight=True, early_stopping=False),
    # A1 overfits (train acc 1.0, val loss rising after epoch 5) -> regularise.
    "A2_simplernn_dropout_es": dict(
        model=dict(rnn_type="simplernn", embed_dim=64, rnn_units=64, dense_units=32,
                   dropout=0.3, learning_rate=1e-3),
        epochs=30, batch_size=32, class_weight=True, early_stopping=True, patience=3),
    # A2 stopped at epoch 4 with weights from epoch 1: val loss is very noisy -> smaller
    # model, lower learning rate, gradient clipping and more patience.
    "A3_simplernn_small_lr": dict(
        model=dict(rnn_type="simplernn", embed_dim=32, rnn_units=32, dense_units=16,
                   dropout=0.3, learning_rate=5e-4, clipnorm=1.0),
        epochs=40, batch_size=32, class_weight=True, early_stopping=True, patience=5),
    # A3 is stable but still noisy; try gated cells, which handle long messages better.
    "A4_lstm": dict(
        model=dict(rnn_type="lstm", embed_dim=64, rnn_units=64, dense_units=32,
                   dropout=0.3, learning_rate=1e-3, clipnorm=1.0),
        epochs=40, batch_size=32, class_weight=True, early_stopping=True, patience=5),
    "A5_gru": dict(
        model=dict(rnn_type="gru", embed_dim=64, rnn_units=64, dense_units=32,
                   dropout=0.3, learning_rate=1e-3, clipnorm=1.0),
        epochs=40, batch_size=32, class_weight=True, early_stopping=True, patience=5),
    # Remaining val errors are mostly ham flagged as spam -> soften the spam class weight.
    "A6_gru_softer_weights": dict(
        model=dict(rnn_type="gru", embed_dim=64, rnn_units=64, dense_units=32,
                   dropout=0.3, learning_rate=1e-3, clipnorm=1.0),
        epochs=40, batch_size=32, class_weight=True, spam_weight_scale=0.5,
        early_stopping=True, patience=5),
}


def train(name):
    cfg = EXPERIMENTS[name]
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    data = prepare_data()
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]

    class_weight = None
    if cfg["class_weight"]:
        weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
        weights[1] *= cfg.get("spam_weight_scale", 1.0)
        class_weight = {0: float(weights[0]), 1: float(weights[1])}

    callbacks = []
    if cfg["early_stopping"]:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg.get("patience", 3), restore_best_weights=True))

    model = build_model(**cfg["model"])
    start = time.time()
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=cfg["epochs"], batch_size=cfg["batch_size"],
                        class_weight=class_weight, callbacks=callbacks, verbose=2)
    train_seconds = time.time() - start

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(EXPERIMENTS_DIR / f"{name}.keras")
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(EXPERIMENTS_DIR / f"{name}_history.json", "w") as f:
        json.dump(hist, f, indent=2)

    best_epoch = int(np.argmin(hist["val_loss"])) + 1 if cfg["early_stopping"] else len(hist["loss"])
    result = {
        "name": name, "config": cfg, "class_weight": class_weight,
        "epochs_run": len(hist["loss"]), "weights_from_epoch": best_epoch,
        "params": int(model.count_params()), "train_seconds": round(train_seconds, 1),
        "train": compute_metrics(y_train, model.predict(X_train, verbose=0)),
        "val": compute_metrics(y_val, model.predict(X_val, verbose=0)),
        "test": compute_metrics(y_test, model.predict(X_test, verbose=0)),
        "final_epoch_train_acc": hist["accuracy"][-1], "final_epoch_val_acc": hist["val_accuracy"][-1],
        "final_epoch_train_loss": hist["loss"][-1], "final_epoch_val_loss": hist["val_loss"][-1],
        "min_val_loss": min(hist["val_loss"]),
    }

    results = {}
    if RESULTS_FILE.exists():
        results = json.loads(RESULTS_FILE.read_text())
    results[name] = result
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(results, indent=2))

    for split in ("train", "val", "test"):
        m = result[split]
        print(f"{name} {split:5s} acc={m['accuracy']:.4f} prec={m['precision']:.4f} "
              f"rec={m['recall']:.4f} f1={m['f1']:.4f} cm={m['confusion_matrix']}")
    print(f"epochs_run={result['epochs_run']} weights_from_epoch={best_epoch} "
          f"min_val_loss={result['min_val_loss']:.4f} final_val_loss={hist['val_loss'][-1]:.4f}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True, choices=sorted(EXPERIMENTS))
    train(parser.parse_args().exp)
