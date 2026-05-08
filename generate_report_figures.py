"""Regenerate report figures for the Overleaf project.

Outputs are written into Overleaf__A2/Photos/ as PNGs, sized to fit the
LaTeX text width when included with width=\linewidth.

Run from the project root:
    /opt/anaconda3/envs/comp5318/bin/python generate_report_figures.py
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Data"
OUTPUTS = ROOT / "Outputs"
PHOTOS = ROOT / "Overleaf__A2" / "Photos"
PHOTOS.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

CLASS_NAMES = [
    "Adipose",
    "Background",
    "Debris",
    "Lymphocytes",
    "Mucus",
    "Smooth Muscle",
    "Normal Mucosa",
    "CA Stroma",
    "CRC Epithelium",
]


def load_data():
    X_train = np.load(DATA / "X_train.npy")
    y_train = np.load(DATA / "y_train.npy").astype(np.int64).ravel()
    X_test = np.load(DATA / "X_test.npy")
    y_test = np.load(DATA / "y_test.npy").astype(np.int64).ravel()
    return X_train, y_train, X_test, y_test


def fig_sample_grid(X_train, y_train):
    rng = np.random.default_rng(SEED)
    n_per_class = 3
    fig, axes = plt.subplots(
        len(CLASS_NAMES), n_per_class,
        figsize=(3.6, 9.6),
        gridspec_kw=dict(wspace=0.05, hspace=0.05),
    )
    for cls in range(len(CLASS_NAMES)):
        idxs = rng.choice(np.where(y_train == cls)[0], size=n_per_class, replace=False)
        for j, idx in enumerate(idxs):
            ax = axes[cls, j]
            ax.imshow(X_train[idx])
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(CLASS_NAMES[cls], fontsize=8, rotation=0,
                              ha="right", va="center", labelpad=4)
    fig.tight_layout()
    out = PHOTOS / "samples_per_class.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_confusion(cm, title, out_path):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    disp = ConfusionMatrixDisplay(cm_norm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format=".2f",
              xticks_rotation=45)
    for txt in disp.text_.ravel():
        txt.set_fontsize(6)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="both", labelsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def confusion_svm(X_train, y_train, X_test, y_test, hp):
    Xtr = X_train.reshape(len(X_train), -1).astype(np.float32)
    Xte = X_test.reshape(len(X_test), -1).astype(np.float32)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=hp["n_components"], random_state=SEED)),
        ("svc", SVC(C=hp["C"], gamma=hp["gamma"], kernel="rbf",
                    decision_function_shape="ovr", random_state=SEED)),
    ])
    pipe.fit(Xtr, y_train)
    y_pred = pipe.predict(Xte)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(9)))
    plot_confusion(cm, "SVM (RBF)", PHOTOS / "cm_svm.png")


def confusion_mlp(X_train, y_train, X_test, y_test, hp):
    import tensorflow as tf  # imported lazily so the script still runs without TF
    tf.keras.utils.set_random_seed(SEED)
    Xtr = X_train.astype(np.float32) / 255.0
    Xte = X_test.astype(np.float32) / 255.0
    units = hp["units"]
    p = hp["dropout"]
    lr = hp["learning_rate"]
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units, activation="relu"),
        tf.keras.layers.Dropout(p),
        tf.keras.layers.Dense(units // 2, activation="relu"),
        tf.keras.layers.Dropout(p),
        tf.keras.layers.Dense(9, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    es = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3,
                                          restore_best_weights=True)
    model.fit(Xtr, y_train, validation_split=0.2, epochs=25, batch_size=128,
              callbacks=[es], verbose=0)
    y_pred = model.predict(Xte, verbose=0).argmax(axis=1)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(9)))
    plot_confusion(cm, "MLP", PHOTOS / "cm_mlp.png")


def confusion_cnn(X_train, y_train, X_test, y_test, hp):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)
    Xtr = X_train.astype(np.float32) / 255.0
    Xte = X_test.astype(np.float32) / 255.0
    f = hp["filters"]
    k = hp["kernel_size"]
    p = hp["dropout"]
    lr = hp["learning_rate"]
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 3)),
        tf.keras.layers.Conv2D(f, k, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(2 * f, k, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(p),
        tf.keras.layers.Dense(9, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    es = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3,
                                          restore_best_weights=True)
    model.fit(Xtr, y_train, validation_split=0.2, epochs=25, batch_size=128,
              callbacks=[es], verbose=0)
    y_pred = model.predict(Xte, verbose=0).argmax(axis=1)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(9)))
    plot_confusion(cm, "CNN", PHOTOS / "cm_cnn.png")


def tuning_bar_chart(df, axes_specs, title, out_path):
    """Plot mean validation accuracy along each hyper-parameter axis.

    axes_specs: list of (column_name, label, formatter)
    """
    fig, axes = plt.subplots(1, len(axes_specs), figsize=(3.0 * len(axes_specs), 3.0))
    if len(axes_specs) == 1:
        axes = [axes]
    for ax, (col, label, fmt) in zip(axes, axes_specs):
        grouped = df.groupby(col)["val_accuracy"].agg(["mean", "max"]).reset_index()
        x_labels = [fmt(v) for v in grouped[col].tolist()]
        x = np.arange(len(grouped))
        width = 0.4
        ax.bar(x - width / 2, grouped["mean"] * 100, width, label="mean")
        ax.bar(x + width / 2, grouped["max"] * 100, width, label="best")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("Val accuracy (%)", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def fig_tune_svm():
    df = pd.read_csv(OUTPUTS / "svm_tuning.csv").rename(columns={
        "param_pca__n_components": "n_components",
        "param_svc__C": "C",
        "param_svc__gamma": "gamma",
        "mean_test_score": "val_accuracy",
    })
    tuning_bar_chart(
        df,
        [("n_components", "PCA components", lambda v: str(int(v))),
         ("C", "C", lambda v: f"{float(v):g}"),
         ("gamma", "gamma", lambda v: str(v))],
        "SVM grid search (3-fold CV on 5,000-sample subset)",
        PHOTOS / "tune_svm.png",
    )


def fig_tune_mlp():
    df = pd.read_csv(OUTPUTS / "mlp_tuning.csv")
    tuning_bar_chart(
        df,
        [("units", "Hidden units", lambda v: str(int(v))),
         ("dropout", "Dropout", lambda v: f"{float(v):g}"),
         ("learning_rate", "Learning rate", lambda v: f"{float(v):.0e}")],
        "MLP grid search (validation partition)",
        PHOTOS / "tune_mlp.png",
    )


def fig_tune_cnn():
    df = pd.read_csv(OUTPUTS / "cnn_tuning.csv")
    tuning_bar_chart(
        df,
        [("filters", "Base filters", lambda v: str(int(v))),
         ("kernel_size", "Kernel size", lambda v: str(int(v))),
         ("dropout", "Dropout", lambda v: f"{float(v):g}"),
         ("learning_rate", "Learning rate", lambda v: f"{float(v):.0e}")],
        "CNN grid search (validation partition)",
        PHOTOS / "tune_cnn.png",
    )


def main():
    do_cm = "--no-cm" not in sys.argv
    X_train, y_train, X_test, y_test = load_data()

    fig_sample_grid(X_train, y_train)
    fig_tune_svm()
    fig_tune_mlp()
    fig_tune_cnn()

    if do_cm:
        with open(OUTPUTS / "best_hparams.json") as f:
            best = json.load(f)
        confusion_svm(X_train, y_train, X_test, y_test, best["svm"])
        confusion_mlp(X_train, y_train, X_test, y_test, best["mlp"])
        confusion_cnn(X_train, y_train, X_test, y_test, best["cnn"])


if __name__ == "__main__":
    main()
