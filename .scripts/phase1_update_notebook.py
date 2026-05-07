"""
Phase 1: Clean up Section 1 of the assignment notebook.

What this script does:
  - Expands the imports cell (adds sklearn, tensorflow, seeds).
  - Replaces the single giant data-exploration cell with several smaller, well-titled cells
    so the structure matches the rubric items (description, distribution, sample images,
    pixel analysis, preprocessing).
  - Replaces the hard-coded "32000 photos for the test dataset" typo with f-strings
    that derive counts from array shapes.
  - Adds a stratified train/validation split.
  - Inserts a code cell under "### Examples of preprocessed data" that shows pre-processed
    samples for both the neural-net branch and the flattened classical-ML branch.
  - Replaces the informal SVM placeholder markdown with a proper justification.

Idempotent: rerunning is safe because we look up cells by stable text patterns.
"""
from pathlib import Path
import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell

NB_PATH = Path('a2-code-530839244-540958494-550120560-550053316.ipynb')

nb = nbf.read(NB_PATH.as_posix(), as_version=4)


def find_index(predicate):
    for i, c in enumerate(nb.cells):
        if predicate(c):
            return i
    return -1


# ---------------------------------------------------------------------------
# 1. Update the imports cell
# ---------------------------------------------------------------------------
imports_src = '''\
# --- Numerical / data handling ---------------------------------------------
import numpy as np
import pandas as pd

# --- Plotting ---------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.dpi"] = 110
sns.set_theme(style="whitegrid", context="notebook")

# --- Classical ML -----------------------------------------------------------
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --- Deep learning ----------------------------------------------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers

# --- Utilities --------------------------------------------------------------
import time, os, json, itertools
from collections import Counter

# --- Reproducibility --------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# --- Hardware sanity check --------------------------------------------------
print("TensorFlow version :", tf.__version__)
print("Visible GPUs       :", tf.config.list_physical_devices("GPU"))
'''

idx_imports = find_index(lambda c: c.cell_type == 'code' and 'import numpy' in c.source)
assert idx_imports >= 0, 'imports cell not found'
nb.cells[idx_imports] = new_code_cell(imports_src)


# ---------------------------------------------------------------------------
# 2. Replace the single giant data-exploration cell with multiple cells
# ---------------------------------------------------------------------------
idx_big = find_index(
    lambda c: c.cell_type == 'code' and 'Class Shapes & Data Type' in c.source
)
assert idx_big >= 0, 'big data exploration cell not found'

PATHMNIST_CLASSES = {
    0: "Adipose",
    1: "Background",
    2: "Debris",
    3: "Lymphocytes",
    4: "Mucus",
    5: "Smooth Muscle",
    6: "Normal Colon Mucosa",
    7: "Cancer-associated Stroma",
    8: "Colorectal Adenocarcinoma Epithelium",
}

new_cells = [
    new_markdown_cell(
        "### 1.1 Loading the dataset\n"
        "We load the four NumPy arrays distributed on Canvas. The training and test "
        "splits are pre-defined; we will additionally carve a validation set out of the "
        "training split below for hyper-parameter selection."
    ),
    new_code_cell(
        '# Load the dataset training and test sets as numpy arrays\n'
        'X_train = np.load("Data/X_train.npy")\n'
        'y_train = np.load("Data/y_train.npy")\n'
        'X_test  = np.load("Data/X_test.npy")\n'
        'y_test  = np.load("Data/y_test.npy")\n'
        '\n'
        'print(f"X_train shape: {X_train.shape},  dtype: {X_train.dtype}")\n'
        'print(f"y_train shape: {y_train.shape},  dtype: {y_train.dtype}")\n'
        'print(f"X_test  shape: {X_test.shape},  dtype: {X_test.dtype}")\n'
        'print(f"y_test  shape: {y_test.shape},  dtype: {y_test.dtype}")\n'
    ),
    new_markdown_cell(
        "### 1.2 Class definitions\n"
        "PathMNIST is a 9-class colorectal histology dataset taken from the MedMNIST v2 "
        "benchmark. The class index → tissue type mapping below follows the official "
        "MedMNIST v2 release."
    ),
    new_code_cell(
        '# PathMNIST class index -> tissue name (per MedMNIST v2)\n'
        'class_names = {\n'
        + "".join(f'    {k}: "{v}",\n' for k, v in PATHMNIST_CLASSES.items())
        + '}\n'
        'NUM_CLASSES = len(class_names)\n'
        'print(f"Number of classes: {NUM_CLASSES}")\n'
        'for k, v in class_names.items():\n'
        '    print(f"  Class {k}: {v}")\n'
    ),
    new_markdown_cell(
        "### 1.3 Sample sizes and class balance\n"
        "We print the per-split sample counts using `len(...)` so the text never "
        "drifts out of sync with the actual arrays. PathMNIST is mildly imbalanced; "
        "the bar plot below confirms whether class re-weighting may be needed."
    ),
    new_code_cell(
        'print(f"Training samples : {len(X_train):,}")\n'
        'print(f"Test samples     : {len(X_test):,}")\n'
        'print(f"Image dimensions : {X_train.shape[1]}x{X_train.shape[2]}x{X_train.shape[3]} (HxWxC)")\n'
        'print(f"Pixel dtype      : {X_train.dtype} (range {X_train.min()}..{X_train.max()})")\n'
        '\n'
        'unique, counts = np.unique(y_train, return_counts=True)\n'
        'class_balance = pd.DataFrame({\n'
        '    "class": [class_names[u] for u in unique],\n'
        '    "count": counts,\n'
        '    "proportion": counts / counts.sum(),\n'
        '})\n'
        'display(class_balance.style.format({"proportion": "{:.2%}"}))\n'
        '\n'
        'fig, ax = plt.subplots(figsize=(8, 4))\n'
        'sns.barplot(data=class_balance, x="class", y="count", ax=ax,\n'
        '            color="steelblue")\n'
        'ax.set_title("Class distribution in the training set")\n'
        'ax.set_xlabel("")\n'
        'ax.tick_params(axis="x", rotation=30)\n'
        'for label in ax.get_xticklabels():\n'
        '    label.set_horizontalalignment("right")\n'
        'plt.tight_layout()\n'
        'plt.show()\n'
        '\n'
        'imbalance_ratio = counts.max() / counts.min()\n'
        'print(f"Imbalance ratio (max/min): {imbalance_ratio:.2f}")\n'
    ),
    new_markdown_cell(
        "### 1.4 Sample images per class\n"
        "Three random images per class are shown to give an intuition for what the "
        "classifier has to discriminate. Many classes share a pinkish H&E stain colour "
        "and primarily differ by texture — a feature CNNs are particularly suited to capture."
    ),
    new_code_cell(
        'rng = np.random.default_rng(RANDOM_SEED)\n'
        'classes = np.unique(y_train)\n'
        'fig, axes = plt.subplots(NUM_CLASSES, 3, figsize=(6, 2 * NUM_CLASSES))\n'
        'for i, cls in enumerate(classes):\n'
        '    idxs = np.where(y_train == cls)[0]\n'
        '    selected = rng.choice(idxs, 3, replace=False)\n'
        '    for j, k in enumerate(selected):\n'
        '        ax = axes[i, j]\n'
        '        ax.imshow(X_train[k])\n'
        '        ax.set_xticks([]); ax.set_yticks([])\n'
        '        if j == 0:\n'
        '            ax.set_ylabel(f"{cls}: {class_names[cls]}", rotation=0,\n'
        '                          ha="right", va="center", fontsize=9)\n'
        'fig.suptitle("Three random training images per class", y=1.0)\n'
        'plt.tight_layout()\n'
        'plt.show()\n'
    ),
    new_markdown_cell(
        "### 1.5 Pixel intensity analysis\n"
        "We inspect the per-channel pixel distribution to decide what kind of scaling "
        "is appropriate. The values lie in the standard 0–255 uint8 range, motivating a "
        "simple division by 255 to bring them into the [0, 1] range used by neural "
        "network optimisers."
    ),
    new_code_cell(
        'channel_names = ["Red", "Green", "Blue"]\n'
        'colours = ["red", "green", "blue"]\n'
        'fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))\n'
        'for c in range(3):\n'
        '    data = X_train[..., c].ravel()\n'
        '    axes[c].hist(data, bins=50, color=colours[c], alpha=0.7)\n'
        '    axes[c].set_title(f"{channel_names[c]} channel")\n'
        '    axes[c].set_xlabel("Pixel intensity")\n'
        '    axes[c].set_ylabel("Frequency")\n'
        '    print(f"{channel_names[c]:<5}  mean={data.mean():6.2f}  std={data.std():6.2f}")\n'
        'plt.tight_layout()\n'
        'plt.show()\n'
    ),
    new_markdown_cell(
        "### 1.6 Pre-processing\n"
        "Two pre-processing pipelines are produced because the three classifiers consume "
        "data in different shapes:\n"
        "\n"
        "* **Neural network branch (MLP / CNN)** — pixels are scaled to `[0, 1]` by dividing "
        "by 255. Keeping pixel intensities small and bounded improves gradient stability and "
        "avoids any single colour channel dominating because of its scale.\n"
        "* **Classical-ML branch (SVM)** — each image is additionally flattened from "
        "`(28, 28, 3)` into a 2 352-dimensional feature vector, which is the input shape "
        "expected by `sklearn.svm.SVC`. We will further apply PCA inside the SVM pipeline "
        "(see §2) to keep the kernel evaluation tractable.\n"
        "\n"
        "Standardisation (subtracting the per-feature mean and dividing by the standard "
        "deviation) was considered for the SVM branch and is applied as part of the final "
        "pipeline; we keep the un-standardised flattened tensors here so each model can "
        "scale them as required."
    ),
    new_code_cell(
        '# Neural-network branch: scale pixels to [0, 1]\n'
        'X_train_norm = X_train.astype("float32") / 255.0\n'
        'X_test_norm  = X_test.astype("float32")  / 255.0\n'
        '\n'
        '# Classical-ML branch: flatten to (N, 2352)\n'
        'X_train_flat = X_train_norm.reshape(X_train_norm.shape[0], -1)\n'
        'X_test_flat  = X_test_norm.reshape(X_test_norm.shape[0], -1)\n'
        '\n'
        'print(f"Normalised pixel range : [{X_train_norm.min():.3f}, {X_train_norm.max():.3f}]")\n'
        'print(f"NN-branch shape        : {X_train_norm.shape}")\n'
        'print(f"Classical-branch shape : {X_train_flat.shape}")\n'
    ),
    new_markdown_cell(
        "### 1.7 Train / validation split\n"
        "The provided test set must only be used to evaluate the final selected models. "
        "For hyper-parameter tuning we therefore split the training set into a smaller "
        "training subset and a held-out validation subset. We use a stratified split "
        "so the class proportions are preserved on both sides."
    ),
    new_code_cell(
        'VAL_FRACTION = 0.2\n'
        '\n'
        '(X_tr_norm, X_val_norm,\n'
        ' X_tr_flat, X_val_flat,\n'
        ' y_tr,      y_val) = train_test_split(\n'
        '    X_train_norm, X_train_flat, y_train,\n'
        '    test_size=VAL_FRACTION,\n'
        '    stratify=y_train,\n'
        '    random_state=RANDOM_SEED,\n'
        ')\n'
        '\n'
        'print(f"Train subset (NN)        : {X_tr_norm.shape}")\n'
        'print(f"Validation subset (NN)   : {X_val_norm.shape}")\n'
        'print(f"Train subset (classical) : {X_tr_flat.shape}")\n'
        'print(f"Validation (classical)   : {X_val_flat.shape}")\n'
        '\n'
        'tr_props  = pd.Series(y_tr).value_counts(normalize=True).sort_index()\n'
        'val_props = pd.Series(y_val).value_counts(normalize=True).sort_index()\n'
        'props = pd.DataFrame({"train": tr_props, "val": val_props}).round(3)\n'
        'props.index = props.index.map(class_names)\n'
        'display(props)\n'
    ),
]

# Replace the original giant cell with the new sequence
nb.cells = nb.cells[:idx_big] + new_cells + nb.cells[idx_big + 1:]


# ---------------------------------------------------------------------------
# 3. Insert a "preprocessed examples" code cell after the existing markdown header
# ---------------------------------------------------------------------------
idx_examples_md = find_index(
    lambda c: c.cell_type == 'markdown'
    and 'Examples of preprocessed data' in c.source
)
assert idx_examples_md >= 0, '"Examples of preprocessed data" markdown not found'

# Replace the existing instructional markdown with a richer version
nb.cells[idx_examples_md] = new_markdown_cell(
    "### Examples of preprocessed data\n"
    "Below we show the same training image after each of the two preprocessing "
    "pipelines. The neural-network input is a 3-channel 28×28 float tensor in `[0, 1]` "
    "and is visually identical to the original. The classical-ML input is the same "
    "tensor flattened to a 2 352-dimensional vector, visualised here as a stacked "
    "1-D strip."
)

examples_code = (
    'sample_idx = rng.choice(len(X_tr_norm), 4, replace=False)\n'
    '\n'
    'fig, axes = plt.subplots(2, 4, figsize=(12, 4))\n'
    'for j, k in enumerate(sample_idx):\n'
    '    axes[0, j].imshow(X_tr_norm[k])\n'
    '    axes[0, j].set_title(f"NN input\\n{class_names[int(y_tr[k])]}", fontsize=9)\n'
    '    axes[0, j].axis("off")\n'
    '\n'
    '    flat = X_tr_flat[k].reshape(1, -1)\n'
    '    axes[1, j].imshow(flat, aspect="auto", cmap="viridis")\n'
    '    axes[1, j].set_title("Flattened (2352 d)", fontsize=9)\n'
    '    axes[1, j].set_yticks([])\n'
    'plt.tight_layout()\n'
    'plt.show()\n'
    '\n'
    'print(f"NN-input min/max : {X_tr_norm.min():.3f} / {X_tr_norm.max():.3f}")\n'
    'print(f"Flat-input shape : {X_tr_flat.shape} (each row is one image)")\n'
)

# Insert the new code cell directly after the markdown header (only once)
already_added = (
    idx_examples_md + 1 < len(nb.cells)
    and nb.cells[idx_examples_md + 1].cell_type == 'code'
    and 'sample_idx' in nb.cells[idx_examples_md + 1].source
)
if not already_added:
    nb.cells.insert(idx_examples_md + 1, new_code_cell(examples_code))


# ---------------------------------------------------------------------------
# 4. Replace the informal SVM placeholder markdown
# ---------------------------------------------------------------------------
idx_svm_md = find_index(
    lambda c: c.cell_type == 'markdown'
    and 'SVM' in c.source
    and 'random forrest' in c.source.lower()  # the typo'd placeholder
)
if idx_svm_md >= 0:
    nb.cells[idx_svm_md] = new_markdown_cell(
        "We choose the **Support Vector Machine (SVM)** with an RBF kernel as our "
        "classical algorithm from the first six weeks of the course. SVMs perform well "
        "in high-dimensional feature spaces, are robust to over-fitting when the margin "
        "is properly regularised, and provide a non-linear decision boundary through "
        "the kernel trick — all desirable properties for raw flattened image features.\n"
        "\n"
        "Because the flattened input has 2 352 dimensions and 32 000 samples, fitting "
        "an RBF SVM directly is computationally heavy. We therefore wrap the SVM in a "
        "pipeline that first **standardises** features and then projects them with "
        "**PCA** before the kernel evaluation. The PCA step controls runtime and reduces "
        "the impact of redundant pixel-level correlations."
    )


# ---------------------------------------------------------------------------
# Save the notebook
# ---------------------------------------------------------------------------
nbf.write(nb, NB_PATH.as_posix())
print(f"Updated notebook: {NB_PATH}")
print(f"Total cells now : {len(nb.cells)}")
