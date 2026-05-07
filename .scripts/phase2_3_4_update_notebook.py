"""
Phase 2-4: Add baseline models, hyper-parameter tuning, and final-model evaluation
cells to the assignment notebook.

Design notes
------------
* §2 ("Algorithm design and setup"): for each of SVM, MLP and CNN we provide a
  factory function plus a small "baseline run" that confirms the architecture
  trains end-to-end. The factory functions are reused by §3 and §4 so the model
  definitions live in exactly one place.
* §3 ("Hyper-parameter tuning"): each model has its own grid search. Results
  are saved to ``Outputs/<model>_tuning.csv`` and a single
  ``Outputs/best_hparams.json`` so §4 cells stay independent of §3 (per spec:
  markers should be able to skip §3 cells).
* §4 ("Final models"): each cell reloads the best hyper-parameters from the
  saved JSON (with hard-coded fallbacks), trains on the full training set
  (= train + val partition), and evaluates on the held-out PathMNIST test set.

The script is idempotent — re-running deletes any §2/§3/§4/§5 code cells we
previously inserted (identified by an ``# A2:`` tag in their first line) and
inserts the current version. Markdown headers are kept intact.
"""
from pathlib import Path
import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell

NB_PATH = Path('a2-code-530839244-540958494-550120560-550053316.ipynb')
nb = nbf.read(NB_PATH.as_posix(), as_version=4)

A2_TAG = '# A2:'  # tag that marks our generated code cells so we can re-run safely


def find_index(predicate, start=0):
    for i in range(start, len(nb.cells)):
        if predicate(nb.cells[i]):
            return i
    return -1


def insert_after(anchor_idx, new_cells):
    """Insert *new_cells* after the cell at *anchor_idx* and return the
    index of the last newly-inserted cell."""
    for offset, c in enumerate(new_cells, start=1):
        nb.cells.insert(anchor_idx + offset, c)
    return anchor_idx + len(new_cells)


def remove_generated_after(anchor_idx, until_predicate):
    """Remove generated code cells (tagged with A2_TAG) between
    *anchor_idx*+1 and the first cell satisfying *until_predicate*."""
    j = anchor_idx + 1
    while j < len(nb.cells) and not until_predicate(nb.cells[j]):
        c = nb.cells[j]
        if c.cell_type == 'code' and c.source.lstrip().startswith(A2_TAG):
            del nb.cells[j]
        else:
            j += 1


def code(src):
    return new_code_cell(src.lstrip('\n'))


# ---------------------------------------------------------------------------
# Locate the major markdown headers we will hang content under.
# ---------------------------------------------------------------------------
def md_match(text):
    return lambda c: c.cell_type == 'markdown' and text in c.source


# §2 — section headers
i_section2          = find_index(md_match('## 2. Algorithm design and setup'))
i_sec2_choice       = find_index(md_match('### Algorithm of choice from first six weeks'), i_section2 + 1)
i_sec2_mlp          = find_index(md_match('### Fully connected neural network'), i_section2 + 1)
i_sec2_cnn          = find_index(md_match('### Convolutional neural network'), i_section2 + 1)

# §3 — section headers
i_section3          = find_index(md_match('## 3. Hyperparameter tuning'))
i_sec3_choice       = find_index(md_match('### Algorithm of choice from first six weeks'), i_section3 + 1)
i_sec3_mlp          = find_index(md_match('### Fully connected neural network'), i_section3 + 1)
i_sec3_cnn          = find_index(md_match('### Convolutional neural network'), i_section3 + 1)

# §4 — section headers
i_section4          = find_index(md_match('## 4. Final models'))
i_sec4_choice       = find_index(md_match('### Algorithm of choice from first six weeks'), i_section4 + 1)
i_sec4_mlp          = find_index(md_match('### Fully connected neural network'), i_section4 + 1)
i_sec4_cnn          = find_index(md_match('### Convolutional neural network'), i_section4 + 1)

# §5
i_section5          = find_index(md_match('## 5. AI Acknowledgement'))

# Sanity-check
for name, idx in [
    ('§2', i_section2), ('§2.choice', i_sec2_choice), ('§2.mlp', i_sec2_mlp), ('§2.cnn', i_sec2_cnn),
    ('§3', i_section3), ('§3.choice', i_sec3_choice), ('§3.mlp', i_sec3_mlp), ('§3.cnn', i_sec3_cnn),
    ('§4', i_section4), ('§4.choice', i_sec4_choice), ('§4.mlp', i_sec4_mlp), ('§4.cnn', i_sec4_cnn),
    ('§5', i_section5),
]:
    if idx < 0:
        raise RuntimeError(f'Could not locate header for {name}')


# ---------------------------------------------------------------------------
# Build the new code cells.
# ---------------------------------------------------------------------------

# A shared helper cell at the very top of §2 — defines factories used by all
# downstream sections.
SHARED_FACTORIES = code('''
# A2: shared model factories used by §2, §3 and §4.
# Defining the architectures here keeps §3 / §4 cells short and consistent.

from pathlib import Path
OUTPUTS_DIR = Path("Outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)


def make_svm(C=1.0, gamma="scale", n_components=50, random_state=RANDOM_SEED):
    """Standardise -> PCA -> RBF SVC."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=n_components, random_state=random_state)),
        ("svc",    SVC(C=C, gamma=gamma, kernel="rbf",
                       random_state=random_state, cache_size=500)),
    ])


def make_mlp(units=128, dropout=0.3, learning_rate=1e-3, num_classes=NUM_CLASSES):
    """Two-hidden-layer feed-forward network."""
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 3)),
        layers.Flatten(),
        layers.Dense(units, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(units // 2, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_cnn(filters=32, dropout=0.3, learning_rate=1e-3,
             kernel_size=3, num_classes=NUM_CLASSES):
    """A small two-block convolutional network."""
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 3)),
        layers.Conv2D(filters, kernel_size, activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=2),
        layers.Conv2D(filters * 2, kernel_size, activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=2),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


print("Model factories defined: make_svm, make_mlp, make_cnn")
''')

# §2.x baselines
SVM_BASELINE = code('''
# A2: §2 — Baseline SVM run on a stratified 5 000-sample subset.
SVM_SUBSET_SIZE = 5_000

sub_idx, _ = train_test_split(
    np.arange(len(X_tr_flat)),
    train_size=SVM_SUBSET_SIZE,
    stratify=y_tr,
    random_state=RANDOM_SEED,
)

t0 = time.time()
svm_baseline = make_svm()
svm_baseline.fit(X_tr_flat[sub_idx], y_tr[sub_idx])
fit_time = time.time() - t0

val_acc = accuracy_score(y_val, svm_baseline.predict(X_val_flat))
print("Baseline SVM (StandardScaler -> PCA(50) -> RBF SVC, default C=1, gamma='scale')")
print(f"  trained on  : {SVM_SUBSET_SIZE:,} samples")
print(f"  fit time    : {fit_time:.1f}s")
print(f"  val accuracy: {val_acc:.4f}")
''')

MLP_BASELINE = code('''
# A2: §2 — Baseline MLP, 5 epochs.
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

mlp_baseline = make_mlp()
mlp_baseline.summary()

t0 = time.time()
mlp_hist = mlp_baseline.fit(
    X_tr_norm, y_tr,
    validation_data=(X_val_norm, y_val),
    epochs=5, batch_size=128, verbose=2,
)
print(f"\\nBaseline MLP fit in {time.time()-t0:.1f}s")
print(f"  best val accuracy: {max(mlp_hist.history['val_accuracy']):.4f}")
''')

CNN_BASELINE = code('''
# A2: §2 — Baseline CNN, 5 epochs.
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

cnn_baseline = make_cnn()
cnn_baseline.summary()

t0 = time.time()
cnn_hist = cnn_baseline.fit(
    X_tr_norm, y_tr,
    validation_data=(X_val_norm, y_val),
    epochs=5, batch_size=128, verbose=2,
)
print(f"\\nBaseline CNN fit in {time.time()-t0:.1f}s")
print(f"  best val accuracy: {max(cnn_hist.history['val_accuracy']):.4f}")
''')

# §3.x tuning
SVM_TUNE = code('''
# A2: §3 — SVM grid search on a 5 000-sample stratified subset.
# Tuning over: PCA n_components, SVM C, SVM gamma (3 hyperparameters).
SVM_TUNE_SUBSET = 5_000
SVM_TUNE_CSV    = OUTPUTS_DIR / "svm_tuning.csv"

svm_grid = {
    "pca__n_components": [30, 50, 100],
    "svc__C":            [0.5, 1.0, 5.0],
    "svc__gamma":        ["scale", 0.01, 0.001],
}

if SVM_TUNE_CSV.exists():
    print(f"Loading cached SVM tuning results from {SVM_TUNE_CSV}")
    svm_results = pd.read_csv(SVM_TUNE_CSV)
else:
    sub_idx, _ = train_test_split(
        np.arange(len(X_tr_flat)),
        train_size=SVM_TUNE_SUBSET,
        stratify=y_tr,
        random_state=RANDOM_SEED,
    )
    n_combos = np.prod([len(v) for v in svm_grid.values()])
    print(f"Running SVM grid search: {n_combos} combos x 3-fold CV "
          f"on {SVM_TUNE_SUBSET:,} samples...")
    t0 = time.time()
    svm_search = GridSearchCV(
        make_svm(), svm_grid,
        cv=3, scoring="accuracy", n_jobs=-1, verbose=1,
    )
    svm_search.fit(X_tr_flat[sub_idx], y_tr[sub_idx])
    print(f"  total search time: {(time.time()-t0)/60:.1f} min")

    svm_results = pd.DataFrame(svm_search.cv_results_)[[
        "param_pca__n_components", "param_svc__C", "param_svc__gamma",
        "mean_fit_time", "mean_test_score", "std_test_score", "rank_test_score",
    ]].sort_values("rank_test_score").reset_index(drop=True)
    svm_results.to_csv(SVM_TUNE_CSV, index=False)
    print(f"Saved -> {SVM_TUNE_CSV}")

display(svm_results.head(10))
best_svm = svm_results.iloc[0]
# Store hyper-parameters with the keyword names accepted by ``make_svm`` so
# that ``make_svm(**best_svm_params)`` is a drop-in factory call in §4.
def _maybe_float(x):
    s = str(x)
    try:
        return float(s)
    except ValueError:
        return s

best_svm_params = {
    "n_components": int(best_svm["param_pca__n_components"]),
    "C":            float(best_svm["param_svc__C"]),
    "gamma":        _maybe_float(best_svm["param_svc__gamma"]),
}
print("Best SVM CV accuracy:", round(best_svm["mean_test_score"], 4))
print("Best SVM hyper-parameters:", best_svm_params)
''')

SVM_TUNE_VIZ = code('''
# A2: §3 — Visualise SVM tuning trends.
fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), sharey=True)

# Aggregate by each hyperparameter individually.
for ax, hp, label in zip(
    axes,
    ["param_pca__n_components", "param_svc__C", "param_svc__gamma"],
    ["PCA n_components", "SVM C", "SVM gamma"],
):
    grouped = svm_results.groupby(hp)["mean_test_score"].agg(["mean", "max"])
    grouped.plot(kind="bar", ax=ax, color=["#4c72b0", "#dd8452"])
    ax.set_title(label)
    ax.set_xlabel(label)
    ax.set_ylabel("CV accuracy")
    ax.legend(["mean", "max"], fontsize=8)
plt.tight_layout()
plt.show()
''')

MLP_TUNE = code('''
# A2: §3 — MLP grid search (3 hyperparameters: units, dropout, learning rate).
MLP_TUNE_CSV = OUTPUTS_DIR / "mlp_tuning.csv"

mlp_grid = {
    "units":         [64, 128, 256],
    "dropout":       [0.0, 0.3, 0.5],
    "learning_rate": [1e-3, 1e-4],
}

if MLP_TUNE_CSV.exists():
    print(f"Loading cached MLP tuning results from {MLP_TUNE_CSV}")
    mlp_results = pd.read_csv(MLP_TUNE_CSV)
else:
    rows = []
    combos = list(itertools.product(*mlp_grid.values()))
    print(f"Running MLP tuning: {len(combos)} combos, up to 25 epochs each (early stop)")
    for i, (units, dropout, lr) in enumerate(combos, start=1):
        tf.random.set_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        model = make_mlp(units=units, dropout=dropout, learning_rate=lr)
        es = callbacks.EarlyStopping(patience=3, restore_best_weights=True,
                                     monitor="val_accuracy", mode="max")
        t0 = time.time()
        h = model.fit(
            X_tr_norm, y_tr,
            validation_data=(X_val_norm, y_val),
            epochs=25, batch_size=128, verbose=0, callbacks=[es],
        )
        fit_time = time.time() - t0
        val_acc = max(h.history["val_accuracy"])
        rows.append(dict(
            units=units, dropout=dropout, learning_rate=lr,
            val_accuracy=val_acc, fit_time=fit_time,
            epochs_trained=len(h.history["loss"]),
        ))
        print(f"  [{i:>2}/{len(combos)}] units={units:>3}, dropout={dropout:.1f}, "
              f"lr={lr:.0e} -> val_acc={val_acc:.4f} ({fit_time:.1f}s)")
        keras.backend.clear_session()
    mlp_results = pd.DataFrame(rows).sort_values("val_accuracy", ascending=False)
    mlp_results.to_csv(MLP_TUNE_CSV, index=False)
    print(f"Saved -> {MLP_TUNE_CSV}")

display(mlp_results.head(10))
best_mlp = mlp_results.iloc[0]
best_mlp_params = {
    "units":         int(best_mlp["units"]),
    "dropout":       float(best_mlp["dropout"]),
    "learning_rate": float(best_mlp["learning_rate"]),
}
print("Best MLP val accuracy :", round(best_mlp["val_accuracy"], 4))
print("Best MLP hyper-params :", best_mlp_params)
''')

MLP_TUNE_VIZ = code('''
# A2: §3 — Visualise MLP tuning trends.
fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), sharey=True)
for ax, hp in zip(axes, ["units", "dropout", "learning_rate"]):
    grouped = mlp_results.groupby(hp)["val_accuracy"].agg(["mean", "max"])
    grouped.plot(kind="bar", ax=ax, color=["#4c72b0", "#dd8452"])
    ax.set_title(hp)
    ax.set_ylabel("Val accuracy")
    ax.legend(["mean", "max"], fontsize=8)
plt.tight_layout()
plt.show()
''')

CNN_TUNE = code('''
# A2: §3 — CNN grid search (filters, dropout, learning_rate, kernel_size).
CNN_TUNE_CSV = OUTPUTS_DIR / "cnn_tuning.csv"

cnn_grid = {
    "filters":       [16, 32],
    "dropout":       [0.0, 0.3],
    "learning_rate": [1e-3, 1e-4],
    "kernel_size":   [3, 5],
}

if CNN_TUNE_CSV.exists():
    print(f"Loading cached CNN tuning results from {CNN_TUNE_CSV}")
    cnn_results = pd.read_csv(CNN_TUNE_CSV)
else:
    rows = []
    combos = list(itertools.product(*cnn_grid.values()))
    print(f"Running CNN tuning: {len(combos)} combos, up to 25 epochs each (early stop)")
    for i, (filters, dropout, lr, ks) in enumerate(combos, start=1):
        tf.random.set_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        model = make_cnn(filters=filters, dropout=dropout,
                         learning_rate=lr, kernel_size=ks)
        es = callbacks.EarlyStopping(patience=3, restore_best_weights=True,
                                     monitor="val_accuracy", mode="max")
        t0 = time.time()
        h = model.fit(
            X_tr_norm, y_tr,
            validation_data=(X_val_norm, y_val),
            epochs=25, batch_size=128, verbose=0, callbacks=[es],
        )
        fit_time = time.time() - t0
        val_acc = max(h.history["val_accuracy"])
        rows.append(dict(
            filters=filters, dropout=dropout, learning_rate=lr, kernel_size=ks,
            val_accuracy=val_acc, fit_time=fit_time,
            epochs_trained=len(h.history["loss"]),
        ))
        print(f"  [{i:>2}/{len(combos)}] filters={filters:>2}, "
              f"dropout={dropout:.1f}, lr={lr:.0e}, k={ks} -> "
              f"val_acc={val_acc:.4f} ({fit_time:.1f}s)")
        keras.backend.clear_session()
    cnn_results = pd.DataFrame(rows).sort_values("val_accuracy", ascending=False)
    cnn_results.to_csv(CNN_TUNE_CSV, index=False)
    print(f"Saved -> {CNN_TUNE_CSV}")

display(cnn_results.head(10))
best_cnn = cnn_results.iloc[0]
best_cnn_params = {
    "filters":       int(best_cnn["filters"]),
    "dropout":       float(best_cnn["dropout"]),
    "learning_rate": float(best_cnn["learning_rate"]),
    "kernel_size":   int(best_cnn["kernel_size"]),
}
print("Best CNN val accuracy :", round(best_cnn["val_accuracy"], 4))
print("Best CNN hyper-params :", best_cnn_params)
''')

CNN_TUNE_VIZ = code('''
# A2: §3 — Visualise CNN tuning trends.
fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), sharey=True)
for ax, hp in zip(axes, ["filters", "dropout", "learning_rate", "kernel_size"]):
    grouped = cnn_results.groupby(hp)["val_accuracy"].agg(["mean", "max"])
    grouped.plot(kind="bar", ax=ax, color=["#4c72b0", "#dd8452"])
    ax.set_title(hp)
    ax.set_ylabel("Val accuracy")
    ax.legend(["mean", "max"], fontsize=8)
plt.tight_layout()
plt.show()


# Persist the best hyper-parameters for §4 so it can run independently of §3.
best_hparams = {
    "svm": best_svm_params,
    "mlp": best_mlp_params,
    "cnn": best_cnn_params,
}
HPARAMS_PATH = OUTPUTS_DIR / "best_hparams.json"
with open(HPARAMS_PATH, "w") as f:
    json.dump(best_hparams, f, indent=2)
print(f"Saved best hyper-parameters -> {HPARAMS_PATH}")
print(json.dumps(best_hparams, indent=2))
''')

# §4 — final-model code cells (must run independently of §3)
FINAL_LOAD = code('''
# A2: §4 — load best hyper-parameters (independent of §3).
# The §3 grid-search cells are slow; markers may skip them. We therefore
# reload the search artefacts from disk and fall back to sensible hard-coded
# values if the file is missing.
HPARAMS_PATH = OUTPUTS_DIR / "best_hparams.json"
DEFAULT_BEST = {
    "svm": {"n_components": 30, "C": 5.0, "gamma": 0.001},
    "mlp": {"units": 128, "dropout": 0.0, "learning_rate": 1e-4},
    "cnn": {"filters": 16, "dropout": 0.3, "learning_rate": 1e-3, "kernel_size": 5},
}
if HPARAMS_PATH.exists():
    BEST = json.load(open(HPARAMS_PATH))
    print(f"Loaded best hyper-parameters from {HPARAMS_PATH}")
else:
    BEST = DEFAULT_BEST
    print("Best hyper-parameter file not found - using hard-coded defaults.")
print(json.dumps(BEST, indent=2))

# Container that the final summary table will populate.
final_results = {}
''')

FINAL_SVM = code('''
# A2: §4 — Final SVM trained on the entire training set (train + validation).
# We refit on (X_train_flat, y_train) and evaluate on the held-out test set.
svm_final = make_svm(**BEST["svm"])

t0 = time.time()
svm_final.fit(X_train_flat, y_train)
svm_train_time = time.time() - t0

t0 = time.time()
svm_pred = svm_final.predict(X_test_flat)
svm_pred_time = time.time() - t0

svm_acc = accuracy_score(y_test, svm_pred)
svm_f1  = f1_score(y_test, svm_pred, average="macro")
final_results["SVM"] = {
    "test_accuracy":  svm_acc,
    "test_macro_f1":  svm_f1,
    "train_time_s":   svm_train_time,
    "predict_time_s": svm_pred_time,
    "hparams":        BEST["svm"],
}

print(f"SVM trained on full {len(y_train):,} samples in {svm_train_time:.1f}s")
print(f"Test accuracy : {svm_acc:.4f}")
print(f"Macro F1      : {svm_f1:.4f}")
print()
print(classification_report(y_test, svm_pred,
                            target_names=[class_names[i] for i in range(NUM_CLASSES)]))

fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(
    confusion_matrix(y_test, svm_pred),
    display_labels=[class_names[i] for i in range(NUM_CLASSES)],
)
disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45, values_format="d")
ax.set_title("SVM — confusion matrix on test set")
plt.tight_layout()
plt.show()
''')

FINAL_MLP = code('''
# A2: §4 — Final MLP trained on the entire training set (train + validation).
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

mlp_final = make_mlp(**BEST["mlp"])
es = callbacks.EarlyStopping(patience=4, restore_best_weights=True,
                             monitor="val_accuracy", mode="max")

# Use 10 % of the full training data as a small held-out monitor for early
# stopping; this is independent from y_test (which we never touch during fit).
X_fit, X_es, y_fit, y_es = train_test_split(
    X_train_norm, y_train,
    test_size=0.1, stratify=y_train, random_state=RANDOM_SEED,
)

t0 = time.time()
mlp_hist_final = mlp_final.fit(
    X_fit, y_fit,
    validation_data=(X_es, y_es),
    epochs=40, batch_size=128, verbose=2, callbacks=[es],
)
mlp_train_time = time.time() - t0

t0 = time.time()
mlp_pred = mlp_final.predict(X_test_norm, verbose=0).argmax(axis=1)
mlp_pred_time = time.time() - t0

mlp_acc = accuracy_score(y_test, mlp_pred)
mlp_f1  = f1_score(y_test, mlp_pred, average="macro")
final_results["MLP"] = {
    "test_accuracy":  mlp_acc,
    "test_macro_f1":  mlp_f1,
    "train_time_s":   mlp_train_time,
    "predict_time_s": mlp_pred_time,
    "hparams":        BEST["mlp"],
    "epochs_trained": len(mlp_hist_final.history["loss"]),
}

print(f"MLP trained for {final_results['MLP']['epochs_trained']} epochs in "
      f"{mlp_train_time:.1f}s")
print(f"Test accuracy : {mlp_acc:.4f}")
print(f"Macro F1      : {mlp_f1:.4f}")
print()
print(classification_report(y_test, mlp_pred,
                            target_names=[class_names[i] for i in range(NUM_CLASSES)]))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
ConfusionMatrixDisplay(
    confusion_matrix(y_test, mlp_pred),
    display_labels=[class_names[i] for i in range(NUM_CLASSES)],
).plot(ax=axes[0], cmap="Blues", colorbar=False, xticks_rotation=45, values_format="d")
axes[0].set_title("MLP — confusion matrix on test set")

axes[1].plot(mlp_hist_final.history["accuracy"], label="train")
axes[1].plot(mlp_hist_final.history["val_accuracy"], label="val")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].set_title("MLP learning curve")
axes[1].legend()
plt.tight_layout()
plt.show()
''')

FINAL_CNN = code('''
# A2: §4 — Final CNN trained on the entire training set (train + validation).
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

cnn_final = make_cnn(**BEST["cnn"])
es = callbacks.EarlyStopping(patience=4, restore_best_weights=True,
                             monitor="val_accuracy", mode="max")

X_fit, X_es, y_fit, y_es = train_test_split(
    X_train_norm, y_train,
    test_size=0.1, stratify=y_train, random_state=RANDOM_SEED,
)

t0 = time.time()
cnn_hist_final = cnn_final.fit(
    X_fit, y_fit,
    validation_data=(X_es, y_es),
    epochs=40, batch_size=128, verbose=2, callbacks=[es],
)
cnn_train_time = time.time() - t0

t0 = time.time()
cnn_pred = cnn_final.predict(X_test_norm, verbose=0).argmax(axis=1)
cnn_pred_time = time.time() - t0

cnn_acc = accuracy_score(y_test, cnn_pred)
cnn_f1  = f1_score(y_test, cnn_pred, average="macro")
final_results["CNN"] = {
    "test_accuracy":  cnn_acc,
    "test_macro_f1":  cnn_f1,
    "train_time_s":   cnn_train_time,
    "predict_time_s": cnn_pred_time,
    "hparams":        BEST["cnn"],
    "epochs_trained": len(cnn_hist_final.history["loss"]),
}

print(f"CNN trained for {final_results['CNN']['epochs_trained']} epochs in "
      f"{cnn_train_time:.1f}s")
print(f"Test accuracy : {cnn_acc:.4f}")
print(f"Macro F1      : {cnn_f1:.4f}")
print()
print(classification_report(y_test, cnn_pred,
                            target_names=[class_names[i] for i in range(NUM_CLASSES)]))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
ConfusionMatrixDisplay(
    confusion_matrix(y_test, cnn_pred),
    display_labels=[class_names[i] for i in range(NUM_CLASSES)],
).plot(ax=axes[0], cmap="Blues", colorbar=False, xticks_rotation=45, values_format="d")
axes[0].set_title("CNN — confusion matrix on test set")

axes[1].plot(cnn_hist_final.history["accuracy"], label="train")
axes[1].plot(cnn_hist_final.history["val_accuracy"], label="val")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].set_title("CNN learning curve")
axes[1].legend()
plt.tight_layout()
plt.show()
''')

FINAL_SUMMARY_MD = new_markdown_cell(
    "### Final-model comparison\n"
    "The table below consolidates the test-set performance and runtime for each model "
    "with its tuned hyper-parameters."
)

FINAL_SUMMARY = code('''
# A2: §4 — Consolidated comparison of the three final models.
summary_rows = []
for name in ["SVM", "MLP", "CNN"]:
    r = final_results[name]
    summary_rows.append({
        "Model":           name,
        "Test accuracy":   round(r["test_accuracy"], 4),
        "Macro F1":        round(r["test_macro_f1"], 4),
        "Train time (s)":  round(r["train_time_s"], 1),
        "Predict time (s)": round(r["predict_time_s"], 1),
        "Hyper-parameters": r["hparams"],
    })
summary_df = pd.DataFrame(summary_rows)
display(summary_df)

summary_df.to_csv(OUTPUTS_DIR / "final_results.csv", index=False)
with open(OUTPUTS_DIR / "final_results.json", "w") as f:
    json.dump(final_results, f, indent=2, default=str)
print(f"Saved final results to {OUTPUTS_DIR/'final_results.json'}")
''')

# §5 AI Acknowledgement (replace the original instructional markdown with the
# real acknowledgement text required by the rubric).
AI_ACK_MD = new_markdown_cell(
    "## 5. AI Acknowledgement\n"
    "We used **Anthropic Claude (Claude Opus 4.7)** through the Claude Code CLI as a "
    "coding-and-writing assistant for this assignment.\n"
    "\n"
    "**How the tool was used:**\n"
    "* Refactoring Section 1 of this notebook into clearly delimited subsections, "
    "fixing a hard-coded sample-count typo, and adding the stratified train/validation "
    "split.\n"
    "* Drafting the model factory functions (`make_svm`, `make_mlp`, `make_cnn`), the "
    "grid-search loops for §3 and the test-set evaluation cells in §4.\n"
    "* Drafting the LaTeX report sections (Introduction, Data, Methods, Results & "
    "Discussion, Conclusion).\n"
    "\n"
    "All AI-suggested code and prose were read, validated, and edited by the group "
    "members; numerical results were obtained by executing the code on our own "
    "hardware (Apple M3 with `tensorflow-metal`)."
)


# ---------------------------------------------------------------------------
# Apply the inserts. Working in reverse order so index references stay valid.
# ---------------------------------------------------------------------------

# Re-resolve indices because earlier inserts shift them. We'll use a robust
# pattern: locate each anchor markdown right before inserting, remove any
# previously-generated A2 cells beneath it, then insert.

def reinsert(after_predicate, until_predicate, new_cells):
    idx = find_index(after_predicate)
    if idx < 0:
        raise RuntimeError("anchor not found")
    remove_generated_after(idx, until_predicate)
    return insert_after(idx, new_cells)


# §2 shared factories — sit directly under "## 2. Algorithm design and setup"
reinsert(
    md_match('## 2. Algorithm design and setup'),
    md_match('### Algorithm of choice from first six weeks'),
    [SHARED_FACTORIES],
)

# §2 SVM baseline — under the "### Algorithm of choice…" subheader inside §2
def is_section2_choice_anchor(c):
    return (c.cell_type == 'markdown'
            and 'Algorithm of choice from first six weeks' in c.source
            and 'Support Vector Machine' in c.source)
reinsert(
    is_section2_choice_anchor,
    md_match('### Fully connected neural network'),
    [SVM_BASELINE],
)

# §2 MLP baseline
def is_section2_mlp_anchor(c):
    return (c.cell_type == 'markdown'
            and c.source.strip().startswith('### Fully connected neural network'))
reinsert(
    is_section2_mlp_anchor,
    md_match('### Convolutional neural network'),
    [MLP_BASELINE],
)

# §2 CNN baseline — anchor is the first "### Convolutional neural network" header
def is_section2_cnn_anchor(c):
    if c.cell_type != 'markdown':
        return False
    if not c.source.strip().startswith('### Convolutional neural network'):
        return False
    # Need the §2 instance, not the §3 / §4 ones. We use the fact that §3
    # is preceded by "## 3.".
    return True

# Apply only to the FIRST CNN subheader (which lives in §2)
i_first_cnn = find_index(is_section2_cnn_anchor)
remove_generated_after(i_first_cnn, md_match('## 3. Hyperparameter tuning'))
insert_after(i_first_cnn, [CNN_BASELINE])


# §3 SVM tuning — anchor is "### Algorithm of choice…" beneath "## 3."
i_section3 = find_index(md_match('## 3. Hyperparameter tuning'))
i_sec3_choice = find_index(
    md_match('### Algorithm of choice from first six weeks'), i_section3 + 1
)
remove_generated_after(i_sec3_choice,
                       md_match('### Fully connected neural network'))
insert_after(i_sec3_choice, [SVM_TUNE, SVM_TUNE_VIZ])

# §3 MLP tuning
i_section3 = find_index(md_match('## 3. Hyperparameter tuning'))
i_sec3_mlp = find_index(
    md_match('### Fully connected neural network'), i_section3 + 1
)
remove_generated_after(i_sec3_mlp, md_match('### Convolutional neural network'))
insert_after(i_sec3_mlp, [MLP_TUNE, MLP_TUNE_VIZ])

# §3 CNN tuning
i_section3 = find_index(md_match('## 3. Hyperparameter tuning'))
i_sec3_cnn = find_index(md_match('### Convolutional neural network'), i_section3 + 1)
remove_generated_after(i_sec3_cnn, md_match('## 4. Final models'))
insert_after(i_sec3_cnn, [CNN_TUNE, CNN_TUNE_VIZ])


# §4 — final cells. Insert FINAL_LOAD right under "## 4. Final models"
i_section4 = find_index(md_match('## 4. Final models'))
remove_generated_after(i_section4,
                       md_match('### Algorithm of choice from first six weeks'))
insert_after(i_section4, [FINAL_LOAD])

# §4 SVM final
i_section4 = find_index(md_match('## 4. Final models'))
i_sec4_choice = find_index(
    md_match('### Algorithm of choice from first six weeks'), i_section4 + 1
)
remove_generated_after(i_sec4_choice,
                       md_match('### Fully connected neural network'))
insert_after(i_sec4_choice, [FINAL_SVM])

# §4 MLP final
i_section4 = find_index(md_match('## 4. Final models'))
i_sec4_mlp = find_index(
    md_match('### Fully connected neural network'), i_section4 + 1
)
remove_generated_after(i_sec4_mlp, md_match('### Convolutional neural network'))
insert_after(i_sec4_mlp, [FINAL_MLP])

# §4 CNN final + summary table
i_section4 = find_index(md_match('## 4. Final models'))
i_sec4_cnn = find_index(md_match('### Convolutional neural network'), i_section4 + 1)
remove_generated_after(i_sec4_cnn, md_match('## 5. AI Acknowledgement'))
# clear any prior summary-md we might have inserted
j = i_sec4_cnn + 1
while (j < len(nb.cells)
       and not (nb.cells[j].cell_type == 'markdown'
                and 'AI Acknowledgement' in nb.cells[j].source)):
    if (nb.cells[j].cell_type == 'markdown'
            and 'Final-model comparison' in nb.cells[j].source):
        del nb.cells[j]
    else:
        j += 1
insert_after(i_sec4_cnn, [FINAL_CNN, FINAL_SUMMARY_MD, FINAL_SUMMARY])


# §5 Replace AI Acknowledgement markdown
i_section5 = find_index(md_match('## 5. AI Acknowledgement'))
nb.cells[i_section5] = AI_ACK_MD


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
nbf.write(nb, NB_PATH.as_posix())
print(f"Updated notebook: {NB_PATH}")
print(f"Total cells now : {len(nb.cells)}")
