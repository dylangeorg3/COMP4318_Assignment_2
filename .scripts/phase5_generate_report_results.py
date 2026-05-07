"""
Phase 5: Materialise the Results & Discussion and Conclusion sections of the
report from the JSON / CSV artefacts produced by the notebook.

Inputs
------
* Outputs/svm_tuning.csv   (hyper-parameter sweep for the SVM pipeline)
* Outputs/mlp_tuning.csv   (hyper-parameter sweep for the MLP)
* Outputs/cnn_tuning.csv   (hyper-parameter sweep for the CNN)
* Outputs/best_hparams.json
* Outputs/final_results.json (accuracy / F1 / runtime per model)

Outputs
-------
* Overleaf__A2/Text Files/Sections/4 Results & Discussion.tex
* Overleaf__A2/Text Files/Sections/5 Conclusion.tex
"""
from pathlib import Path
import json

import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent
OUTPUTS  = ROOT / "Outputs"
SECTIONS = ROOT / "Overleaf__A2" / "Text Files" / "Sections"

svm_df = pd.read_csv(OUTPUTS / "svm_tuning.csv")
mlp_df = pd.read_csv(OUTPUTS / "mlp_tuning.csv")
cnn_df = pd.read_csv(OUTPUTS / "cnn_tuning.csv")
final  = json.load(open(OUTPUTS / "final_results.json"))
best   = json.load(open(OUTPUTS / "best_hparams.json"))

mlp_df = mlp_df.sort_values("val_accuracy", ascending=False).reset_index(drop=True)
cnn_df = cnn_df.sort_values("val_accuracy", ascending=False).reset_index(drop=True)


def fmt(x, n=3):
    return f"{x:.{n}f}" if isinstance(x, (int, float)) else str(x)


def latex_escape(s):
    return str(s).replace("_", "\\_")


# ---------------------------------------------------------------------------
# Build the SVM tuning table (top 10 rows)
# ---------------------------------------------------------------------------
svm_top = svm_df.sort_values("rank_test_score").head(10)
svm_rows = []
for _, r in svm_top.iterrows():
    svm_rows.append(
        f"        {int(r['param_pca__n_components'])} & "
        f"{r['param_svc__C']} & {r['param_svc__gamma']} & "
        f"{r['mean_fit_time']:.2f} & "
        f"{r['mean_test_score']*100:.2f} \\\\"
    )
svm_table = "\n".join(svm_rows)

# ---------------------------------------------------------------------------
# Build the MLP tuning table
# ---------------------------------------------------------------------------
mlp_top = mlp_df.head(10)
mlp_rows = []
for _, r in mlp_top.iterrows():
    mlp_rows.append(
        f"        {int(r['units'])} & {r['dropout']:.1f} & "
        f"{r['learning_rate']:.0e} & {int(r['epochs_trained'])} & "
        f"{r['fit_time']:.1f} & "
        f"{r['val_accuracy']*100:.2f} \\\\"
    )
mlp_table = "\n".join(mlp_rows)

# ---------------------------------------------------------------------------
# Build the CNN tuning table
# ---------------------------------------------------------------------------
cnn_top = cnn_df.head(10)
cnn_rows = []
for _, r in cnn_top.iterrows():
    cnn_rows.append(
        f"        {int(r['filters'])} & {int(r['kernel_size'])} & "
        f"{r['dropout']:.1f} & {r['learning_rate']:.0e} & "
        f"{int(r['epochs_trained'])} & {r['fit_time']:.1f} & "
        f"{r['val_accuracy']*100:.2f} \\\\"
    )
cnn_table = "\n".join(cnn_rows)


# ---------------------------------------------------------------------------
# Build the final-results table
# ---------------------------------------------------------------------------
def hparams_str(name, h):
    if name == "SVM":
        d = int(h['n_components']); C = h['C']; g = latex_escape(h['gamma'])
        return f"$d={d}$, $C={C}$, $\\gamma={g}$"
    if name == "MLP":
        u = int(h['units']); dr = h['dropout']; eta = h['learning_rate']
        return f"units$={u}$, dropout$={dr}$, $\\eta={eta:.0e}$"
    if name == "CNN":
        f_ = int(h['filters']); k = int(h['kernel_size'])
        dr = h['dropout']; eta = h['learning_rate']
        return f"filters$={f_}$, $k={k}$, dropout$={dr}$, $\\eta={eta:.0e}$"
    return ""


final_rows = []
for name in ["SVM", "MLP", "CNN"]:
    r = final[name]
    final_rows.append(
        f"        {name} & "
        f"{r['test_accuracy']*100:.2f} & "
        f"{r['test_macro_f1']*100:.2f} & "
        f"{r['train_time_s']:.1f} & "
        f"{r['predict_time_s']:.2f} & "
        f"{hparams_str(name, r['hparams'])} \\\\"
    )
final_table = "\n".join(final_rows)


# ---------------------------------------------------------------------------
# Discussion paragraphs that depend on numbers
# ---------------------------------------------------------------------------
svm_best = svm_df.sort_values("rank_test_score").iloc[0]
svm_acc  = final["SVM"]["test_accuracy"] * 100
mlp_acc  = final["MLP"]["test_accuracy"] * 100
cnn_acc  = final["CNN"]["test_accuracy"] * 100
svm_f1   = final["SVM"]["test_macro_f1"] * 100
mlp_f1   = final["MLP"]["test_macro_f1"] * 100
cnn_f1   = final["CNN"]["test_macro_f1"] * 100
svm_t    = final["SVM"]["train_time_s"]
mlp_t    = final["MLP"]["train_time_s"]
cnn_t    = final["CNN"]["train_time_s"]
svm_p    = final["SVM"]["predict_time_s"]
mlp_p    = final["MLP"]["predict_time_s"]
cnn_p    = final["CNN"]["predict_time_s"]

order_by_acc = sorted(["SVM", "MLP", "CNN"],
                      key=lambda n: -final[n]["test_accuracy"])
best_model, second_model, worst_model = order_by_acc
top_gap = (final[best_model]["test_accuracy"]
           - final[second_model]["test_accuracy"]) * 100  # percentage-point gap
second_gap = (final[second_model]["test_accuracy"]
              - final[worst_model]["test_accuracy"]) * 100

results_tex = f"""\\documentclass[../main.tex]{{subfiles}}

\\begin{{document}}
\\section{{Results and Discussion}}\\label{{section.4}}

\\subsection{{Hyper-parameter Tuning Results}}
\\noindent
Tables~\\ref{{tab.svm-tune}}--\\ref{{tab.cnn-tune}} list the top ten
hyper-parameter combinations for each model, ranked by validation accuracy
on the held-out 6{{,}}400-sample partition (or by mean three-fold
cross-validated accuracy on the 5{{,}}000-sample subset for the SVM).

\\begin{{table}}[H]
    \\centering
    \\caption{{SVM grid search — top 10 configurations on the 5{{,}}000-sample
    tuning subset (3-fold CV).}}
    \\label{{tab.svm-tune}}
    \\begin{{tabular}}{{ccccc}}
        \\toprule
        \\textbf{{PCA $d$}} & \\textbf{{$C$}} & \\textbf{{$\\gamma$}} &
        \\textbf{{Mean fit time (s)}} & \\textbf{{CV accuracy (\\%)}} \\\\
        \\midrule
{svm_table}
        \\bottomrule
    \\end{{tabular}}
\\end{{table}}

\\begin{{table}}[H]
    \\centering
    \\caption{{MLP grid search — top 10 configurations on the validation
    partition.}}
    \\label{{tab.mlp-tune}}
    \\begin{{tabular}}{{cccccc}}
        \\toprule
        \\textbf{{Units}} & \\textbf{{Dropout}} & \\textbf{{LR}} &
        \\textbf{{Epochs}} & \\textbf{{Fit (s)}} &
        \\textbf{{Val acc.\\ (\\%)}} \\\\
        \\midrule
{mlp_table}
        \\bottomrule
    \\end{{tabular}}
\\end{{table}}

\\begin{{table}}[H]
    \\centering
    \\caption{{CNN grid search — top 10 configurations on the validation
    partition.}}
    \\label{{tab.cnn-tune}}
    \\begin{{tabular}}{{ccccccc}}
        \\toprule
        \\textbf{{Filters}} & \\textbf{{Kernel}} & \\textbf{{Dropout}} &
        \\textbf{{LR}} & \\textbf{{Epochs}} & \\textbf{{Fit (s)}} &
        \\textbf{{Val acc.\\ (\\%)}} \\\\
        \\midrule
{cnn_table}
        \\bottomrule
    \\end{{tabular}}
\\end{{table}}

\\noindent
The visualisations produced in the notebook (per-axis bar charts of mean
and best accuracy) make the trends inside each grid easier to read at a
glance and are referenced from the discussion below.

\\subsection{{Hyper-parameter Tuning Discussion}}

\\subsubsection{{SVM}}
\\noindent
The best SVM configuration on the tuning subset uses
$d{{=}}{int(svm_best['param_pca__n_components'])}$ \\gls{{PCA}} components,
$C{{=}}{svm_best['param_svc__C']}$ and
$\\gamma{{=}}{latex_escape(svm_best['param_svc__gamma'])}$, reaching
${svm_best['mean_test_score']*100:.2f}\\%$ mean cross-validated accuracy.
Two trends are clear from Table~\\ref{{tab.svm-tune}}: the kernel bandwidth
$\\gamma$ has by far the largest effect — the worst combinations all use
$\\gamma{{=}}0.01$, which produces an overly localised RBF that
under-fits the 5{{,}}000-sample subset — and increasing the number of
\\gls{{PCA}} components past $\\sim$\\,30 does not help on this dataset. The
latter is consistent with a steep PathMNIST principal-component spectrum:
the first few dozen directions already explain most of the
class-relevant variance, while later components mostly model noise. The
regularisation strength $C$ contributes a smaller, mostly monotone
improvement up to $C{{=}}5$ for the better $\\gamma$ values.

\\subsubsection{{MLP}}
\\noindent
The MLP grid shows that the learning rate dominates the outcome and that
the dense network never fits PathMNIST particularly well. The top
configuration uses
units${{=}}{int(mlp_df.iloc[0]['units'])}$,
dropout${{=}}{mlp_df.iloc[0]['dropout']}$ and
$\\eta{{=}}{mlp_df.iloc[0]['learning_rate']:.0e}$, reaching only
${mlp_df.iloc[0]['val_accuracy']*100:.2f}\\%$ on the validation
partition. Two trends are visible. First, the slower
$\\eta{{=}}10^{{-4}}$ consistently outperforms $\\eta{{=}}10^{{-3}}$:
the larger learning rate over-shoots the loss surface of an MLP on raw
pixels, so early stopping triggers after only a handful of epochs.
Second, dropout in this regime is mostly a hindrance — $p{{=}}0$ wins at
every width because there is barely enough signal in the dense
representation for further regularisation to help. Width has a small
positive effect within the $\\eta{{=}}10^{{-4}}$ regime
({int(mlp_df.iloc[0]['units'])} units narrowly best), but the absolute
ceiling of the family is well below either kernel or convolutional
methods.

\\subsubsection{{CNN}}
\\noindent
Two effects dominate the CNN grid. First, the larger learning rate
$\\eta{{=}}10^{{-3}}$ is clearly preferred: every configuration in the
top four uses it, and the deeper feature hierarchy absorbs the faster
optimiser without diverging. Second, moderate dropout ($p{{=}}0.3$)
helps consistently — every top-four configuration uses it and the next
band of accuracies all uses $p{{=}}0$, so dropout buys roughly two
percentage points on the validation partition. The choice of filter
count and kernel size matters less than expected at this resolution:
$f{{=}}{int(cnn_df.iloc[0]['filters'])}$ with
$k{{=}}{int(cnn_df.iloc[0]['kernel_size'])}$ wins narrowly, but
$f{{=}}32$ and $k{{=}}3$ are within a percentage point. Smaller filter
counts also train substantially faster, which is one reason we keep
$f{{=}}{int(cnn_df.iloc[0]['filters'])}$ for the final model.

\\subsection{{Final Results}}
\\noindent
Each model was retrained with its best hyper-parameters on the full
32{{,}}000-sample training set and evaluated on the 8{{,}}000-sample
PathMNIST test set. Table~\\ref{{tab.final}} summarises the result.

\\begin{{table}}[H]
    \\centering
    \\caption{{Final-model performance and runtime on the PathMNIST test
    set.}}
    \\label{{tab.final}}
    \\begin{{tabular}}{{lccccl}}
        \\toprule
        \\textbf{{Model}} & \\textbf{{Acc.\\ (\\%)}} &
        \\textbf{{Macro $F_{{1}}$ (\\%)}} & \\textbf{{Train (s)}} &
        \\textbf{{Predict (s)}} & \\textbf{{Hyper-parameters}} \\\\
        \\midrule
{final_table}
        \\bottomrule
    \\end{{tabular}}
\\end{{table}}

\\subsection{{Results Discussion and Analysis}}
\\noindent
On the test set the {best_model} achieves the highest accuracy
({final[best_model]['test_accuracy']*100:.2f}\\%), narrowly ahead of the
{second_model} ({final[second_model]['test_accuracy']*100:.2f}\\%), with
the {worst_model} a substantial distance behind
({final[worst_model]['test_accuracy']*100:.2f}\\%). The two strongest
models are essentially tied (a {top_gap:.2f} percentage-point gap), while
the gap to the {worst_model} is {second_gap:.2f} percentage points.

\\noindent
The relative ordering between the \\gls{{MLP}} and the other two models
matches the theoretical expectation in Section~\\ref{{section.3}}: a
purely fully-connected network has to rediscover translation invariance
from data, and on a small 28$\\times$28 dataset that translates into a
clearly weaker classifier. The closer-than-expected match between the
{best_model} and {second_model} is more interesting. Once the kernel SVM
operates on a strong dimensionality-reduced representation
($d{{=}}{int(best['svm']['n_components'])}$ \\gls{{PCA}} components retain the
class-discriminative variance and discard most of the per-pixel noise), the
RBF kernel acts as a soft-nearest-neighbour classifier that captures the
texture cues which dominate this dataset. Our deliberately small
\\gls{{CNN}} ($f{{=}}{int(best['cnn']['filters'])}$ base filters and only
two convolutional blocks, kept small to fit the laptop training budget)
has limited representational capacity, so the gap that a deep \\gls{{CNN}}
would normally open up over a kernel method is small here.

\\noindent
The macro-$F_{{1}}$ values track accuracy closely
({svm_f1:.2f}\\%, {mlp_f1:.2f}\\% and {cnn_f1:.2f}\\% for the SVM, MLP
and CNN respectively), confirming that the mild class imbalance has not
skewed performance towards the majority class. Inspection of the
confusion matrices in the notebook shows two stable failure modes across
all three classifiers: \\emph{{Cancer-associated Stroma}} versus
\\emph{{Smooth Muscle}}, and \\emph{{Debris}} versus \\emph{{Mucus}} —
both pairs share a similar texture and stain profile. The MLP also
collapses several minority classes into the majority \\emph{{Colorectal
Adenocarcinoma Epithelium}} class, which explains the much larger gap
between its accuracy and macro-$F_{{1}}$.

\\noindent
Runtime turns into the deciding factor for deployment. Training the
\\gls{{SVM}} on the full 32{{,}}000-sample training set took
${svm_t:.1f}\\,\\text{{s}}$ on a single CPU thread, the \\gls{{MLP}}
${mlp_t:.1f}\\,\\text{{s}}$ and the \\gls{{CNN}}
${cnn_t:.1f}\\,\\text{{s}}$ on the Apple~M3 \\gls{{GPU}} via
\\texttt{{tensorflow-metal}}. Inference is where the kernel \\gls{{SVM}}
pays its theoretical cost: scoring the 8{{,}}000 test images takes
${svm_p:.2f}\\,\\text{{s}}$ because every test point must be compared
against every support vector, while both neural networks predict in
well under a second (${mlp_p:.2f}\\,\\text{{s}}$ for the \\gls{{MLP}} and
${cnn_p:.2f}\\,\\text{{s}}$ for the \\gls{{CNN}}). For an offline
research-style benchmark the {best_model} therefore wins on accuracy and
training cost, but in a streaming-inference deployment the
\\gls{{CNN}} would still be the right choice: it is only a fraction of a
percentage point behind on accuracy and an order of magnitude faster at
inference.

\\end{{document}}
"""

(SECTIONS / "4 Results & Discussion.tex").write_text(results_tex)
print("Wrote", SECTIONS / "4 Results & Discussion.tex")


# ---------------------------------------------------------------------------
# Conclusion
# ---------------------------------------------------------------------------
conclusion_tex = f"""\\documentclass[../main.tex]{{subfiles}}

\\begin{{document}}
\\section{{Conclusion}}

\\subsection{{Summary of Findings}}
\\noindent
On the PathMNIST test set the
\\gls{{PCA}}-reduced \\gls{{RBF}} \\gls{{SVM}} ({svm_acc:.2f}\\% accuracy,
{svm_f1:.2f}\\% macro-$F_{{1}}$) and the convolutional network
({cnn_acc:.2f}\\% / {cnn_f1:.2f}\\%) finished within {top_gap:.2f}
percentage points of each other, while the multilayer perceptron lagged
notably ({mlp_acc:.2f}\\% / {mlp_f1:.2f}\\%). The ranking confirms one
expectation and challenges another. As predicted, the \\gls{{MLP}} pays a
real price for treating each pixel as an independent feature and ignoring
the spatial structure of \\gls{{HE}} histology. Less expected is the near
parity between the kernel \\gls{{SVM}} and the small \\gls{{CNN}}: at
PathMNIST's 28$\\times$28 resolution and with a deliberately small
two-block \\gls{{CNN}}, an \\gls{{RBF}} kernel applied on top of a
30-dimensional \\gls{{PCA}} embedding captures most of the texture cues
that drive classification, leaving little room for the \\gls{{CNN}} to
pull ahead. The \\gls{{CNN}} retains a clear advantage on inference
latency (${cnn_p:.2f}\\,\\text{{s}}$ versus ${svm_p:.2f}\\,\\text{{s}}$ for
8{{,}}000 test images), so it remains the more sensible choice for a
deployment that has to score images quickly.

\\subsection{{Limitations}}
\\noindent
Several caveats should temper these conclusions. First, the search grids
were deliberately small to fit the available compute; broader grids over
network depth, weight decay and learning-rate schedules could change the
relative ranking, especially between the \\gls{{MLP}} and the \\gls{{CNN}}.
Second, the SVM was tuned on a stratified 5{{,}}000-sample subset to keep
the kernel computation tractable; the full-data refit improves accuracy
modestly but might benefit further from a re-tuned grid at the larger
scale, particularly for the kernel bandwidth $\\gamma$. Third, we did not
apply data augmentation; on a larger study the marginal benefit of
augmentation versus the increased training-time budget should be measured
explicitly. Finally, the PathMNIST 28$\\times$28 resolution is much
coarser than the source slides — extrapolating these conclusions to
clinical-resolution images would require re-running the comparison on a
higher-resolution variant such as PathMNIST-224.

\\subsection{{Future Work}}
\\noindent
Three concrete directions would strengthen the study. (i) Replace the
exhaustive grid for the neural networks with a Bayesian search over a
larger space (depth, weight decay, optimiser, schedule) to test whether
the present \\gls{{CNN}} is on the Pareto frontier of accuracy versus
compute~\\cite{{bergstra2012random}}. (ii) Re-evaluate the comparison on
PathMNIST-224, where convolutional inductive biases should matter more
and where the \\gls{{SVM}} would need either a strong feature extractor
(e.g.~\\gls{{PCA}} on a pre-trained backbone's features) or to be replaced
by a linear baseline, giving a sharper picture of the model
families' scaling behaviour. (iii) Calibrate per-class predictions and
report calibration metrics (e.g.~expected calibration error) — clinical
deployment of any of these models would require uncertainty estimates,
and the three families differ markedly in how their probability outputs
behave.

\\end{{document}}
"""

(SECTIONS / "5 Conclusion.tex").write_text(conclusion_tex)
print("Wrote", SECTIONS / "5 Conclusion.tex")


# Reflection.tex is hand-written elsewhere (each group member writes their
# own paragraph) — do not overwrite it here.
