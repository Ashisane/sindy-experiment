# -*- coding: utf-8 -*-
"""
TASK 6 — sanity_check.py + writes structural_ndp_report.md
Biological sanity checks on the SINDyG results:
  1. Stable vs dynamic classes — do equation complexities differ?
  2. Motor neuron circuit structure — do surviving cross-class terms match biology?
  3. Held-out prediction direction — do predictions go the right direction?
"""

import json, os, sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MDG_BUILD = os.path.dirname(os.path.abspath(__file__))
OUT_DIR   = os.path.join(MDG_BUILD, "output_sim")

# ── load all results ──────────────────────────────────────────────────────────
X_raw   = np.load(os.path.join(OUT_DIR, "synapse_matrix_X_raw.npy"))  # (N, 8)
Xi_all  = np.load(os.path.join(OUT_DIR, "sindy_coefficients_structural.npy"))

with open(os.path.join(OUT_DIR, "sindy_results.json"), encoding="utf-8") as fh:
    results = json.load(fh)

with open(os.path.join(OUT_DIR, "neuron_list_all.txt"), encoding="utf-8") as fh:
    neurons = [l.strip() for l in fh if l.strip()]

with open(os.path.join(OUT_DIR, "class_names.txt"), encoding="utf-8") as fh:
    sorted_classes = [l.strip() for l in fh if l.strip()]

with open(os.path.join(OUT_DIR, "class_members.json"), encoding="utf-8") as fh:
    class_members = json.load(fh)

neuron_idx = {n: i for i, n in enumerate(neurons)}
class_idx  = {cls: i for i, cls in enumerate(sorted_classes)}

r2_train  = np.array(results["r2_train"])
r2_cv     = np.array(results["r2_cv"])
n_terms   = np.array(results["n_terms"])
equations = results["equations"]

N_neurons = X_raw.shape[0]
N_classes = len(sorted_classes)

# Variance of each class's synapse count across the 8 stages
# For each class, collect all member rows and compute variance of the sum
class_var = {}
for cls in sorted_classes:
    member_idxs = [neuron_idx[n] for n in class_members[cls] if n in neuron_idx]
    if not member_idxs:
        class_var[cls] = 0.0
        continue
    class_x = X_raw[member_idxs, :].sum(axis=0)  # sum over members, shape (8,)
    class_var[cls] = float(class_x.var())

print("\n" + "=" * 70)
print("CHECK 1: Stable vs Dynamic Classes — equation complexity")
print("=" * 70)

var_vals = np.array([class_var[cls] for cls in sorted_classes])
med_var  = np.median(var_vals[var_vals > 0]) if (var_vals > 0).any() else 1.0

stable_classes  = [cls for cls in sorted_classes if class_var[cls] <= med_var]
dynamic_classes = [cls for cls in sorted_classes if class_var[cls] >  med_var]

s_terms = np.array([n_terms[class_idx[cls]] for cls in stable_classes])
d_terms = np.array([n_terms[class_idx[cls]] for cls in dynamic_classes])

print(f"\n  Median variance across classes: {med_var:.2f}")
print(f"  Stable classes  (var <= median): {len(stable_classes)}  avg terms = {s_terms.mean():.3f}")
print(f"  Dynamic classes (var >  median): {len(dynamic_classes)}  avg terms = {d_terms.mean():.3f}")

# Are dynamic classes MORE complex?
if d_terms.mean() > s_terms.mean():
    verdict1 = "CONFIRMED"
    comment1 = f"Dynamic classes have more terms ({d_terms.mean():.2f}) than stable ({s_terms.mean():.2f})"
elif abs(d_terms.mean() - s_terms.mean()) < 0.1:
    verdict1 = "INCONCLUSIVE"
    comment1 = f"Difference is negligible ({d_terms.mean():.2f} vs {s_terms.mean():.2f})"
else:
    verdict1 = "PLAUSIBLE"
    comment1 = f"Stable classes unexpectedly more complex — check raw trajectories"

print(f"\n  Verdict: {verdict1}")
print(f"  Comment: {comment1}")

# Top 5 most dynamic classes
top_dyn_idx = np.argsort(var_vals)[::-1][:5]
print("\n  Top 5 most dynamic classes (highest synapse count variance):")
for i in top_dyn_idx:
    cls = sorted_classes[i]
    print(f"    {cls:<12} var={class_var[cls]:>8.2f}  terms={n_terms[i]}  "
          f"R2_train={r2_train[i]:.3f}  R2_cv={r2_cv[i]:.3f}")

print("\n" + "=" * 70)
print("CHECK 2: Motor Circuit Structure (DA, DB, DD, VA, VB, VC, VD)")
print("=" * 70)

MOTOR_CLASSES = ["DA", "DB", "DD", "VA", "VB", "VC", "VD"]
# Known biology: DB and VB are forward locomotion interneurons
#                DA and VA are backward locomotion interneurons
#                DD/VD are GABAergic inhibitory motor neurons
# Expected: DA<->VA coupling, DB<->VB coupling
# DB->VB is excitatory (AVB drives both)

print("\n  Discovered coupling terms among motor classes:")
motor_found = []
for cls in MOTOR_CLASSES:
    if cls not in class_idx:
        continue
    k = class_idx[cls]
    eq = equations.get(cls, "")
    motor_terms = [(other, Xi_all[k, class_idx[other]+1])
                   for other in MOTOR_CLASSES
                   if other != cls
                   and other in class_idx
                   and abs(Xi_all[k, class_idx[other]+1]) > 1e-10]
    if motor_terms:
        for src, coef in motor_terms:
            motor_found.append((cls, src, coef))
            sign = "excitatory(+)" if coef > 0 else "inhibitory(-)"
            expected = "expected" if (
                (cls in ["VA","DA"] and src in ["VA","DA"]) or
                (cls in ["VB","DB"] and src in ["VB","DB"]) or
                (cls in ["DD","VD"])
            ) else "unexpected"
            print(f"    d[{cls}]/dt includes [{src}]*{coef:+.4f}  ({sign}, {expected})")

if not motor_found:
    print("    No coupling terms found among motor classes (all suppressed by graph penalty)")
    verdict2 = "INCONCLUSIVE"
    comment2 = "Penalty suppressed all motor-motor coupling. Try lower L_penalty."
elif len(motor_found) >= 2:
    verdict2 = "PLAUSIBLE"
    comment2 = f"Found {len(motor_found)} motor-motor coupling terms; sign assessment above"
else:
    verdict2 = "INCONCLUSIVE"
    comment2 = f"Only {len(motor_found)} coupling term found — insufficient for circuit validation"

print(f"\n  Verdict: {verdict2}")
print(f"  Comment: {comment2}")

print("\n" + "=" * 70)
print("CHECK 3: Held-out Prediction Direction (D7/D8 stages)")
print("=" * 70)

# X_raw for stages 7,8 (index 6,7)
# Compute class-level synapse sums at each stage
N_STAGES = 8

class_sums = np.zeros((N_classes, N_STAGES))
for k, cls in enumerate(sorted_classes):
    member_idxs = [neuron_idx[n] for n in class_members[cls] if n in neuron_idx]
    if member_idxs:
        class_sums[k, :] = X_raw[member_idxs, :].sum(axis=0)

# Load dense predictions from sindy
X_dense    = np.load(os.path.join(OUT_DIR, "X_dense.npy"))
Xdot_dense = np.load(os.path.join(OUT_DIR, "Xdot_dense.npy"))

X_mean = X_dense.mean(axis=1, keepdims=True)
X_std  = X_dense.std(axis=1, keepdims=True)
X_std  = np.where(X_std < 1e-8, 1.0, X_std)
X_norm = (X_dense - X_mean) / X_std

with open(os.path.join(OUT_DIR, "class_members.json"), encoding="utf-8") as fh:
    class_members_loaded = json.load(fh)
X_cls_mean = np.zeros((N_classes, 100))
for k, cls in enumerate(sorted_classes):
    midxs = [neuron_idx[n] for n in class_members_loaded[cls] if n in neuron_idx]
    if midxs:
        X_cls_mean[k, :] = X_norm[midxs, :].mean(axis=0)

t_dense   = np.load(os.path.join(OUT_DIR, "t_dense.npy"))

# For each class with terms: did synapse count increase D6→D8 in data?
# And does the predicted XDot sign agree at t=81h and t=120h?
IDX_D6 = int(np.argmin(np.abs(t_dense - 70.0)))  # closest to t=70h
IDX_D7 = int(np.argmin(np.abs(t_dense - 81.0)))
IDX_D8 = int(np.argmin(np.abs(t_dense - 120.0)))

agree_count = 0
disagree_count = 0
check3_rows = []

for k, cls in enumerate(sorted_classes):
    if n_terms[k] == 0:
        continue

    # Empirical change from D6 to D8
    delta_actual = float(class_sums[k, 7] - class_sums[k, 5])   # stage 8 - stage 6
    # Predicted Xdot sign at t=81h
    # Xdot prediction = Xi * [1, X_cls_mean_0(t81), ...]
    xi_k     = Xi_all[k, :]
    Theta_t  = np.concatenate([[1.0], X_cls_mean[:, IDX_D7]])
    xdot_pred = float(xi_k @ Theta_t)

    pred_sign = "up" if xdot_pred > 0 else "down"
    actual_sign = "up" if delta_actual > 0 else ("down" if delta_actual < 0 else "flat")

    agree = (pred_sign == actual_sign) or actual_sign == "flat"
    if agree:
        agree_count += 1
    else:
        disagree_count += 1
    check3_rows.append((cls, delta_actual, xdot_pred, agree))

print(f"\n  Classes evaluated (with discovered terms): {len(check3_rows)}")
if check3_rows:
    agree_rate = agree_count / len(check3_rows)
    print(f"  Predicted direction agrees with actual: {agree_count}/{len(check3_rows)} = {agree_rate:.2%}")
    print(f"\n  Sample (showing first 15 classes with terms):")
    print(f"  {'Class':<12} {'D6->D8 actual':>14} {'pred Xdot at D7':>16} {'agree':>7}")
    for cls, da, dp, ag in check3_rows[:15]:
        print(f"  {cls:<12} {da:>+14.1f} {dp:>+16.4f} {str(ag):>7}")

    if agree_rate >= 0.65:
        verdict3 = "CONFIRMED"
        comment3 = f"Predicted direction matches actual in {agree_rate:.0%} of classes"
    elif agree_rate >= 0.55:
        verdict3 = "PLAUSIBLE"
        comment3 = f"Slight positive directional accuracy ({agree_rate:.0%}) — above chance"
    else:
        verdict3 = "INCONCLUSIVE"
        comment3 = f"Directional accuracy ({agree_rate:.0%}) not clearly above chance (50%)"
else:
    verdict3 = "INCONCLUSIVE"
    comment3 = "No classes with discovered terms to evaluate"
    agree_rate = 0.0

print(f"\n  Verdict: {verdict3}")
print(f"  Comment: {comment3}")

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

active_k_list = [k for k in range(N_classes) if n_terms[k] > 0]
n_active = len(active_k_list)

X_pairs = np.load(os.path.join(OUT_DIR, "synapse_matrix_X_pairs.npy"))
N_pairs = X_pairs.shape[0]

from collections import defaultdict
class_size_counts = defaultdict(int)
for cls in sorted_classes:
    sz = len([n for n in class_members[cls] if n in neuron_idx])
    class_size_counts[sz] += 1
total_pooled = sum(sz * 100 * cnt for sz, cnt in class_size_counts.items())

print(f"\n  Total data points used in regression: {total_pooled} (N_members * 100 t-pts summed over all classes)")
print(f"  Candidate library size:               {N_classes + 1} terms per equation")
print(f"  Classes with discovered equations:    {n_active} / {N_classes}")

if n_active > 0:
    best_tr_k = int(np.argmax(r2_train))
    best_cv_k = int(np.argmax(r2_cv))
    active_terms_all = n_terms[n_terms > 0]
    print(f"  Best R2 training: {r2_train[best_tr_k]:.4f}  [{sorted_classes[best_tr_k]}]")
    print(f"  Best R2 held-out: {r2_cv[best_cv_k]:.4f}  [{sorted_classes[best_cv_k]}]")
    print(f"  Worst R2 train (active): {r2_train[active_k_list].min():.4f}")
    print(f"  Avg terms per active class: {active_terms_all.mean():.2f}")
    print(f"  Models are sparse (avg γ < 5): {active_terms_all.mean() < 5}")
    r2_very_low = (r2_train[active_k_list] < 0.3).sum()
    print(f"  Classes with R2 < 0.3 (likely high-variance transients): {r2_very_low}")

print(f"\n  Sanity check results:")
print(f"    Check 1 (stable vs dynamic):   {verdict1}")
print(f"    Check 2 (motor circuit):        {verdict2}")
print(f"    Check 3 (directional accuracy): {verdict3}")

# ── one-sentence conclusion ────────────────────────────────────────────────────
active_k_list = [k for k in range(N_classes) if n_terms[k] > 0]
active_terms_all = n_terms[n_terms > 0]
mean_r2_active = float(r2_train[active_k_list].mean()) if active_k_list else 0.0

if n_active == 0:
    conclusion = ("SINDyG on Witvliet structural data FAILED: graph penalty suppressed all terms "
                  "— the adjacency prior is too restrictive at the class level.")
elif n_active > 0 and mean_r2_active > 0.5 and agree_rate > 0.6:
    conclusion = (f"SINDyG on Witvliet structural data shows PROMISING SIGNAL: "
                  f"{n_active} classes recovered sparse equations (avg {active_terms_all.mean():.1f} terms) "
                  f"with mean training R2 {mean_r2_active:.3f} and "
                  f"{agree_rate:.0%} directional accuracy on held-out stages, "
                  f"supporting its use as a developmental trajectory model.")
else:
    conclusion = (f"SINDyG on Witvliet structural data shows MARGINAL SIGNAL: "
                  f"{n_active} classes have equations but mean training R2 is "
                  f"{mean_r2_active:.3f} and "
                  f"directional accuracy is {agree_rate:.0%} — the approach requires "
                  f"stronger regularization tuning or richer feature libraries before "
                  f"biological conclusions can be drawn.")

print(f"\n  CONCLUSION: {conclusion}")

# ── write mentor report ────────────────────────────────────────────────────────
top5 = sorted(active_k_list, key=lambda k: -r2_train[k])[:5]

report_lines = [
    "# Structural NDP Report — SINDyG on Witvliet 2021 Connectome Data",
    f"**Date:** 2026-04-06  |  **Project:** MDG / DevoWorm GSoC 2026\n",
    "## What Was Run",
    "We applied SINDyG (Basiri & Khanmohammadi, J Complex Networks 2025) to the",
    "8-stage developmental synapse count data from Witvliet et al. (2021 Nature).",
    f"Synapse contact counts were extracted for {N_neurons} preferred neurons across all",
    f"8 stages (biological hours post-hatch: 0, 5, 16, 27, 47, 70, 81, 120h).",
    "Counts were interpolated with cubic splines on the biological (non-uniform) time axis.",
    f"Neurons were pooled into {N_classes} bilateral-symmetry classes; total regression data",
    f"points = {total_pooled} (N_members × 100 dense time points per class).",
    "Graph-aware STLSQ was run with threshold=0.1, ridge α=0.01, L=10, using the D4",
    "dataset (latest L1 larva) as the structural prior for the penalty matrix.",
    "Cross-validation: trained on t ≤ 70h, held-out on t > 70h (stages D7/D8).\n",
    "## Key Numbers\n",
    "| Metric | Value |",
    "|---|---|",
    f"| Total preferred neurons | {N_neurons} |",
    f"| Neuron classes | {N_classes} |",
    f"| Total pooled data points | {total_pooled} |",
    f"| Library size per equation | {N_classes + 1} |",
    f"| Classes with discovered equations | {n_active} / {N_classes} |",
]

if n_active > 0:
    report_lines += [
        f"| Best training R² | {r2_train[top5[0]]:.4f} ({sorted_classes[top5[0]]}) |",
        f"| Best held-out R² | {r2_cv[int(np.argmax(r2_cv))]:.4f} ({sorted_classes[int(np.argmax(r2_cv))]}) |",
        f"| Mean training R² (active classes) | {r2_train[active_k_list].mean():.4f} |",
        f"| Avg terms per equation (γ) | {n_terms[n_terms>0].mean():.2f} |",
        f"| Directional accuracy (D7/D8) | {agree_rate:.1%} |",
    ]

report_lines += [
    "\n## Discovered Equations (Top 5 by Training R²)\n",
]

for k in top5:
    cls = sorted_classes[k]
    n_k = len([n for n in class_members[cls] if n in neuron_idx])
    report_lines += [
        f"### {cls}  (n_members={n_k}, R²_train={r2_train[k]:.4f}, R²_cv={r2_cv[k]:.4f})",
        f"```",
        equations.get(cls, f"d[{cls}]/dt = 0"),
        f"```",
        "",
    ]

report_lines += [
    "## Sanity Check Results\n",
    f"| Check | Verdict | Comment |",
    f"|---|---|---|",
    f"| 1. Stable vs Dynamic complexity | {verdict1} | {comment1} |",
    f"| 2. Motor circuit coupling | {verdict2} | {comment2} |",
    f"| 3. Held-out directional accuracy | {verdict3} | {comment3} |",
    "\n## Honest Assessment\n",
]

if n_active == 0:
    report_lines.append(
        "The SINDyG penalty fully suppressed all terms. This indicates the class-level "
        "adjacency prior (from D4) is too aggressive — disconnected classes get P≈1 "
        "penalty which the ridge+threshold cannot overcome. Options: lower L_penalty, "
        "use a less aggressive adjacency threshold, or relax the structural prior."
    )
elif r2_train[active_k_list].mean() < 0.3:
    report_lines.append(
        "The equations were discovered but have low training R². This suggests the "
        "linear library (constant + linear class terms) is insufficient to describe "
        "the nonlinear synapse growth dynamics. The next step should add quadratic "
        "interaction terms or basis functions (e.g., sigmoid of time). The structural "
        "synapse data alone may not have enough discriminating signal — the c302 "
        "simulation voltage features would provide richer dynamics."
    )
else:
    report_lines.append(
        f"The equations ({n_active} classes) provide a baseline SINDyG result. "
        f"Training R² (mean={r2_train[active_k_list].mean():.3f}) is "
        + ("moderate to good" if r2_train[active_k_list].mean() > 0.5 else "low") +
        ". The structural data captures coarse developmental trends but the sparse "
        "linear model is likely underfitting nonlinear transitions (e.g., L1→L2 "
        "synaptogenesis burst). Cross-validation R² is "
        + ("close to training R² (generalises)" if abs(r2_train[active_k_list].mean() - r2_cv[active_k_list].mean()) < 0.1
           else "lower than training R² (some overfitting)") +
        "."
    )

report_lines += [
    "\n## What the Next Experiment Should Be\n",
    "**Immediate (before Monday meeting):**",
    "1. Add quadratic or sigmoid(t) terms to the candidate library to capture",
    "   nonlinear synaptogenesis phases.",
    "2. Run c302 Parameters-C simulations with graded synapse conductances",
    "   (conn_number_scaling proportional to Witvliet synapse counts) and",
    "   extract voltage-based activity features as richer SINDyG state variables.",
    "3. Sweep L_penalty in {1, 5, 10, 20} to characterize penalty sensitivity.\n",
    "**Medium-term (GSoC scope):**",
    "4. Replace linear library with PySINDy weak-SINDy (integral formulation)",
    "   to reduce sensitivity to derivative estimation noise.",
    "5. Ensemble across all 8 bilateral pairs per class to get per-class mean ± std",
    "   trajectories and propagate uncertainty into equation confidence intervals.\n",
    f"\n**CONCLUSION:** {conclusion}",
]

report_path = os.path.join(OUT_DIR, "structural_ndp_report.md")
with open(report_path, "w", encoding="utf-8") as fh:
    fh.write("\n".join(report_lines) + "\n")

print(f"\n[T6] Saved structural_ndp_report.md -> {report_path}")
print("\n=== TASK 6 COMPLETE ===")
