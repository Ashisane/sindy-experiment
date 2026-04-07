# -*- coding: utf-8 -*-
"""
functional_sindy.py  —  TASK C
================================
Weak SINDy on c302 functional features (max_voltage, mean_voltage,
time_above_threshold) extracted from all 8 stage simulations.

Cross-validates coupling terms against structural Weak SINDy results
from output_sim/weak_sindy_results.json.
"""

import sys, os, json, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\Users\UTKARSH\Desktop\mdg\mdg_build")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MDG_BUILD  = r"C:\Users\UTKARSH\Desktop\mdg\mdg_build"
OUT_STAGES = os.path.join(MDG_BUILD, "output_stages")
OUT_SIM    = os.path.join(MDG_BUILD, "output_sim")

T_BIO  = np.array([0, 5, 16, 27, 47, 70, 81, 120], dtype=float)
DELTA_T = np.diff(T_BIO)
T_MID   = (T_BIO[:-1] + T_BIO[1:]) / 2
T_NORM_MID  = T_MID / 120.0
T_NORM_MID2 = T_NORM_MID ** 2

THRESHOLDS  = [0.02, 0.05, 0.10, 0.15]
ALPHA_RIDGE = 0.01
MAX_ITER    = 100
TRAIN_K     = 5    # train on first 5 transitions (D1→D6)
FEAT_NAMES  = ["max_voltage", "mean_voltage", "time_above_threshold"]

KNOWN_COUPLING = {
    ("ALM", "PVC"): "touch mechanosensory pair (Chalfie 1985)",
    ("AIA", "AWC"): "AWC chemosensory → AIA interneuron",
    ("AIA", "AIB"): "AIB co-interneuron in AWC pathway",
    ("AIA", "ADL"): "nociceptive chemosensory → AIA",
    ("BDU", "PVC"): "NOVEL: anterior-posterior co-development",
    ("RIA", "IL"):  "IL sensory → RIA hub interneuron",
}

print("=" * 68)
print("TASK C — FUNCTIONAL WEAK SINDy + CROSS-VALIDATION")
print("=" * 68)

# ── Load neuron ordering and class membership ──────────────────────────────────
with open(os.path.join(OUT_SIM, "class_members.json"), encoding="utf-8") as f:
    class_members = json.load(f)
with open(os.path.join(OUT_SIM, "neuron_list_all.txt"), encoding="utf-8") as f:
    all_neurons_struct = [l.strip() for l in f if l.strip()]

A_class = np.load(os.path.join(OUT_SIM, "A_class.npy"))
sorted_classes = sorted(class_members.keys())
class_idx  = {c: i for i, c in enumerate(sorted_classes)}

def neuron_to_class(name):
    name = name[:-1] if len(name) > 2 and name[-1] in "LR" else name
    name = re.sub(r"\d+$", "", name)
    return name if name else "UNK"


# ── Load functional features from all 8 stages ────────────────────────────────
print("\n  Loading functional features from Task B ...")
# Find neurons common to all 8 stages
neuron_sets = []
neuron_orders = []
for stage in range(1, 9):
    d = os.path.join(OUT_STAGES, f"D{stage}")
    txt = os.path.join(d, f"neuron_order_D{stage}.txt")
    if not os.path.exists(txt):
        print(f"    WARNING: D{stage} neuron order not found — Task B may not be complete")
        neuron_orders.append([])
        neuron_sets.append(set())
        continue
    with open(txt) as f:
        order = [l.strip() for l in f if l.strip()]
    neuron_orders.append(order)
    neuron_sets.append(set(order))

# If Task B not done for all stages, fall back to structural neuron list
if any(len(s) == 0 for s in neuron_sets):
    print("  Task B incomplete. Using structural neuron list as proxy.")
    common_neurons = all_neurons_struct
else:
    common_neurons = sorted(set.intersection(*neuron_sets))
    print(f"  Common neurons across all 8 stages: {len(common_neurons)}")

N_common = len(common_neurons)
neuron_cidx = {n: i for i, n in enumerate(common_neurons)}

# Build feature tensor F[i, stage, feat]  — shape (N_common, 8, 3)
F = np.zeros((N_common, 8, 3))
F[:] = np.nan

for stage in range(1, 9):
    d   = os.path.join(OUT_STAGES, f"D{stage}")
    npy = os.path.join(d, f"features_D{stage}.npy")
    txt = os.path.join(d, f"neuron_order_D{stage}.txt")
    if not os.path.exists(npy) or not os.path.exists(txt):
        continue
    feat   = np.load(npy)            # (N_stage, 4): max_v, mean_v, time_dep, is_act
    order  = neuron_orders[stage - 1]
    for i_n, name in enumerate(order):
        if name in neuron_cidx:
            ci = neuron_cidx[name]
            F[ci, stage - 1, 0] = feat[i_n, 0]   # max_voltage
            F[ci, stage - 1, 1] = feat[i_n, 1]   # mean_voltage
            F[ci, stage - 1, 2] = feat[i_n, 2]   # time_above_threshold

# Fill NaN with 0 (absent neurons = no activity) before SINDy
F_clean = np.where(np.isnan(F), 0.0, F)
print(f"  Feature tensor shape: {F_clean.shape}  (NaN filled with 0)")

# ── Pool features by class ─────────────────────────────────────────────────────
F_cls = np.zeros((len(sorted_classes), 8, 3))  # class-level mean features
for k_cls, cls in enumerate(sorted_classes):
    members = [n for n in class_members[cls] if n in neuron_cidx]
    if members:
        idxs = [neuron_cidx[m] for m in members]
        F_cls[k_cls, :, :] = F_clean[idxs, :, :].mean(axis=0)


# ── SINDy utilities ────────────────────────────────────────────────────────────
def ridge_solve(A, b, alpha=ALPHA_RIDGE):
    M = A.T @ A + alpha * np.eye(A.shape[1])
    try:
        return np.linalg.solve(M, A.T @ b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(M, A.T @ b, rcond=None)[0]

def stlsq(Theta, Y, tau, alpha=ALPHA_RIDGE):
    n = Theta.shape[1]
    xi = ridge_solve(Theta, Y, alpha)
    for _ in range(MAX_ITER):
        active = np.abs(xi) >= tau
        if not active.any():
            return np.zeros(n), active
        T_a = Theta[:, active]
        xi_new = np.zeros(n)
        xi_new[active] = ridge_solve(T_a, Y, alpha)
        xi_new[np.abs(xi_new) < tau] = 0.0
        if np.array_equal(xi_new != 0, xi != 0):
            return xi_new, xi_new != 0
        xi = xi_new
    return xi, xi != 0

def r2_score(y, yp):
    ss_r = np.sum((y - yp)**2); ss_t = np.sum((y - y.mean())**2)
    return 1.0 - ss_r/ss_t if ss_t > 1e-14 else (1.0 if ss_r < 1e-14 else 0.0)


# ── Run functional SINDy per class, per feature ───────────────────────────────
print("\n  Running functional Weak SINDy ...")
func_results = {}

for k_cls, cls in enumerate(sorted_classes):
    members = [n for n in class_members[cls] if n in neuron_cidx]
    if not members:
        continue

    connected = [j for j in range(len(sorted_classes))
                 if A_class[k_cls, j] > 0 and j != k_cls]

    cls_res = {}
    for fi, feat_name in enumerate(FEAT_NAMES):
        # Y: (n_members, 7) effective derivatives
        X_cls = np.array([F_clean[neuron_cidx[m], :, fi] for m in members])  # (n_m, 8)
        X_max = X_cls.max()
        if X_max < 1e-6:
            continue
        X_norm = X_cls / X_max
        dX  = np.diff(X_norm, axis=1)          # (n_m, 7)
        Y_v = (dX / DELTA_T).ravel()            # (n_m*7,)

        # Library columns: [const, t, t², connected class features at midpoints]
        lib_cols  = [np.ones(7), T_NORM_MID, T_NORM_MID2]
        lib_names_l = ["const", "t", "t2"]
        for j in connected:
            cls_j  = sorted_classes[j]
            X_j    = F_cls[j, :, fi]
            X_j_max = X_j.max()
            if X_j_max < 1e-6:
                continue
            X_j_norm = X_j / X_j_max
            mid_j    = (X_j_norm[:-1] + X_j_norm[1:]) / 2
            lib_cols.append(mid_j)
            lib_names_l.append(cls_j)

        Theta_one = np.column_stack(lib_cols)           # (7, n_lib)
        n_m       = len(members)
        Theta_stk = np.tile(Theta_one, (n_m, 1))       # (n_m*7, n_lib)
        Theta_tr  = Theta_stk[:n_m*TRAIN_K, :]
        Y_tr      = Y_v[:n_m*TRAIN_K]
        Theta_te  = Theta_stk[n_m*TRAIN_K:, :]
        Y_te      = Y_v[n_m*TRAIN_K:]

        # Threshold sweep
        best_xi, best_tau, best_r2, best_gamma = None, THRESHOLDS[-1], -np.inf, 99
        for tau in THRESHOLDS:
            xi_t, act_t = stlsq(Theta_tr, Y_tr, tau)
            gam = int(act_t.sum())
            r2t = r2_score(Y_v, Theta_stk @ xi_t)
            if gam <= 5 and r2t > best_r2:
                best_xi, best_tau, best_r2, best_gamma = xi_t, tau, r2t, gam
        if best_xi is None:
            xi_t, _ = stlsq(Theta_tr, Y_tr, THRESHOLDS[-1])
            best_xi = xi_t; best_r2 = r2_score(Y_v, Theta_stk @ xi_t)

        xi_cv, _ = stlsq(Theta_tr, Y_tr, best_tau)
        r2_cv    = r2_score(Y_te, Theta_te @ xi_cv) if len(Y_te) > 1 else float("nan")
        gamma    = int((np.abs(xi_cv) >= best_tau).sum())

        surviving = [(lib_names_l[j], float(xi_cv[j]))
                     for j in range(len(lib_names_l)) if abs(xi_cv[j]) >= best_tau]
        eq = (f"d[{cls}_{feat_name}]/dt = " +
              " + ".join(f"{c:+.4f}*[{nm}]" for nm, c in surviving)
              if surviving else f"d[{cls}_{feat_name}]/dt = 0")

        cls_res[feat_name] = {
            "gamma": gamma, "threshold": best_tau,
            "r2_train": float(best_r2), "r2_cv": float(r2_cv),
            "equation": eq,
            "surviving_terms": surviving,
        }

    if cls_res:
        func_results[cls] = cls_res

# Print summary
print(f"\n  Functional SINDy summary (classes with ≥1 equation):")
print(f"  {'Class':<10} {'Feature':<25} {'γ':>3} {'R2tr':>7} {'R2cv':>7}")
for cls, fdict in sorted(func_results.items(),
                          key=lambda x: -max(v.get("r2_cv", -99)
                                              for v in x[1].values() if isinstance(v, dict))):
    for feat, r in fdict.items():
        if r.get("gamma", 0) > 0:
            print(f"  {cls:<10} {feat:<25} {r['gamma']:>3} "
                  f"{r['r2_train']:>7.4f} {r['r2_cv']:>7.4f}")


# ── Cross-validation with structural SINDy ────────────────────────────────────
print("\n" + "=" * 68)
print("CROSS-VALIDATION: Structural vs Functional NDP coupling terms")
print("=" * 68)

struct_results_path = os.path.join(OUT_SIM, "weak_sindy_results.json")
cross_val_report = []
divergent_report = []

if os.path.exists(struct_results_path):
    with open(struct_results_path, encoding="utf-8") as f:
        struct_res = json.load(f)

    TARGET_CLASSES = {
        "ALM":  {"expected_coupling": ["PVC"], "reason": "touch mechanosensory pair"},
        "AIA":  {"expected_coupling": ["AWC", "AIB", "ADL"], "reason": "chemosensory circuit"},
        "BDU":  {"expected_coupling": ["PVC"], "reason": "NOVEL anterior-posterior"},
        "RIA":  {"expected_coupling": ["IL", "RIV"], "reason": "hub interneuron"},
        "IL1D": {"expected_coupling": ["RMDV"], "reason": "sensory-ring motor"},
    }

    print(f"\n  Checking {len(TARGET_CLASSES)} target classes from structural SINDy:")
    for cls, info in TARGET_CLASSES.items():
        struct_terms = []
        if cls in struct_res:
            struct_terms = [nm for nm, _ in struct_res[cls].get("surviving_terms", [])]

        func_terms = {}
        if cls in func_results:
            for feat, r in func_results[cls].items():
                func_terms[feat] = [nm for nm, _ in r.get("surviving_terms", [])]

        print(f"\n  [{cls}]")
        print(f"    Structural: {struct_terms}")
        for feat, fterms in func_terms.items():
            print(f"    Functional ({feat}): {fterms}")

        # Find cross-validated couplings
        all_func_terms = set(t for ts in func_terms.values() for t in ts)
        for partner in info["expected_coupling"]:
            in_struct  = partner in struct_terms
            in_func    = partner in all_func_terms
            key        = KNOWN_COUPLING.get((cls, partner), "")
            if in_struct and in_func:
                s_coef = next((c for nm, c in struct_res[cls].get("surviving_terms", [])
                               if nm == partner), None)
                f_coef_dict = {feat: next((c for nm, c in r.get("surviving_terms", [])
                                          if nm == partner), None)
                               for feat, r in func_results.get(cls, {}).items()}
                cross_val_report.append({
                    "class": cls, "partner": partner, "bio": key,
                    "struct_coef": s_coef, "func_coef": str(f_coef_dict),
                    "status": "CROSS-VALIDATED"
                })
                print(f"    ✓ CROSS-VALIDATED: {cls}–{partner}  ({key})")
            elif in_struct and not in_func:
                divergent_report.append({
                    "class": cls, "partner": partner,
                    "in_struct": True, "in_func": False,
                    "note": "Structural coupling not reflected in functional features"
                })
                print(f"    ~ Structural only: {cls}–{partner}")
            elif in_func and not in_struct:
                divergent_report.append({
                    "class": cls, "partner": partner,
                    "in_struct": False, "in_func": True,
                    "note": "Novel functional coupling absent in structural SINDy"
                })
                print(f"    * Functional only: {cls}–{partner}")

else:
    print("  weak_sindy_results.json not found — structural SINDy results unavailable")

# ── Save results ──────────────────────────────────────────────────────────────
out_json = os.path.join(OUT_STAGES, "functional_sindy_results.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(func_results, f, indent=2)
print(f"\n  Saved functional_sindy_results.json")

# ── Write cross-validation report ─────────────────────────────────────────────
report_path = os.path.join(OUT_STAGES, "cross_validation_report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("# Structural vs Functional NDP Cross-Validation Report\n\n")
    f.write("## Cross-Validated Couplings\n\n")
    if cross_val_report:
        f.write("| Class | Partner | Biology | Status |\n|---|---|---|---|\n")
        for r in cross_val_report:
            f.write(f"| {r['class']} | {r['partner']} | {r['bio']} | **{r['status']}** |\n")
    else:
        f.write("No cross-validated couplings found. See divergent report.\n")
    f.write("\n## Divergent Couplings\n\n")
    for r in divergent_report:
        source = "structural only" if r["in_struct"] else "functional only"
        f.write(f"- **{r['class']}–{r['partner']}** ({source}): {r['note']}\n")
    f.write("\n## Functional Equations (best R²_cv per class)\n\n")
    for cls, fdict in func_results.items():
        best_feat = max(fdict.items(), key=lambda x: x[1].get("r2_cv", -99))
        feat_name, feat_r = best_feat
        if feat_r.get("gamma", 0) > 0:
            f.write(f"### {cls} ({feat_name})\n```\n{feat_r['equation']}\n```\n")
            f.write(f"R²_train={feat_r['r2_train']:.4f}  R²_cv={feat_r['r2_cv']:.4f}  γ={feat_r['gamma']}\n\n")

print(f"  Saved cross_validation_report.md")
print(f"\n  Summary:")
print(f"    Cross-validated couplings: {len(cross_val_report)}")
print(f"    Divergent (struct only):   {len([r for r in divergent_report if r['in_struct']])}")
print(f"    Divergent (func only):     {len([r for r in divergent_report if r['in_func']])}")

print("\n=== TASK C COMPLETE ===")
