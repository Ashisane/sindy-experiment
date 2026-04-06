# -*- coding: utf-8 -*-
"""
TASK 5 — sindy_structural.py
SINDyG (Basiri & Khanmohammadi, J Complex Networks 2025) applied to Witvliet
developmental synapse count trajectories.

For each neuron class k:
  - State  X_k:    normalized trajectory of all class members (n_k, 100)
  - Target Xdot_k: normalized derivative
  - Library Theta: [1, mean_class_1, ..., mean_class_N]  shape (n_k*100, N+1)
  - Penalty P[c,k]: graph-aware weight from A_class
  - Solve penalized STLSQ iteratively

Outputs
-------
output_sim/sindy_coefficients_structural.npy  (N_classes, N_lib)
output_sim/sindy_equations.txt
output_sim/sindy_results.json
"""

import json, os, sys, re, time
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MDG_BUILD = os.path.dirname(os.path.abspath(__file__))
OUT_DIR   = os.path.join(MDG_BUILD, "output_sim")

# ── SINDyG hyper-parameters ──────────────────────────────────────────────────
THRESHOLD   = 0.05  # STLSQ hard-threshold on normalized coefficients
ALPHA       = 0.001 # ridge regularization (lighter — let graph penalty do the work)
MAX_ITER    = 100   # max STLSQ iterations
L_PENALTY   = 3.0   # sharpness of sigmoid penalty — lowered from 10:
                    # at density=5.8%, L=10 gives mean P≈0.99 (kills everything);
                    # L=3 gives mean P≈0.85 for disconnected, ≈0.18 for connected
IDX_SPLIT   = 58    # time index split: t_dense[58] ≈ 70.5h → train ≤ this, test > this


def neuron_to_class(name: str) -> str:
    if len(name) > 2 and name[-1] in "LR":
        name = name[:-1]
    name = re.sub(r"\d+$", "", name)
    return name if name else "UNK"


def sindy_penalty(A_col: np.ndarray, L: float = L_PENALTY) -> np.ndarray:
    """
    Compute penalty vector for candidate terms pointing to a given class k.
    P[j] = 1 / (1 + exp(L * (A_col[j] - 0.5)))
    A_col[j] = normalized adjacency from source class j to sink class k.
    P near 0 → well-connected, low penalty (retain term).
    P near 1 → disconnected,  high penalty (suppress term).
    """
    return 1.0 / (1.0 + np.exp(L * (A_col - 0.5)))


def graph_stlsq(Theta: np.ndarray,
                Y: np.ndarray,
                P_vec: np.ndarray,
                alpha:   float = ALPHA,
                threshold: float = THRESHOLD,
                max_iter: int = MAX_ITER):
    """
    Graph-aware Sequential Thresholded Least Squares.

    Parameters
    ----------
    Theta   : (n_samples, n_lib) candidate library
    Y       : (n_samples,)       target (Xdot for one class)
    P_vec   : (n_lib,)           penalty per library column
    alpha   : ridge λ
    threshold: hard threshold on |xi| (relative to max)

    Returns
    -------
    xi      : (n_lib,) coefficient vector
    active  : bool mask of surviving terms
    """
    n_lib = Theta.shape[1]
    active = np.ones(n_lib, dtype=bool)

    # Initial penalized ridge
    def penalized_ridge(Theta_a, P_a):
        # min ||Y - Theta_a @ xi||^2 + alpha * ||P_a * xi||^2
        # Normal equations: (Theta_a.T Theta_a + alpha * diag(P_a^2)) xi = Theta_a.T Y
        A = Theta_a.T @ Theta_a + alpha * np.diag(P_a ** 2)
        b = Theta_a.T @ Y
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b, rcond=None)[0]

    xi = np.zeros(n_lib)

    for it in range(max_iter):
        Theta_a = Theta[:, active]
        P_a     = P_vec[active]

        if Theta_a.shape[1] == 0:   # no active terms left
            break

        xi_a    = penalized_ridge(Theta_a, P_a)

        # Threshold: suppress terms smaller than threshold * max(|xi|)
        max_xi = float(np.abs(xi_a).max()) if len(xi_a) > 0 else 0.0
        if max_xi < 1e-14:          # all coefficients already zero
            active[:] = False
            break
        keep   = np.abs(xi_a) >= threshold * max_xi

        new_active = active.copy()
        new_active[active] = keep

        if np.array_equal(new_active, active):
            # Converged — final unpenalized refit on active set
            if keep.any():
                Theta_final = Theta[:, new_active]
                A = Theta_final.T @ Theta_final + alpha * np.eye(keep.sum())
                b = Theta_final.T @ Y
                try:
                    xi_final = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    xi_final = np.linalg.lstsq(A, b, rcond=None)[0]
                xi[new_active] = xi_final
            break

        active = new_active
        xi[~active] = 0.0

    return xi, active


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0
    return float(1.0 - ss_res / ss_tot)


# ── load data ─────────────────────────────────────────────────────────────────
X_dense    = np.load(os.path.join(OUT_DIR, "X_dense.npy"))       # (N, 100)
Xdot_dense = np.load(os.path.join(OUT_DIR, "Xdot_dense.npy"))    # (N, 100)
A_class    = np.load(os.path.join(OUT_DIR, "A_class.npy"))        # (N_cls, N_cls)

with open(os.path.join(OUT_DIR, "class_names.txt"), encoding="utf-8") as fh:
    sorted_classes = [l.strip() for l in fh if l.strip()]
with open(os.path.join(OUT_DIR, "class_members.json"), encoding="utf-8") as fh:
    class_members = json.load(fh)
with open(os.path.join(OUT_DIR, "neuron_list_all.txt"), encoding="utf-8") as fh:
    neurons = [l.strip() for l in fh if l.strip()]

N_classes = len(sorted_classes)
neuron_idx = {n: i for i, n in enumerate(neurons)}
class_idx  = {cls: i for i, cls in enumerate(sorted_classes)}

print(f"[T5] {N_classes} classes, X_dense {X_dense.shape}")

# ── per-neuron normalization (zero mean, unit std) over dense time axis ───────
X_mean = X_dense.mean(axis=1, keepdims=True)      # (N,1)
X_std  = X_dense.std(axis=1, keepdims=True)       # (N,1)
X_std  = np.where(X_std < 1e-8, 1.0, X_std)       # avoid /0 for flat neurons

X_norm    = (X_dense    - X_mean) / X_std          # (N, 100)
Xdot_norm = Xdot_dense / X_std                    # derivative: only divide by std

# ── class-level mean trajectories for library Theta ───────────────────────────
# X_cls_mean[k] = mean over all members of class k, shape (100,)
X_cls_mean = np.zeros((N_classes, 100))
for k, cls in enumerate(sorted_classes):
    member_idxs = [neuron_idx[n] for n in class_members[cls] if n in neuron_idx]
    if member_idxs:
        X_cls_mean[k, :] = X_norm[member_idxs, :].mean(axis=0)

# Library columns: [constant(1), X_cls_mean_0, ..., X_cls_mean_{N-1}]
N_lib       = N_classes + 1
lib_names   = ["const"] + sorted_classes          # human-readable names

# Build one global Theta_global (N_neurons*100, N_lib) for efficiency
# Rows indexed by (neuron i, time t): row = i*100 + t
# Theta_global[i*100+t, :] = [1, X_cls_mean[0,t], ..., X_cls_mean[N-1,t]]
print(f"[T5] Building global Theta ({len(neurons)*100} × {N_lib}) ...")
Theta_global = np.ones((len(neurons) * 100, N_lib))
for j, cls in enumerate(sorted_classes):
    # column j+1 = X_cls_mean[j] tiled for each neuron
    Theta_global[:, j+1] = np.tile(X_cls_mean[j, :], len(neurons))

print(f"[T5] Theta_global shape: {Theta_global.shape}  ({Theta_global.nbytes/1e6:.1f} MB)")

# ── penalty matrix P: (N_lib, N_classes) ─────────────────────────────────────
# P_mat[:, k] is the penalty vector for equation of class k
# Constant term gets P=0 (no structural constraint on bias)
P_mat = np.zeros((N_lib, N_classes))
for k in range(N_classes):
    A_col = A_class[:, k]          # adjacency from all classes TO class k
    P_lib  = np.zeros(N_lib)
    P_lib[0] = 0.0                  # constant: no penalty
    P_lib[1:] = sindy_penalty(A_col)
    P_mat[:, k] = P_lib

print(f"[T5] Penalty matrix built. Mean penalty = {P_mat[1:, :].mean():.4f}")

# ── run SINDyG for each class ─────────────────────────────────────────────────
Xi_all      = np.zeros((N_classes, N_lib))   # discovered coefficients
r2_train    = np.zeros(N_classes)
r2_cv       = np.zeros(N_classes)            # held-out R² (t > 70h)
n_terms     = np.zeros(N_classes, dtype=int)
eq_strings  = {}

t0 = time.time()

for k, cls in enumerate(sorted_classes):
    member_names = [n for n in class_members[cls] if n in neuron_idx]
    if not member_names:
        continue

    member_rows_all = [neuron_idx[n] for n in member_names]
    n_k = len(member_rows_all)

    # ── build Theta_k and Y_k (stacked over members) ─────────────────────────
    row_idxs_all = []   # indices into Theta_global / Xdot_norm flat arrays
    for ni in member_rows_all:
        row_idxs_all.extend(range(ni * 100, ni * 100 + 100))

    Theta_k = Theta_global[row_idxs_all, :]          # (n_k*100, N_lib)
    Y_k     = Xdot_norm[member_rows_all, :].ravel()  # (n_k*100,)

    # ── split train / CV ──────────────────────────────────────────────────────
    train_mask = np.zeros(n_k * 100, dtype=bool)
    test_mask  = np.zeros(n_k * 100, dtype=bool)
    for m in range(n_k):
        base = m * 100
        train_mask[base: base + IDX_SPLIT]  = True
        test_mask[base + IDX_SPLIT: base + 100] = True

    Theta_tr = Theta_k[train_mask, :]
    Y_tr     = Y_k[train_mask]
    Theta_te = Theta_k[test_mask, :]
    Y_te     = Y_k[test_mask]

    P_vec_k  = P_mat[:, k]

    # ── graph-aware STLSQ ────────────────────────────────────────────────────
    xi_k, active_k = graph_stlsq(Theta_tr, Y_tr, P_vec_k)

    # ── metrics ───────────────────────────────────────────────────────────────
    Y_pred_tr = Theta_k @ xi_k
    Y_pred_te = Theta_te @ xi_k

    r2_train[k] = r_squared(Y_k, Y_pred_tr)        # training (all data)
    r2_cv[k]    = r_squared(Y_te, Y_pred_te)        # held-out

    n_terms[k]  = int(active_k.sum())

    # ── build equation string ─────────────────────────────────────────────────
    active_terms = [(lib_names[j], xi_k[j]) for j in range(N_lib) if abs(xi_k[j]) > 1e-10]
    if active_terms:
        terms_str = " + ".join(f"{c:+.4f}*[{nm}]" for nm, c in active_terms)
        eq_str = f"d[{cls}]/dt = {terms_str}"
    else:
        eq_str = f"d[{cls}]/dt = 0  (no terms survived)"

    eq_strings[cls] = eq_str
    Xi_all[k, :] = xi_k

    # ── progress print every 20 classes ──────────────────────────────────────
    if k % 20 == 0 or n_terms[k] > 0:
        print(f"  [{k:>3}/{N_classes}] {cls:<12}  n_k={n_k}  "
              f"terms={n_terms[k]}  R2_train={r2_train[k]:.3f}  R2_cv={r2_cv[k]:.3f}")

elapsed = time.time() - t0
print(f"\n[T5] All classes processed in {elapsed:.1f}s")

# ── summary statistics ────────────────────────────────────────────────────────
classes_with_terms = [cls for k, cls in enumerate(sorted_classes) if n_terms[k] > 0]
print(f"\n[T5] Classes with ≥1 discovered term: {len(classes_with_terms)} / {N_classes}")

if classes_with_terms:
    r2_active = r2_train[[k for k, cls in enumerate(sorted_classes) if n_terms[k] > 0]]
    r2_cv_act = r2_cv[[k for k, cls in enumerate(sorted_classes) if n_terms[k] > 0]]
    print(f"     R2_train: mean={r2_active.mean():.3f}  max={r2_active.max():.3f}  "
          f"min={r2_active.min():.3f}")
    print(f"     R2_cv:    mean={r2_cv_act.mean():.3f}  max={r2_cv_act.max():.3f}  "
          f"min={r2_cv_act.min():.3f}")
    print(f"     avg terms per active class: {n_terms[n_terms>0].mean():.2f}")

# Top 10 classes by training R²
print(f"\n[T5] Top 10 classes by training R2 (with ≥1 term):")
print(f"  {'Class':<12} {'n_k':>4} {'terms':>6} {'R2_train':>10} {'R2_cv':>10}  Equation")
sorted_by_r2 = sorted(enumerate(sorted_classes), key=lambda x: -r2_train[x[0]])
shown = 0
for k, cls in sorted_by_r2:
    if n_terms[k] == 0:
        continue
    n_k = len([n for n in class_members[cls] if n in neuron_idx])
    print(f"  {cls:<12} {n_k:>4} {n_terms[k]:>6} {r2_train[k]:>10.4f} {r2_cv[k]:>10.4f}  "
          f"{eq_strings[cls][:80]}")
    shown += 1
    if shown >= 10:
        break

# ── save ─────────────────────────────────────────────────────────────────────
np.save(os.path.join(OUT_DIR, "sindy_coefficients_structural.npy"), Xi_all)

with open(os.path.join(OUT_DIR, "sindy_equations.txt"), "w", encoding="utf-8") as fh:
    for cls in sorted_classes:
        fh.write(eq_strings.get(cls, f"d[{cls}]/dt = 0") + "\n")

results = {
    "classes":          sorted_classes,
    "n_terms":          n_terms.tolist(),
    "r2_train":         r2_train.tolist(),
    "r2_cv":            r2_cv.tolist(),
    "equations":        eq_strings,
    "classes_with_terms": classes_with_terms,
    "hyperparams": {
        "threshold": THRESHOLD, "alpha": ALPHA,
        "max_iter": MAX_ITER, "L_penalty": L_PENALTY,
        "idx_split": IDX_SPLIT,
    }
}
with open(os.path.join(OUT_DIR, "sindy_results.json"), "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=2)

print(f"\n[T5] Saved sindy_coefficients_structural.npy {Xi_all.shape}")
print(f"[T5] Saved sindy_equations.txt  ({len(sorted_classes)} lines)")
print(f"[T5] Saved sindy_results.json")

print("\n=== TASK 5 COMPLETE ===")
n_active = len(classes_with_terms)
print(f"  Classes with discovered equations: {n_active} / {N_classes}")
print()
active_k_list = [k for k, cls in enumerate(sorted_classes) if n_terms[k] > 0]
if active_k_list:
    best_train_k = int(np.argmax([r2_train[k] for k in active_k_list]))
    best_cv_k    = int(np.argmax([r2_cv[k]    for k in active_k_list]))
    print(f"  Best R2 training: {r2_train[active_k_list[best_train_k]]:.4f}  [{sorted_classes[active_k_list[best_train_k]]}]")
    print(f"  Best R2 held-out: {r2_cv[active_k_list[best_cv_k]]:.4f}  [{sorted_classes[active_k_list[best_cv_k]]}]")
    print(f"  Avg terms per active class: {n_terms[n_terms>0].mean():.2f}")
    print(f"  Sparse (avg gamma < 5): {n_terms[n_terms>0].mean() < 5}")
