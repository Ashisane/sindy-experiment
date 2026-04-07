# -*- coding: utf-8 -*-
"""
this_inference.py  —  TASK D
==============================
THIS (Taylor-based Hypergraph Inference using SINDy) on D1 and D8 voltage traces.
Infers pairwise edges and triadic hyperedges from c302 simulation data.

Based on Delabays et al. 2025 formulation:
  x_dot_i ≈ D(X) v_i  where D includes linear (x_j) and quadratic (x_j*x_k) terms
  Nonzero v_i[x_j*x_k] → triadic hyperedge {i,j,k}
"""

import sys, os, json, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MDG_BUILD  = r"C:\Users\UTKARSH\Desktop\mdg\mdg_build"
OUT_STAGES = os.path.join(MDG_BUILD, "output_stages")
OUT_SIM    = os.path.join(MDG_BUILD, "output_sim")

THIS_THRESH = 0.10   # STLSQ threshold for THIS
ALPHA       = 0.01
MAX_ITER    = 100
DT_MS       = 0.05   # simulation dt in ms
MAX_ACTIVE  = 60     # cap active neurons to keep quadratic library tractable
N_NEAR_BASE = 2000   # sample points nearest to base point for THIS

# Known circuit modules
CIRCUIT_MODULES = {
    "locomotion":     {"AVBL","AVBR","AVAL","AVAR","PVCL","PVCR",
                       "DB1","DB2","DB3","DB4","DB5","DB6","DB7",
                       "VB1","VB2","VB3","VB4","VB5","VB6","VB7","VB8","VB9","VB10","VB11"},
    "mechanosensory": {"ALML","ALMR","PVDL","PVDR","PVCL","PVCR","PLML","PLMR"},
    "chemosensory":   {"AWCL","AWCR","ASEL","ASER","AIAL","AIAR","AIBL","AIBR",
                       "ASHL","ASHR","ADLL","ADLR"},
}

print("=" * 68)
print("TASK D — THIS Hyperedge Inference on D1 and D8")
print("=" * 68)

# ── Load class membership and adjacency ───────────────────────────────────────
with open(os.path.join(OUT_SIM, "class_members.json"), encoding="utf-8") as f:
    class_members = json.load(f)
with open(os.path.join(OUT_SIM, "class_names.txt"), encoding="utf-8") as f:
    sorted_classes = [l.strip() for l in f if l.strip()]
A_class   = np.load(os.path.join(OUT_SIM, "A_class.npy"))
class_idx = {c: i for i, c in enumerate(sorted_classes)}

def neuron_to_class(name):
    n = name[:-1] if len(name) > 2 and name[-1].upper() in "LR" else name
    n = re.sub(r"\d+$", "", n)
    return n if n else "UNK"

def classes_connected(n1, n2):
    c1, c2 = neuron_to_class(n1), neuron_to_class(n2)
    i1, i2 = class_idx.get(c1, -1), class_idx.get(c2, -1)
    if i1 < 0 or i2 < 0:
        return False
    return A_class[i1, i2] > 0 or A_class[i2, i1] > 0


# ── THIS utilities ────────────────────────────────────────────────────────────
def ridge_solve(A, b, alpha=ALPHA):
    M = A.T @ A + alpha * np.eye(A.shape[1])
    try:
        return np.linalg.solve(M, A.T @ b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(M, A.T @ b, rcond=None)[0]

def stlsq(Theta, Y, tau):
    n = Theta.shape[1]
    xi = ridge_solve(Theta, Y)
    for _ in range(MAX_ITER):
        active = np.abs(xi) >= tau
        if not active.any():
            return np.zeros(n)
        T_a = Theta[:, active]
        xi_new = np.zeros(n)
        xi_new[active] = ridge_solve(T_a, Y)
        xi_new[np.abs(xi_new) < tau] = 0.0
        if np.array_equal(xi_new != 0, xi != 0):
            return xi_new
        xi = xi_new
    return xi


def run_this(dat_path, stage_tag, neuron_list):
    """Run THIS algorithm on a voltage .dat file. Returns dict of results."""
    print(f"\n  Loading {dat_path} ...")
    if not os.path.exists(dat_path):
        print(f"    File not found: {dat_path}")
        return None

    V   = np.loadtxt(dat_path)
    Vmv = V[:, 1:] * 1000.0                  # (T, N)
    T, N = Vmv.shape
    N_neurons = min(N, len(neuron_list))
    neuron_list = neuron_list[:N_neurons]
    Vmv = Vmv[:, :N_neurons]

    print(f"    Shape: {V.shape}  mV range: {Vmv.min():.1f} to {Vmv.max():.1f}")

    # Select active neurons only
    Vmax = Vmv.max(axis=0)
    active_mask = Vmax > -20.0
    n_active    = int(active_mask.sum())
    print(f"    Active neurons (Vmax > -20mV): {n_active}/{N_neurons}")

    if n_active < 3:
        print(f"    Too few active neurons for THIS. Skipping.")
        return {"n_active": n_active, "n_pairwise": 0, "n_triadic": 0, "error": "too_few_active"}

    # Cap at MAX_ACTIVE for tractability
    if n_active > MAX_ACTIVE:
        top_idx = np.argsort(Vmax)[::-1][:MAX_ACTIVE]
        active_mask = np.zeros(N_neurons, dtype=bool)
        active_mask[top_idx] = True
        n_active = MAX_ACTIVE
        print(f"    Capped at {MAX_ACTIVE} most-active neurons for tractability")

    active_idx    = np.where(active_mask)[0]
    active_neurons = [neuron_list[i] for i in active_idx]
    X_active      = Vmv[:, active_idx]   # (T, n_active)

    # THIS pre-processing
    x0       = np.median(X_active, axis=0)          # base point = median
    X_dev    = X_active - x0                        # deviations
    X_std    = X_dev.std(axis=0)
    X_std    = np.where(X_std < 1e-6, 1.0, X_std)
    X_norm   = X_dev / X_std                        # normalized deviations

    # Sample points closest to base point (||X_dev||_2)
    dist_to_base = np.linalg.norm(X_dev, axis=1)
    near_idx     = np.argsort(dist_to_base)[:N_NEAR_BASE]
    near_idx     = np.sort(near_idx)
    X_s          = X_norm[near_idx, :]              # (N_near, n_active)

    # Compute time derivative by central finite differences
    # Only valid for near_idx points that are interior
    interior = (near_idx > 0) & (near_idx < T - 1)
    near_int = near_idx[interior]

    Xdot_s = np.zeros((len(near_int), n_active))
    for ti, t_abs in enumerate(near_int):
        Xdot_s[ti, :] = (X_norm[t_abs + 1, :] - X_norm[t_abs - 1, :]) / (2 * DT_MS)

    # Build monomial library D(X):
    # Columns: [const, x_0, ..., x_{n-1}, x_i*x_j for connected (i,j)]
    X_int = X_norm[near_int, :]   # (n_int, n_active)
    n_int = len(near_int)

    # Linear columns
    lib_cols  = [np.ones(n_int)]
    lib_names = ["const"]
    for i in range(n_active):
        lib_cols.append(X_int[:, i])
        lib_names.append(active_neurons[i])

    # Quadratic columns (structurally connected pairs only)
    quad_pairs = []
    for i in range(n_active):
        for j in range(i + 1, n_active):
            if classes_connected(active_neurons[i], active_neurons[j]):
                lib_cols.append(X_int[:, i] * X_int[:, j])
                lib_names.append(f"{active_neurons[i]}*{active_neurons[j]}")
                quad_pairs.append((i, j))

    Theta = np.column_stack(lib_cols)   # (n_int, n_lib)
    n_lib = Theta.shape[1]
    n_linear = 1 + n_active
    print(f"    Library: {n_int} rows × {n_lib} cols  ({n_active} linear + {len(quad_pairs)} quadratic)")

    # Run STLSQ for each active neuron
    Xi = np.zeros((n_active, n_lib))
    for i in range(n_active):
        xi_i = stlsq(Theta, Xdot_s[:, i], THIS_THRESH)
        Xi[i, :] = xi_i

    # Extract edges and hyperedges
    pairwise_edges = []
    triadic_hyperedges = []

    for i in range(n_active):
        for j in range(1, n_linear):
            if abs(Xi[i, j]) >= THIS_THRESH and j - 1 != i:
                pairwise_edges.append((active_neurons[j-1], active_neurons[i], float(Xi[i, j])))

    for q_idx, (j_, k_) in enumerate(quad_pairs):
        col_idx = n_linear + q_idx
        for i in range(n_active):
            if abs(Xi[i, col_idx]) >= THIS_THRESH:
                triadic_hyperedges.append({
                    "i": active_neurons[i],
                    "j": active_neurons[j_],
                    "k": active_neurons[k_],
                    "coef": float(Xi[i, col_idx])
                })

    print(f"    Pairwise edges:     {len(pairwise_edges)}")
    print(f"    Triadic hyperedges: {len(triadic_hyperedges)}")

    # Circuit consistency check
    circuit_consistent = 0
    for h in triadic_hyperedges:
        neurons = {h["i"], h["j"], h["k"]}
        for module_name, module_set in CIRCUIT_MODULES.items():
            if len(neurons & module_set) >= 2:
                circuit_consistent += 1
                h["circuit"] = module_name
                break
        else:
            h["circuit"] = "unknown"

    print(f"    Circuit-consistent triadic: {circuit_consistent}/{len(triadic_hyperedges)}")

    return {
        "stage": stage_tag,
        "n_active": n_active,
        "active_neurons": active_neurons,
        "n_pairwise": len(pairwise_edges),
        "n_triadic": len(triadic_hyperedges),
        "n_circuit_consistent": circuit_consistent,
        "pairwise_edges": pairwise_edges[:50],    # top 50
        "triadic_hyperedges": triadic_hyperedges[:30],
        "Xi": Xi.tolist(),
    }


# ── Find D1 and D8 .dat files ─────────────────────────────────────────────────
def find_dat(stage_label):
    """Try task B stages first, then fall back to 20pA sims."""
    candidates = []
    # Task B outputs
    d = os.path.join(OUT_STAGES, stage_label)
    for f in os.listdir(d) if os.path.isdir(d) else []:
        if f.endswith(".dat") and "activity" not in f:
            candidates.append(os.path.join(d, f))
    # 20pA sims in mdg_build
    for f in os.listdir(MDG_BUILD):
        if f.endswith(".dat") and "activity" not in f and stage_label.lower() in f.lower():
            candidates.append(os.path.join(MDG_BUILD, f))
    # output_c0
    for f in os.listdir(os.path.join(MDG_BUILD, "output_c0")):
        if f.endswith(".dat") and "activity" not in f and stage_label.lower() in f.lower():
            candidates.append(os.path.join(MDG_BUILD, "output_c0", f))
    return candidates[0] if candidates else None


def load_neuron_order(stage_num):
    d   = os.path.join(OUT_STAGES, f"D{stage_num}")
    txt = os.path.join(d, f"neuron_order_D{stage_num}.txt")
    if os.path.exists(txt):
        with open(txt) as f:
            return [l.strip() for l in f if l.strip()]
    # fallback: try LEMS file column extraction
    return [f"neuron_{i}" for i in range(250)]


# Use 20pA D1 and D8 traces (already available) as fallback
D1_DAT  = os.path.join(MDG_BUILD, "MDG_C0_D1.dat")
D8_DAT  = os.path.join(MDG_BUILD, "MDG_C0_D8.dat")

# Try to get neuron orders from LEMS files
def neuron_order_from_lems(lems_path, fallback_n=200):
    neurons = []
    if not os.path.exists(lems_path):
        return [f"n_{i}" for i in range(fallback_n)]
    with open(lems_path, encoding="utf-8") as f:
        for line in f:
            if "OutputColumn" in line or "outputColumn" in line:
                import re
                m = re.search(r'id=["\']([^"\']+)["\']', line)
                if m:
                    n = m.group(1).replace("_v", "").replace("Pop_", "")
                    neurons.append(n)
    return neurons if neurons else [f"n_{i}" for i in range(fallback_n)]

neurons_d1 = neuron_order_from_lems(
    os.path.join(MDG_BUILD, "output_c0", "LEMS_MDG_C0_D1.xml"))
neurons_d8 = neuron_order_from_lems(
    os.path.join(MDG_BUILD, "output_c0", "LEMS_MDG_C0_D8.xml"))

print(f"\n  D1 neuron order: {len(neurons_d1)} neurons from LEMS")
print(f"  D8 neuron order: {len(neurons_d8)} neurons from LEMS")

results_d1 = run_this(D1_DAT, "D1", neurons_d1)
results_d8 = run_this(D8_DAT, "D8", neurons_d8)


# ── Compare D1 vs D8 ──────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("D1 vs D8 HYPEREDGE COMPARISON")
print("=" * 68)
for tag, r in [("D1", results_d1), ("D8", results_d8)]:
    if r:
        print(f"  {tag}: {r['n_active']} active | "
              f"{r['n_pairwise']} pairwise edges | "
              f"{r['n_triadic']} triadic hyperedges | "
              f"{r['n_circuit_consistent']} circuit-consistent")

if results_d1 and results_d8 and not results_d1.get("error") and not results_d8.get("error"):
    n1, n8 = results_d1["n_triadic"], results_d8["n_triadic"]
    if n8 > n1:
        print(f"\n  MORE triadic hyperedges in adult (D8={n8}) vs hatchling (D1={n1})")
        print(f"  Developmental hyperedge gain: +{n8-n1}")
    elif n8 == n1:
        print(f"\n  Same number of triadic hyperedges D1 vs D8 ({n1})")
    else:
        print(f"\n  FEWER triadic hyperedges in adult (D8={n8}) vs hatchling (D1={n1})")
        print(f"  (Possible: stronger inhibitory pruning in adult)")

    # Stable hyperedges (in both)
    if results_d1.get("triadic_hyperedges") and results_d8.get("triadic_hyperedges"):
        h1_set = {frozenset([h["i"], h["j"], h["k"]]) for h in results_d1["triadic_hyperedges"]}
        h8_set = {frozenset([h["i"], h["j"], h["k"]]) for h in results_d8["triadic_hyperedges"]}
        stable  = h1_set & h8_set
        novel_d8 = h8_set - h1_set
        print(f"  Stable hyperedges (D1 AND D8): {len(stable)}")
        print(f"  Novel in D8 (adult-specific):  {len(novel_d8)}")
        if novel_d8:
            print(f"  First 5 adult-specific: {list(novel_d8)[:5]}")


# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(OUT_STAGES, exist_ok=True)
with open(os.path.join(OUT_STAGES, "THIS_report.md"), "w", encoding="utf-8") as f:
    f.write("# THIS Hyperedge Inference — D1 vs D8\n\n")
    for tag, r in [("D1", results_d1), ("D8", results_d8)]:
        if not r:
            continue
        f.write(f"## {tag}\n")
        f.write(f"- Active neurons: {r['n_active']}\n")
        f.write(f"- Pairwise edges: {r['n_pairwise']}\n")
        f.write(f"- Triadic hyperedges: {r['n_triadic']}\n")
        f.write(f"- Circuit-consistent triads: {r['n_circuit_consistent']}\n\n")
        if r.get("triadic_hyperedges"):
            f.write("### Top triadic hyperedges\n\n| i | j | k | coef | circuit |\n|---|---|---|---|---|\n")
            for h in r["triadic_hyperedges"][:10]:
                f.write(f"| {h['i']} | {h['j']} | {h['k']} | {h['coef']:+.4f} | {h.get('circuit','?')} |\n")
            f.write("\n")

for tag, r_data, fname in [("D1", results_d1, "THIS_D1_results.json"),
                             ("D8", results_d8, "THIS_D8_results.json")]:
    if r_data:
        save = {k: v for k, v in r_data.items() if k != "Xi"}
        with open(os.path.join(OUT_STAGES, fname), "w", encoding="utf-8") as f:
            json.dump(save, f, indent=2)

print(f"\n  Saved THIS_report.md, THIS_D1_results.json, THIS_D8_results.json")
print("\n=== TASK D COMPLETE ===")
