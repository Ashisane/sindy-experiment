# -*- coding: utf-8 -*-
"""
TASK 2 — spline_interpolation.py
Fit cubic splines to each neuron's 8-stage synapse count trajectory and
compute dense time-series + analytical derivatives.

Biological timepoints (h post-hatch): [0, 5, 16, 27, 47, 70, 81, 120]
NOT evenly spaced.

Outputs
-------
output_sim/X_dense.npy     (N_neurons, 100) – spline values at 100 dense points
output_sim/Xdot_dense.npy  (N_neurons, 100) – spline derivative at same points
output_sim/t_dense.npy     (100,)           – time axis in hours
output_sim/spline_examples.png
"""

import os, sys
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use("Agg")  # no display on Windows server
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MDG_BUILD = os.path.dirname(os.path.abspath(__file__))
OUT_DIR   = os.path.join(MDG_BUILD, "output_sim")

# ── biological time axis ─────────────────────────────────────────────────────
T_BIO   = np.array([0, 5, 16, 27, 47, 70, 81, 120], dtype=float)
T_DENSE = np.linspace(0, 120, 100)

# ── load data ────────────────────────────────────────────────────────────────
X_raw = np.load(os.path.join(OUT_DIR, "synapse_matrix_X_raw.npy")).astype(float)
with open(os.path.join(OUT_DIR, "neuron_list_all.txt"), encoding="utf-8") as fh:
    neurons = [l.strip() for l in fh if l.strip()]

N = X_raw.shape[0]
assert N == len(neurons), f"Mismatch: {N} vs {len(neurons)}"
neuron_idx = {n: i for i, n in enumerate(neurons)}

print(f"[T2] Loaded X_raw shape: {X_raw.shape}")
print(f"[T2] Fitting cubic splines for {N} neurons across t = {T_BIO.tolist()} h")

# ── fit splines ───────────────────────────────────────────────────────────────
X_dense    = np.zeros((N, 100))
Xdot_dense = np.zeros((N, 100))

transient_flags = []   # neurons whose spline goes negative somewhere

for i in range(N):
    y = X_raw[i, :]      # 8 measured values
    cs = CubicSpline(T_BIO, y, bc_type="not-a-knot")
    x_dense_i = cs(T_DENSE)
    X_dense[i, :]    = x_dense_i
    Xdot_dense[i, :] = cs(T_DENSE, 1)   # 1st derivative

    # Flag transient: spline dips below 0 (oscillation around 0-value stages)
    if len(x_dense_i) > 0 and float(x_dense_i.min()) < -0.5:
        transient_flags.append(neurons[i])

print(f"[T2] Splines fit. Transient neurons (spline dips < -0.5): {len(transient_flags)}")
if transient_flags[:10]:
    print(f"     Examples: {transient_flags[:10]}")

# ── print 3 example neurons ──────────────────────────────────────────────────
# Try preferred examples first; if zero counts, fall back to top-count neurons
_PREFERRED = ["AVBL", "AVAL", "RID"]
_totals     = X_raw.sum(axis=1)   # total outgoing synapses across all stages
_sorted_by_count = [neurons[i] for i in np.argsort(_totals)[::-1]]

EXAMPLES = []
for name in _PREFERRED:
    if name in neuron_idx and _totals[neuron_idx[name]] > 0:
        EXAMPLES.append(name)
for name in _sorted_by_count:            # fill up to 3 from highest-count neurons
    if name not in EXAMPLES:
        EXAMPLES.append(name)
    if len(EXAMPLES) == 3:
        break

print(f"\n[T2] Example neurons chosen: {EXAMPLES}")
print("[T2] Raw counts, spline values, derivatives:")
for name in EXAMPLES:
    i  = neuron_idx[name]
    cs = CubicSpline(T_BIO, X_raw[i, :], bc_type="not-a-knot")
    raw_vals    = X_raw[i, :].tolist()
    spline_vals = [round(float(cs(t)),3) for t in T_BIO]
    deriv_vals  = [round(float(cs(t,1)),4) for t in T_BIO]
    print(f"\n  {name}:")
    print(f"    raw counts   : {[int(v) for v in raw_vals]}")
    print(f"    spline values: {spline_vals}  (should match raw)")
    print(f"    derivatives  : {deriv_vals} syn/h")

# ── plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(len(EXAMPLES), 2, figsize=(14, 4*len(EXAMPLES)))
if len(EXAMPLES) == 1:
    axes = axes.reshape(1, 2)  # ensure 2-D indexing
fig.suptitle("Cubic Spline Fits — Witvliet Synapse Counts", fontsize=13, fontweight="bold")

for row, name in enumerate(EXAMPLES):
    i  = neuron_idx[name]
    cs = CubicSpline(T_BIO, X_raw[i, :], bc_type="not-a-knot")
    y_vals  = X_dense[i, :]
    dy_vals = Xdot_dense[i, :]
    ymax    = max(float(y_vals.max()),  1.0)   # at least 1 to avoid autoscale crash
    dymax   = max(float(abs(dy_vals).max()), 0.01)

    ax1 = axes[row, 0]
    ax2 = axes[row, 1]

    ax1.plot(T_DENSE, y_vals, "b-", lw=2, label="Spline")
    ax1.scatter(T_BIO, X_raw[i, :], color="red", zorder=5, s=50, label="Data")
    ax1.axhline(0, color="gray", lw=0.5, ls="--")
    ax1.set_ylim(min(-1.0, float(y_vals.min()) - 0.1), ymax * 1.15)
    ax1.set_title(f"{name} — synapse count")
    ax1.set_xlabel("Time (h post-hatch)")
    ax1.set_ylabel("Synapse count")
    ax1.legend(fontsize=8)
    ax1.set_xlim(0, 120)

    ax2.plot(T_DENSE, dy_vals, "g-", lw=2, label="d/dt (spline)")
    ax2.axhline(0, color="gray", lw=0.5, ls="--")
    ax2.scatter(T_BIO, cs(T_BIO, 1), color="orange", zorder=5, s=50, label="d/dt at stages")
    ax2.set_ylim(-dymax * 1.3, dymax * 1.3)
    ax2.set_title(f"{name} — derivative")
    ax2.set_xlabel("Time (h post-hatch)")
    ax2.set_ylabel("d(synapse)/dt  [syn/h]")
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, 120)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "spline_examples.png"), dpi=120, bbox_inches="tight")
plt.close()
print(f"\n[T2] Saved spline_examples.png")

# ── save ─────────────────────────────────────────────────────────────────────
np.save(os.path.join(OUT_DIR, "X_dense.npy"),    X_dense)
np.save(os.path.join(OUT_DIR, "Xdot_dense.npy"), Xdot_dense)
np.save(os.path.join(OUT_DIR, "t_dense.npy"),    T_DENSE)

print(f"[T2] Saved X_dense.npy    {X_dense.shape}")
print(f"[T2] Saved Xdot_dense.npy {Xdot_dense.shape}")
print(f"[T2] Saved t_dense.npy    {T_DENSE.shape}")

# runtime stats
print(f"\n[T2] Dense trajectory range: {X_dense.min():.3f} to {X_dense.max():.3f}")
print(f"[T2] Derivative range:        {Xdot_dense.min():.4f} to {Xdot_dense.max():.4f} syn/h")

print("\n=== TASK 2 COMPLETE ===")
print(f"  Neurons splined: {N}")
print(f"  Transient (spline < -0.5): {len(transient_flags)}")
print(f"  X_dense shape:  {X_dense.shape}")
print(f"  Xdot_dense shape: {Xdot_dense.shape}")
print(f"  t_dense: linspace(0, 120, 100), step = {T_DENSE[1]-T_DENSE[0]:.4f} h")
