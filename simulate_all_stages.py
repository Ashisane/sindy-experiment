# -*- coding: utf-8 -*-
"""
simulate_all_stages.py  —  TASK B
===================================
Simulate all 8 Witvliet stages at the sensitivity amplitude found by Task A.
Extract per-neuron functional features from each simulation.
"""

import sys, os, json, shutil, time, csv
import numpy as np

sys.path.insert(0, r"C:\Users\UTKARSH\Desktop\mdg\c302")
sys.path.insert(0, r"C:\Users\UTKARSH\Desktop\mdg\mdg_build")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from c302 import parameters_C0
from pyneuroml import pynml
import c302
import witvliet_reader as wr

MDG_BUILD   = r"C:\Users\UTKARSH\Desktop\mdg\mdg_build"
OUT_SWEEP   = os.path.join(MDG_BUILD, "output_sweep")
OUT_STAGES  = os.path.join(MDG_BUILD, "output_stages")
os.makedirs(OUT_STAGES, exist_ok=True)

STIM_NEURONS = ["AVBL", "AVBR", "AVAL", "AVAR", "PVCL", "PVCR"]
FIRED_THRESH = -20.0   # mV — active
DEPOL_THRESH = -40.0   # mV — any depolarisation
DURATION     = 500
DT           = 0.05
T_POST_MS    = 10.0    # ms — post-stimulus window for mean voltage
IDX_POST     = int(T_POST_MS / DT)  # timestep index

# ── Load optimal amplitude from Task A ────────────────────────────────────────
sweep_json = os.path.join(OUT_SWEEP, "sweep_results.json")
if os.path.exists(sweep_json):
    with open(sweep_json, encoding="utf-8") as f:
        sweep_data = json.load(f)
    OPT_AMP = sweep_data.get("optimal_amp", 1.0)
    print(f"  Loaded optimal amplitude from Task A: {OPT_AMP} pA")
else:
    OPT_AMP = 1.0
    print(f"  sweep_results.json not found — using default: {OPT_AMP} pA")

print("=" * 68)
print(f"TASK B — SIMULATE ALL 8 STAGES (amp={OPT_AMP} pA)")
print("=" * 68)

T_BIO   = [0, 5, 16, 27, 47, 70, 81, 120]
STAGES  = list(range(1, 9))

activation_rows = []

params = parameters_C0.ParameterisedModel()
params.set_bioparameter("unphysiological_offset_current",
                        f"{OPT_AMP} pA", "TaskB", "0")
params.set_bioparameter("unphysiological_offset_current_del",
                        "50 ms", "TaskB", "0")
params.set_bioparameter("unphysiological_offset_current_dur",
                        "400 ms", "TaskB", "0")


for stage in STAGES:
    t0 = time.time()
    bio_h = T_BIO[stage - 1]
    net_id = f"Stage_D{stage}_{str(OPT_AMP).replace('.','p')}pA"
    out_dir = os.path.join(OUT_STAGES, f"D{stage}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n  Stage {stage} (D{stage}, {bio_h}h) — {net_id}")

    wr._DEFAULT_STAGE = stage
    wr._instance = None
    cells, _ = wr.read_data(include_nonconnected_cells=False)
    stim = [c for c in STIM_NEURONS if c in cells]
    print(f"    cells={len(cells)}, stimulated={len(stim)}")

    lems = os.path.join(out_dir, f"LEMS_{net_id}.xml")
    c302.generate(
        net_id, params,
        data_reader="witvliet_reader",
        cells=cells,
        cells_to_stimulate=stim,
        cells_to_plot=cells,
        duration=DURATION, dt=DT,
        target_directory=out_dir,
        verbose=False,
    )

    ok = pynml.run_lems_with_jneuroml(
        lems, max_memory="4G", nogui=True, plot=False, verbose=False)

    # Move .dat from CWD to stage dir
    dat_cwd = os.path.join(MDG_BUILD, f"{net_id}.dat")
    dat_dst = os.path.join(out_dir, f"{net_id}.dat")
    if os.path.exists(dat_cwd):
        shutil.move(dat_cwd, dat_dst)
    for ext in [".activity.dat"]:
        f_cwd = os.path.join(MDG_BUILD, f"{net_id}{ext}")
        if os.path.exists(f_cwd):
            shutil.move(f_cwd, os.path.join(out_dir, f"{net_id}{ext}"))

    if not os.path.exists(dat_dst):
        print(f"    ERROR: {dat_dst} not found")
        activation_rows.append({
            "stage": stage, "hours": bio_h,
            "n_neurons": len(cells), "n_active": -1,
            "pct_active": -1, "error": True
        })
        continue

    # ── Extract features ──────────────────────────────────────────────────────
    V    = np.loadtxt(dat_dst)
    Vmv  = V[:, 1:] * 1000.0          # (T, N)
    T, N = Vmv.shape
    post_slice = slice(IDX_POST, T)    # post-stimulus window

    max_v    = Vmv.max(axis=0)                              # (N,)
    mean_v   = Vmv[post_slice, :].mean(axis=0)             # (N,)
    time_dep = (Vmv > DEPOL_THRESH).mean(axis=0)           # fraction of time > -40mV
    is_act   = (max_v > FIRED_THRESH).astype(float)        # 1 or 0

    # Stack features: shape (N, 4)
    features = np.column_stack([max_v, mean_v, time_dep, is_act])

    # Save
    np.save(os.path.join(out_dir, f"features_D{stage}.npy"), features)
    with open(os.path.join(out_dir, f"neuron_order_D{stage}.txt"), "w") as fh:
        for c in cells:
            fh.write(c + "\n")

    n_active = int(is_act.sum())
    pct = n_active / N * 100
    with open(os.path.join(out_dir, f"n_active_D{stage}.txt"), "w") as fh:
        fh.write(str(n_active))

    print(f"    active={n_active}/{N} ({pct:.0f}%)  Vmax={max_v.max():.1f}mV  "
          f"({time.time()-t0:.0f}s)")

    activation_rows.append({
        "stage": stage, "hours": bio_h,
        "n_neurons": N, "n_active": n_active,
        "pct_active": round(pct, 1), "error": False
    })

# ── Summary table ──────────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("ACTIVATION TRAJECTORY ACROSS DEVELOPMENT")
print("=" * 68)
print(f"  {'Stage':>6}  {'Hours':>6}  {'Neurons':>8}  {'Active':>8}  {'%Active':>8}")
for row in activation_rows:
    flag = "" if not row.get("error") else "  ERROR"
    print(f"  D{row['stage']:>5}  {row['hours']:>6}h  {row['n_neurons']:>8}  "
          f"{row['n_active']:>8}  {row['pct_active']:>7}%{flag}")

# Check monotonicity
actives_ok = [r["n_active"] for r in activation_rows if not r.get("error") and r["n_active"] >= 0]
if len(actives_ok) >= 2:
    diffs = [actives_ok[i+1] - actives_ok[i] for i in range(len(actives_ok)-1)]
    n_pos = sum(1 for d in diffs if d > 0)
    n_neg = sum(1 for d in diffs if d < 0)
    if n_pos > n_neg:
        trend = "INCREASING  (D1→D8 more neurons activate — DEVELOPMENTAL SIGNAL CONFIRMED)"
    elif n_neg > n_pos:
        trend = "DECREASING  (unexpected — possible inhibitory circuit dominance)"
    else:
        trend = "FLAT — amplitude needs further tuning to reveal developmental difference"
    print(f"\n  Trend: {trend}")

# Save CSV
csv_path = os.path.join(OUT_STAGES, "activation_table.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["stage","hours","n_neurons","n_active","pct_active","error"])
    w.writeheader()
    w.writerows(activation_rows)
print(f"\n  Saved activation_table.csv")

# Also save full activation data as JSON for downstream tasks
with open(os.path.join(OUT_STAGES, "activation_data.json"), "w") as f:
    json.dump({
        "optimal_amp": OPT_AMP, "stages": activation_rows
    }, f, indent=2)

print("\n=== TASK B COMPLETE ===")
