# -*- coding: utf-8 -*-
"""
amplitude_sweep.py  —  TASK A
==============================
Find the developmental sensitivity window: the stimulation amplitude at which
D8 (adult, denser connectome) activates more neurons than D1 (hatchling).

Amplitudes tested: [0.2, 0.5, 1.0, 2.0, 5.0] pA
For each amplitude: run D1, parse voltage, run D8, parse voltage.
Key output: sweep_results.json and sweep_plot.png
"""

import sys, os, json, shutil, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\Users\UTKARSH\Desktop\mdg\c302")
sys.path.insert(0, r"C:\Users\UTKARSH\Desktop\mdg\mdg_build")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from c302 import parameters_C0
from pyneuroml import pynml
import c302
import witvliet_reader as wr

MDG_BUILD  = r"C:\Users\UTKARSH\Desktop\mdg\mdg_build"
OUT_SWEEP  = os.path.join(MDG_BUILD, "output_sweep")
os.makedirs(OUT_SWEEP, exist_ok=True)

AMPLITUDES    = [0.2, 0.5, 1.0, 2.0, 5.0]
STIM_NEURONS  = ["AVBL", "AVBR", "AVAL", "AVAR", "PVCL", "PVCR"]
FIRED_THRESH  = -20.0  # mV
DURATION      = 500
DT            = 0.05

print("=" * 68)
print("TASK A — AMPLITUDE SWEEP (Parameters C0, D1 vs D8)")
print("=" * 68)
print(f"  Amplitudes: {AMPLITUDES} pA")
print(f"  Stimulated: {STIM_NEURONS}")
print(f"  Threshold for 'fired': Vmax > {FIRED_THRESH} mV\n")


def amp_str(amp):
    return str(amp).replace(".", "p")


def run_stage(amp, stage):
    """Generate NeuroML, run jNeuroML, parse .dat. Returns (n_total, n_fired, v_max)."""
    params = parameters_C0.ParameterisedModel()
    params.set_bioparameter("unphysiological_offset_current",
                            f"{amp} pA", "sweep", "0")
    params.set_bioparameter("unphysiological_offset_current_del",
                            "50 ms", "sweep", "0")
    params.set_bioparameter("unphysiological_offset_current_dur",
                            "400 ms", "sweep", "0")

    wr._DEFAULT_STAGE = stage
    wr._instance = None
    cells, _ = wr.read_data(include_nonconnected_cells=False)
    stim = [c for c in STIM_NEURONS if c in cells]

    net_id = f"Sweep_D{stage}_{amp_str(amp)}pA"
    lems   = os.path.join(OUT_SWEEP, f"LEMS_{net_id}.xml")

    c302.generate(
        net_id, params,
        data_reader="witvliet_reader",
        cells=cells,
        cells_to_stimulate=stim,
        cells_to_plot=cells,
        duration=DURATION, dt=DT,
        target_directory=OUT_SWEEP,
        verbose=False,
    )

    ok = pynml.run_lems_with_jneuroml(
        lems, max_memory="4G", nogui=True, plot=False, verbose=False)

    # jNeuroML writes .dat to CWD — move it to output_sweep
    dat_cwd = os.path.join(MDG_BUILD, f"{net_id}.dat")
    dat_dst = os.path.join(OUT_SWEEP, f"{net_id}.dat")
    if os.path.exists(dat_cwd):
        shutil.move(dat_cwd, dat_dst)
        # Also move activity.dat if exists
        act_cwd = os.path.join(MDG_BUILD, f"{net_id}.activity.dat")
        if os.path.exists(act_cwd):
            shutil.move(act_cwd, os.path.join(OUT_SWEEP, f"{net_id}.activity.dat"))

    if not os.path.exists(dat_dst):
        print(f"    ERROR: {dat_dst} not found — sim may have failed")
        return len(cells), 0, 0.0

    V    = np.loadtxt(dat_dst)
    Vmv  = V[:, 1:] * 1000.0
    Vmax = Vmv.max(axis=0)

    n_fired = int((Vmax > FIRED_THRESH).sum())
    n_total = Vmv.shape[1]
    vpc     = float(Vmax.max())
    return n_total, n_fired, vpc


# ── Main sweep loop ────────────────────────────────────────────────────────────
results = []
t_sweep = time.time()

for amp in AMPLITUDES:
    print(f"\n{'─'*60}")
    print(f"  Amplitude: {amp} pA")
    print(f"{'─'*60}")

    # D1
    t0 = time.time()
    print(f"  [D1] Generating + running ...")
    try:
        n1, fired1, vmax1 = run_stage(amp, 1)
        ok1 = True
    except Exception as e:
        print(f"  [D1] FAILED: {e}")
        n1, fired1, vmax1, ok1 = 161, 0, 0.0, False
    print(f"  [D1] {fired1}/{n1} fired  Vmax={vmax1:.1f}mV  ({time.time()-t0:.0f}s)")

    # D8
    t0 = time.time()
    print(f"  [D8] Generating + running ...")
    try:
        n8, fired8, vmax8 = run_stage(amp, 8)
        ok8 = True
    except Exception as e:
        print(f"  [D8] FAILED: {e}")
        n8, fired8, vmax8, ok8 = 180, 0, 0.0, False
    print(f"  [D8] {fired8}/{n8} fired  Vmax={vmax8:.1f}mV  ({time.time()-t0:.0f}s)")

    diff = fired8 - fired1
    pct1  = fired1 / n1 * 100 if n1 > 0 else 0
    pct8  = fired8 / n8 * 100 if n8 > 0 else 0
    print(f"\n  >> amp={amp}pA | D1: {fired1}/{n1} ({pct1:.0f}%) | "
          f"D8: {fired8}/{n8} ({pct8:.0f}%) | diff: {diff:+d}")

    results.append({
        "amp": amp,
        "n_d1": n1, "fired_d1": fired1, "pct_d1": round(pct1, 1),
        "n_d8": n8, "fired_d8": fired8, "pct_d8": round(pct8, 1),
        "diff": diff,
        "ok_d1": ok1, "ok_d8": ok8,
    })

print(f"\n{'='*68}")
print(f"  Total sweep time: {(time.time()-t_sweep)/60:.1f} min")
print(f"{'='*68}")

# ── Find sensitivity window ────────────────────────────────────────────────────
print("\n  SENSITIVITY WINDOW ANALYSIS:")
print(f"  {'Amp':>6}  {'D1%':>6}  {'D8%':>6}  {'Diff':>6}  {'In range?'}")
in_range = []
for r in results:
    in_range_flag = (10 <= r["pct_d1"] <= 70) and (r["diff"] > 0)
    print(f"  {r['amp']:>6.1f}  {r['pct_d1']:>6.1f}  {r['pct_d8']:>6.1f}  "
          f"{r['diff']:>+6}  {'YES' if in_range_flag else ''}")
    if in_range_flag:
        in_range.append(r)

if in_range:
    best = max(in_range, key=lambda x: x["diff"])
    print(f"\n  OPTIMAL AMPLITUDE: {best['amp']} pA")
    print(f"    D1: {best['fired_d1']}/{best['n_d1']} ({best['pct_d1']:.0f}%)")
    print(f"    D8: {best['fired_d8']}/{best['n_d8']} ({best['pct_d8']:.0f}%)")
    print(f"    Developmental contrast: +{best['diff']} neurons")
else:
    # Pick the lowest amplitude that worked
    worked = [r for r in results if r["ok_d1"] and r["ok_d8"] and r["fired_d1"] > 0]
    if worked:
        best = min(worked, key=lambda x: x["amp"])
        print(f"\n  No clear sensitivity window found.")
        print(f"  Using lowest working amplitude: {best['amp']} pA as default for Task B")
    else:
        best = {"amp": 1.0}
        print(f"\n  All simulations silent or failed. Defaulting to 1.0 pA for Task B.")

# ── Save results ───────────────────────────────────────────────────────────────
sweep_out = os.path.join(OUT_SWEEP, "sweep_results.json")
with open(sweep_out, "w", encoding="utf-8") as f:
    json.dump({"results": results, "optimal_amp": best["amp"]}, f, indent=2)
print(f"\n  Saved {sweep_out}")

# ── Plot ─────────────────────────────────────────────────────────────────────
amps  = [r["amp"] for r in results]
d1s   = [r["pct_d1"] for r in results]
d8s   = [r["pct_d8"] for r in results]

x     = np.arange(len(amps))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - width/2, d1s, width, label="D1 (hatchling, 0h)", color="#4A90D9", alpha=0.85)
bars2 = ax.bar(x + width/2, d8s, width, label="D8 (adult, 120h)",  color="#E8694A", alpha=0.85)

ax.set_xlabel("Stimulation amplitude (pA)", fontsize=12)
ax.set_ylabel("% neurons fired (Vmax > -20 mV)", fontsize=12)
ax.set_title("c302 Parameters C0 — Amplitude Sweep: D1 vs D8 Network Activation",
             fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f"{a}pA" for a in amps])
ax.legend(fontsize=11)
ax.axhline(30, color="gray", ls="--", lw=1, alpha=0.5, label="30% target floor")
ax.axhline(70, color="gray", ls=":",  lw=1, alpha=0.5)
ax.set_ylim(0, 105)

# Annotate sensitivity window
for r, xi in zip(results, x):
    ax.text(xi - width/2, d1s[results.index(r)] + 1.5, f"{r['fired_d1']}", ha="center",
            fontsize=8, color="#4A90D9")
    ax.text(xi + width/2, d8s[results.index(r)] + 1.5, f"{r['fired_d8']}", ha="center",
            fontsize=8, color="#E8694A")

plt.tight_layout()
fig.savefig(os.path.join(OUT_SWEEP, "sweep_plot.png"), dpi=130, bbox_inches="tight")
plt.close()
print(f"  Saved sweep_plot.png")

print(f"\n=== TASK A COMPLETE ===")
print(f"  Optimal amplitude: {best['amp']} pA")
print(f"  Results in: {OUT_SWEEP}")
