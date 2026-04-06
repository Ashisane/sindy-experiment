# -*- coding: utf-8 -*-
"""
run_simulation_c0.py
====================
c302 Parameters C0 (graded/analogue GradedSynapse2) simulation for D1 and D8.

Parameters C0 uses:
  Cells:        Simplified conductance-based (Morris-Lecar-like), no fast K channel
  Chem synapses: GradedSynapse2 — ANALOGUE (voltage-dependent, continuously transmitting)
  Gap junctions: standard GapJunction

Unlike Parameters C (spike-triggered ExpOneSynapse), analogue synapses transmit
proportionally to membrane potential at ALL times — no threshold needed.
Expected: broad network activation propagating through graded depolarisation.
"""

import sys, os, time, importlib
import numpy as np

t_start = time.time()

# ── paths ──────────────────────────────────────────────────────────────────────
C302_DIR    = r"C:\Users\UTKARSH\Desktop\mdg\c302"
MDG_BUILD   = r"C:\Users\UTKARSH\Desktop\mdg\mdg_build"
OUTPUT_DIR  = os.path.join(MDG_BUILD, "output_c0")

sys.path.insert(0, C302_DIR)
sys.path.insert(0, MDG_BUILD)

os.makedirs(OUTPUT_DIR, exist_ok=True)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── STEP 1: Load Parameters C0 ────────────────────────────────────────────────
print("=" * 64)
print("STEP 1 — Load Parameters C0 and inspect synapse model")
print("=" * 64)

from c302 import parameters_C0
params = parameters_C0.ParameterisedModel()
print(f"  Loaded: {type(params).__module__}.{type(params).__name__}")
print(f"  Level:  {params.level}")

# Confirm graded synapses
synapse_params = [(bp.name, bp.value) for bp in params.bioparameters
                  if any(k in bp.name for k in ("syn", "gbase", "exc_syn", "inh_syn"))]
print(f"\n  Synapse-related bioparameters:")
for name, val in synapse_params:
    print(f"    {name!r:50s} = {val!r}")

# ── STEP 2: Set stimulation ────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("STEP 2 — Set stimulation bioparameters")
print("=" * 64)

# C0 inherits from C and has the same offset current params (confirmed at line 132-140)
STIM_PARAMS = {
    "unphysiological_offset_current":     "20 pA",
    "unphysiological_offset_current_del": "50 ms",
    "unphysiological_offset_current_dur": "400 ms",
}

for name, val in STIM_PARAMS.items():
    try:
        params.set_bioparameter(name, val, "MDG correction run", "0")
        actual = params.get_bioparameter(name).value
        print(f"  SET   {name!r:50s} → {actual!r}  ✓")
    except Exception as e:
        print(f"  ERROR setting {name!r}: {e}")
        try:
            params.add_bioparameter(name, val, "MDG correction run", "0")
            print(f"  ADDED {name!r} = {val!r}")
        except Exception as e2:
            print(f"  FAILED add too: {e2}")

# Verify excitatory synapse conductance (must be nonzero for signal propagation)
exc_g = params.get_bioparameter("neuron_to_neuron_exc_syn_conductance").value
print(f"\n  neuron_to_neuron_exc_syn_conductance = {exc_g}")
print(f"  (GradedSynapse2 — continuous voltage-dependent transmission)")

# ── STEP 3: Generate and run D1 ───────────────────────────────────────────────
print("\n" + "=" * 64)
print("STEP 3 — Generate + Run D1 (Parameters C0)")
print("=" * 64)

import c302

# Import witvliet_reader module once — stage is controlled via module global
import witvliet_reader as wr

# --- D1 ---
wr._DEFAULT_STAGE = 1
wr._instance = None   # force re-instantiation

cells_d1, _ = wr.read_data(include_nonconnected_cells=False)
print(f"  D1 cells: {len(cells_d1)}")

STIMULATE = [c for c in ["AVBL", "AVBR", "AVAL", "AVAR", "PVCL", "PVCR",
                          "AWCL", "AWCR", "ASEL", "ASER"]
             if c in cells_d1]
print(f"  Stimulated ({len(STIMULATE)}): {STIMULATE}")

NET_ID_D1 = "MDG_C0_D1"
LEMS_D1   = os.path.join(OUTPUT_DIR, f"LEMS_{NET_ID_D1}.xml")

print(f"\n  Generating NeuroML for {NET_ID_D1} ...")
c302.generate(
    NET_ID_D1,
    params,
    data_reader="witvliet_reader",
    cells=cells_d1,
    cells_to_stimulate=STIMULATE,
    cells_to_plot=cells_d1,
    duration=500,
    dt=0.05,
    target_directory=OUTPUT_DIR,
    verbose=False,
)
print(f"  Generated: {LEMS_D1}")
print(f"  Running jNeuroML simulation (duration=500ms, dt=0.05ms) ...")

from pyneuroml import pynml
success_d1 = pynml.run_lems_with_jneuroml(
    LEMS_D1,
    max_memory="4G",
    nogui=True,
    plot=False,
    verbose=True,
)
print(f"  Simulation exit status: {success_d1}")

# ── STEP 4: Parse D1 outputs ───────────────────────────────────────────────────
print("\n" + "=" * 64)
print("STEP 4 — Parse D1 outputs")
print("=" * 64)

DAT_D1  = os.path.join(OUTPUT_DIR, f"{NET_ID_D1}.dat")
ACT_D1  = os.path.join(OUTPUT_DIR, f"{NET_ID_D1}.activity.dat")
SPIKE_THRESH  = -20.0   # mV — any neuron exceeding this fired an action/Ca-spike
SUBTHRESH_LIM = -40.0   # mV — depolarised but not spiking

fired_d1      = 0
subthresh_d1  = 0
silent_d1     = 0
ca_active_d1  = 0
ca_threshold  = None
V_mv_d1       = None
cells_ordered = None

if os.path.exists(DAT_D1):
    V = np.loadtxt(DAT_D1)
    print(f"  Voltage .dat  shape: {V.shape}")
    V_mv_d1 = V[:, 1:] * 1000.0     # V → mV
    Vmax     = V_mv_d1.max(axis=0)

    fired_d1     = int((Vmax > SPIKE_THRESH).sum())
    subthresh_d1 = int(((Vmax > SUBTHRESH_LIM) & (Vmax <= SPIKE_THRESH)).sum())
    silent_d1    = int((Vmax <= SUBTHRESH_LIM).sum())

    print(f"\n  VOLTAGE RESULTS (D1):")
    print(f"    Fired (Vmax > {SPIKE_THRESH} mV):    {fired_d1} / {V_mv_d1.shape[1]}")
    print(f"    Sub-threshold depolarised:            {subthresh_d1}")
    print(f"    Silent (Vmax <= {SUBTHRESH_LIM} mV): {silent_d1}")
    print(f"    Voltage range: {V_mv_d1.min():.1f} to {V_mv_d1.max():.1f} mV")
else:
    print(f"  WARNING: {DAT_D1} not found — simulation may have failed")

if os.path.exists(ACT_D1):
    Ca  = np.loadtxt(ACT_D1)
    print(f"\n  Calcium .activity.dat shape: {Ca.shape}")
    Ca_data    = Ca[:, 1:]
    Ca_base    = Ca_data[:100, :].mean(axis=0)   # first ~5 ms = pre-stim baseline
    Ca_peak    = Ca_data.max(axis=0)
    Ca_delta   = Ca_peak - Ca_base

    # Threshold: >10% above baseline mean, minimum 1e-9 to avoid division by 0
    ca_threshold = max(Ca_base.mean() * 0.10, Ca_base.mean() + Ca_delta.std() * 0.5, 1e-9)
    ca_active_d1 = int((Ca_delta > ca_threshold).sum())
    ca_inactive  = int((Ca_delta <= ca_threshold).sum())

    print(f"\n  CALCIUM RESULTS (D1):")
    print(f"    Ca-active neurons (ΔCa > threshold): {ca_active_d1} / {Ca_data.shape[1]}")
    print(f"    Silent Ca:                            {ca_inactive}")
    print(f"    Ca range:      {Ca_data.min():.5f} to {Ca_data.max():.5f}")
    print(f"    Ca baseline:   {Ca_base.mean():.5f}")
    print(f"    Ca threshold:  {ca_threshold:.6f}")

    # Show top 10 most-active neurons by Ca delta
    print(f"\n  Top 10 Ca-active (sorted by ΔCa):")
    top10 = np.argsort(Ca_delta)[::-1][:10]
    for idx in top10:
        print(f"    col_{idx:03d}  base={Ca_base[idx]:.5f}  peak={Ca_peak[idx]:.5f}  "
              f"delta={Ca_delta[idx]:.5f}  {'ACTIVE' if Ca_delta[idx] > ca_threshold else 'silent'}")
else:
    print(f"  WARNING: {ACT_D1} not found")
    ca_active_d1 = 0

# ── STEP 5: Conditionally run D8 ──────────────────────────────────────────────
print("\n" + "=" * 64)
print("STEP 5 — Conditionally run D8")
print("=" * 64)

ca_active_d8 = 0
success_d8   = None

if ca_active_d1 > 20:
    print(f"  D1 shows {ca_active_d1} Ca-active neurons (> 20 threshold). Running D8...")

    wr._DEFAULT_STAGE = 8
    wr._instance = None   # force re-instantiation

    cells_d8, _ = wr.read_data(include_nonconnected_cells=False)
    STIMULATE_D8 = [c for c in STIMULATE if c in cells_d8]
    absent = [c for c in STIMULATE if c not in cells_d8]
    print(f"  D8 cells: {len(cells_d8)}")
    print(f"  Stimulating: {STIMULATE_D8}")
    if absent:
        print(f"  ABSENT in D8: {absent}")

    NET_ID_D8 = "MDG_C0_D8"
    LEMS_D8   = os.path.join(OUTPUT_DIR, f"LEMS_{NET_ID_D8}.xml")

    print(f"\n  Generating NeuroML for {NET_ID_D8} ...")
    c302.generate(
        NET_ID_D8,
        params,
        data_reader="witvliet_reader",
        cells=cells_d8,
        cells_to_stimulate=STIMULATE_D8,
        cells_to_plot=cells_d8,
        duration=500,
        dt=0.05,
        target_directory=OUTPUT_DIR,
        verbose=False,
    )

    print(f"  Running jNeuroML for D8 ...")
    success_d8 = pynml.run_lems_with_jneuroml(
        LEMS_D8,
        max_memory="4G",
        nogui=True,
        plot=False,
        verbose=True,
    )

    ACT_D8 = os.path.join(OUTPUT_DIR, f"{NET_ID_D8}.activity.dat")
    if os.path.exists(ACT_D8):
        Ca8       = np.loadtxt(ACT_D8)
        Ca8_data  = Ca8[:, 1:]
        Ca8_base  = Ca8_data[:100, :].mean(axis=0)
        Ca8_delta = Ca8_data.max(axis=0) - Ca8_base
        ca_active_d8 = int((Ca8_delta > ca_threshold).sum())

        print(f"\n  CALCIUM RESULTS (D8):")
        print(f"    Ca-active: {ca_active_d8} / {Ca8_data.shape[1]}")

        if ca_active_d8 > ca_active_d1:
            print(f"  ✓ MORE active at D8 ({ca_active_d8}) than D1 ({ca_active_d1})")
            print(f"  DEVELOPMENTAL SIGNAL CONFIRMED: denser adult connectome → broader activation")
        elif ca_active_d8 == ca_active_d1:
            print(f"  ~ Same activation D1 vs D8 — graded synapses saturate early")
        else:
            print(f"  D8 shows FEWER active neurons ({ca_active_d8} vs D1 {ca_active_d1})")
            print(f"  Possible cause: D8 has stronger inhibitory connections (GABAergic)")
    else:
        print(f"  WARNING: {ACT_D8} not found")
else:
    print(f"  D1 shows only {ca_active_d1} Ca-active neurons (threshold: 20).")
    print(f"  Skipping D8 — Parameters C0 network propagation insufficient.")
    print(f"  Diagnostic: check if graded synapses are activating at all.")
    print(f"  Possible fixes:")
    print(f"    1. Increase stimulation current (try 50-100 pA instead of 20 pA)")
    print(f"    2. Increase exc_syn_conductance or lower exc_syn_vth")
    print(f"    3. Check voltage range — are stimulated neurons even depolarising?")

# ── STEP 6: Final verdict ──────────────────────────────────────────────────────
t_total = time.time() - t_start
print("\n" + "=" * 64)
print("C302 PARAMETERS C0 VERDICT")
print("=" * 64)
print(f"  Parameter set: Parameters C0 (GradedSynapse2, Morris-Lecar-like)")
print(f"  Stimulated: {STIMULATE}")
print(f"  Duration: 500ms, dt=0.05ms")
print(f"  D1 voltage-fired (>-20mV):  {fired_d1}")
print(f"  D1 subthreshold (>-40mV):   {subthresh_d1}")
print(f"  D1 Ca-active neurons:        {ca_active_d1}")
if ca_active_d8:
    print(f"  D8 Ca-active neurons:        {ca_active_d8}")
    print(f"  D1 → D8 change:              {ca_active_d8 - ca_active_d1:+d} neurons")
print(f"\n  Total time: {t_total:.1f}s")

if ca_active_d1 > 40:
    print("\n  VERDICT: STRONG ACTIVATION")
    print("    Parameters C0 analogue synapses propagate broadly")
    print("    c302 C0 is viable for developmental feature extraction")
elif ca_active_d1 > 20:
    print("\n  VERDICT: MODERATE ACTIVATION")
    print("    Improvement over Parameters C (7 neurons)")
    print("    D1 vs D8 comparison may capture developmental signal")
elif ca_active_d1 > 7:
    print("\n  VERDICT: MARGINAL IMPROVEMENT over Parameters C")
    print("    Some propagation via graded synapses, but limited")
    print("    Need higher stimulation amplitude or conductance tuning")
else:
    print("\n  VERDICT: NO MEANINGFUL IMPROVEMENT")
    print("    Graded synapses alone insufficient for network propagation")
    print("    Root cause: analogue synapses require sustained voltage input")
    print("    but stimulation current (20pA) causes only small depolarisation")
    print("    The leak conductance (0.05 mS/cm2) clamps neurons near resting potential")
    print("    Fix: increase current to 100-500 pA, or reduce neuron_leak_cond_density")

# ── Report snippet for mentor meeting ─────────────────────────────────────────
print("\n" + "=" * 64)
print("MENTOR MEETING NUMBERS")
print("=" * 64)
print(f"  Parameters C  (spike-triggered, previous): 7/161 neurons fired")
print(f"  Parameters C0 (graded/analogue, NOW):      {ca_active_d1} Ca-active")
if ca_active_d8:
    print(f"  D1 vs D8 Ca-active:                        {ca_active_d1} vs {ca_active_d8}")
    if ca_active_d8 > ca_active_d1:
        delta_pct = (ca_active_d8 - ca_active_d1)/ca_active_d1*100
        print(f"  Developmental increase:                    +{delta_pct:.0f}%")
print(f"\n  This compares Parameters C vs C0 directly under identical stimulation.")
print(f"  A clear negative result (no improvement) is also reportable —")
print(f"  it tells us the bottleneck is conductance/stimulation tuning, not synapse model.")
