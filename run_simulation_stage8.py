# -*- coding: utf-8 -*-
"""
run_simulation_stage8.py
------------------------
Generates and runs a c302 Parameters-C simulation driven by the
Witvliet 2021 (Nature) Dataset 8 connectome (adult stage).

Identical stimulation parameters to run_simulation.py (stage 1):
  - 20 pA offset current, delay 50 ms, duration 400 ms
  - cells_to_stimulate = ["AVBL", "AVBR", "AVAL", "AVAR", "PVCL", "PVCR"]
  - duration = 500 ms, dt = 0.05 ms

Key differences:
  - WitvlietDataReader(stage=8)  →  180 neurons
  - net_id = "MDG_D8"
  - Saves: neuron_order_D8.txt, voltage_matrix_D8.npy

IMPORTANT: c302.generate() calls the module-level read_data() on the
c302.witvliet_reader module, which uses _DEFAULT_STAGE. We patch
_DEFAULT_STAGE = 8 and reset _instance = None on the imported module
before calling generate(), so it picks up stage-8 connectivity.
"""

import os
import sys
import re
import shutil
import traceback
import importlib

import numpy as np

# ── UTF-8 stdout (Windows cp1252 workaround) ──────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Paths ─────────────────────────────────────────────────────────────────────
STAGE       = 8
MDG_BUILD   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(MDG_BUILD, "output_sim")
NET_ID      = "MDG_D8"
LEMS_FILE   = os.path.join(OUTPUT_DIR, f"LEMS_{NET_ID}.xml")
DAT_FILE    = os.path.join(OUTPUT_DIR, f"{NET_ID}.dat")
ORDER_FILE  = os.path.join(OUTPUT_DIR, "neuron_order_D8.txt")
NPY_FILE    = os.path.join(OUTPUT_DIR, "voltage_matrix_D8.npy")

SPIKE_THRESH     = -20.0   # mV — c302 Parameters C spike threshold
SUBTHRESH_THRESH = -40.0   # mV

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1.  Ensure witvliet_reader is in c302 package dir ────────────────────────
#   load_data_reader() does: importlib.import_module("c302.<name>")

import c302 as _c302_pkg

C302_PKG_DIR = os.path.dirname(os.path.abspath(_c302_pkg.__file__))
SRC_READER   = os.path.join(MDG_BUILD, "witvliet_reader.py")
DST_READER   = os.path.join(C302_PKG_DIR, "witvliet_reader.py")

if not os.path.exists(DST_READER) or (
    os.path.getmtime(SRC_READER) > os.path.getmtime(DST_READER)
):
    shutil.copy2(SRC_READER, DST_READER)
    print(f"[setup] Copied witvliet_reader.py -> {DST_READER}")
else:
    print(f"[setup] witvliet_reader.py already current in c302 package.")

if MDG_BUILD not in sys.path:
    sys.path.insert(0, MDG_BUILD)

# ── 2.  Patch the module-level stage in c302.witvliet_reader ─────────────────
#   generate() calls: load_data_reader("witvliet_reader").read_data(...)
#   which calls the module-level read_data() → get_instance(_DEFAULT_STAGE)
#   We must set _DEFAULT_STAGE = 8 and clear _instance BEFORE generate().

print(f"\n[stage] Patching c302.witvliet_reader module-level stage -> {STAGE}")

# Force fresh import (in case a stage-1 cached copy is in sys.modules)
if "c302.witvliet_reader" in sys.modules:
    del sys.modules["c302.witvliet_reader"]

_wr_mod = importlib.import_module("c302.witvliet_reader")
_wr_mod._DEFAULT_STAGE = STAGE
_wr_mod._instance      = None        # clear any cached stage-1 instance

print(f"[stage] _DEFAULT_STAGE = {_wr_mod._DEFAULT_STAGE}  |  _instance reset")

# ── 3.  Set bioparameters on Parameters C ────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 1: Setting bioparameters on parameters_C")
print("=" * 60)

from c302 import parameters_C

params = parameters_C.ParameterisedModel()

BIOPARAMS = {
    "unphysiological_offset_current":     "20 pA",
    "unphysiological_offset_current_del": "50 ms",
    "unphysiological_offset_current_dur": "400 ms",
}

for name, value in BIOPARAMS.items():
    params.set_bioparameter(name, value, "MDG_run_simulation_stage8", "0")
    actual = params.get_bioparameter(name).value
    ok = "OK" if actual == value else f"MISMATCH (got '{actual}')"
    print(f"  {name} = {value!r}  [{ok}]")

# ── 4.  Get stage-8 cell list and verify stimulated cells ────────────────────

print("\n" + "=" * 60)
print(f"STEP 2: Loading Dataset{STAGE} cells from witvliet_reader")
print("=" * 60)

import witvliet_reader as wr

reader8   = wr.WitvlietDataReader(stage=STAGE)
cells, _  = reader8.read_data()
cells_set = set(cells)
print(f"  Total cells in Dataset{STAGE}: {len(cells)}")

REQUESTED_STIM = ["AVBL", "AVBR", "AVAL", "AVAR", "PVCL", "PVCR"]
present  = [c for c in REQUESTED_STIM if c in cells_set]
absent   = [c for c in REQUESTED_STIM if c not in cells_set]

print(f"\n  Stimulate cells requested : {REQUESTED_STIM}")
print(f"  Present in stage-{STAGE} cells : {present}  ({len(present)}/6)")
if absent:
    print(f"  ABSENT                     : {absent}  *** WARNING ***")
else:
    print(f"  Absent                     : (none — all 6 confirmed present)")

CELLS_TO_STIMULATE = present   # only pass confirmed-present cells

print(f"\n  cells_to_stimulate -> {CELLS_TO_STIMULATE}")
print(f"  cells_to_plot      -> all {len(cells)} cells")

# ── 5.  Run c302.generate() ──────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 3: Running c302.generate()")
print("=" * 60)

import c302

try:
    nml_doc = c302.generate(
        net_id             = NET_ID,
        params             = params,
        data_reader        = "witvliet_reader",
        cells              = cells,
        cells_to_plot      = cells,
        cells_to_stimulate = CELLS_TO_STIMULATE,
        duration           = 500,
        dt                 = 0.05,
        target_directory   = OUTPUT_DIR,
        verbose            = True,
    )
    print("\n[generate] c302.generate() completed successfully.")

    created = sorted(os.listdir(OUTPUT_DIR))
    print(f"[generate] Files in {OUTPUT_DIR}:")
    for f in created:
        fp      = os.path.join(OUTPUT_DIR, f)
        size_kb = os.path.getsize(fp) / 1024
        print(f"           {f:45s} {size_kb:9.1f} KB")

except Exception:
    print("\n[generate] FAILED — full traceback:")
    traceback.print_exc()
    sys.exit(1)

# ── 6.  Verify LEMS file ──────────────────────────────────────────────────────

if not os.path.isfile(LEMS_FILE):
    print(f"\n[ERROR] LEMS file not found: {LEMS_FILE}")
    sys.exit(1)
print(f"\n[lems] LEMS file confirmed: {LEMS_FILE}")

# ── 7.  Run simulation with jNeuroML ─────────────────────────────────────────

print("\n" + "=" * 60)
print(f"STEP 4: Running LEMS simulation via jNeuroML")
print(f"        ({len(cells)} neurons x 500 ms — may take 10-20 minutes)")
print("=" * 60)

from pyneuroml import pynml

orig_dir = os.getcwd()
os.chdir(OUTPUT_DIR)
try:
    success = pynml.run_lems_with_jneuroml(
        f"LEMS_{NET_ID}.xml",
        nogui           = True,
        load_saved_data = False,
        max_memory      = "4G",
        verbose         = True,
    )
    print(f"\n[jNeuroML] run returned: {success}")
except Exception:
    print("\n[jNeuroML] FAILED — full traceback:")
    traceback.print_exc()
    os.chdir(orig_dir)
    sys.exit(1)
finally:
    os.chdir(orig_dir)

# ── 8.  Inspect .dat file (basic) ────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 5: Inspecting MDG_D8.dat")
print("=" * 60)

if not os.path.isfile(DAT_FILE):
    print(f"[ERROR] {DAT_FILE} not found.")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  {f}")
    sys.exit(1)

try:
    raw = np.loadtxt(DAT_FILE)
except Exception:
    print(f"[ERROR] Failed to load {DAT_FILE}:")
    traceback.print_exc()
    sys.exit(1)

time_s  = raw[:, 0]
volt_V  = raw[:, 1:]
volt_mV = volt_V * 1000.0

print(f"  Shape:     {raw.shape}  (timesteps x columns)")
print(f"  Time:      {time_s[0]*1000:.3f} ms -> {time_s[-1]*1000:.3f} ms")
print(f"  dt:        {(time_s[1]-time_s[0])*1000:.4f} ms")
print(f"  Columns:   {raw.shape[1]}  (1 time + {volt_mV.shape[1]} neurons)")

# ── 9.  Extract neuron order from LEMS_MDG_D8.xml ────────────────────────────

print("\n" + "=" * 60)
print("STEP 6: Extracting neuron column order from LEMS file")
print("=" * 60)

with open(LEMS_FILE, encoding="utf-8") as fh:
    lems_text = fh.read()

voltage_block_match = re.search(
    r'<OutputFile[^>]*fileName="MDG_D8\.dat"[^>]*>(.*?)</OutputFile>',
    lems_text,
    re.DOTALL,
)
if not voltage_block_match:
    raise RuntimeError("Could not find <OutputFile ... MDG_D8.dat ...> in LEMS file.")

voltage_block = voltage_block_match.group(1)
col_ids  = re.findall(r'<OutputColumn\s+id="([^"]+)"', voltage_block)
neurons  = [cid.removesuffix("_v") for cid in col_ids]

print(f"  Neurons in LEMS order: {len(neurons)}")
print()
print("  First 10:")
for i, n in enumerate(neurons[:10], 1):
    print(f"    col {i:>3}: {n}")
print()
print(f"  Last 10 (col {len(neurons)-9} to {len(neurons)}):")
for i, n in enumerate(neurons[-10:], len(neurons) - 9):
    print(f"    col {i:>3}: {n}")

assert len(neurons) == volt_mV.shape[1], (
    f"Column count mismatch: LEMS has {len(neurons)}, .dat has {volt_mV.shape[1]}"
)

# ── 10.  Classify neuron activity ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 7: Classifying neuron activity")
print("=" * 60)

max_v = volt_mV.max(axis=0)

fired     = [(n, mv) for n, mv in zip(neurons, max_v) if mv >  SPIKE_THRESH]
subthresh = [(n, mv) for n, mv in zip(neurons, max_v) if SUBTHRESH_THRESH < mv <= SPIKE_THRESH]
silent    = [(n, mv) for n, mv in zip(neurons, max_v) if mv <= SUBTHRESH_THRESH]

print(f"\n  Spike threshold   : > {SPIKE_THRESH} mV")
print(f"  Subthresh bracket : {SUBTHRESH_THRESH} to {SPIKE_THRESH} mV")
print()
print(f"  Neurons FIRED     (max > {SPIKE_THRESH} mV)             : {len(fired):>3}")
print(f"  Neurons SUBTHRESH ({SUBTHRESH_THRESH} < max <= {SPIKE_THRESH} mV) : {len(subthresh):>3}")
print(f"  Neurons SILENT    (max <= {SUBTHRESH_THRESH} mV)          : {len(silent):>3}")
print(f"  TOTAL                                              : {len(neurons):>3}")

print()
print("  --- Neurons that FIRED (sorted by peak voltage) ---")
if fired:
    for name, mv in sorted(fired, key=lambda x: -x[1]):
        stim_flag = " [stimulated]" if name in CELLS_TO_STIMULATE else " [network-driven]"
        print(f"    {name:<10} max={mv:+8.2f} mV{stim_flag}")
else:
    print("    (none)")

print()
print("  --- Subthreshold neurons (up to 20 shown) ---")
for name, mv in sorted(subthresh, key=lambda x: -x[1])[:20]:
    print(f"    {name:<10} max={mv:+8.2f} mV")
if len(subthresh) > 20:
    print(f"    ... and {len(subthresh)-20} more")

# ── 11.  Save outputs ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 8: Saving neuron_order_D8.txt and voltage_matrix_D8.npy")
print("=" * 60)

with open(ORDER_FILE, "w", encoding="utf-8") as fh:
    fh.write("\n".join(neurons) + "\n")
print(f"  Written: {ORDER_FILE}  ({len(neurons)} lines)")

voltage_matrix = volt_mV.T     # (N_neurons, T_timesteps)
np.save(NPY_FILE, voltage_matrix)
check = np.load(NPY_FILE)
print(f"  Written: {NPY_FILE}")
print(f"    Shape        : {voltage_matrix.shape}  (neurons x timesteps)")
print(f"    Reloaded     : {check.shape}")
print(f"    dtype        : {check.dtype}")
print(f"    Value range  : {check.min():+.4f} mV  to  {check.max():+.4f} mV")

# ── 12.  Side-by-side comparison ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("STAGE 1 vs STAGE 8 — COMPARISON")
print("=" * 60)

# Load stage-1 data for comparison
D1_DAT    = os.path.join(OUTPUT_DIR, "MDG_D1.dat")
D1_LEMS   = os.path.join(OUTPUT_DIR, "LEMS_MDG_D1.xml")
D1_NPY    = os.path.join(OUTPUT_DIR, "voltage_matrix.npy")

if os.path.isfile(D1_DAT) and os.path.isfile(D1_LEMS):
    with open(D1_LEMS, encoding="utf-8") as fh:
        d1_lems = fh.read()
    d1_block = re.search(
        r'<OutputFile[^>]*fileName="MDG_D1\.dat"[^>]*>(.*?)</OutputFile>',
        d1_lems, re.DOTALL)
    d1_neurons_list = []
    if d1_block:
        d1_ids = re.findall(r'<OutputColumn\s+id="([^"]+)"', d1_block.group(1))
        d1_neurons_list = [cid.removesuffix("_v") for cid in d1_ids]

    d1_raw  = np.loadtxt(D1_DAT)
    d1_mV   = d1_raw[:, 1:] * 1000.0
    d1_max  = d1_mV.max(axis=0)
    d1_fired_count = int((d1_max > SPIKE_THRESH).sum())
    d1_sub_count   = int(((d1_max > SUBTHRESH_THRESH) & (d1_max <= SPIKE_THRESH)).sum())
    d1_sil_count   = int((d1_max <= SUBTHRESH_THRESH).sum())

    d8_fired_count = len(fired)
    d8_sub_count   = len(subthresh)
    d8_sil_count   = len(silent)

    print(f"\n  {'Metric':<40} {'Stage 1':>10} {'Stage 8':>10}")
    print(f"  {'-'*60}")
    print(f"  {'Neurons simulated':<40} {len(d1_neurons_list):>10} {len(neurons):>10}")
    print(f"  {'Neurons FIRED (max > -20 mV)':<40} {d1_fired_count:>10} {d8_fired_count:>10}")
    print(f"  {'Neurons SUBTHRESH (-40 to -20 mV)':<40} {d1_sub_count:>10} {d8_sub_count:>10}")
    print(f"  {'Neurons SILENT (max < -40 mV)':<40} {d1_sil_count:>10} {d8_sil_count:>10}")
    print(f"  {'Voltage min (mV)':<40} {d1_mV.min():>10.3f} {volt_mV.min():>10.3f}")
    print(f"  {'Voltage max (mV)':<40} {d1_mV.max():>10.3f} {volt_mV.max():>10.3f}")
    print(f"  {'dat shape':<40} {str(d1_raw.shape):>10} {str(raw.shape):>10}")

    # Neurons unique to stage 8
    d1_set = set(d1_neurons_list)
    d8_set = set(neurons)
    new_in_d8  = sorted(d8_set - d1_set)
    lost_in_d8 = sorted(d1_set - d8_set)
    print(f"\n  Neurons added in stage 8  ({len(new_in_d8):>3}): {new_in_d8}")
    print(f"  Neurons absent in stage 8 ({len(lost_in_d8):>3}): {lost_in_d8}")
else:
    print("  Stage-1 dat/LEMS not found — skipping comparison.")

print("\n" + "=" * 60)
print("run_simulation_stage8.py completed successfully.")
print("=" * 60)
