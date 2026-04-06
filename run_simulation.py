# -*- coding: utf-8 -*-
"""
run_simulation.py
-----------------
Generates and runs a c302 Parameters-C simulation driven by the
Witvliet 2021 (Nature) Dataset 1 connectome (stage 1).

Steps
-----
1. Patch sys.path so c302 can find witvliet_reader as "c302.witvliet_reader".
   load_data_reader() in c302 imports non-cect readers as "c302.<name>",
   so we copy witvliet_reader.py into the installed c302 package directory.

2. Instantiate parameters_C.ParameterisedModel and set the three
   unphysiological offset-current bioparameters.

3. Call c302.generate() with:
     net_id              = "MDG_D1"
     data_reader         = "witvliet_reader"
     cells               = all 161 neurons from Dataset1
     cells_to_stimulate  = ["AVBL","AVBR","AVAL","AVAR","PVCL","PVCR"]
     cells_to_plot       = same as cells
     duration            = 500 ms
     dt                  = 0.05 ms
     target_directory    = output_sim/

4. Run the generated LEMS file with jNeuroML.

5. Inspect MDG_D1.dat.
"""

import os
import sys
import shutil
import traceback
import importlib

# Force UTF-8 stdout on Windows to avoid cp1252 encoding errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── 0.  Paths ────────────────────────────────────────────────────────────────

MDG_BUILD   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(MDG_BUILD, "output_sim")
NET_ID      = "MDG_D1"
LEMS_FILE   = os.path.join(OUTPUT_DIR, f"LEMS_{NET_ID}.xml")
DAT_FILE    = os.path.join(OUTPUT_DIR, f"{NET_ID}.dat")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1.  Make witvliet_reader importable as "c302.witvliet_reader" ────────────
#
# c302.__init__.load_data_reader() does:
#     importlib.import_module("c302.%s" % data_reader)
# so the module must live inside the c302 package.

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
    print(f"[setup] witvliet_reader.py already in c302 package at {DST_READER}")

# Also add mdg_build/ to sys.path so standalone imports of witvliet_reader work
if MDG_BUILD not in sys.path:
    sys.path.insert(0, MDG_BUILD)

# ── 2.  Set bioparameters on Parameters C ────────────────────────────────────

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
    params.set_bioparameter(name, value, "MDG_run_simulation", "0")
    actual = params.get_bioparameter(name).value
    status = "✓ OK" if actual == value else f"✗ MISMATCH (got '{actual}')"
    print(f"  set_bioparameter({name!r}, {value!r}) → {status}")

# ── 3.  Get cell list from witvliet_reader ───────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2: Loading Dataset1 cells from witvliet_reader")
print("=" * 60)

import witvliet_reader as wr

cells, _ = wr.WitvlietDataReader(stage=1).read_data()
print(f"  Cells to include: {len(cells)}")

CELLS_TO_STIMULATE = ["AVBL", "AVBR", "AVAL", "AVAR", "PVCL", "PVCR"]
# Filter to only those present in stage-1 cells
CELLS_TO_STIMULATE = [c for c in CELLS_TO_STIMULATE if c in cells]
print(f"  Cells to stimulate: {CELLS_TO_STIMULATE}")
print(f"  Cells to plot: {len(cells)} (all)")

# ── 4.  Run c302.generate() ──────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 3: Running c302.generate()")
print("=" * 60)

import c302

try:
    nml_doc = c302.generate(
        net_id              = NET_ID,
        params              = params,
        data_reader         = "witvliet_reader",
        cells               = cells,
        cells_to_plot       = cells,
        cells_to_stimulate  = CELLS_TO_STIMULATE,
        duration            = 500,
        dt                  = 0.05,
        target_directory    = OUTPUT_DIR,
        verbose             = True,
    )
    print("\n[generate] c302.generate() completed successfully.")

    # List files created in output_sim/
    created = sorted(os.listdir(OUTPUT_DIR))
    print(f"[generate] Files in {OUTPUT_DIR}:")
    for f in created:
        fp = os.path.join(OUTPUT_DIR, f)
        size_kb = os.path.getsize(fp) / 1024
        print(f"           {f:40s}  {size_kb:8.1f} KB")

except Exception:
    print("\n[generate] FAILED — full traceback:")
    traceback.print_exc()
    sys.exit(1)

# ── 5.  Verify LEMS file exists ──────────────────────────────────────────────

if not os.path.isfile(LEMS_FILE):
    print(f"\n[ERROR] Expected LEMS file not found: {LEMS_FILE}")
    print("[ERROR] Cannot proceed with simulation.")
    sys.exit(1)

print(f"\n[lems] LEMS file confirmed: {LEMS_FILE}")

# ── 6.  Run simulation with jNeuroML ─────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4: Running LEMS simulation via jNeuroML")
print("        (This may take 5–15 minutes for 161 neurons × 500 ms)")
print("=" * 60)

from pyneuroml import pynml

# jNeuroML must be run from the directory containing the LEMS file
orig_dir = os.getcwd()
os.chdir(OUTPUT_DIR)

try:
    success = pynml.run_lems_with_jneuroml(
        f"LEMS_{NET_ID}.xml",
        nogui       = True,
        load_saved_data = False,
        max_memory  = "4G",
        verbose     = True,
    )
    print(f"\n[jNeuroML] run returned: {success}")
except Exception:
    print("\n[jNeuroML] FAILED — full traceback:")
    traceback.print_exc()
    os.chdir(orig_dir)
    sys.exit(1)
finally:
    os.chdir(orig_dir)

# ── 7.  Inspect output .dat file ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 5: Inspecting output .dat file")
print("=" * 60)

if not os.path.isfile(DAT_FILE):
    print(f"[ERROR] {DAT_FILE} not found after simulation.")
    print("[ERROR] Listing output_sim/ contents:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"        {f}")
    sys.exit(1)

import numpy as np

try:
    data = np.loadtxt(DAT_FILE)
    shape = data.shape
    time_col = data[:, 0]
    t_min_ms = time_col[0]  * 1000   # LEMS time is in seconds
    t_max_ms = time_col[-1] * 1000
    n_cols   = shape[1]
    n_neurons_recorded = n_cols - 1   # first column is time

    print(f"  File:                {DAT_FILE}")
    print(f"  Shape:               {shape}  (rows × columns)")
    print(f"  Time column range:   {t_min_ms:.3f} ms  →  {t_max_ms:.3f} ms")
    print(f"  Number of columns:   {n_cols}  (1 time + {n_neurons_recorded} neuron voltages)")
    print(f"  dt (actual):         {(time_col[1]-time_col[0])*1000:.4f} ms")

except Exception:
    print(f"[ERROR] Failed to load {DAT_FILE}:")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("run_simulation.py completed successfully.")
print("=" * 60)
