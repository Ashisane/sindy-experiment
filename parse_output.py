import os
import sys
import re
import numpy as np

# ── UTF-8 stdout on Windows ────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Paths ─────────────────────────────────────────────────────────────────
MDG_BUILD  = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(MDG_BUILD, "output_sim")
LEMS_FILE  = os.path.join(OUT_DIR, "LEMS_MDG_D1.xml")
DAT_FILE   = os.path.join(OUT_DIR, "MDG_D1.dat")
ORDER_FILE = os.path.join(OUT_DIR, "neuron_order.txt")
NPY_FILE   = os.path.join(OUT_DIR, "voltage_matrix.npy")

# Spike / activity thresholds (mV)
SPIKE_THRESH      = -20.0   # crossed => fired
SUBTHRESH_THRESH  = -40.0   # crossed but not spike => subthreshold

# ═════════════════════════════════════════════════════════════════════════════
# TASK 1 — Extract neuron column order from LEMS_MDG_D1.xml
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TASK 1: Extracting neuron column order from LEMS file")
print("=" * 60)

with open(LEMS_FILE, encoding="utf-8") as fh:
    lems_text = fh.read()

# The LEMS file has two OutputFile blocks:
#   1. neurons_v  -> MDG_D1.dat          (voltage,  v)
#   2. neurons_activity -> MDG_D1.activity.dat  (caConc)
#
# We want only the OutputColumns that belong to the voltage OutputFile.
# Strategy: find the <OutputFile ... fileName="MDG_D1.dat"> block and extract
# all <OutputColumn> id values within it, stopping at </OutputFile>.

voltage_block_match = re.search(
    r'<OutputFile[^>]*fileName="MDG_D1\.dat"[^>]*>(.*?)</OutputFile>',
    lems_text,
    re.DOTALL,
)
if not voltage_block_match:
    raise RuntimeError("Could not find <OutputFile ... MDG_D1.dat ...> block in LEMS file.")

voltage_block = voltage_block_match.group(1)

# Each OutputColumn id is like "ADAL_v" — strip "_v" suffix
col_ids  = re.findall(r'<OutputColumn\s+id="([^"]+)"', voltage_block)
neurons  = [cid.removesuffix("_v") for cid in col_ids]

print(f"  Total neurons in dat file: {len(neurons)}")
print()
print("  First 10 neurons (col 1 to 10):")
for i, name in enumerate(neurons[:10], start=1):
    print(f"    col {i:>3}: {name}")
print()
print("  Last 10 neurons (col {lo} to {hi}):".format(
    lo=len(neurons) - 9, hi=len(neurons)))
for i, name in enumerate(neurons[-10:], start=len(neurons) - 9):
    print(f"    col {i:>3}: {name}")

# ═════════════════════════════════════════════════════════════════════════════
# TASK 2 — Load .dat, convert to mV, classify activity
# ═════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("TASK 2: Loading MDG_D1.dat and classifying neuron activity")
print("=" * 60)

raw = np.loadtxt(DAT_FILE)           # shape (10001, 162): col0=time(s), cols1-161=V
print(f"  Raw .dat shape: {raw.shape}")

time_s    = raw[:, 0]                # seconds
volt_V    = raw[:, 1:]               # volts, shape (10001, 161)
volt_mV   = volt_V * 1000.0         # convert to mV

assert volt_mV.shape[1] == len(neurons), (
    f"Mismatch: {volt_mV.shape[1]} voltage columns vs {len(neurons)} neuron names"
)

# Per-neuron max voltage
max_v = volt_mV.max(axis=0)         # shape (161,)

fired       = []   # max > -20 mV
subthresh   = []   # -40 <= max <= -20 mV
silent      = []   # max < -40 mV

for idx, name in enumerate(neurons):
    mv = max_v[idx]
    if mv > SPIKE_THRESH:
        fired.append((name, mv))
    elif mv > SUBTHRESH_THRESH:
        subthresh.append((name, mv))
    else:
        silent.append((name, mv))

print()
print(f"  Spike threshold   : > {SPIKE_THRESH} mV")
print(f"  Subthresh bracket : {SUBTHRESH_THRESH} to {SPIKE_THRESH} mV")
print()
print(f"  Neurons FIRED     (max > {SPIKE_THRESH} mV)          : {len(fired):>3}")
print(f"  Neurons SUBTHRESH ({SUBTHRESH_THRESH} < max <= {SPIKE_THRESH} mV) : {len(subthresh):>3}")
print(f"  Neurons SILENT    (max < {SUBTHRESH_THRESH} mV)         : {len(silent):>3}")
print(f"  TOTAL             : {len(fired)+len(subthresh)+len(silent):>3}")

print()
print("  --- Neurons that FIRED ---")
if fired:
    fired_sorted = sorted(fired, key=lambda x: -x[1])
    for name, mv in fired_sorted:
        print(f"    {name:<10} max={mv:+8.2f} mV")
else:
    print("    (none — check stimulation or parameters)")

print()
print("  --- Subthreshold depolarised neurons (sample, up to 20) ---")
subthr_sorted = sorted(subthresh, key=lambda x: -x[1])
for name, mv in subthr_sorted[:20]:
    print(f"    {name:<10} max={mv:+8.2f} mV")
if len(subthr_sorted) > 20:
    print(f"    ... and {len(subthr_sorted)-20} more")

# ═════════════════════════════════════════════════════════════════════════════
# TASK 3 — Save outputs
# ═════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("TASK 3: Saving neuron_order.txt and voltage_matrix.npy")
print("=" * 60)

# neuron_order.txt — one name per line, in column order
with open(ORDER_FILE, "w", encoding="utf-8") as fh:
    fh.write("\n".join(neurons) + "\n")
print(f"  Written: {ORDER_FILE}")
print(f"    Lines: {len(neurons)}")

# voltage_matrix.npy — shape (N_neurons, T_timesteps), in mV
# Transpose so rows=neurons, cols=time (SINDyG convention)
voltage_matrix = volt_mV.T          # (161, 10001)
np.save(NPY_FILE, voltage_matrix)

# Reload to verify
check = np.load(NPY_FILE)
print(f"  Written: {NPY_FILE}")
print(f"    Saved shape  : {voltage_matrix.shape}  (N_neurons x T_timesteps)")
print(f"    Reloaded shape: {check.shape}")
print(f"    dtype        : {check.dtype}")
print(f"    Value range  : {check.min():+.4f} mV  to  {check.max():+.4f} mV")

print()
print("=" * 60)
print("parse_output.py completed successfully.")
print("=" * 60)
