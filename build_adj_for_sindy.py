# -*- coding: utf-8 -*-
"""
TASK 4 — build_adj_for_sindy.py
Build a class-level adjacency matrix using D4 (latest L1) as structural prior.
A_class[i,j] = normalized total synaptic weight from class i to class j.
Used to build the SINDyG penalty matrix P.

Outputs
-------
output_sim/A_class.npy    (N_classes, N_classes) float64, normalized [0,1]
"""

import json, os, sys, re
from collections import defaultdict
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MDG_BUILD = os.path.dirname(os.path.abspath(__file__))
SYNAPSE_DIR = r"C:\Users\UTKARSH\Desktop\mdg\nature2021\data\synapses"
OUT_DIR   = os.path.join(MDG_BUILD, "output_sim")


def neuron_to_class(name: str) -> str:
    if len(name) > 2 and name[-1] in "LR":
        name = name[:-1]
    name = re.sub(r"\d+$", "", name)
    return name if name else "UNK"


# ── load class info ───────────────────────────────────────────────────────────
with open(os.path.join(OUT_DIR, "class_names.txt"), encoding="utf-8") as fh:
    sorted_classes = [l.strip() for l in fh if l.strip()]

with open(os.path.join(OUT_DIR, "class_members.json"), encoding="utf-8") as fh:
    class_members = json.load(fh)

N_classes  = len(sorted_classes)
class_idx  = {cls: i for i, cls in enumerate(sorted_classes)}

print(f"[T4] {N_classes} neuron classes loaded")

# ── load D4 synapses (latest L1 larva, most complete L1) ─────────────────────
D4_FILE = os.path.join(SYNAPSE_DIR, "Dataset4_synapses.json")
with open(D4_FILE, encoding="utf-8") as fh:
    d4_data = json.load(fh)
print(f"[T4] Loaded Dataset4 ({len(d4_data)} entries) as structural prior")

# Build per-pair synapse count
from c302.ConnectomeReader import PREFERRED_NEURON_NAMES
PREF = set(PREFERRED_NEURON_NAMES)

# Reuse the count-distinct-vast-ids logic
def _uid(entry: dict) -> int:
    v = entry.get("catmaid_id", entry.get("vast_id", id(entry)))
    if isinstance(v, list):
        return v[0]
    return v

pair_vast: dict[tuple, set] = defaultdict(set)
for entry in d4_data:
    pre  = entry["pre"]
    if pre not in PREF:
        continue
    uid = _uid(entry)
    for post in entry["post"]:
        if post in PREF:
            pair_vast[(pre, post)].add(uid)

d4_counts = {pair: len(vids) for pair, vids in pair_vast.items()}
print(f"[T4] D4: {len(d4_counts)} preferred-neuron (pre,post) pairs with ≥1 synapse")

# ── build class-level adjacency ───────────────────────────────────────────────
A_class = np.zeros((N_classes, N_classes), dtype=np.float64)

skipped = 0
for (pre, post), cnt in d4_counts.items():
    cls_pre  = neuron_to_class(pre)
    cls_post = neuron_to_class(post)
    if cls_pre in class_idx and cls_post in class_idx:
        i = class_idx[cls_pre]
        j = class_idx[cls_post]
        A_class[i, j] += cnt
    else:
        skipped += 1

print(f"[T4] Built A_class ({N_classes}×{N_classes}). Skipped {skipped} pairs (class not in list)")

# ── normalize to [0,1] ────────────────────────────────────────────────────────
A_max = A_class.max()
if A_max > 0:
    A_class /= A_max

# ── statistics ───────────────────────────────────────────────────────────────
n_nonzero = (A_class > 0).sum()
density   = n_nonzero / (N_classes * N_classes)
print(f"[T4] A_class: {N_classes}×{N_classes}, nonzero = {n_nonzero}, density = {density:.4f} ({density*100:.2f}%)")
print(f"[T4] Value range: {A_class.min():.4f} to {A_class.max():.4f}")

# ── top 10 class-class connections ───────────────────────────────────────────
print(f"\n[T4] Top 10 class→class connections (normalized weight):")
flat = [(A_class[i,j], sorted_classes[i], sorted_classes[j])
        for i in range(N_classes) for j in range(N_classes) if A_class[i,j] > 0]
flat.sort(reverse=True)
for w, src, dst in flat[:10]:
    print(f"  {src:<10} -> {dst:<10}  {w:.4f}")

# ── save ─────────────────────────────────────────────────────────────────────
np.save(os.path.join(OUT_DIR, "A_class.npy"), A_class)
print(f"\n[T4] Saved A_class.npy {A_class.shape}")

print("\n=== TASK 4 COMPLETE ===")
print(f"  Classes: {N_classes}")
print(f"  A_class density: {density:.4f} ({density*100:.2f}%)")
print(f"  Structural prior: D4 dataset (latest L1 larva)")
print(f"  Normalization: all values in [0, 1]  (max raw = {A_max:.0f})")
