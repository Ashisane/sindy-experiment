# -*- coding: utf-8 -*-
"""
TASK 1 — build_synapse_matrix.py
Build per-stage synapse count matrices for all 8 Witvliet developmental datasets.

Outputs
-------
output_sim/synapse_matrix_X_raw.npy   (N_neurons, 8)  – outgoing synapse count per neuron per stage
output_sim/synapse_matrix_X_pairs.npy (N_pairs,   8)  – per-(pre,post)-pair synapse count
output_sim/neuron_list_all.txt         N_neurons neuron names, one per line
output_sim/pair_list.txt              N_pairs "pre->post" entries, one per line
"""

import json, os, sys
from collections import defaultdict
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── paths ────────────────────────────────────────────────────────────────────
MDG_BUILD = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = r"C:\Users\UTKARSH\Desktop\mdg\nature2021\data\synapses"
OUT_DIR   = os.path.join(MDG_BUILD, "output_sim")
os.makedirs(OUT_DIR, exist_ok=True)

# biological timepoints (hours post-hatch), NOT evenly spaced
T_BIO = [0, 5, 16, 27, 47, 70, 81, 120]

# ── load PREFERRED_NEURON_NAMES from c302 ────────────────────────────────────
sys.path.insert(0, MDG_BUILD)
from c302.ConnectomeReader import PREFERRED_NEURON_NAMES
PREF = set(PREFERRED_NEURON_NAMES)
print(f"[T1] PREFERRED_NEURON_NAMES: {len(PREF)} entries")

# ── load all 8 datasets ──────────────────────────────────────────────────────
# stage_counts[k][(pre,post)] = synapse count at stage k
# Helper: catmaid_id can be a list (e.g. D7) or int — always make hashable
def _uid(entry: dict) -> int:
    v = entry.get("catmaid_id", entry.get("vast_id", id(entry)))
    if isinstance(v, list):
        return v[0]   # single-element list → extract the int
    return v

stage_counts = []

for stage in range(1, 9):
    fname = os.path.join(DATA_DIR, f"Dataset{stage}_synapses.json")
    with open(fname, encoding="utf-8") as fh:
        data = json.load(fh)

    pair_vast: dict[tuple, set] = defaultdict(set)
    for entry in data:
        pre = entry["pre"]
        uid = _uid(entry)
        for post_neuron in entry["post"]:
            pair_vast[(pre, post_neuron)].add(uid)

    counts = {pair: len(vids) for pair, vids in pair_vast.items()}
    stage_counts.append(counts)
    print(f"  Stage {stage}: {len(data)} raw entries → {len(counts)} unique (pre,post) pairs")

# ── master neuron list (union, filtered to PREFERRED) ────────────────────────
all_neurons_raw = set()
for sc in stage_counts:
    for (pre, post) in sc:
        all_neurons_raw.add(pre)
        all_neurons_raw.add(post)

neurons_all = sorted(all_neurons_raw & PREF)
neuron_idx  = {n: i for i, n in enumerate(neurons_all)}
N = len(neurons_all)
print(f"\n[T1] Total unique preferred neurons across all stages: {N}")

# ── X_raw: (N, 8) outgoing synapse count per neuron per stage ────────────────
X_raw = np.zeros((N, 8), dtype=np.int32)
for k, sc in enumerate(stage_counts):
    for (pre, post), cnt in sc.items():
        if pre in neuron_idx and post in neuron_idx:
            X_raw[neuron_idx[pre], k] += cnt   # outgoing count for 'pre'

# ── X_pairs: (P, 8) per-pair synapse count ───────────────────────────────────
# Only pairs appearing in at least 2 stages
all_pairs_raw: set[tuple] = set()
for sc in stage_counts:
    all_pairs_raw.update(
        (pre, post) for (pre, post) in sc
        if pre in neuron_idx and post in neuron_idx
    )

# Count how many stages each pair appears in
pair_stage_counts: dict[tuple, int] = defaultdict(int)
for k, sc in enumerate(stage_counts):
    for pair in sc:
        if pair[0] in neuron_idx and pair[1] in neuron_idx:
            pair_stage_counts[pair] += 1

# Distribution of stage presence
dist = defaultdict(int)
for pair, cnt in pair_stage_counts.items():
    dist[cnt] += 1
print("\n[T1] Distribution of (pre,post) pairs by number of stages present:")
for n_stages in range(1, 9):
    print(f"  present in {n_stages} stage(s): {dist[n_stages]:>5} pairs")

# Filter to pairs present in ≥2 stages
filtered_pairs = sorted(
    [pair for pair, cnt in pair_stage_counts.items() if cnt >= 2]
)
P = len(filtered_pairs)
pair_idx = {pair: i for i, pair in enumerate(filtered_pairs)}
print(f"\n[T1] Pairs present in ≥2 stages: {P}")

X_pairs = np.zeros((P, 8), dtype=np.int32)
for k, sc in enumerate(stage_counts):
    for pair, cnt in sc.items():
        if pair in pair_idx:
            X_pairs[pair_idx[pair], k] = cnt

# ── top 10 pairs by total synapse count ──────────────────────────────────────
totals = X_pairs.sum(axis=1)
top10_idx = np.argsort(totals)[::-1][:10]
print("\n[T1] Top 10 pairs by total synapse count (summed across 8 stages):")
print(f"  {'pre':<12} {'post':<12} {'total':>8}  per-stage counts")
for idx in top10_idx:
    pre, post = filtered_pairs[idx]
    stage_str = "  ".join(f"{X_pairs[idx, k]:>3}" for k in range(8))
    print(f"  {pre:<12} {post:<12} {totals[idx]:>8}  [{stage_str}]")

# ── save ─────────────────────────────────────────────────────────────────────
np.save(os.path.join(OUT_DIR, "synapse_matrix_X_raw.npy"),   X_raw)
np.save(os.path.join(OUT_DIR, "synapse_matrix_X_pairs.npy"), X_pairs)

with open(os.path.join(OUT_DIR, "neuron_list_all.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(neurons_all) + "\n")

with open(os.path.join(OUT_DIR, "pair_list.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(f"{pre}->{post}" for pre, post in filtered_pairs) + "\n")

print(f"\n[T1] Saved synapse_matrix_X_raw.npy  {X_raw.shape}")
print(f"[T1] Saved synapse_matrix_X_pairs.npy {X_pairs.shape}")
print(f"[T1] Saved neuron_list_all.txt ({N} neurons)")
print(f"[T1] Saved pair_list.txt ({P} pairs)")
print("\n=== TASK 1 COMPLETE ===")
print(f"  Unique preferred neurons: {N}")
print(f"  Pairs in ≥2 stages:       {P}")
print(f"  X_raw shape:              {X_raw.shape}")
print(f"  X_pairs shape:            {X_pairs.shape}")
