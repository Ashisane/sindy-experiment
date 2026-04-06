# -*- coding: utf-8 -*-
"""
TASK 3 — pool_by_class.py
Map neurons to their class (strip L/R suffix or trailing digits),
then pool dense trajectories by class to create the data structures
that make SINDyG regression overdetermined.

Class mapping rules (applied in order):
  1. If name ends in L or R and len > 2 → strip last char
  2. If result ends in digits → strip trailing digits
  3. Use result as class name

Outputs (saved to output_sim/)
-------------------------------
class_members.json  {class_name: [member1, member2, ...]}
class_names.txt     class names in sorted order (one per line)

Printed: top classes by member count, total pooled data points
"""

import json, os, sys, re
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MDG_BUILD = os.path.dirname(os.path.abspath(__file__))
OUT_DIR   = os.path.join(MDG_BUILD, "output_sim")


def neuron_to_class(name: str) -> str:
    """Map a neuron name to its class name."""
    # Step 1: strip terminal L/R
    if len(name) > 2 and name[-1] in "LR":
        name = name[:-1]
    # Step 2: strip trailing digits
    name = re.sub(r"\d+$", "", name)
    return name if name else "UNK"


# ── load ─────────────────────────────────────────────────────────────────────
X_dense    = np.load(os.path.join(OUT_DIR, "X_dense.npy"))
Xdot_dense = np.load(os.path.join(OUT_DIR, "Xdot_dense.npy"))

with open(os.path.join(OUT_DIR, "neuron_list_all.txt"), encoding="utf-8") as fh:
    neurons = [l.strip() for l in fh if l.strip()]

N = len(neurons)
print(f"[T3] Loaded {N} neurons, X_dense shape = {X_dense.shape}")

# ── build class → member mapping ─────────────────────────────────────────────
from collections import defaultdict
class_members: dict[str, list[str]] = defaultdict(list)
for n in neurons:
    cls = neuron_to_class(n)
    class_members[cls].append(n)

# Sort members within each class for reproducibility
class_members = {cls: sorted(mbrs) for cls, mbrs in class_members.items()}
# Sort classes
sorted_classes = sorted(class_members.keys())
class_idx  = {cls: i for i, cls in enumerate(sorted_classes)}
N_classes  = len(sorted_classes)

neuron_idx = {n: i for i, n in enumerate(neurons)}

# ── print class size distribution ────────────────────────────────────────────
sizes = np.array([len(class_members[cls]) for cls in sorted_classes])
print(f"\n[T3] Total neuron classes: {N_classes}")
print(f"     Size distribution:")
for sz in sorted(set(sizes.tolist())):
    n_cls = (sizes == sz).sum()
    pct   = 100 * n_cls / N_classes
    print(f"       size {sz:>2}: {n_cls:>4} classes  ({pct:.1f}%)")

print(f"\n[T3] Top 10 classes by member count:")
top_idx = np.argsort(sizes)[::-1][:10]
print(f"  {'Class':<12} {'Members':>8}  Neuron names")
for i in top_idx:
    cls = sorted_classes[i]
    print(f"  {cls:<12} {sizes[i]:>8}  {class_members[cls]}")

# ── compute total pooled data points ─────────────────────────────────────────
# Only classes with ≥2 members contribute "pooled" benefit
multi_classes  = [cls for cls in sorted_classes if len(class_members[cls]) >= 2]
total_pts_all  = sum(len(class_members[cls]) * 100 for cls in sorted_classes)
total_pts_multi = sum(len(class_members[cls]) * 100 for cls in multi_classes)

print(f"\n[T3] Classes with ≥2 members: {len(multi_classes)}")
print(f"     Total data points (all classes × 100 t-pts):          {total_pts_all:>8}")
print(f"     Total data points (multi-member classes only):         {total_pts_multi:>8}")
print(f"     Effective independent rows for regression ~ {N_classes * 100:>6} (N_classes × 100)")

# ── verify example classes ────────────────────────────────────────────────────
print("\n[T3] Example class verifications:")
for name in ["AVBL", "AVAL", "AVAR", "VB1", "DA1", "RMDL"]:
    if name in neuron_idx:
        cls = neuron_to_class(name)
        mbrs = class_members.get(cls, [])
        print(f"  {name} -> class '{cls}' members: {mbrs}")

# ── save ─────────────────────────────────────────────────────────────────────
with open(os.path.join(OUT_DIR, "class_members.json"), "w", encoding="utf-8") as fh:
    json.dump(class_members, fh, indent=2)

with open(os.path.join(OUT_DIR, "class_names.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(sorted_classes) + "\n")

print(f"\n[T3] Saved class_members.json ({N_classes} classes)")
print(f"[T3] Saved class_names.txt")

print("\n=== TASK 3 COMPLETE ===")
print(f"  Neuron classes: {N_classes}")
print(f"  Multi-member (≥2) classes: {len(multi_classes)}")
print(f"  Total pooled data points: {total_pts_all}  "
      f"(vs N_lib = N_classes + 1 = {N_classes + 1} candidate terms)")
print(f"  System IS overdetermined: {total_pts_all > N_classes + 1}")
