# -*- coding: utf-8 -*-
"""
run_all_tasks.py — Master runner for the SINDyG structural NDP pipeline.
Runs Tasks 1–6 in sequence in the same Python process.
"""
import sys, os, time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MDG_BUILD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MDG_BUILD)

tasks = [
    ("Task 1 — Build synapse matrix",          "build_synapse_matrix"),
    ("Task 2 — Cubic spline interpolation",    "spline_interpolation"),
    ("Task 3 — Neuron class pooling",          "pool_by_class"),
    ("Task 4 — Adjacency matrix for SINDyG",  "build_adj_for_sindy"),
    ("Task 5 — SINDyG fitting",               "sindy_structural"),
    ("Task 6 — Sanity check + report",         "sanity_check"),
]

t_pipeline_start = time.time()
print("=" * 70)
print("MDG SINDyG STRUCTURAL NDP PIPELINE")
print("=" * 70)

for label, module_name in tasks:
    print(f"\n{'#'*70}")
    print(f"# {label}")
    print(f"{'#'*70}\n")
    t0 = time.time()
    import importlib
    # Force fresh import each time (in case of state from prior runs)
    if module_name in sys.modules:
        del sys.modules[module_name]
    mod = importlib.import_module(module_name)
    elapsed = time.time() - t0
    print(f"\n  [{label}] Done in {elapsed:.1f}s")

total_elapsed = time.time() - t_pipeline_start
print(f"\n{'='*70}")
print(f"PIPELINE COMPLETE in {total_elapsed:.1f}s")
print(f"{'='*70}")
print(f"\nOutputs in: {os.path.join(MDG_BUILD, 'output_sim')}")
print(f"Mentor report: output_sim/structural_ndp_report.md")
