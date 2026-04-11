from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from pipeline_utils import (
    DT_MS,
    OUT_STAGES,
    OUT_SWEEP,
    STIM_NEURONS,
    T_BIO_HOURS,
    ensure_directory,
    load_stage_cells,
    run_stage_simulation,
)


POST_STIM_INDEX = int(round(10.0 / DT_MS))
STAGES = list(range(1, 9))


ensure_directory(OUT_STAGES)
sweep_json = OUT_SWEEP / "sweep_results.json"
if sweep_json.exists():
    sweep_data = json.loads(sweep_json.read_text(encoding="utf-8"))
    optimal_amp = float(sweep_data.get("optimal_amp", 1.0))
else:
    optimal_amp = 1.0

print("=" * 72)
print("TASK B: SIMULATE ALL 8 STAGES AT THE SENSITIVITY AMPLITUDE")
print("=" * 72)
print(f"Using amplitude: {optimal_amp} pA")
print(f"Stimulated neurons: {STIM_NEURONS}")

activation_rows: list[dict] = []

for stage in STAGES:
    stage_dir = ensure_directory(OUT_STAGES / f"D{stage}")
    net_id = f"Stage_D{stage}_{str(optimal_amp).replace('.', 'p')}pA"
    print("-" * 72)
    print(f"Stage D{stage} ({T_BIO_HOURS[stage - 1]} h)")
    try:
        result = run_stage_simulation(
            stage,
            optimal_amp,
            net_id,
            stage_dir,
            cells_to_stimulate=STIM_NEURONS,
            source_tag="task_b_all_stages",
            reuse_existing=True,
            verbose=False,
        )
        raw = np.loadtxt(result.dat_path)
        voltage_mv = raw[:, 1:] * 1000.0
        neuron_order = list(result.neuron_order)

        max_voltage = voltage_mv.max(axis=0)
        mean_voltage = voltage_mv[POST_STIM_INDEX:, :].mean(axis=0)
        time_above_threshold = (voltage_mv > -40.0).mean(axis=0)
        is_active = (max_voltage > -20.0).astype(float)
        features = np.column_stack(
            [max_voltage, mean_voltage, time_above_threshold, is_active]
        )

        features_path = OUT_STAGES / f"features_D{stage}.npy"
        order_path = OUT_STAGES / f"neuron_order_D{stage}.txt"
        active_path = OUT_STAGES / f"n_active_D{stage}.txt"
        np.save(features_path, features)
        order_path.write_text("\n".join(neuron_order) + "\n", encoding="utf-8")
        n_active = int(is_active.sum())
        active_path.write_text(str(n_active), encoding="utf-8")

        n_neurons = int(features.shape[0])
        pct_active = round(100.0 * n_active / max(1, n_neurons), 2)
        print(
            f"  active={n_active}/{n_neurons} ({pct_active:.2f}%) | "
            f"reused={result.reused_existing}"
        )
        activation_rows.append(
            {
                "stage": stage,
                "hours": float(T_BIO_HOURS[stage - 1]),
                "n_neurons": n_neurons,
                "n_active": n_active,
                "pct_active": pct_active,
                "dat_path": result.dat_path,
                "lems_path": result.lems_path,
                "error": "",
            }
        )
    except Exception as exc:
        expected_neurons = len(load_stage_cells(stage))
        print(f"  FAILED: {exc}")
        activation_rows.append(
            {
                "stage": stage,
                "hours": float(T_BIO_HOURS[stage - 1]),
                "n_neurons": expected_neurons,
                "n_active": -1,
                "pct_active": -1.0,
                "dat_path": "",
                "lems_path": "",
                "error": str(exc),
            }
        )

print("=" * 72)
print("Stage | N neurons | N active | % active")
print("=" * 72)
for row in activation_rows:
    error_suffix = f" | ERROR: {row['error']}" if row["error"] else ""
    print(
        f"D{row['stage']} | {row['n_neurons']} | {row['n_active']} | "
        f"{row['pct_active']}%{error_suffix}"
    )

valid_counts = [row["n_active"] for row in activation_rows if row["n_active"] >= 0]
if len(valid_counts) == len(STAGES):
    monotonic = all(valid_counts[idx] <= valid_counts[idx + 1] for idx in range(len(valid_counts) - 1))
    strictly_changes = any(valid_counts[idx] < valid_counts[idx + 1] for idx in range(len(valid_counts) - 1))
    if monotonic and strictly_changes:
        trend_label = "DEVELOPMENTAL SIGNAL CONFIRMED"
    elif len(set(valid_counts)) == 1:
        trend_label = "Activation is flat across stages; amplitude needs further tuning"
    else:
        trend_label = "Activation is non-monotonic; inspect stage-specific differences"
else:
    trend_label = "Activation table incomplete because at least one stage failed"
print(trend_label)

csv_path = OUT_STAGES / "activation_table.csv"
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["stage", "hours", "n_neurons", "n_active", "pct_active", "dat_path", "lems_path", "error"],
    )
    writer.writeheader()
    writer.writerows(activation_rows)

json_path = OUT_STAGES / "activation_data.json"
json_path.write_text(
    json.dumps(
        {
            "optimal_amp": optimal_amp,
            "trend_label": trend_label,
            "stages": activation_rows,
        },
        indent=2,
    ),
    encoding="utf-8",
)

print(f"Saved {csv_path}")
print("=== TASK B COMPLETE ===")
