from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipeline_utils import (
    FIRED_THRESHOLD_MV,
    OUT_SWEEP,
    STIM_NEURONS,
    amp_to_tag,
    ensure_directory,
    load_stage_cells,
    run_stage_simulation,
)


AMPLITUDES = [0.2, 0.5, 1.0, 2.0, 5.0]
DURATION_MS = 500.0
DT_MS = 0.05


def run_one(stage: int, amplitude_pa: float) -> dict:
    net_id = f"Sweep_D{stage}_{amp_to_tag(amplitude_pa)}pA"
    result = run_stage_simulation(
        stage,
        amplitude_pa,
        net_id,
        OUT_SWEEP,
        duration_ms=DURATION_MS,
        dt_ms=DT_MS,
        cells_to_stimulate=STIM_NEURONS,
        source_tag="task_a_sweep",
        reuse_existing=True,
        verbose=False,
    )
    payload = {
        "stage": stage,
        "net_id": net_id,
        "dat_path": result.dat_path,
        "lems_path": result.lems_path,
        "reused_existing": result.reused_existing,
        "jneuroml_ok": result.jneuroml_ok,
    }
    payload.update(result.summary)
    return payload


ensure_directory(OUT_SWEEP)
expected_counts = {stage: len(load_stage_cells(stage)) for stage in (1, 8)}

print("=" * 72)
print("TASK A: AMPLITUDE SWEEP - DEVELOPMENTAL SENSITIVITY WINDOW")
print("=" * 72)
print(f"Stimulated neurons: {STIM_NEURONS}")
print(f"Fired threshold: Vmax > {FIRED_THRESHOLD_MV} mV")
print(f"Expected neurons: D1={expected_counts[1]}, D8={expected_counts[8]}")

results: list[dict] = []
failed_amps: list[dict] = []
all_silent_at_floor = False

for amplitude_pa in AMPLITUDES:
    print("-" * 72)
    print(f"Running amplitude {amplitude_pa} pA")
    row = {"amp": amplitude_pa}

    for stage in (1, 8):
        label = f"D{stage}"
        try:
            stage_result = run_one(stage, amplitude_pa)
            row[f"{label.lower()}_result"] = stage_result
        except Exception as exc:
            stage_result = {
                "stage": stage,
                "net_id": f"Sweep_D{stage}_{amp_to_tag(amplitude_pa)}pA",
                "dat_path": str(OUT_SWEEP / f"Sweep_D{stage}_{amp_to_tag(amplitude_pa)}pA.dat"),
                "lems_path": str(OUT_SWEEP / f"LEMS_Sweep_D{stage}_{amp_to_tag(amplitude_pa)}pA.xml"),
                "reused_existing": False,
                "jneuroml_ok": False,
                "error": str(exc),
                "n_timesteps": 0,
                "n_neurons": expected_counts[stage],
                "dt_ms": DT_MS,
                "min_voltage_mv": float("nan"),
                "max_voltage_mv": float("nan"),
                "fired": 0,
                "subthreshold": 0,
                "silent": expected_counts[stage],
            }
            row[f"{label.lower()}_result"] = stage_result
            failed_amps.append({"amp": amplitude_pa, "stage": stage, "error": str(exc)})
            print(f"  {label} FAILED: {exc}")

    d1 = row["d1_result"]
    d8 = row["d8_result"]
    diff = int(d8["fired"] - d1["fired"])
    row["n_d1"] = int(d1["fired"])
    row["n_d8"] = int(d8["fired"])
    row["diff"] = diff
    row["d1_pct"] = round(100.0 * d1["fired"] / max(1, d1["n_neurons"]), 2)
    row["d8_pct"] = round(100.0 * d8["fired"] / max(1, d8["n_neurons"]), 2)

    print(
        f"amp={amplitude_pa}pA | D1: {d1['fired']}/{d1['n_neurons']} fired | "
        f"D8: {d8['fired']}/{d8['n_neurons']} fired | diff: {diff}"
    )
    print(
        f"  D1 detail: subthreshold={d1['subthreshold']}, silent={d1['silent']}, "
        f"reused={d1['reused_existing']}"
    )
    print(
        f"  D8 detail: subthreshold={d8['subthreshold']}, silent={d8['silent']}, "
        f"reused={d8['reused_existing']}"
    )

    if amplitude_pa == 0.2 and (d1["fired"] == 0 and d8["fired"] == 0):
        all_silent_at_floor = True
        print("  NOTE: 0.2 pA produced all-silent output across both stages.")

    results.append(row)

successful = [
    row for row in results
    if "error" not in row["d1_result"] and "error" not in row["d8_result"]
]
window_candidates = [row for row in successful if row["diff"] > 0]
if window_candidates:
    best = max(window_candidates, key=lambda item: (item["diff"], -abs(item["d1_pct"] - 40.0)))
    sensitivity_note = (
        f"Sensitivity window identified at {best['amp']} pA: "
        f"D8 exceeds D1 by {best['diff']} fired neurons."
    )
elif successful:
    best = next((row for row in successful if row["amp"] == 1.0), successful[0])
    sensitivity_note = (
        "Sweep was inconclusive because no amplitude produced D8 > D1. "
        f"Defaulting to {best['amp']} pA for Task B."
    )
else:
    best = {"amp": 1.0}
    sensitivity_note = "All amplitudes failed. Defaulting to 1.0 pA for Task B."

if all_silent_at_floor:
    sensitivity_note += " 0.2 pA is below the usable floor for this setup."

print("=" * 72)
print("SENSITIVITY WINDOW")
print("=" * 72)
print(sensitivity_note)
if failed_amps:
    print("Failed amplitudes/stages:")
    for entry in failed_amps:
        print(f"  amp={entry['amp']} stage=D{entry['stage']}: {entry['error']}")

json_payload = {
    "amplitudes": AMPLITUDES,
    "stim_neurons": STIM_NEURONS,
    "optimal_amp": best["amp"],
    "sensitivity_note": sensitivity_note,
    "all_silent_at_0_2": all_silent_at_floor,
    "failures": failed_amps,
    "results": results,
}
json_path = OUT_SWEEP / "sweep_results.json"
json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

x = np.arange(len(results))
width = 0.36
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width / 2, [row["n_d1"] for row in results], width, label="D1", color="#2f6db0")
ax.bar(x + width / 2, [row["n_d8"] for row in results], width, label="D8", color="#d45b3f")
ax.set_xticks(x)
ax.set_xticklabels([f"{row['amp']}" for row in results])
ax.set_xlabel("Amplitude (pA)")
ax.set_ylabel("Fired neurons")
ax.set_title("Task A amplitude sweep: D1 vs D8 fired neurons")
ax.legend()
for index, row in enumerate(results):
    ax.text(index - width / 2, row["n_d1"] + 1, str(row["n_d1"]), ha="center", va="bottom", fontsize=8)
    ax.text(index + width / 2, row["n_d8"] + 1, str(row["n_d8"]), ha="center", va="bottom", fontsize=8)
fig.tight_layout()
plot_path = OUT_SWEEP / "sweep_plot.png"
fig.savefig(plot_path, dpi=140)
plt.close(fig)

print(f"Saved {json_path}")
print(f"Saved {plot_path}")
print("=== TASK A COMPLETE ===")
print(f"Optimal amplitude: {best['amp']} pA")
