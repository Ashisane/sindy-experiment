from __future__ import annotations

import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from pipeline_utils import (
    C302_PACKAGE,
    MDG_BUILD,
    ensure_import_paths,
    ensure_directory,
    parse_output_columns,
    pushd,
    sync_witvliet_reader,
)


OUTPUT_DIR = ensure_directory(MDG_BUILD / "output_kenta_benchmark")
BENCHMARK_JSON = MDG_BUILD / "benchmark_results.json"
TRACE_FILES = {
    "P1_baseline": MDG_BUILD / "traces_P1_baseline.npy",
    "P2_AVAL_lesion": MDG_BUILD / "traces_P2_AVAL_lesion.npy",
    "P3_unilateral_touch": MDG_BUILD / "traces_P3_unilateral.npy",
    "P1_low": MDG_BUILD / "traces_P1_low.npy",
    "P2_low": MDG_BUILD / "traces_P2_low.npy",
    "P3_low": MDG_BUILD / "traces_P3_low.npy",
}
NEURON_INDEX_JSON = MDG_BUILD / "neuron_index.json"
REPORT_MD = MDG_BUILD / "KENTA_BENCHMARK_REPORT.md"

STAGE = 8
STANDARD_AMPLITUDE_PA = 2.0
LOW_AMPLITUDE_PA = 0.5
STIM_DELAY_MS = 50.0
STIM_DURATION_MS = 200.0
SIM_DURATION_MS = 500.0
DT_MS = 0.05

REQUESTED_CIRCUIT = [
    "ALML",
    "ALMR",
    "PVML",
    "PVMR",
    "PVM",
    "AVM",
    "AVAL",
    "AVAR",
    "AVDL",
    "AVDR",
    "AVBL",
    "AVBR",
    "PVCL",
    "PVCR",
]
BILATERAL_PAIRS = [("AVAL", "AVAR"), ("AVBL", "AVBR"), ("PVCL", "PVCR")]
BACKWARD_NEURONS = ["AVAL", "AVAR", "AVDL", "AVDR"]
FORWARD_NEURONS = ["AVBL", "AVBR", "PVCL", "PVCR"]
PERTURBATIONS = {
    "P1_baseline": {
        "net_id": "KENTA_P1_baseline",
        "stimulated": ["ALML", "ALMR"],
        "lesioned": [],
        "amplitude_pA": STANDARD_AMPLITUDE_PA,
    },
    "P2_AVAL_lesion": {
        "net_id": "KENTA_P2_AVAL_lesion",
        "stimulated": ["ALML", "ALMR"],
        "lesioned": ["AVAL"],
        "amplitude_pA": STANDARD_AMPLITUDE_PA,
    },
    "P3_unilateral_touch": {
        "net_id": "KENTA_P3_unilateral",
        "stimulated": ["ALML"],
        "lesioned": [],
        "amplitude_pA": STANDARD_AMPLITUDE_PA,
    },
    "P1_low": {
        "net_id": "KENTA_P1_low",
        "stimulated": ["ALML", "ALMR"],
        "lesioned": [],
        "amplitude_pA": LOW_AMPLITUDE_PA,
    },
    "P2_low": {
        "net_id": "KENTA_P2_low",
        "stimulated": ["ALML", "ALMR"],
        "lesioned": ["AVAL"],
        "amplitude_pA": LOW_AMPLITUDE_PA,
    },
    "P3_low": {
        "net_id": "KENTA_P3_low",
        "stimulated": ["ALML"],
        "lesioned": [],
        "amplitude_pA": LOW_AMPLITUDE_PA,
    },
}


def sync_kenta_reader() -> Path:
    source = MDG_BUILD / "kenta_stage8_reader.py"
    dest = C302_PACKAGE / "kenta_stage8_reader.py"
    if not dest.exists() or source.read_text(encoding="utf-8") != dest.read_text(encoding="utf-8"):
        shutil.copy2(source, dest)
    return dest


def import_kenta_reader(excluded: list[str]):
    ensure_import_paths()
    sync_witvliet_reader()
    sync_kenta_reader()
    importlib.invalidate_caches()
    module_name = "c302.kenta_stage8_reader"
    if module_name in sys.modules:
        module = importlib.reload(sys.modules[module_name])
    else:
        module = importlib.import_module(module_name)
    module.set_stage(STAGE)
    module.set_excluded_neurons(excluded)
    return module


def load_stage8_circuit() -> tuple[list[str], list[str]]:
    ensure_import_paths()
    from witvliet_reader import WitvlietDataReader

    cells, _ = WitvlietDataReader(stage=STAGE).read_data(include_nonconnected_cells=False)
    present = [name for name in REQUESTED_CIRCUIT if name in set(cells)]
    absent = [name for name in REQUESTED_CIRCUIT if name not in set(cells)]
    return present, absent


def configure_params(amplitude_pa: float):
    import c302
    from c302 import parameters_C0

    params = parameters_C0.ParameterisedModel()
    params.set_bioparameter("unphysiological_offset_current", f"{amplitude_pa} pA", "kenta_benchmark", "0")
    params.set_bioparameter("unphysiological_offset_current_del", f"{STIM_DELAY_MS} ms", "kenta_benchmark", "0")
    params.set_bioparameter("unphysiological_offset_current_dur", f"{STIM_DURATION_MS} ms", "kenta_benchmark", "0")
    return c302, params


def run_lems(lems_path: Path, target_directory: Path) -> bool:
    from pyneuroml import pynml

    with pushd(target_directory):
        return bool(
            pynml.run_lems_with_jneuroml(
                lems_path.name,
                nogui=True,
                load_saved_data=False,
                plot=False,
                max_memory="4G",
                verbose=False,
            )
        )


def run_perturbation(
    label: str,
    present_circuit: list[str],
    stimulated: list[str],
    lesioned: list[str],
    amplitude_pa: float,
) -> dict[str, Any]:
    import c302

    perturb_dir = ensure_directory(OUTPUT_DIR / label)
    net_id = PERTURBATIONS[label]["net_id"]
    circuit_neurons = [name for name in present_circuit if name not in set(lesioned)]
    if not set(stimulated).issubset(set(circuit_neurons)):
        missing_stim = [name for name in stimulated if name not in circuit_neurons]
        raise ValueError(f"Stimulated neurons missing from circuit: {missing_stim}")

    import_kenta_reader(lesioned)
    c302_module, params = configure_params(amplitude_pa)
    c302_module.generate(
        net_id,
        params,
        data_reader="kenta_stage8_reader",
        cells=circuit_neurons,
        cells_to_stimulate=stimulated,
        cells_to_plot=circuit_neurons,
        duration=SIM_DURATION_MS,
        dt=DT_MS,
        target_directory=str(perturb_dir),
        verbose=False,
    )

    lems_path = perturb_dir / f"LEMS_{net_id}.xml"
    dat_path = perturb_dir / f"{net_id}.dat"
    ok = run_lems(lems_path, perturb_dir)
    if not ok or not dat_path.exists():
        raise RuntimeError(f"Simulation failed for {label}")

    raw = np.loadtxt(dat_path)
    time_ms = raw[:, 0] * 1000.0
    traces_mv = raw[:, 1:] * 1000.0
    neuron_order = parse_output_columns(lems_path, dat_path.name)
    if not neuron_order:
        neuron_order = list(circuit_neurons)
    if len(neuron_order) != traces_mv.shape[1]:
        raise RuntimeError(f"Column mismatch for {label}: {len(neuron_order)} names vs {traces_mv.shape[1]} traces")

    baseline_mask = time_ms < STIM_DELAY_MS
    if not baseline_mask.any():
        raise RuntimeError("No pre-stimulus window available for baseline computation")
    baseline_mv = traces_mv[baseline_mask, :].mean(axis=0)
    peak_mv = traces_mv.max(axis=0)
    peak_delta_mv = peak_mv - baseline_mv
    percent_depolarization = np.divide(
        peak_delta_mv,
        -baseline_mv,
        out=np.zeros_like(peak_delta_mv),
        where=np.abs(baseline_mv) > 1e-9,
    ) * 100.0

    peak_dict = {name: round(float(value), 4) for name, value in zip(neuron_order, peak_mv)}
    metric2_per_neuron = {
        name: round(float(value), 4) for name, value in zip(neuron_order, percent_depolarization)
    }
    mean_percent_depolarization = round(float(np.mean(percent_depolarization)), 6)

    name_to_peak_delta = {name: float(value) for name, value in zip(neuron_order, peak_delta_mv)}
    backward_available = [name for name in BACKWARD_NEURONS if name in name_to_peak_delta]
    forward_available = [name for name in FORWARD_NEURONS if name in name_to_peak_delta]
    backward_mean = float(np.mean([name_to_peak_delta[name] for name in backward_available])) if backward_available else float("nan")
    forward_mean = float(np.mean([name_to_peak_delta[name] for name in forward_available])) if forward_available else float("nan")
    if np.isnan(backward_mean) or np.isnan(forward_mean) or forward_mean <= 0:
        backward_forward_ratio = None
    else:
        backward_forward_ratio = round(backward_mean / forward_mean, 6)

    symmetry = {}
    symmetry_values = []
    for left_name, right_name in BILATERAL_PAIRS:
        key = f"{left_name}_{right_name}"
        if left_name in name_to_peak_delta and right_name in name_to_peak_delta:
            left_value = max(name_to_peak_delta[left_name], 0.0)
            right_value = max(name_to_peak_delta[right_name], 0.0)
            denom = max(left_value, right_value, 1e-9)
            value = round(abs(left_value - right_value) / denom, 6)
            symmetry[key] = value
            symmetry_values.append(value)
        else:
            symmetry[key] = None
    symmetry["mean"] = round(float(np.mean(symmetry_values)), 6) if symmetry_values else None

    return {
        "label": label,
        "amplitude_pA": amplitude_pa,
        "stimulated": stimulated,
        "lesioned": lesioned,
        "circuit_neurons_used": neuron_order,
        "time_ms": time_ms,
        "traces_mv": traces_mv,
        "peak_dict": peak_dict,
        "metric2_per_neuron_percent_depolarization": metric2_per_neuron,
        "mean_percent_depolarization": mean_percent_depolarization,
        "backward_mean_delta_mV": round(backward_mean, 6) if not np.isnan(backward_mean) else None,
        "forward_mean_delta_mV": round(forward_mean, 6) if not np.isnan(forward_mean) else None,
        "backward_forward_ratio": backward_forward_ratio,
        "symmetry": symmetry,
        "raw_dat_path": str(dat_path),
    }


def build_report(metadata: dict[str, Any], perturbations: dict[str, dict[str, Any]]) -> str:
    p1 = perturbations["P1_baseline"]
    p2 = perturbations["P2_AVAL_lesion"]
    p3 = perturbations["P3_unilateral_touch"]
    p1_low = perturbations["P1_low"]
    p2_low = perturbations["P2_low"]
    p3_low = perturbations["P3_low"]

    def fmt(value):
        if value is None:
            return "NA"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    p1_ratio = p1["metric3_backward_forward_ratio"]
    p2_ratio = p2["metric3_backward_forward_ratio"]
    p3_ratio = p3["metric3_backward_forward_ratio"]
    p1_low_ratio = p1_low["metric3_backward_forward_ratio"]
    p2_low_ratio = p2_low["metric3_backward_forward_ratio"]
    p3_low_ratio = p3_low["metric3_backward_forward_ratio"]
    lesion_delta = None if p1_ratio is None or p2_ratio is None else p2_ratio - p1_ratio
    avar_baseline = p1["metric1_peak_voltage"].get("AVAR")
    avar_lesion = p2["metric1_peak_voltage"].get("AVAR")
    avar_change = None if avar_baseline is None or avar_lesion is None else avar_lesion - avar_baseline
    p1_sym = p1["metric4_bilateral_symmetry"].get("mean")
    p3_sym = p3["metric4_bilateral_symmetry"].get("mean")
    p1_low_sym = p1_low["metric4_bilateral_symmetry"].get("mean")
    p3_low_sym = p3_low["metric4_bilateral_symmetry"].get("mean")
    asymmetry_increase = None if p1_sym is None or p3_sym is None else p3_sym - p1_sym
    low_asymmetry_increase = None if p1_low_sym is None or p3_low_sym is None else p3_low_sym - p1_low_sym
    strongest_p3_pair = None
    valid_p3_pairs = {k: v for k, v in p3["metric4_bilateral_symmetry"].items() if k != "mean" and v is not None}
    if valid_p3_pairs:
        strongest_p3_pair = max(valid_p3_pairs.items(), key=lambda item: item[1])
    strongest_p3_low_pair = None
    valid_p3_low_pairs = {
        k: v for k, v in p3_low["metric4_bilateral_symmetry"].items() if k != "mean" and v is not None
    }
    if valid_p3_low_pairs:
        strongest_p3_low_pair = max(valid_p3_low_pairs.items(), key=lambda item: item[1])

    lines = [
        "# KENTA Benchmark Report",
        "",
        "## 1. Circuit setup",
        "",
        f"Stage 8 circuit neurons requested: {', '.join(metadata['requested_circuit_neurons'])}",
        f"Stage 8 circuit neurons present: {', '.join(metadata['circuit_neurons'])}",
        f"Stage 8 requested neurons absent: {', '.join(metadata['absent_requested_neurons']) if metadata['absent_requested_neurons'] else 'none'}",
        "The benchmark uses only the present stage-8 circuit neurons so the protocol remains small and exactly reproducible for KENTA.",
        "",
        "## 2. Results table",
        "",
        "| Metric | P1 baseline (2.0 pA) | P2 AVAL lesion (2.0 pA) | P3 unilateral touch (2.0 pA) | P1 low (0.5 pA) | P2 low (0.5 pA) | P3 low (0.5 pA) |",
        "|---|---|---|---|---|---|---|",
        f"| Metric 1 summary: mean circuit peak voltage (mV) | {fmt(np.mean(list(p1['metric1_peak_voltage'].values())))} | {fmt(np.mean(list(p2['metric1_peak_voltage'].values())))} | {fmt(np.mean(list(p3['metric1_peak_voltage'].values())))} | {fmt(np.mean(list(p1_low['metric1_peak_voltage'].values())))} | {fmt(np.mean(list(p2_low['metric1_peak_voltage'].values())))} | {fmt(np.mean(list(p3_low['metric1_peak_voltage'].values())))} |",
        f"| Metric 2: mean percent depolarization | {fmt(p1['metric2_mean_percent_depolarization'])} | {fmt(p2['metric2_mean_percent_depolarization'])} | {fmt(p3['metric2_mean_percent_depolarization'])} | {fmt(p1_low['metric2_mean_percent_depolarization'])} | {fmt(p2_low['metric2_mean_percent_depolarization'])} | {fmt(p3_low['metric2_mean_percent_depolarization'])} |",
        f"| Metric 3: backward / forward ratio | {fmt(p1['metric3_backward_forward_ratio'])} | {fmt(p2['metric3_backward_forward_ratio'])} | {fmt(p3['metric3_backward_forward_ratio'])} | {fmt(p1_low['metric3_backward_forward_ratio'])} | {fmt(p2_low['metric3_backward_forward_ratio'])} | {fmt(p3_low['metric3_backward_forward_ratio'])} |",
        f"| Metric 4: mean bilateral symmetry index | {fmt(p1['metric4_bilateral_symmetry'].get('mean'))} | {fmt(p2['metric4_bilateral_symmetry'].get('mean'))} | {fmt(p3['metric4_bilateral_symmetry'].get('mean'))} | {fmt(p1_low['metric4_bilateral_symmetry'].get('mean'))} | {fmt(p2_low['metric4_bilateral_symmetry'].get('mean'))} | {fmt(p3_low['metric4_bilateral_symmetry'].get('mean'))} |",
        "",
        "Full Metric 1 peak-voltage dictionaries and per-neuron Metric 2 percent-depolarization values are stored in benchmark_results.json for KENTA comparison.",
        "",
        "## 3. P1 biological validation",
        "",
    ]
    if p1_ratio is not None and p1_ratio > 1.0:
        lines.append(
            f"P1 passes the anterior-touch sanity check: backward / forward = {p1_ratio:.4f} > 1, so the c302 sub-circuit prefers backward command activation under bilateral anterior touch."
        )
    else:
        lines.append(
            f"P1 does not pass the anterior-touch sanity check cleanly: backward / forward = {fmt(p1_ratio)}. This suggests the result is either too parameter-sensitive or the reduced sub-circuit omits compensating structural context that exists in the full connectome."
        )
    if p1_low_ratio is not None and p1_low_ratio > 1.0:
        lines.append(
            f"At 0.5 pA, P1_low does pass the same check with backward / forward = {p1_low_ratio:.4f} > 1, which makes the lower-amplitude regime more biologically plausible for KENTA comparison."
        )
    else:
        lines.append(
            f"At 0.5 pA, P1_low still does not show backward dominance cleanly: backward / forward = {fmt(p1_low_ratio)}."
        )

    lines.extend([
        "",
        "## 4. P2 lesion effect",
        "",
        f"The AVAL lesion changes Metric 3 by {fmt(lesion_delta)} relative to baseline.",
        f"AVAR peak voltage changes by {fmt(avar_change)} mV relative to baseline.",
        f"At 0.5 pA, Metric 3 changes from {fmt(p1_low_ratio)} in P1_low to {fmt(p2_low_ratio)} in P2_low.",
    ])
    if avar_change is not None and avar_change > 0:
        lines.append("AVAR increases after the lesion, which is consistent with partial compensation through the remaining bilateral pathway.")
    else:
        lines.append("AVAR does not increase after the lesion, so the model does not show clear compensatory recruitment under this reduced-circuit benchmark.")

    lines.extend([
        "",
        "## 5. P3 asymmetry",
        "",
        f"Mean bilateral symmetry index changes by {fmt(asymmetry_increase)} from P1 to P3.",
        f"At 0.5 pA, mean bilateral symmetry index changes by {fmt(low_asymmetry_increase)} from P1_low to P3_low.",
    ])
    if strongest_p3_pair is not None:
        lines.append(f"The strongest bilateral asymmetry in P3 is {strongest_p3_pair[0]} = {strongest_p3_pair[1]:.4f}.")
    if strongest_p3_low_pair is not None:
        lines.append(f"At 0.5 pA, the strongest bilateral asymmetry in P3_low is {strongest_p3_low_pair[0]} = {strongest_p3_low_pair[1]:.4f}.")
    if p1_sym is not None and p3_sym is not None and p3_sym > p1_sym:
        lines.append("P3 is more asymmetric than P1, so unilateral sensory drive is preserved functionally in the benchmark output.")
    else:
        lines.append("P3 does not increase asymmetry over P1, so the current c302 setup is washing out the expected lateralized response.")
    if p1_low_sym is not None and p3_low_sym is not None and p3_low_sym > p1_low_sym:
        lines.append("At 0.5 pA, unilateral touch also increases asymmetry over bilateral touch.")
    else:
        lines.append("At 0.5 pA, unilateral touch does not increase asymmetry over bilateral touch, so the lower-amplitude regime improves the command ratio but weakens the lateralization signature.")

    lines.extend([
        "",
        "## 6. Known limitations for KENTA comparison",
        "",
        "- c302 uses adult HH-derived parameters adapted for graded synapses.",
        "- Witvliet D8 is a population-average connectome with substantial inter-individual variability.",
        "- Gap junctions are included, but the benchmark does not distinguish their uncertainty from chemical synapse uncertainty.",
        "- c302 does not model neuropeptide modulation.",
        "- The stimulus is artificial current injection, not mechanosensory transduction.",
        "- Activation depth metric (Metric 2) should be interpreted with caution in graded-synapse models — the -50 mV threshold is reached by most neurons even at low stimulation amplitudes. The KENTA collaborator should compare raw peak voltages (Metric 1) and bilateral symmetry (Metric 4) as the primary comparison targets.",
        "",
        "## 7. Scientific interpretation for the collaboration",
        "",
        "If KENTA matches the c302 benchmark across the baseline, lesion, and unilateral-touch perturbations, that would argue that the core functional signatures of this circuit are imposed mainly by the adult stage-8 wiring pattern rather than by the detailed c302 membrane equations. If KENTA and c302 disagree, the mismatch becomes informative: it would identify which signatures are structurally robust and which depend strongly on continuous-time conductance dynamics, synaptic parameterization, or how graded transmission is implemented. That separation is exactly what makes this a useful collaboration benchmark rather than just another simulation run.",
    ])
    return "\n".join(lines) + "\n"


def main():
    ensure_import_paths()
    sync_witvliet_reader()
    sync_kenta_reader()

    present_circuit, absent_circuit = load_stage8_circuit()
    print("=" * 72)
    print("MDG KENTA CIRCUIT BENCHMARK")
    print("=" * 72)
    print(f"Stage 8 present circuit neurons: {present_circuit}")
    if absent_circuit:
        print(f"Stage 8 absent requested neurons: {absent_circuit}")

    perturbation_results: dict[str, dict[str, Any]] = {}
    neuron_index: dict[str, dict[str, str]] = {}

    import c302
    c302_version = getattr(c302, "__version__", "unknown")

    for label, config in PERTURBATIONS.items():
        print("-" * 72)
        print(f"Running {label}")
        result = run_perturbation(
            label,
            present_circuit=present_circuit,
            stimulated=config["stimulated"],
            lesioned=config["lesioned"],
            amplitude_pa=config["amplitude_pA"],
        )
        np.save(TRACE_FILES[label], result["traces_mv"])
        neuron_index[label] = {str(idx): name for idx, name in enumerate(result["circuit_neurons_used"])}
        perturbation_results[label] = {
            "amplitude_pA": config["amplitude_pA"],
            "stimulated": config["stimulated"],
            "lesioned": config["lesioned"],
            "circuit_neurons_used": result["circuit_neurons_used"],
            "metric1_peak_voltage": result["peak_dict"],
            "metric2_mean_percent_depolarization": result["mean_percent_depolarization"],
            "metric2_per_neuron_percent_depolarization": result["metric2_per_neuron_percent_depolarization"],
            "metric3_backward_forward_ratio": result["backward_forward_ratio"],
            "metric3_backward_mean_delta_mV": result["backward_mean_delta_mV"],
            "metric3_forward_mean_delta_mV": result["forward_mean_delta_mV"],
            "metric4_bilateral_symmetry": result["symmetry"],
            "raw_traces_available": True,
        }
        print(
            f"  mean percent depolarization = {perturbation_results[label]['metric2_mean_percent_depolarization']} | "
            f"ratio = {perturbation_results[label]['metric3_backward_forward_ratio']} | "
            f"mean symmetry = {perturbation_results[label]['metric4_bilateral_symmetry'].get('mean')}"
        )

    metadata = {
        "c302_version": c302_version,
        "parameters": "C0_GradedSynapse2",
        "connectivity_stage": "Witvliet_D8",
        "stimulation_amplitudes_pA": [STANDARD_AMPLITUDE_PA, LOW_AMPLITUDE_PA],
        "stimulation_duration_ms": STIM_DURATION_MS,
        "stimulation_delay_ms": STIM_DELAY_MS,
        "simulation_duration_ms": SIM_DURATION_MS,
        "requested_circuit_neurons": REQUESTED_CIRCUIT,
        "circuit_neurons": present_circuit,
        "absent_requested_neurons": absent_circuit,
        "metric2_definition": "mean percent depolarization across circuit neurons, computed as ((peak voltage - prestimulus resting voltage) / (0 - prestimulus resting voltage)) * 100",
        "metric3_definition": "mean peak depolarization above prestimulus baseline in backward command neurons divided by the same quantity in forward command neurons",
        "metric4_definition": "bilateral symmetry index computed on peak depolarization above prestimulus baseline",
    }
    benchmark_payload = {"metadata": metadata, "perturbations": perturbation_results}
    BENCHMARK_JSON.write_text(json.dumps(benchmark_payload, indent=2), encoding="utf-8")
    NEURON_INDEX_JSON.write_text(json.dumps(neuron_index, indent=2), encoding="utf-8")
    REPORT_MD.write_text(build_report(metadata, perturbation_results), encoding="utf-8-sig")

    print("=" * 72)
    print(f"Saved {BENCHMARK_JSON}")
    print(f"Saved {NEURON_INDEX_JSON}")
    for label, path in TRACE_FILES.items():
        print(f"Saved {path}")
    print(f"Saved {REPORT_MD}")
    print("=== KENTA BENCHMARK COMPLETE ===")


if __name__ == "__main__":
    main()
