from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pipeline_utils import OUT_C0, OUT_SIM, OUT_STAGES, locate_trace_file, neuron_to_class, parse_output_columns


THIS_THRESHOLD = 0.10
ALPHA = 0.01
MAX_ITER = 100
N_NEAR_BASE = 1000
MODULES = {
    "locomotion": {"AVBL", "AVBR", "AVAL", "AVAR", "PVCL", "PVCR", "DB1", "DB2", "DB3", "DB4", "DB5", "DB6", "DB7", "VB1", "VB2", "VB3", "VB4", "VB5", "VB6", "VB7", "VB8", "VB9", "VB10", "VB11"},
    "mechanosensory": {"ALML", "ALMR", "PVDL", "PVDR", "PVCL", "PVCR"},
    "chemosensory": {"AWCL", "AWCR", "ASEL", "ASER", "AIAL", "AIAR", "AIBL", "AIBR"},
}


def ridge_solve(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    lhs = matrix.T @ matrix + ALPHA * np.eye(matrix.shape[1])
    rhs = matrix.T @ target
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def stlsq(theta: np.ndarray, target: np.ndarray, threshold: float = THIS_THRESHOLD) -> np.ndarray:
    xi = ridge_solve(theta, target)
    for _ in range(MAX_ITER):
        active = np.abs(xi) >= threshold
        if not active.any():
            return np.zeros(theta.shape[1])
        xi_new = np.zeros(theta.shape[1])
        xi_new[active] = ridge_solve(theta[:, active], target)
        xi_new[np.abs(xi_new) < threshold] = 0.0
        if np.array_equal(xi_new != 0.0, xi != 0.0):
            return xi_new
        xi = xi_new
    return xi


def load_activation_data() -> dict[str, Any]:
    activation_path = OUT_STAGES / "activation_data.json"
    if activation_path.exists():
        return json.loads(activation_path.read_text(encoding="utf-8"))
    return {}


def stage_trace_paths(stage: int) -> tuple[Path | None, Path | None]:
    activation_data = load_activation_data()
    for row in activation_data.get("stages", []):
        if int(row.get("stage", -1)) == stage and row.get("dat_path"):
            dat_path = Path(row["dat_path"])
            lems_path = Path(row["lems_path"]) if row.get("lems_path") else None
            if dat_path.exists():
                return dat_path, lems_path

    stage_label = f"D{stage}"
    dat_path = locate_trace_file(
        OUT_STAGES / stage_label / f"Stage_{stage_label}_{activation_data.get('optimal_amp', 1.0)}pA.dat",
        OUT_C0 / f"MDG_C0_{stage_label}.dat",
        OUT_STAGES / stage_label / f"Stage_{stage_label}_1p0pA.dat",
        Path(__file__).resolve().parent / f"MDG_C0_{stage_label}.dat",
    )
    lems_path = None
    if dat_path is not None:
        candidate_lems = dat_path.parent / f"LEMS_{dat_path.stem}.xml"
        if candidate_lems.exists():
            lems_path = candidate_lems
    return dat_path, lems_path


def build_neuron_order(stage: int, dat_path: Path, lems_path: Path | None) -> list[str]:
    if lems_path is not None and lems_path.exists():
        neuron_order = parse_output_columns(lems_path, dat_path.name)
        if neuron_order:
            return neuron_order
    root_order = OUT_STAGES / f"neuron_order_D{stage}.txt"
    if root_order.exists():
        return [line.strip() for line in root_order.read_text(encoding="utf-8").splitlines() if line.strip()]
    raise FileNotFoundError(f"Neuron order unavailable for D{stage}")


def connected_pair(neuron_a: str, neuron_b: str, class_index: dict[str, int], adjacency: np.ndarray) -> bool:
    class_a = class_index.get(neuron_to_class(neuron_a), -1)
    class_b = class_index.get(neuron_to_class(neuron_b), -1)
    if class_a < 0 or class_b < 0:
        return False
    return bool(adjacency[class_a, class_b] > 0 or adjacency[class_b, class_a] > 0)


def consistent_module(neurons: tuple[str, str, str]) -> str | None:
    triple = set(neurons)
    for module_name, module_members in MODULES.items():
        if triple.issubset(module_members):
            return module_name
    return None


def run_this(stage: int, dat_path: Path, neuron_order: list[str], class_index: dict[str, int], adjacency: np.ndarray) -> dict[str, Any]:
    raw = np.loadtxt(dat_path)
    time_s = raw[:, 0]
    voltage_mv = raw[:, 1:] * 1000.0
    dt_ms = float((time_s[1] - time_s[0]) * 1000.0)
    neuron_order = neuron_order[: voltage_mv.shape[1]]

    vmax = voltage_mv.max(axis=0)
    active_indices = np.where(vmax > -20.0)[0]
    active_neurons = [neuron_order[idx] for idx in active_indices]
    active_voltage = voltage_mv[:, active_indices]
    if len(active_neurons) < 3:
        return {
            "stage": f"D{stage}",
            "dat_path": str(dat_path),
            "active_neurons": active_neurons,
            "n_active": len(active_neurons),
            "n_pairwise_edges": 0,
            "n_triadic_hyperedges": 0,
            "pairwise_edges": [],
            "triadic_hyperedges": [],
            "stable_vs_other": {},
            "incidence_neurons": active_neurons,
            "incidence_shape": [len(active_neurons), 0],
        }

    base_point = np.median(active_voltage, axis=0)
    deviations = active_voltage - base_point
    std = deviations.std(axis=0)
    std[std < 1e-8] = 1.0
    normalized = deviations / std
    distances = np.linalg.norm(deviations, axis=1)
    near_indices = np.sort(np.argsort(distances)[: min(N_NEAR_BASE, len(distances))])
    interior = near_indices[(near_indices > 0) & (near_indices < len(time_s) - 1)]
    sampled = normalized[interior, :]
    derivatives = (normalized[interior + 1, :] - normalized[interior - 1, :]) / (2.0 * dt_ms)

    library_columns = [np.ones(len(interior))]
    library_terms = [("const", None)]
    for neuron_name, column in zip(active_neurons, sampled.T):
        library_columns.append(column)
        library_terms.append(("linear", neuron_name))

    quadratic_pairs: list[tuple[int, int]] = []
    for idx_a in range(len(active_neurons)):
        for idx_b in range(idx_a + 1, len(active_neurons)):
            if connected_pair(active_neurons[idx_a], active_neurons[idx_b], class_index, adjacency):
                quadratic_pairs.append((idx_a, idx_b))
                library_columns.append(sampled[:, idx_a] * sampled[:, idx_b])
                library_terms.append(("quadratic", (active_neurons[idx_a], active_neurons[idx_b])))

    theta = np.column_stack(library_columns)
    coefficients = np.zeros((len(active_neurons), theta.shape[1]), dtype=float)
    for target_idx in range(len(active_neurons)):
        coefficients[target_idx] = stlsq(theta, derivatives[:, target_idx])

    pairwise_edges: list[dict[str, Any]] = []
    triadic_hyperedges: list[dict[str, Any]] = []
    for target_idx, target_name in enumerate(active_neurons):
        for term_idx, term in enumerate(library_terms[1:], start=1):
            coef = float(coefficients[target_idx, term_idx])
            if abs(coef) < THIS_THRESHOLD:
                continue
            if term[0] == "linear":
                source_name = term[1]
                if source_name != target_name:
                    pairwise_edges.append({"source": source_name, "target": target_name, "coef": coef})
            else:
                src_a, src_b = term[1]
                hyperedge_nodes = tuple(sorted((target_name, src_a, src_b)))
                triadic_hyperedges.append(
                    {
                        "nodes": hyperedge_nodes,
                        "target": target_name,
                        "coef": coef,
                        "module": consistent_module(hyperedge_nodes) or "random",
                    }
                )

    unique_hyperedges: dict[tuple[str, str, str], dict[str, Any]] = {}
    for edge in triadic_hyperedges:
        key = edge["nodes"]
        existing = unique_hyperedges.get(key)
        if existing is None or abs(edge["coef"]) > abs(existing["coef"]):
            unique_hyperedges[key] = edge

    sorted_hyperedges = sorted(unique_hyperedges.values(), key=lambda item: abs(item["coef"]), reverse=True)
    incidence = np.zeros((len(active_neurons), len(sorted_hyperedges)), dtype=np.int8)
    neuron_index = {name: idx for idx, name in enumerate(active_neurons)}
    for column_idx, edge in enumerate(sorted_hyperedges):
        for neuron_name in edge["nodes"]:
            incidence[neuron_index[neuron_name], column_idx] = 1

    circuit_consistent = sum(1 for edge in sorted_hyperedges if edge["module"] != "random")
    return {
        "stage": f"D{stage}",
        "dat_path": str(dat_path),
        "active_neurons": active_neurons,
        "n_active": len(active_neurons),
        "n_pairwise_edges": len(pairwise_edges),
        "n_triadic_hyperedges": len(sorted_hyperedges),
        "n_circuit_consistent": circuit_consistent,
        "pairwise_edges": pairwise_edges,
        "triadic_hyperedges": sorted_hyperedges,
        "incidence_matrix": incidence,
        "incidence_neurons": active_neurons,
        "incidence_shape": list(incidence.shape),
        "quadratic_pair_count": len(quadratic_pairs),
    }


print("=" * 72)
print("TASK D: THIS HYPEREDGE INFERENCE")
print("=" * 72)

class_names = [line.strip() for line in (OUT_SIM / "class_names.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
class_index = {name: idx for idx, name in enumerate(class_names)}
adjacency = np.load(OUT_SIM / "A_class.npy")

results: dict[int, dict[str, Any]] = {}
for stage in (1, 8):
    dat_path, lems_path = stage_trace_paths(stage)
    if dat_path is None:
        raise FileNotFoundError(f"No voltage trace found for D{stage}")
    neuron_order = build_neuron_order(stage, dat_path, lems_path)
    print(f"Running THIS on D{stage}: {dat_path}")
    result = run_this(stage, dat_path, neuron_order, class_index, adjacency)
    results[stage] = result
    incidence_path = OUT_STAGES / f"THIS_D{stage}_incidence.npy"
    np.save(incidence_path, result["incidence_matrix"])
    (OUT_STAGES / f"THIS_D{stage}_active_neurons.txt").write_text(
        "\n".join(result["incidence_neurons"]) + "\n",
        encoding="utf-8",
    )
    print(
        f"  D{stage}: active={result['n_active']} | pairwise={result['n_pairwise_edges']} | "
        f"triadic={result['n_triadic_hyperedges']} | circuit-consistent={result['n_circuit_consistent']}"
    )

hyperedges_d1 = {tuple(edge["nodes"]) for edge in results[1]["triadic_hyperedges"]}
hyperedges_d8 = {tuple(edge["nodes"]) for edge in results[8]["triadic_hyperedges"]}
stable = sorted(hyperedges_d1 & hyperedges_d8)
novel_d8 = sorted(hyperedges_d8 - hyperedges_d1)

report_lines = [
    "# THIS Hyperedge Inference",
    "",
    "## D1 vs D8 summary",
    "",
    "| Stage | Active neurons | Pairwise edges | Triadic hyperedges | Circuit-consistent |",
    "|---|---|---|---|---|",
]
for stage in (1, 8):
    result = results[stage]
    report_lines.append(
        f"| D{stage} | {result['n_active']} | {result['n_pairwise_edges']} | {result['n_triadic_hyperedges']} | {result['n_circuit_consistent']} |"
    )
report_lines.extend(
    [
        "",
        f"Stable hyperedges across D1 and D8: **{len(stable)}**",
        f"Novel D8 hyperedges: **{len(novel_d8)}**",
        "",
        "## Top D8 triadic hyperedges",
        "",
        "| Nodes | Coefficient | Module flag |",
        "|---|---|---|",
    ]
)
for edge in results[8]["triadic_hyperedges"][:20]:
    nodes = ", ".join(edge["nodes"])
    report_lines.append(f"| {nodes} | {edge['coef']:+.4f} | {edge['module']} |")
report_lines.extend(["", "## Stable hyperedges", ""])
if stable:
    for nodes in stable[:20]:
        report_lines.append(f"- {', '.join(nodes)}")
else:
    report_lines.append("No stable hyperedges identified.")
report_lines.extend(["", "## Novel adult D8 hyperedges", ""])
if novel_d8:
    for nodes in novel_d8[:20]:
        report_lines.append(f"- {', '.join(nodes)}")
else:
    report_lines.append("No adult-specific D8 hyperedges identified.")
report_path = OUT_STAGES / "THIS_report.md"
report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

serializable_results = {}
for stage, result in results.items():
    serializable = {key: value for key, value in result.items() if key != "incidence_matrix"}
    serializable_results[f"D{stage}"] = serializable
(OUT_STAGES / "THIS_results.json").write_text(json.dumps(serializable_results, indent=2), encoding="utf-8")

print(f"Stable hyperedges: {len(stable)}")
print(f"Novel D8 hyperedges: {len(novel_d8)}")
print(f"Saved {report_path}")
print("=== TASK D COMPLETE ===")
