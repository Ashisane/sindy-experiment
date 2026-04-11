from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pipeline_utils import OUT_SIM, OUT_STAGES, T_BIO_HOURS, neuron_to_class


FEATURE_NAMES = ["max_voltage", "mean_voltage", "time_above_threshold"]
DELTA_T = np.diff(T_BIO_HOURS)
T_MID = (T_BIO_HOURS[:-1] + T_BIO_HOURS[1:]) / 2.0
T_NORM = T_MID / 120.0
T_NORM2 = T_NORM ** 2
THRESHOLD = 0.05
ALPHA = 0.01
MAX_ITER = 100
TRAIN_TRANSITIONS = 5
TARGET_CLASSES = {
    "ALM": ["PVC"],
    "AIA": ["AWC", "AIB"],
    "BDU": ["PVC"],
    "RIA": ["IL", "RIV"],
}


def ridge_solve(matrix: np.ndarray, target: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    lhs = matrix.T @ matrix + alpha * np.eye(matrix.shape[1])
    rhs = matrix.T @ target
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def stlsq(theta: np.ndarray, target: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
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


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return 1.0 if ss_res <= 1e-12 else 0.0
    return 1.0 - ss_res / ss_tot


def scale_for_normalization(values: np.ndarray) -> float:
    scale = float(np.max(np.abs(values)))
    return scale if scale > 1e-8 else 1.0


print("=" * 72)
print("TASK C: WEAK SINDY ON FUNCTIONAL FEATURES")
print("=" * 72)

class_members = json.loads((OUT_SIM / "class_members.json").read_text(encoding="utf-8"))
class_names = [line.strip() for line in (OUT_SIM / "class_names.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
class_index = {name: idx for idx, name in enumerate(class_names)}
A_class = np.load(OUT_SIM / "A_class.npy")
structural_results = json.loads((OUT_SIM / "weak_sindy_results.json").read_text(encoding="utf-8"))

stage_orders: list[list[str]] = []
stage_features: list[np.ndarray] = []
for stage in range(1, 9):
    order_path = OUT_STAGES / f"neuron_order_D{stage}.txt"
    feature_path = OUT_STAGES / f"features_D{stage}.npy"
    if not order_path.exists() or not feature_path.exists():
        raise FileNotFoundError(f"Task B output missing for D{stage}")
    stage_orders.append([line.strip() for line in order_path.read_text(encoding="utf-8").splitlines() if line.strip()])
    stage_features.append(np.load(feature_path))

common_neurons = sorted(set.intersection(*(set(order) for order in stage_orders)))
common_index = {name: idx for idx, name in enumerate(common_neurons)}
print(f"Common neurons across all 8 stages: {len(common_neurons)}")

feature_tensor = np.full((len(common_neurons), 8, 4), np.nan, dtype=float)
for stage_idx, (order, features) in enumerate(zip(stage_orders, stage_features)):
    for neuron_idx, neuron_name in enumerate(order):
        common_idx = common_index.get(neuron_name)
        if common_idx is not None:
            feature_tensor[common_idx, stage_idx, :] = features[neuron_idx, :]

if np.isnan(feature_tensor).any():
    raise RuntimeError("Functional tensor contains missing values after common-neuron intersection")

class_to_common_members: dict[str, list[str]] = {}
for class_name, members in class_members.items():
    overlap = [member for member in members if member in common_index]
    if overlap:
        class_to_common_members[class_name] = overlap

class_feature_tensor = np.zeros((len(class_names), 8, len(FEATURE_NAMES)), dtype=float)
for class_name, members in class_to_common_members.items():
    idx = class_index[class_name]
    member_indices = [common_index[name] for name in members]
    class_feature_tensor[idx] = feature_tensor[member_indices, :, : len(FEATURE_NAMES)].mean(axis=0)

functional_results: dict[str, Any] = {}

for class_name in class_names:
    members = class_to_common_members.get(class_name, [])
    if not members:
        continue
    class_idx = class_index[class_name]
    connected_classes = [
        class_names[other_idx]
        for other_idx in range(len(class_names))
        if other_idx != class_idx and A_class[class_idx, other_idx] > 0
    ]

    class_payload: dict[str, Any] = {
        "n_members_common": len(members),
        "features": {},
    }
    member_indices = [common_index[name] for name in members]

    for feature_idx, feature_name in enumerate(FEATURE_NAMES):
        class_values = feature_tensor[member_indices, :, feature_idx]
        scale = scale_for_normalization(class_values)
        normalized = class_values / scale
        effective_derivative = np.diff(normalized, axis=1) / DELTA_T[np.newaxis, :]
        target_vector = effective_derivative.reshape(-1)

        library_columns = [np.ones(7), T_NORM, T_NORM2]
        library_names = ["const", "t", "t2"]
        for partner_class in connected_classes:
            partner_idx = class_index[partner_class]
            partner_traj = class_feature_tensor[partner_idx, :, feature_idx]
            partner_scale = scale_for_normalization(partner_traj)
            midpoint = ((partner_traj / partner_scale)[:-1] + (partner_traj / partner_scale)[1:]) / 2.0
            library_columns.append(midpoint)
            library_names.append(partner_class)

        theta_one = np.column_stack(library_columns)
        theta = np.tile(theta_one, (len(members), 1))
        theta_train = theta[: len(members) * TRAIN_TRANSITIONS, :]
        target_train = target_vector[: len(members) * TRAIN_TRANSITIONS]
        theta_test = theta[len(members) * TRAIN_TRANSITIONS :, :]
        target_test = target_vector[len(members) * TRAIN_TRANSITIONS :]

        xi = stlsq(theta_train, target_train)
        gamma = int(np.count_nonzero(np.abs(xi) >= THRESHOLD))
        surviving_terms = [
            {"term": library_names[idx], "coef": float(xi[idx])}
            for idx in range(len(library_names))
            if abs(xi[idx]) >= THRESHOLD
        ]
        equation = (
            f"d[{class_name}_{feature_name}]/dt = "
            + " + ".join(f"{term['coef']:+.4f}*[{term['term']}]" for term in surviving_terms)
            if surviving_terms
            else f"d[{class_name}_{feature_name}]/dt = 0"
        )
        class_payload["features"][feature_name] = {
            "scale": scale,
            "gamma": gamma,
            "threshold": THRESHOLD,
            "alpha": ALPHA,
            "r2_train": float(r2_score(target_train, theta_train @ xi)),
            "r2_cv": float(r2_score(target_test, theta_test @ xi)) if len(target_test) else float("nan"),
            "equation": equation,
            "surviving_terms": surviving_terms,
            "connected_classes": connected_classes,
        }

    best_feature_name, best_feature_payload = max(
        class_payload["features"].items(),
        key=lambda item: item[1]["r2_cv"],
    )
    class_payload["best_feature"] = best_feature_name
    class_payload["best_r2_cv"] = best_feature_payload["r2_cv"]
    class_payload["best_equation"] = best_feature_payload["equation"]
    functional_results[class_name] = class_payload


def structural_partner_map(class_name: str) -> dict[str, float]:
    terms = structural_results.get(class_name, {}).get("surviving_terms", [])
    return {
        term_name: float(coef)
        for term_name, coef in terms
        if term_name not in {"const", "t", "t2"}
    }


def functional_partner_map(class_name: str) -> dict[str, list[dict[str, Any]]]:
    payload = functional_results.get(class_name, {})
    feature_payloads = payload.get("features", {})
    partners: dict[str, list[dict[str, Any]]] = {}
    for feature_name, result in feature_payloads.items():
        for term in result.get("surviving_terms", []):
            partner = term["term"]
            if partner in {"const", "t", "t2"}:
                continue
            partners.setdefault(partner, []).append(
                {
                    "feature": feature_name,
                    "coef": float(term["coef"]),
                    "r2_cv": float(result["r2_cv"]),
                }
            )
    return partners


cross_validated: list[dict[str, Any]] = []
divergent: list[dict[str, Any]] = []
checked_classes = sorted(set(structural_results) & set(functional_results))

for class_name in checked_classes:
    structural_partners = structural_partner_map(class_name)
    functional_partners = functional_partner_map(class_name)
    all_partners = sorted(set(structural_partners) | set(functional_partners))
    for partner in all_partners:
        in_structural = partner in structural_partners
        in_functional = partner in functional_partners
        if in_structural and in_functional:
            best_functional_hit = max(functional_partners[partner], key=lambda item: abs(item["coef"]))
            cross_validated.append(
                {
                    "class": class_name,
                    "partner": partner,
                    "structural_coef": structural_partners[partner],
                    "functional_feature": best_functional_hit["feature"],
                    "functional_coef": best_functional_hit["coef"],
                    "functional_r2_cv": best_functional_hit["r2_cv"],
                }
            )
        elif in_structural:
            divergent.append(
                {
                    "class": class_name,
                    "partner": partner,
                    "status": "structural_only",
                    "note": "Structural growth coupling was not recovered from the stimulation-driven functional features.",
                }
            )
        else:
            best_functional_hit = max(functional_partners[partner], key=lambda item: abs(item["coef"]))
            divergent.append(
                {
                    "class": class_name,
                    "partner": partner,
                    "status": "functional_only",
                    "feature": best_functional_hit["feature"],
                    "note": "Functional response coupling appears without a matching structural weak-SINDy term.",
                }
            )

print("CROSS-VALIDATED COUPLINGS (appear in both structural AND functional NDP):")
for row in cross_validated:
    print(
        f"  {row['class']} - {row['partner']} | "
        f"struct={row['structural_coef']:+.4f} | "
        f"func={row['functional_coef']:+.4f} ({row['functional_feature']})"
    )

print("DIVERGENT COUPLINGS (structural only or functional only):")
for row in divergent:
    print(f"  {row['class']} - {row['partner']} | {row['status']} | {row['note']}")

print("Known structural checks:")
for class_name, partners in TARGET_CLASSES.items():
    func_partners = functional_partner_map(class_name)
    struct_partners = structural_partner_map(class_name)
    for partner in partners:
        print(
            f"  {class_name} - {partner}: "
            f"structural={'yes' if partner in struct_partners else 'no'}, "
            f"functional={'yes' if partner in func_partners else 'no'}"
        )

json_payload = {
    "feature_names": FEATURE_NAMES,
    "threshold": THRESHOLD,
    "alpha": ALPHA,
    "max_iter": MAX_ITER,
    "common_neurons": common_neurons,
    "class_results": functional_results,
    "cross_validated_couplings": cross_validated,
    "divergent_couplings": divergent,
}
json_path = OUT_STAGES / "functional_sindy_results.json"
json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

report_lines = [
    "# Functional Weak SINDy Cross-Validation",
    "",
    f"Common neurons across all 8 stages: **{len(common_neurons)}**",
    "",
    "## Cross-Validated Couplings",
    "",
]
if cross_validated:
    report_lines.append("| Class | Partner | Structural coef | Functional coef | Feature | Functional R2_cv |")
    report_lines.append("|---|---|---|---|---|---|")
    for row in cross_validated:
        report_lines.append(
            f"| {row['class']} | {row['partner']} | {row['structural_coef']:+.4f} | "
            f"{row['functional_coef']:+.4f} | {row['functional_feature']} | {row['functional_r2_cv']:.4f} |"
        )
else:
    report_lines.append("No cross-validated couplings were recovered.")
report_lines.extend(["", "## Divergent Couplings", ""])
if divergent:
    for row in divergent:
        report_lines.append(f"- **{row['class']} - {row['partner']}** ({row['status']}): {row['note']}")
else:
    report_lines.append("No divergent couplings were identified.")
report_lines.extend(["", "## Best Functional Equation Per Class", ""])
for class_name, payload in sorted(
    functional_results.items(),
    key=lambda item: item[1]["best_r2_cv"],
    reverse=True,
):
    best_feature = payload["best_feature"]
    best_payload = payload["features"][best_feature]
    report_lines.append(f"### {class_name} ({best_feature})")
    report_lines.append("")
    report_lines.append("```text")
    report_lines.append(best_payload["equation"])
    report_lines.append("```")
    report_lines.append(
        f"R2_train={best_payload['r2_train']:.4f}, R2_cv={best_payload['r2_cv']:.4f}, gamma={best_payload['gamma']}"
    )
    report_lines.append("")
report_path = OUT_STAGES / "cross_validation_report.md"
report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

print(f"Saved {json_path}")
print(f"Saved {report_path}")
print("=== TASK C COMPLETE ===")
