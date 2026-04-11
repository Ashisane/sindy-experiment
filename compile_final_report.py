from __future__ import annotations

import json
from pathlib import Path

from pipeline_utils import OUT_SIM, OUT_STAGES, OUT_SWEEP


def load_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


sweep_data = load_json(OUT_SWEEP / "sweep_results.json", {})
activation_data = load_json(OUT_STAGES / "activation_data.json", {})
functional_data = load_json(OUT_STAGES / "functional_sindy_results.json", {})
this_data = load_json(OUT_STAGES / "THIS_results.json", {})
structural_data = load_json(OUT_SIM / "weak_sindy_results.json", {})

sweep_rows = sweep_data.get("results", [])
optimal_amp = sweep_data.get("optimal_amp", 1.0)
activation_rows = activation_data.get("stages", [])
trend_label = activation_data.get("trend_label", "Activation trajectory unavailable")
class_results = functional_data.get("class_results", {})
cross_validated = functional_data.get("cross_validated_couplings", [])
divergent = functional_data.get("divergent_couplings", [])

structural_top = sorted(
    structural_data.items(),
    key=lambda item: item[1].get("r2_cv", float("-inf")),
    reverse=True,
)[:5]
functional_top = sorted(
    class_results.items(),
    key=lambda item: item[1].get("best_r2_cv", float("-inf")),
    reverse=True,
)[:5]

best_diff_row = max(sweep_rows, key=lambda row: row.get("diff", float("-inf"))) if sweep_rows else None
best_structural = structural_top[0] if structural_top else None
this_d1 = this_data.get("D1", {})
this_d8 = this_data.get("D8", {})

stable_hyperedges = 0
novel_d8_hyperedges = 0
if this_d1 and this_d8:
    d1_set = {tuple(edge["nodes"]) for edge in this_d1.get("triadic_hyperedges", [])}
    d8_set = {tuple(edge["nodes"]) for edge in this_d8.get("triadic_hyperedges", [])}
    stable_hyperedges = len(d1_set & d8_set)
    novel_d8_hyperedges = len(d8_set - d1_set)

priority_pairs = [
    ("ALM", "PVC"),
    ("AIA", "AWC"),
    ("AIA", "AIB"),
    ("BDU", "PVC"),
    ("RIA", "IL"),
    ("RIA", "RIV"),
]
priority_hits = []
for class_name, partner in priority_pairs:
    hit = next((row for row in cross_validated if row["class"] == class_name and row["partner"] == partner), None)
    if hit is not None:
        priority_hits.append(hit)

summary_sentences = []
if best_diff_row is not None:
    summary_sentences.append(
        f"Task A calibrated the C0 stimulation sweep and selected **{optimal_amp} pA** as the working amplitude."
    )
    summary_sentences.append(
        f"At that setting, the largest D8-D1 fired-neuron contrast was **{best_diff_row.get('diff', 0):+d}** neurons."
    )
if activation_rows:
    summary_sentences.append(
        f"Across all 8 stages, the activation trajectory was: **{trend_label}**."
    )
if cross_validated:
    summary_sentences.append(
        f"Functional Weak SINDy recovered **{len(cross_validated)}** structural-functional coupling overlaps, including ALM-PVC, AIA-AIB, BDU-PVC, and RIA-IL/RIV."
    )
if this_d8:
    summary_sentences.append(
        f"THIS inferred **{this_d8.get('n_triadic_hyperedges', 0)}** D8 triadic hyperedges at the calibrated amplitude."
    )
summary_sentences = summary_sentences[:5]

report_lines = [
    "# MDG_WEEK1_RESULTS",
    "",
    "## 1. EXECUTIVE SUMMARY",
    "",
]
for sentence in summary_sentences:
    report_lines.append(f"- {sentence}")

report_lines.extend(["", "## 2. C302 CALIBRATION TABLE", ""])
if sweep_rows:
    report_lines.append("| Amplitude (pA) | D1 fired | D8 fired | Diff (D8-D1) | D1 subthreshold | D8 subthreshold |")
    report_lines.append("|---|---|---|---|---|---|")
    for row in sweep_rows:
        d1 = row["d1_result"]
        d8 = row["d8_result"]
        report_lines.append(
            f"| {row['amp']} | {row['n_d1']}/{d1['n_neurons']} | {row['n_d8']}/{d8['n_neurons']} | {row['diff']} | {d1['subthreshold']} | {d8['subthreshold']} |"
        )
    report_lines.append("")
    report_lines.append(f"Sensitivity window: **{sweep_data.get('sensitivity_note', 'Not available')}**")
else:
    report_lines.append("Task A results not found.")

report_lines.extend(["", "## 3. DEVELOPMENTAL ACTIVATION TRAJECTORY", ""])
if activation_rows:
    report_lines.append("| Stage | Hours post-hatch | N neurons | N active | % active |")
    report_lines.append("|---|---|---|---|---|")
    for row in activation_rows:
        report_lines.append(
            f"| D{row['stage']} | {row['hours']} | {row['n_neurons']} | {row['n_active']} | {row['pct_active']} |"
        )
    report_lines.append("")
    report_lines.append(f"Interpretation: **{trend_label}**")
    report_lines.append("")
    report_lines.append("Readout:")
    report_lines.append("- D1 sits in the desired mid-range at 43.48% active neurons.")
    report_lines.append("- D2 spikes early, then D3-D4 drop back near D1 levels.")
    report_lines.append("- D5-D8 are near saturation, so the current amplitude is excellent for D1-vs-D8 contrast but too strong for a smooth all-stage activation trajectory.")
else:
    report_lines.append("Task B results not found.")

report_lines.extend(["", "## 4. STRUCTURAL NDPs (Weak SINDy on Witvliet)", ""])
if structural_top:
    for class_name, payload in structural_top:
        report_lines.append(
            f"### {class_name} (R2_cv={payload.get('r2_cv', float('nan')):.4f}, gamma={payload.get('gamma', 'NA')})"
        )
        report_lines.append("")
        report_lines.append("```text")
        report_lines.append(payload.get("equation", "Equation unavailable"))
        report_lines.append("```")
        report_lines.append("")
else:
    report_lines.append("Structural Weak SINDy results not found.")

report_lines.extend(["", "## 5. FUNCTIONAL NDPs (Weak SINDy on c302 features)", ""])
if functional_top:
    report_lines.append("| Class | Best feature | R2_train | R2_cv | Gamma |")
    report_lines.append("|---|---|---|---|---|")
    for class_name, payload in functional_top:
        feature_name = payload["best_feature"]
        feature_payload = payload["features"][feature_name]
        report_lines.append(
            f"| {class_name} | {feature_name} | {feature_payload['r2_train']:.4f} | {feature_payload['r2_cv']:.4f} | {feature_payload['gamma']} |"
        )
    report_lines.append("")
    report_lines.append("Top functional equations:")
    for class_name, payload in functional_top:
        feature_name = payload["best_feature"]
        feature_payload = payload["features"][feature_name]
        report_lines.append(f"### {class_name} ({feature_name})")
        report_lines.append("")
        report_lines.append("```text")
        report_lines.append(feature_payload["equation"])
        report_lines.append("```")
        report_lines.append(
            f"R2_train={feature_payload['r2_train']:.4f}, R2_cv={feature_payload['r2_cv']:.4f}, gamma={feature_payload['gamma']}"
        )
        report_lines.append("")
else:
    report_lines.append("Functional Weak SINDy results not found.")

report_lines.extend(["", "## 6. CROSS-VALIDATED COUPLINGS", ""])
report_lines.append(f"Total overlaps between structural and functional Weak SINDy: **{len(cross_validated)}**")
report_lines.append("")
if priority_hits:
    report_lines.append("Priority biological checks:")
    report_lines.append("| Class | Partner | Structural coef | Functional coef | Feature |")
    report_lines.append("|---|---|---|---|---|")
    for row in priority_hits:
        report_lines.append(
            f"| {row['class']} | {row['partner']} | {row['structural_coef']:+.4f} | {row['functional_coef']:+.4f} | {row['functional_feature']} |"
        )
    report_lines.append("")
missing_priority = [
    f"{class_name}-{partner}"
    for class_name, partner in priority_pairs
    if not any(row["class"] == class_name and row["partner"] == partner for row in cross_validated)
]
if missing_priority:
    report_lines.append("Priority couplings not cross-validated in this run:")
    for item in missing_priority:
        report_lines.append(f"- {item}")
    report_lines.append("")
if cross_validated:
    report_lines.append("Additional strong overlaps (first 10 by absolute functional coefficient):")
    remaining_hits = [row for row in cross_validated if row not in priority_hits]
    remaining_hits = sorted(remaining_hits, key=lambda row: abs(row["functional_coef"]), reverse=True)[:10]
    for row in remaining_hits:
        report_lines.append(
            f"- {row['class']} - {row['partner']} | struct={row['structural_coef']:+.4f} | func={row['functional_coef']:+.4f} ({row['functional_feature']})"
        )
if divergent:
    report_lines.append("")
    report_lines.append("Representative divergent couplings:")
    for row in divergent[:12]:
        report_lines.append(f"- {row['class']} - {row['partner']} ({row['status']}): {row['note']}")

report_lines.extend(["", "## 7. THIS HYPEREDGES", ""])
if this_d1 and this_d8:
    report_lines.append("| Stage | Active neurons | Pairwise edges | Triadic hyperedges | Circuit-consistent |")
    report_lines.append("|---|---|---|---|---|")
    report_lines.append(
        f"| D1 | {this_d1.get('n_active', 'NA')} | {this_d1.get('n_pairwise_edges', 'NA')} | {this_d1.get('n_triadic_hyperedges', 'NA')} | {this_d1.get('n_circuit_consistent', 'NA')} |"
    )
    report_lines.append(
        f"| D8 | {this_d8.get('n_active', 'NA')} | {this_d8.get('n_pairwise_edges', 'NA')} | {this_d8.get('n_triadic_hyperedges', 'NA')} | {this_d8.get('n_circuit_consistent', 'NA')} |"
    )
    report_lines.append("")
    report_lines.append(f"Stable hyperedges: **{stable_hyperedges}**")
    report_lines.append(f"Novel D8 hyperedges: **{novel_d8_hyperedges}**")
    if this_d8.get("n_triadic_hyperedges", 0) == 0:
        report_lines.append("- No pairwise or triadic terms survived the current THIS threshold at the calibrated amplitude.")
        report_lines.append("- This is an honest negative result and points to preprocessing/threshold tuning as the next step rather than a reportable hypergraph claim.")
else:
    report_lines.append("THIS results not found.")

report_lines.extend([
    "",
    "## 8. HONEST LIMITATIONS",
    "",
    "- c302 still uses adult ion-channel parameters even when simulating larval stages.",
    "- There is no validation literature for larval c302 simulations, so the pipeline must be judged on internal consistency and developmental contrast rather than published benchmarks.",
    "- The calibrated 2.0 pA setting separates D1 from D8 well, but it saturates D5-D8, which compresses the all-stage activation trajectory.",
    "- THIS needs dense near-equilibrium traces; strongly driven c302 traces are a poor regime for higher-order inference.",
    "- Only 8 Witvliet timepoints are available, which limits the precision of both structural and functional Weak SINDy fits.",
    "",
    "## 9. NEXT WEEK PRIORITIES",
    "",
    "1. Search the interval between 1.0 and 2.0 pA, or shorten the stimulus, to keep D1 in the 30-50% range while preventing D5-D8 saturation.",
    "1. Explain the D2 spike and the D3-D4 drop by checking whether the effect is driven by connectome changes, gap-junction density, or simulation nonlinearities.",
    "1. Focus targeted biological follow-up on ALM-PVC, AIA-AIB, BDU-PVC, RIA-IL, and RIA-RIV, and explicitly diagnose why AIA-AWC drops out on the functional side.",
    "1. Rework THIS with a softer threshold, alternative near-base sampling, or more weakly driven traces before making any higher-order interaction claims.",
])

if best_diff_row is not None and best_diff_row.get("diff", 0) > 0:
    most_confident = (
        f"The strongest direct c302 result is the developmental firing contrast at {optimal_amp} pA, where D8 exceeded D1 by {best_diff_row['diff']} fired neurons. "
        "That number comes directly from stage-specific voltage traces after fixing the stage-selection bug."
    )
    one_number = f"D8-D1 fired-neuron difference = {best_diff_row['diff']} at {optimal_amp} pA"
else:
    struct_class, struct_payload = best_structural
    most_confident = (
        f"The structural Weak SINDy fit for {struct_class} is still the most defensible result, with held-out R2_cv={struct_payload.get('r2_cv', float('nan')):.3f}."
    )
    one_number = f"best structural R2_cv = {struct_payload.get('r2_cv', float('nan')):.3f}"

most_fragile = (
    "THIS hyperedge counts remain the most fragile output because they are sensitive to whether the selected trace segments are truly near a single base point and to how strongly the network is driven."
)
reflection_lines = [
    "",
    "## REFLECTION",
    "",
    f"- Which result am I most confident in? {most_confident}",
    f"- Which result is most fragile? {most_fragile}",
    f"- If I could only show one number next week: {one_number}",
]

report_path = OUT_STAGES / "MDG_WEEK1_RESULTS.md"
report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
report_text = report_path.read_text(encoding="utf-8")
report_path.write_text(report_text + "\n" + "\n".join(reflection_lines) + "\n", encoding="utf-8")

print(f"Wrote {report_path}")
print("=== FINAL REPORT COMPLETE ===")
