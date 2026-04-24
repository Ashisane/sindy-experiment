# KENTA Benchmark Report

## 1. Circuit setup

Stage 8 circuit neurons requested: ALML, ALMR, PVML, PVMR, PVM, AVM, AVAL, AVAR, AVDL, AVDR, AVBL, AVBR, PVCL, PVCR
Stage 8 circuit neurons present: ALML, ALMR, AVM, AVAL, AVAR, AVDL, AVDR, AVBL, AVBR, PVCL, PVCR
Stage 8 requested neurons absent: PVML, PVMR, PVM
The benchmark uses only the present stage-8 circuit neurons so the protocol remains small and exactly reproducible for KENTA.

## 2. Results table

| Metric | P1 baseline (2.0 pA) | P2 AVAL lesion (2.0 pA) | P3 unilateral touch (2.0 pA) | P1 low (0.5 pA) | P2 low (0.5 pA) | P3 low (0.5 pA) |
|---|---|---|---|---|---|---|
| Metric 1 summary: mean circuit peak voltage (mV) | -10.2369 | -10.4006 | -35.6779 | -42.6479 | -42.4565 | -43.6825 |
| Metric 2: mean percent depolarization | 79.0538 | 78.7189 | 26.9947 | 12.7325 | 13.1238 | 10.6156 |
| Metric 3: backward / forward ratio | 0.9862 | 0.9890 | 0.9023 | 1.2162 | 1.3190 | 1.0610 |
| Metric 4: mean bilateral symmetry index | 0.0219 | 0.0329 | 0.1274 | 0.0094 | 0.0141 | 0.0078 |

Full Metric 1 peak-voltage dictionaries and per-neuron Metric 2 percent-depolarization values are stored in benchmark_results.json for KENTA comparison.

## 3. P1 biological validation

P1 does not pass the anterior-touch sanity check cleanly: backward / forward = 0.9862. This suggests the result is either too parameter-sensitive or the reduced sub-circuit omits compensating structural context that exists in the full connectome.
At 0.5 pA, P1_low does pass the same check with backward / forward = 1.2162 > 1, which makes the lower-amplitude regime more biologically plausible for KENTA comparison.

## 4. P2 lesion effect

The AVAL lesion changes Metric 3 by 0.0028 relative to baseline.
AVAR peak voltage changes by 0.0000 mV relative to baseline.
At 0.5 pA, Metric 3 changes from 1.2162 in P1_low to 1.3190 in P2_low.
AVAR does not increase after the lesion, so the model does not show clear compensatory recruitment under this reduced-circuit benchmark.

## 5. P3 asymmetry

Mean bilateral symmetry index changes by 0.1055 from P1 to P3.
At 0.5 pA, mean bilateral symmetry index changes by -0.0016 from P1_low to P3_low.
The strongest bilateral asymmetry in P3 is PVCL_PVCR = 0.3823.
At 0.5 pA, the strongest bilateral asymmetry in P3_low is PVCL_PVCR = 0.0233.
P3 is more asymmetric than P1, so unilateral sensory drive is preserved functionally in the benchmark output.
At 0.5 pA, unilateral touch does not increase asymmetry over bilateral touch, so the lower-amplitude regime improves the command ratio but weakens the lateralization signature.

## 6. Known limitations for KENTA comparison

- c302 uses adult HH-derived parameters adapted for graded synapses.
- Witvliet D8 is a population-average connectome with substantial inter-individual variability.
- Gap junctions are included, but the benchmark does not distinguish their uncertainty from chemical synapse uncertainty.
- c302 does not model neuropeptide modulation.
- The stimulus is artificial current injection, not mechanosensory transduction.
- Activation depth metric (Metric 2) should be interpreted with caution in graded-synapse models — the -50 mV threshold is reached by most neurons even at low stimulation amplitudes. The KENTA collaborator should compare raw peak voltages (Metric 1) and bilateral symmetry (Metric 4) as the primary comparison targets.

## 7. Scientific interpretation for the collaboration

If KENTA matches the c302 benchmark across the baseline, lesion, and unilateral-touch perturbations, that would argue that the core functional signatures of this circuit are imposed mainly by the adult stage-8 wiring pattern rather than by the detailed c302 membrane equations. If KENTA and c302 disagree, the mismatch becomes informative: it would identify which signatures are structurally robust and which depend strongly on continuous-time conductance dynamics, synaptic parameterization, or how graded transmission is implemented. That separation is exactly what makes this a useful collaboration benchmark rather than just another simulation run.
