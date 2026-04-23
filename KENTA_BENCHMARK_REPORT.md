# KENTA Benchmark Report

## 1. Circuit setup

Stage 8 circuit neurons requested: ALML, ALMR, PVML, PVMR, PVM, AVM, AVAL, AVAR, AVDL, AVDR, AVBL, AVBR, PVCL, PVCR
Stage 8 circuit neurons present: ALML, ALMR, AVM, AVAL, AVAR, AVDL, AVDR, AVBL, AVBR, PVCL, PVCR
Stage 8 requested neurons absent: PVML, PVMR, PVM
The benchmark uses only the present stage-8 circuit neurons so the protocol remains small and exactly reproducible for KENTA.

## 2. Results table

| Metric | P1 baseline | P2 AVAL lesion | P3 unilateral touch |
|---|---|---|---|
| Metric 1 summary: mean circuit peak voltage (mV) | -10.2369 | -10.4006 | -35.6779 |
| Metric 2: activation depth | 1.0000 (11/11) | 1.0000 (10/10) | 1.0000 (11/11) |
| Metric 3: backward / forward ratio | 0.9862 | 0.9890 | 0.9023 |
| Metric 4: mean bilateral symmetry index | 0.0219 | 0.0329 | 0.1274 |

Full Metric 1 peak-voltage dictionaries are stored in benchmark_results.json for KENTA comparison.

## 3. P1 biological validation

P1 does not pass the anterior-touch sanity check cleanly: backward / forward = 0.9862. This suggests the result is either too parameter-sensitive or the reduced sub-circuit omits compensating structural context that exists in the full connectome.

## 4. P2 lesion effect

The AVAL lesion changes Metric 3 by 0.0028 relative to baseline.
AVAR peak voltage changes by 0.0000 mV relative to baseline.
AVAR does not increase after the lesion, so the model does not show clear compensatory recruitment under this reduced-circuit benchmark.

## 5. P3 asymmetry

Mean bilateral symmetry index changes by 0.1055 from P1 to P3.
The strongest bilateral asymmetry in P3 is PVCL_PVCR = 0.3823.
P3 is more asymmetric than P1, so unilateral sensory drive is preserved functionally in the benchmark output.

## 6. Known limitations for KENTA comparison

- c302 uses adult HH-derived parameters adapted for graded synapses.
- Witvliet D8 is a population-average connectome with substantial inter-individual variability.
- Gap junctions are included, but the benchmark does not distinguish their uncertainty from chemical synapse uncertainty.
- c302 does not model neuropeptide modulation.
- The stimulus is artificial current injection, not mechanosensory transduction.

## 7. Scientific interpretation for the collaboration

If KENTA matches the c302 benchmark across the baseline, lesion, and unilateral-touch perturbations, that would argue that the core functional signatures of this circuit are imposed mainly by the adult stage-8 wiring pattern rather than by the detailed c302 membrane equations. If KENTA and c302 disagree, the mismatch becomes informative: it would identify which signatures are structurally robust and which depend strongly on continuous-time conductance dynamics, synaptic parameterization, or how graded transmission is implemented. That separation is exactly what makes this a useful collaboration benchmark rather than just another simulation run.
