# FINAL FF4 Sequential Architecture Ablation N10 — Summary

- Status: **COMPLETE**
- Version: `FINAL_FF4_SEQUENTIAL_ARCHITECTURE_ABLATION_N10_v1`
- Scientific status: `POST_HOC_FF4_CLOSURE; PLAN_AND_SEEDS_FIXED_BEFORE_EXECUTION`
- CAL/FINAL accessed: **No / No**
- Cascade recomputed: **No**
- Representation: **RUT**, fixed URL-/Text-DAPT checkpoints; fixed DOM-SSL encoder.
- Training: 20,000 balanced SSL labels per replicate, N10 paired, 1 epoch, fixed batch 8.
- DEV protocol: threshold on ENG_TUNE negatives at target FPR 0.5%; endpoint on disjoint ENG_META.

## Variant aggregates

| Variant | Mean TPR | Mean FPR | Mean AP | Mean FPR@TPR90 |
|---|---:|---:|---:|---:|
| P0_URL | 87.932% | 0.400% | 0.995619 | 0.528% |
| P1_DUAL_EQUAL | 93.116% | 0.488% | 0.996275 | 0.316% |
| P2_TRI_EQUAL | 93.844% | 0.568% | 0.997269 | 0.296% |
| P3_TRI_GATED | 94.420% | 0.616% | 0.997477 | 0.260% |

## Pre-specified adjacent primary contrasts

| Contrast | ΔTPR mean | 95% CI | Positive pairs | exact p | Holm p | ΔFPR mean |
|---|---:|---:|---:|---:|---:|---:|
| TEXT_ADD: P1_DUAL_EQUAL − P0_URL | +5.184 pp | [+4.324; +6.044] pp | 10/10 | 0.00195312 | 0.00585938 | +0.088 pp |
| DOM_ADD: P2_TRI_EQUAL − P1_DUAL_EQUAL | +0.728 pp | [+0.305; +1.151] pp | 10/10 | 0.00195312 | 0.00585938 | +0.080 pp |
| GATING_ADD: P3_TRI_GATED − P2_TRI_EQUAL | +0.576 pp | [+0.226; +0.926] pp | 9/10 | 0.00390625 | 0.00585938 | +0.048 pp |

## Interpretation boundary

This is a post-hoc FF4 closure analysis after the final architecture was already known. The plan and fresh seeds were fixed before this execution. The results support or fail to support the incremental component claims; they are not presented as a retroactive architecture-selection procedure.
The separate frozen Full-TRI → URL-first Cascade evaluation remains the evidence for the final inference-resource trade-off.