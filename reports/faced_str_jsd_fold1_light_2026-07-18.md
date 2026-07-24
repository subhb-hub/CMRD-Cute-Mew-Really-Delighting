# FACED STR-JSD Early-State Temporal Reference — fold-1 lightweight validation

Date: 2026-07-18  
Status: complete exploratory single-fold experiment  
Protocol hash: `83c4615d7021fb55`  
Seed: 42  
Tasks: 20/20 complete  

## Answer first

The registered five-second STR-JSD proposal is **not validated as a better zero-adaptation cross-subject emotion recognizer** in this experiment.

- Under the conventional subject holdout, the primary C4 representation reached 27.68% accuracy / 26.52% Macro-F1, versus 28.57% / 25.45% for capacity-matched absolute-DE C0. The paired subject-bootstrap accuracy difference was -0.89 percentage points (95% CI -3.87 to +2.08); the Macro-F1 difference was +1.08 points (CI -2.61 to +4.66).
- Under the stronger new-subject + unseen-video holdout, C4 reached 12.96% accuracy / 11.81% Macro-F1, versus 13.89% / 10.03% for C0. Accuracy was essentially at the nine-class chance level (11.11%), and neither paired interval excluded no difference.
- Signed pointwise JSD did not reliably beat unsigned pointwise JSD. The C3-minus-C2 paired intervals crossed zero under both protocols.
- The fixed early-reference quality gate C6 did not improve C4. It reduced conventional accuracy from 27.68% to 25.89%; under the strict split it changed 12.96% to 13.89%, again with intervals spanning zero.
- Five-second pointwise log-ratio C5 was worse than C4. In the conventional split, C4 exceeded C5 by 4.46 accuracy points (95% CI 0.00 to 8.63) and 4.77 Macro-F1 points (CI 0.08 to 9.42). The strict comparison remained uncertain.

One diagnostic result is worth carrying forward, but it is not yet confirmatory: the **one-second C5 control** reached 14.81% accuracy / 14.81% Macro-F1 under the strict split, compared with 13.89% / 7.75% for one-second C0. The paired Macro-F1 difference was +7.06 points (95% CI +1.39 to +12.52), while the accuracy interval still crossed zero. Because this branch was identified after looking at fold-1 target results, it must be frozen and evaluated on untouched folds/seeds before it can support an efficacy claim.

## What was tested

Each official FACED processed trial contains the final 30 one-second windows of one video. The first five windows of that same trial and subject form an **Early-State Temporal Reference**. They are not fixation, rest, or a neutral physiological baseline. Response states begin at window 6.

The primary temporal setting averages five consecutive response PSD windows, giving 21 response states (starts 6 through 26 in one-based indexing). A one-second control retains all 25 later response windows.

Native physical frequency grids were preserved:

| Band | Native frequency points |
|---|---:|
| delta | 3 |
| theta | 4 |
| alpha | 6 |
| beta | 16 |
| gamma | 17 |

The existing cache stores normalized within-band native PSD shape and a separate Gaussian differential-entropy feature. A relative band-energy scale was reconstructed as `exp(2 * DE)` and multiplied by the native PSD shape. The common factor cancels in the band normalization and energy ratios. This produces an energy-calibrated native PSD proxy, but it combines a Butterworth/variance DE estimator with a Welch/Hann spectral-shape estimator; exact raw Welch power was not retained in the old cache and this estimator mismatch is a limitation.

For normalized response/reference spectra `p` and `q`, the pointwise JSD contribution was

`j(f) = 0.5 p(f) log[p(f)/m(f)] + 0.5 q(f) log[q(f)/m(f)]`, where `m=(p+q)/2`.

The signed field was `v(f) = sign[p(f)-q(f)] sqrt[j(f)]`. Across all 20 completed tasks, the maximum numerical error in `sum_f v(f)^2 = JSD(p,q)` was `5.96e-8`.

All seven five-second ablations used the same 1,830-dimensional registered layout and the exact same 174,729-parameter vector-band HCBT. Inactive feature slots were zero, so condition differences cannot be attributed to different network capacity.

| Condition | Active information |
|---|---|
| C0 | absolute response DE |
| C1 | C0 + scalar JSD |
| C2 | C0 + unsigned `sqrt(j(f))` field |
| C3 | C0 + signed `v(f)` field |
| C4 | C0 + relative log-band-energy + signed field |
| C5 | C0 + pointwise energy-calibrated log-PSD ratio |
| C6 | C4 multiplied by `exp(-u / median_source(u))` per band, with `gamma=1` |

The C6 instability statistic `u` was the mean across native frequencies of the variance across the five early log-PSD windows. Its five scale parameters were fitted using source-fit trials only.

## Leakage-safe protocol

Official fold 1 used target subjects 0–11. Source development subjects were fixed to 12–23, leaving source-fit subjects 24–122.

- Conventional subject holdout: all 28 videos appear for source fit, source development, and target subjects. This tests new subjects but does not isolate video identity.
- Subject + stimulus holdout: source fit videos were `[0,3,6,9,12,13,16,19,22,25]`; source development videos were `[1,4,7,10,14,17,20,23,26]`; target videos were `[2,5,8,11,15,18,21,24,27]`. Thus every target emotion has a held-out video identity.

Every model trained for at most 15 epochs with fixed seed 42, balanced source subject/emotion batches, and source-development Macro-F1 checkpoint selection. Source-only subject/video probes ran after checkpoint lock. Target arrays were loaded only after that boundary. The completed-result audit found:

- 20 result files and one protocol hash;
- one parameter count (174,729) across all conditions;
- all 20 target-isolation flags passing;
- no target gradient, checkpoint, hyperparameter, normalization, or gate-scale use.

## Results

### Five-second primary ablation

| Condition | Conventional Acc | Conventional Macro-F1 | Strict Acc | Strict Macro-F1 |
|---|---:|---:|---:|---:|
| C0 absolute DE | **28.57%** | 25.45% | 13.89% | 10.03% |
| C1 + scalar JSD | 26.79% | 24.14% | 10.19% | 8.24% |
| C2 + unsigned field | 19.05% | 16.30% | **14.81%** | 12.79% |
| C3 + signed field | 19.35% | 17.55% | 13.89% | **12.87%** |
| C4 + delta energy + signed | 27.68% | **26.52%** | 12.96% | 11.81% |
| C5 pointwise log-ratio | 23.21% | 21.75% | 8.33% | 7.20% |
| C6 C4 + quality gate | 25.89% | 25.31% | 13.89% | 10.41% |

Subject-bootstrap marginal 95% intervals for selected conditions were:

| Protocol / condition | Accuracy 95% CI | Macro-F1 95% CI |
|---|---:|---:|
| conventional C0 | 24.11–32.74% | 20.63–29.31% |
| conventional C4 | 23.81–31.55% | 22.65–30.34% |
| conventional C5 | 19.05–27.98% | 18.15–25.39% |
| conventional C6 | 21.42–30.95% | 20.87–30.06% |
| strict C0 | 8.33–19.44% | 4.63–14.87% |
| strict C4 | 9.26–16.67% | 8.54–14.49% |
| strict C5 | 3.70–13.89% | 2.55–12.64% |
| strict C6 | 9.26–18.52% | 6.88–13.66% |

No primary C4-versus-C0 paired interval excluded zero. C2 and C3 were significantly worse than C0 in the conventional split, despite reducing identity-probe accuracy. This is evidence that the relative fields discard useful class-associated structure along with nuisance structure rather than cleanly isolating emotion.

### One-second control

| Condition | Conventional Acc | Conventional Macro-F1 | Strict Acc | Strict Macro-F1 |
|---|---:|---:|---:|---:|
| C0 absolute DE | 27.68% | 26.56% | 13.89% | 7.75% |
| C4 delta energy + signed | **28.87%** | **27.07%** | 12.96% | 12.55% |
| C5 pointwise log-ratio | 23.81% | 21.75% | **14.81%** | **14.81%** |

Five-second averaging did not help C4: the paired five-minus-one-second intervals crossed zero under both protocols. For C5 under the strict split, five-second averaging was worse than one second by 6.48 accuracy points (95% CI -12.04 to -0.93) and 7.61 Macro-F1 points (CI -13.64 to -0.92). This suggests that the short-lived direction and magnitude of spectral change may be the useful part of the early-reference idea, and five-window smoothing erases it.

The strict one-second C5 class recalls were `[8.3, 8.3, 8.3, 0.0, 16.7, 25.0, 16.7, 33.3, 16.7]%`. Its Macro-F1 improvement comes from broader, although still weak, class coverage rather than a large accuracy gain. It should therefore be treated as a hypothesis-generating result.

## Identity and stimulus probes

The probe labels were source-only and never influenced the encoder.

| Five-second condition | Conventional subject probe | Conventional video probe | Strict subject probe | Strict video probe |
|---|---:|---:|---:|---:|
| C0 | 88.89% | 37.80% | 94.44% | 31.25% |
| C2 | 63.89% | 16.37% | 69.44% | **11.61%** |
| C3 | **51.85%** | **17.26%** | 59.26% | 15.18% |
| C4 | 53.70% | 21.73% | **51.85%** | 22.02% |
| C5 | **51.85%** | 21.43% | 63.89% | 21.13% |
| C6 | 53.70% | 22.92% | 53.70% | 22.32% |

Random chances are 8.33% for the 12-way development-subject probe and 3.57% for the 28-way video probe. Therefore:

1. Early-reference relative fields substantially reduce recoverable subject and video identity compared with absolute DE.
2. They do not make the representation invariant; even the best probes remain far above chance.
3. Reduced nuisance identification did not translate into higher strict emotion accuracy. The transformation is not sufficiently selective and also removes label-relevant signal.
4. The collapse from roughly 20–29% conventional accuracy to roughly 8–15% strict accuracy confirms that conventional FACED subject holdout contains a large video/stimulus-identification contribution.

## Why the current version likely failed

These explanations have different evidential status.

### Directly supported by this experiment

- **Five-second smoothing is harmful for pointwise log-ratio.** The strict C5 five-versus-one-second paired intervals exclude zero in the wrong direction.
- **The signed field alone is not sufficient.** C3 did not reliably beat C2 and both were much worse than C0 conventionally.
- **Energy magnitude is needed to recover much of the conventional performance.** Adding relative log-energy in C4 recovered C3 from 19.35% to 27.68% conventional accuracy, but did not beat C0 or transfer to unseen videos.
- **The simple quality gate is misaligned with useful reliability.** C6 did not improve C4, so early log-PSD variance alone is not a good proxy for whether the relative feature helps emotion classification.
- **Identity/stimulus suppression is incomplete and nonselective.** Probes fall but remain far above chance, while strict emotion accuracy remains near chance.

### Plausible but not resolved here

- The first five seconds after video onset may already contain emotion-specific onset dynamics. Treating their average as a stationary reference can subtract the very response of interest.
- A single fixed reference cannot represent nonstationary within-trial drift. The informative quantity may be a short-range temporal derivative around multiple anchors, not distance from one early average.
- The DE-calibrated PSD reconstruction mixes two estimators. The resulting pointwise log-ratio may contain band-dependent calibration error.
- The strict source-fit partition has only ten training video identities. This is deliberately leakage-resistant but makes stimulus-general learning difficult; any improvement must be robust enough to survive that low-diversity regime.
- Fifteen epochs and one seed are sufficient for a lightweight diagnostic, not for a stable performance ceiling. However, extra optimization alone cannot explain the absence of a paired C4 advantage under the matched budget.

## Recommended next experiment

Do not expand the current five-second C4/C6 model or add adversarial/prototype/graph modules yet. The smallest credible next step is:

1. **Freeze the newly discovered one-second C5 branch before any new target evaluation.** Use exact raw Welch band power and PSD from the same estimator, rather than DE-calibrated reconstruction.
2. **Use a multi-scale causal representation:** retain the one-second pointwise log-ratio, add a robust three- or five-second summary as a separate token, and never replace the one-second path by smoothing.
3. **Replace mean early reference with a robust reference ensemble:** median/trimmed mean across the first five PSDs plus dispersion channels. This tests whether outlier sensitivity, rather than the early-state concept itself, caused failure.
4. **Select every design choice only on source subjects with the video-held-out development partition.** After locking, evaluate on a new official fold (recommended fold 2) with seed 42. If the branch survives, then run three seeds and additional folds.
5. **Pre-register success criteria:** strict accuracy and Macro-F1 both above matched C0; paired subject-bootstrap lower bound above zero for at least Macro-F1; no increase in video-ID probe; no target use before lock.
6. **Run a reference-validity diagnostic:** measure whether early-reference instability predicts per-trial errors or feature disagreement on source development. If not, abandon C6-style gates rather than learning a more complex gate.

The current evidence therefore supports a narrower interpretation: same-trial early referencing can suppress subject/video identity and one-second log-ratio may improve class balance, but the registered five-second STR-JSD representation is not yet an effective zero-adaptation cross-subject solution.

## Reproducibility artifacts

- Configuration: `configs/faced/str_jsd_fold1_light.yaml`
- Feature implementation: `src/cmrd/features/str_jsd.py`
- Capacity-matched model: `src/cmrd/models/str_jsd.py`
- Source-isolated runner: `src/cmrd/faced_str_jsd_runner.py`
- Launcher: `scripts/run_faced_str_jsd.ps1`
- Direct paired analysis: `scripts/analyze_faced_str_jsd.py`
- Unit tests: `tests/test_faced_str_jsd.py`
- Run summary: `runs/faced_str_jsd_fold1_light_seed42/summary.json`
- Direct comparison analysis: `runs/faced_str_jsd_fold1_light_seed42/analysis.json`
- Per-task logs and source-selected checkpoints: `runs/faced_str_jsd_fold1_light_seed42/`

Reproduction commands from the repository root:

```powershell
C:\Users\Lin\miniconda3\envs\cmrd\python.exe -m unittest tests.test_faced_str_jsd -v
.\scripts\run_faced_str_jsd.ps1 status
.\scripts\run_faced_str_jsd.ps1 run
C:\Users\Lin\miniconda3\envs\cmrd\python.exe scripts\analyze_faced_str_jsd.py --run-root runs\faced_str_jsd_fold1_light_seed42 --repeats 5000
```

The 20 task-level elapsed times sum to 626.8 seconds. The matrix command completed in 698.9 seconds including feature loading, probes, bootstrap, and persistence.
