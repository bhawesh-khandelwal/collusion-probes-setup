# Second model family: Gemma 4 31B (paper-readiness item 1)

Same pipeline as 27B: probe and group detectors trained on Gemma Core (62 released runs, layers 24-28), tested zero-shot. Logs: `results/gemma_{discussion,final}.log`; scores: `results/gemma_scores_*.json`; differences vs Peak Suspicion: `group-detector/vs_peak.py` (same metric as the main table: mean over layers of per-layer AUROC, stratified bootstrap, 1000 draws).

## AUROC (layer mean, 95% CI), Core read = discussion / final

| Tier | Peak | Deception Split | Asymmetry | set_stats (preselected) | pair_max |
|---|---|---|---|---|---|
| Transfer | 0.73 / 0.64 | 0.73 / 0.54 | 0.75 / 0.59 | 0.71 / 0.64 | 0.73 / 0.60 |
| Implicit vs control | 0.64 / 0.63 | 0.59 / 0.54 | 0.64 / 0.58 | 0.61 / 0.61 | 0.65 / 0.60 |
| implicit_coord (36) | 0.63 / 0.58 | 0.61 / 0.55 | 0.67 / 0.56 | 0.65 / 0.62 | 0.65 / 0.64 |
| implicit2_coord (166) | 0.72 / 0.62 | 0.40 / 0.54 | 0.65 / 0.58 | 0.69 / 0.59 | 0.73 / 0.61 |
| Game (exploratory) | 0.62 / 0.59 | 0.31 / 0.49 | 0.52 / 0.54 | 0.62 / 0.52 | 0.65 / 0.61 |

## Group minus Peak Suspicion (95% CI)

| Tier | set_stats disc | pair_max disc | set_stats final | pair_max final |
|---|---|---|---|---|
| Transfer | -0.023 [-0.056, +0.008] | +0.005 [-0.025, +0.034] | +0.002 [-0.030, +0.035] | -0.036 [-0.104, +0.035] |
| Implicit | -0.027 [-0.063, +0.012] | +0.013 [-0.018, +0.042] | -0.019 [-0.047, +0.012] | -0.025 [-0.096, +0.047] |
| implicit_coord | +0.022 [-0.037, +0.081] | +0.017 [-0.030, +0.075] | +0.040 [+0.009, +0.078] | +0.058 [-0.044, +0.163] |
| implicit2_coord | -0.029 [-0.052, -0.004] | +0.007 [-0.009, +0.023] | -0.030 [-0.068, +0.006] | -0.015 [-0.062, +0.033] |
| Game | +0.008 [-0.038, +0.054] | +0.031 [-0.007, +0.068] | -0.064 [-0.146, +0.019] | +0.020 [-0.049, +0.096] |

27B for comparison (implicit tiers): implicit2_coord pair_max minus Peak +0.053 [+0.018, +0.087] (discussion), +0.023 [-0.007, +0.052] (final); set_stats minus Peak -0.156 and -0.118, both significant.

## Reading

1. Gemma's per-agent probe transfers worse than 27B's (Transfer 0.64 to 0.75 vs 0.94 to 0.96). Every method is weaker.
2. Against Asymmetry and Deception Split (the H1 comparators), group methods win in several Gemma cuts (for example Transfer final set_stats minus Asymmetry +0.051 [+0.020, +0.083]; implicit2_coord discussion pair_max minus Asymmetry +0.077 [+0.037, +0.116]). Peak Suspicion is equal or stronger in those same cuts, so those wins reflect weak comparators, not a group advantage.
3. Against Peak Suspicion: 2 of 20 comparisons significant, one each direction. Consistent with no effect.
4. The 27B exploratory hint (pair_max beats Peak on implicit2_coord, discussion read) does not replicate on Gemma (+0.007 [-0.009, +0.023]).
5. Relational test (Gemma, `group-detector/results/relational_v2_gemma*.log`): partner signal on Core at the discussion read (cos_std DiD +0.158 [+0.033, +0.278]), none at the Core final read, none on Transfer or implicit. 27B showed the Core signal at the final read instead. No transferable partner signal on either model.
6. Receiver reads: see `group-detector/results/RECEIVER.md`. Negative on both models.
