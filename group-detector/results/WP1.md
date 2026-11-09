# WP1 results: low-dimensional group detectors on released Qwen3-32B data

Train: Core (62 runs). Test: Transfer (72 runs) zero-shot. Layers 26-30, layer-mean AUROC.
95% CI: stratified bootstrap over runs, 1,000 resamples (sensitivity runs: 200).

## Selection
- Core leave-one-domain-out AUROC is 1.000 for every baseline and every group config.
  Core cannot rank configs, so ties go to the simplest config: k=1, C=0.01 on every layer.
- With k=1 the group detectors only see per-agent probe scores, i.e. they are learned
  per-agent-then-combine rules, not detectors that use extra group structure.

## Transfer, Core read at discussion (paper convention)
| Method | AUROC [95% CI] | Normal-alone | Recall@1% |
|---|---|---|---|
| Asymmetry Probe | 0.891 [0.82, 0.95] | 0.873 | 0.37 |
| Deception Split | 0.867 [0.79, 0.93] | 0.854 | 0.44 |
| pair_max (k=1) | 0.828 [0.72, 0.92] | 0.801 | 0.09 |
| set_stats (k=1) | 0.758 [0.63, 0.87] | 0.726 | 0.03 |

pair_max minus Asymmetry: -0.063 [-0.116, -0.019]. Minus Deception Split: -0.039 [-0.163, +0.069].

## Sensitivity: forcing more directions (post hoc, not used for claims)
| k | set_stats | pair_max | pair_interact_max |
|---|---|---|---|
| 1 | 0.758 | 0.828 | 0.828 |
| 2 | 0.764 | 0.826 | 0.825 |
| 4 | 0.743 | 0.817 | 0.818 |
| 8 | 0.637 | 0.755 | 0.802 |
More group directions transfer worse. v0 on raw 5,120 dims was about 0.50.

## Core read at final turn (matches Transfer read position)
Peak Suspicion 0.895, Asymmetry 0.892, pair_max 0.859, Deception Split 0.816.
pair_max minus Deception Split +0.043 [-0.077, +0.163]; minus Asymmetry -0.033 [-0.098, +0.025].
Normal-alone slice shrinks to 10-16 runs and Stego drops for most methods.

## Status of hypotheses on released 32B data
- H1: not supported. No group detector beats Asymmetry; CIs vs Deception Split include 0.
- H2: not supported. Gap on normal-alone slice is no larger.
- H3: untested (needs WP2/WP3).

Files: results/v1_discussion.json, v1_final.json, v1_discussion_k{2,4,8}.json and .log.
