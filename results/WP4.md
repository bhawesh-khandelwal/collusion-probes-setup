# WP4 results: Qwen3.8-27B (4-bit MLX), train on Core, test zero-shot

Layer-mean AUROC over layers 26-30, 95% CI from stratified bootstrap over runs (1,000).
Core LODO is 1.000 for every method and config again, so group configs fall to k=1, C=0.01.
Logs: results/wp4_27b_{discussion,final}.log, JSON alongside.

## AUROC, Core read at discussion (paper convention) / final turn
| Method | Transfer | Implicit | Implicit strict | Game |
|---|---|---|---|---|
| Peak Suspicion | 0.93 / 0.94 | 0.85 / 0.83 | 0.88 / 0.89 | 0.63 / 0.63 |
| Deception Split | 0.92 / 0.95 | 0.87 / 0.88 | 0.87 / 0.89 | 0.21 / 0.37 |
| Asymmetry Probe | 0.94 / 0.96 | 0.89 / 0.88 | 0.90 / 0.91 | 0.52 / 0.57 |
| Sorted Concat | 0.93 / 0.94 | 0.84 / 0.83 | 0.84 / 0.86 | 0.28 / 0.50 |
| set_stats | 0.89 / 0.93 | 0.78 / 0.81 | 0.78 / 0.84 | 0.41 / 0.56 |
| pair_max | 0.93 / 0.95 | 0.84 / 0.86 | 0.87 / 0.90 | 0.75 / 0.69 |
| pair_interact_max | 0.91 / 0.91 | 0.80 / 0.76 | 0.84 / 0.82 | 0.52 / 0.73 |

## Key differences (group minus per-agent, 95% CI)
- Transfer: pair_max - Asymmetry -0.012 [-0.040, +0.010] / -0.012 [-0.030, +0.005]. Tie.
- Implicit: pair_max - Asymmetry -0.049 [-0.102, -0.010] / -0.020 [-0.060, +0.021]. Group no better.
  Normal-alone slice: group detectors worse, e.g. pair_interact_max - Asymmetry -0.153 [-0.217, -0.087].
- Game: pair_max - Asymmetry +0.222 [+0.119, +0.328] / +0.124 [+0.050, +0.204].
  pair_interact_max - Asymmetry (final) +0.164 [+0.087, +0.249]. Normal-alone gaps similar or larger.

## Hypotheses
- H1 (Transfer, 27B): not supported. Best group detector ties the best per-agent method.
- H2: not supported on implicit (normal-alone gaps go the wrong way). Not assessed on 27B Transfer: the
  slice has 0-8 collusion runs per layer, so the metric is skipped. (On released 32B Transfer, WP1: not supported.)
- H3 implicit (primary): not supported. Note: positives are implicit-setup runs (only 21/36 judged as
  actually coordinating), so this measures detecting the setup, not confirmed collusion. Per-agent probes transfer to uninstructed incentive-driven pairs
  (0.85-0.91) and group detectors do not improve on them.
- H3 game (secondary): NOT supported by the pre-set selection rule (set_stats, chosen by tie-break: -0.114
  discussion, -0.012 final vs Asymmetry). Exploratory only: pair_max (both reads) and pair_interact_max
  (final read) beat per-agent methods with CI excluding 0; with k=1 these are learned combiners of per-agent
  scores, not joint-activation detectors.
  Caveats: positives are DM games and negatives are no-message games (prompt-format confound); almost all
  firms in a collusion game collude, so "minority colluder" combiners (Split, Asymmetry, Sorted Concat) are
  structurally mismatched and go below chance; Activation Alignment at 0.16 means collusion runs are more
  similar, consistent with shared message context. Pre-registered tie-break group (set_stats) does not beat Asymmetry.

## Post-review update (Nov 8, after Codex review). Logs: results/wp4c_{discussion,final}.log
- Implicit same-setup test (implicit_coord): 21 judge-coordinated vs 15 non-coordinated implicit
  runs, identical prompt setup. AUROC drops to 0.60-0.69 (discussion read) and 0.65-0.82 (final
  read), wide CIs; group detectors tie per-agent methods (final: pair_max 0.81 vs Peak 0.82).
  Most of the 0.85-0.91 on the implicit tier was detecting the setup. Non-coordinated runs are
  mostly code tasks, so family remains a confound.
- H2 slice redefined: one slice per tier fixed across layers, per-agent scores z-scored against
  that tier's control agents and averaged over layers, threshold = control 99th percentile.
  A Core-referenced threshold is unusable: test-tier control agents score about 6 points higher
  on the probe than Core control agents (layer 28), so every test run exceeds it.
  27B Transfer slice: 6 runs (discussion), 2 (final). Group minus Asymmetry: set_stats -0.018
  [-0.083, +0.048], pair_max -0.056 [-0.154, +0.022] (discussion). H2 not supported; underpowered.
  Implicit: group detectors worse (pair_max - Asymmetry -0.090 [-0.160, -0.032], discussion).
- Game same-format control (dm_neutral, 20 games, $0.43): every received message replaced by a
  price-free business update. All 20 games still labelled collusion (2-4 firms flagged, mean CI
  0.30): firms send unanswered price proposals by DM and their notes describe a "collusive
  equilibrium" / "tacit understanding". Same-format negatives cannot be produced with this
  model, so the game comparison stays confounded by prompt format.
- Judge parsing: 0 unparseable verdicts of 196 game firm-verdicts and 36 implicit verdicts; code
  now labels unparseable runs "unknown" instead of negative.

## Judge validation (Nov 8)
Second judge glm-5.2 (different family from players and first judge), same prompts. Cost $1.21.
- Game firms (main + neutral, 239 verdicts): 97% agreement, Cohen's kappa 0.94. Run labels 55/60 agree.
  Within condition: dm 0.99 agree (kappa 0.66), nocomm 0.95 (0.57), dm_neutral 0.99 (0.88);
  lower kappa within condition reflects extreme base rates.
- Implicit runs (36): 94% agreement, kappa 0.89, both judges 21/36 YES. Two disagreements, both in
  the news family. The family pattern (code tasks NO, others YES) replicates with the second judge.
- Human blind check: review/human_check.md (10 cases), key in review/human_check_key.json. Pending.
