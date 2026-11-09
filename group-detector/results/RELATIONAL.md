# Relational ("Frankenstein") test: is there partner signal beyond per-agent suspicion?

DiD per collusion run = [s(real partners) - s(colluder + colluder from another run)]
                      - [s(same-run honest pair) - s(honest + honest from another run)].
Mean over runs, layers 26-30 averaged, 95% bootstrap CI over runs. "family" = cross-run
partners drawn only from the same scenario family. Code: group-detector/relational_test.py.

| Data | Model | cos (family) DiD [CI] | learned (family) DiD [CI] |
|---|---|---|---|
| Core, discussion avg (in-sample) | 32B | +0.077 [+0.024, +0.124] | +2.01 [+1.32, +2.69] |
| Core, final turn (in-sample) | 32B | +0.028 [+0.007, +0.049] | +1.03 [+0.09, +1.97] |
| Core, discussion avg (in-sample) | 27B | +0.048 [+0.009, +0.088] | +1.13 [+0.53, +1.74] |
| Core, final turn (in-sample) | 27B | +0.049 [+0.030, +0.066] | +1.25 [+0.73, +1.78] |
| Transfer, final | 32B | -0.034 [-0.102, +0.037] | -0.69 [-2.02, +0.62] |
| Transfer, final | 27B | -0.028 [-0.090, +0.039] | -0.34 [-1.27, +0.62] |
| Implicit, final | 27B | +0.029 [-0.027, +0.083] | +0.62 [-0.20, +1.39] |
| Implicit strict, final | 27B | +0.036 [-0.051, +0.120] | +0.71 [-0.60, +1.92] |

Reading:
- Core: real partners are more alike than same-conversation non-partners, beyond each agent
  alone, on both models and at both read positions. Unsupervised cosine finds it, so it is not
  a training artefact. Core colluders both receive the same target option, so this is plausibly
  shared secret content rather than partnership itself.
- Transfer: no partner signal (point estimates negative, CIs include 0). Transfer colluders get
  complementary roles (e.g. one writes a bug, the other avoids testing it), so they need not look alike.
- Implicit: positive direction on every metric, CIs include 0 (36 and 21 runs; underpowered).
- Read position is not the cause: the Core signal survives at the final turn.
- The learned pair scorer trained with cross-run hard negatives does not transfer to Transfer.

Caveats (Codex review, Nov 8):
- The learned scorer is trained and scored on the same Core runs; only the cosine rows are
  training-free, and Transfer/implicit rows are out-of-sample.
- Cross-run comparisons reuse agents across runs, while the bootstrap resamples per-run statistics
  as if independent, so the CIs are somewhat too narrow.
- Content given only to colluders (the shared target option) cannot be removed by the honest-pair
  subtraction; same-target non-partner pairs would be needed to attribute the signal to partnership.

## v2 (after review): training-free scores on Core, partner pools recomputed per bootstrap draw
Logs: results/relational_v2_{32b,32b_cf,27b,27b_cf}.log (cf = Core read at final turn).
- Core, discussion read: DiD positive but CIs include 0 except 32B cos_std family (+0.077 [+0.003, +0.159]).
- Core, final-turn read: 27B all four cosine rows significant (e.g. cos_std any +0.039 [+0.020, +0.057]);
  32B "any" rows significant (+0.024 [+0.005, +0.048]), "family" rows not.
- Transfer: no signal on either model (e.g. 27B cos_std any -0.013 [-0.073, +0.044]).
- Implicit: positive on every metric, all CIs include 0.
Conclusion: a small partner signal exists in Core (instructed, same target option) and is not
present in Transfer; for implicit pairs it is suggestive only. Earlier v1 CIs were too narrow.
