# Receiver-side reads (paper-readiness item 3)

Code: `group-detector/receiver_test.py`. Data: `acts/core_final_recv`, `acts/transfer_recv`, `acts/implicit2` (27B, layers 26-30); `acts_gemma/transfer`, `acts_gemma/implicit2` (Gemma 4 31B, layers 24-28).

## What is measured

R[r, s] = reader r's mean activation over the first 256 tokens of each public message from sender s, inside r's final-turn prompt. All four readers see the same public transcript (prompt lengths identical), so reader differences come only from each reader's own system prompt.

Pair scorer: logistic regression on standardised R (32-dim PCA, C=0.01, balanced). Positive = colluder reading its partner. Negatives = all other ordered pairs, including honest agents reading colluders, so sender content alone cannot separate the classes.

- DiD per run = [colluder reading partner minus colluder reading honest] minus [honest reading colluder minus honest reading honest]. Sender-content effects cancel.
- Same-setup contrast (labelled implicit tiers): DiD in judge-coordinated runs minus DiD in non-coordinated runs with the same setup, using the setup roles. This separates "knows who the ally is" from "actually coordinated".
- Detection: run score = max over ordered pairs; AUROC.
- Bootstrap over runs, 2000 draws.

## Results

Train on 27B Core (62 released runs):

| Test tier | DiD | 95% CI | Coordinated minus non-coordinated DiD | Detection AUROC |
|---|---|---|---|---|
| Transfer | -0.149 | [-0.280, -0.023] | n/a | 0.922 [0.85, 0.98] |
| Implicit vs control | +0.077 | [-0.007, +0.171] | n/a | 0.786 [0.67, 0.89] |
| implicit_coord (36) | +0.111 | [-0.007, +0.239] | +0.081 [-0.079, +0.250] | 0.638 [0.45, 0.82] |
| implicit2_coord (166) | -0.092 | [-0.164, -0.011] | -0.279 [-0.384, -0.168] | 0.322 [0.22, 0.43] |

Train on Transfer collusion and control runs (no Core receiver reads exist for Gemma; implicit runs never in training):

| Model | Test tier | DiD | 95% CI | Coordinated minus non-coordinated DiD | Detection AUROC |
|---|---|---|---|---|---|
| 27B | implicit_coord | +0.405 | [+0.231, +0.602] | -0.057 [-0.409, +0.303] | 0.810 [0.65, 0.93] |
| 27B | implicit2_coord | +0.148 | [+0.042, +0.266] | -0.254 [-0.402, -0.106] | 0.561 [0.46, 0.67] |
| Gemma | implicit_coord | +0.193 | [+0.097, +0.287] | +0.156 [-0.164, +0.470] | 0.895 [0.78, 0.98] |
| Gemma | implicit2_coord | -0.058 | [-0.148, +0.029] | -0.059 [-0.171, +0.050] | 0.549 [0.44, 0.66] |

Refusal-excluded implicit2_coord detection (27B, Core-trained): 0.355.

## Reading

1. Instructed collusion (Core to Transfer): no reader-specific partner signal. DiD is negative: honest agents' reads of colluders shift more than colluders' reads of their partner. Detection (0.92) comes from sender content and does not beat Asymmetry Probe (0.957).
2. Implicit setups: when trained on Transfer, colluders read their partner differently (positive DiD). The same effect appears in runs where the pair did not coordinate (27B non-coordinated DiD 0.46 and 0.40). It tracks the setup (the system prompt names a shared incentive), not coordination.
3. Same-setup contrast is never positive and significant. On 27B implicit2 it is significantly negative under both training sets.
4. Verdict: receiver-side reads do not give a group signal for emergent coordination on either model. Item 3 closed as a negative result.

Caveats: Gemma has no Core receiver reads, so the Gemma rows use Transfer training only. 27B Transfer-trained rows use the same scenarios (different runs) as training. Scorer settings fixed in advance, no tuning.
