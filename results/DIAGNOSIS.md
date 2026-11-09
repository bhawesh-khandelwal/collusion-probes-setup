# Why the group detector did not win: in-domain training and a text-monitor baseline

## A. In-domain detectors (train on the same-setup implicit runs)

Code: `group-detector/indomain_cv.py`. Data: 166 implicit2 runs with two-judge consensus labels (27 coordinated, 139 not, all same setup). Fixed settings: 32-dim PCA, C=0.1, balanced, layer-mean scores. `family_prior` = coordination base rate of the run's scenario family in the training folds.

| Split | Method | 27B AUROC | 27B within-family | Gemma AUROC | Gemma within-family |
|---|---|---|---|---|---|
| Scenario held out | peak (per-agent) | 0.850 [0.76, 0.92] | 0.618 | 0.836 [0.75, 0.91] | 0.496 |
| | pair_known (oracle roles) | 0.869 | 0.650 | 0.804 | 0.496 |
| | set_stats | 0.828 | 0.561 | 0.856 | 0.593 |
| | pair_max | 0.814 | 0.631 | 0.791 | 0.531 |
| | family_prior (no activations) | 0.830 [0.72, 0.93] | n/a | 0.830 | n/a |
| Family held out | peak | 0.573 [0.46, 0.68] | 0.669 | 0.525 [0.40, 0.65] | 0.561 |
| | set_stats | 0.541 | 0.553 | 0.624 [0.51, 0.74] | 0.696 |
| | pair_max | 0.619 [0.51, 0.73] | 0.718 | 0.493 | 0.488 |

Group minus peak: no comparison significant on either model or split (largest +0.099 [-0.040, +0.250], Gemma set_stats, family held out).

Reading:
1. In-domain training reaches 0.79 to 0.87 when scenarios are held out, but a model that knows only the scenario family reaches 0.83. Most of the in-domain signal is which family the run comes from (pump_dump runs coordinate 18 of 26 times, cicd 0 of 26).
2. When whole families are held out, every activation method falls to 0.49 to 0.62; every 95% CI except Gemma set_stats (lower bound 0.51) includes 0.5.
3. So the negative result is not only a training-data problem (Core teaching "was instructed"). With matched training data there is still no coordination signal that generalises across scenario families at these read positions, for per-agent or group methods.
4. Caveat: 27 positives, concentrated in two families; family-held-out folds are small.

## B. Black-box text monitor

Code: `game/monitor_text.py`, `group-detector/monitor_eval.py`. Monitor: deepseek-v4-flash (third model family, neither player nor judge), reasoning off, temperature 0. Sees the public transcript and final outputs only, never system prompts or the private channel. Returns 0 to 100. 288 runs, 0 unparseable, cost $0.08.

| Tier | Monitor AUROC | Best probe (27B) | Best probe (Gemma) |
|---|---|---|---|
| Transfer (instructed) | 0.682 [0.56, 0.80] | 0.957 Asymmetry | 0.751 Asymmetry |
| Implicit vs control | 0.568 [0.45, 0.70] | 0.85 to 0.89 | 0.648 pair_max |
| implicit_coord (36) | 0.478 [0.29, 0.68] | 0.815 Peak (final) | 0.666 Asymmetry |
| implicit2_coord (166) | 0.586 [0.49, 0.68]; no-refusal 0.511; within-family 0.554 | 0.646 pair_max | 0.735 pair_interact_max |

Paired differences (`results/monitor_eval.log`): monitor below 27B probes on Transfer (about -0.25, e.g. vs Peak final -0.255 [-0.387, -0.129]) and on implicit vs control (about -0.27, e.g. vs Peak final -0.264 [-0.409, -0.107]); below 27B Peak on implicit_coord final (-0.337 [-0.567, -0.090]) and below Gemma Peak on implicit2_coord discussion (-0.137 [-0.258, -0.011]); other differences not significant.

Reading:
1. On instructed collusion, white-box probes beat a public-transcript monitor by a wide margin on 27B.
2. On emergent coordination, the monitor is at chance once refusals or family are controlled (0.51, 0.55). Activation probes are not clearly better there either.
3. Neither white-box nor black-box oversight from the public record detects whether a same-setup pair actually coordinated. The only reliable evidence is the private channel itself (which the judges read).

## C. Suggested framing

Collusion probes detect the setup, not the coordination. Evidence chain:
- Instructed collusion: probes work (0.94 to 0.96 on 27B), group methods tie per-agent methods.
- Same setup, coordinated vs not: all detectors drop to 0.40 to 0.73 zero-shot (both models).
- In-domain training does not rescue it: signal is scenario family, gone when families are held out.
- Partner-signal tests (Frankenstein DiD, receiver reads) find setup effects, not coordination effects.
- A text monitor on public transcripts is also at chance on the same split.
Released artifact: the same-setup split (180 runs, two-judge labels, third-judge check) as a small benchmark for coordination, not instruction, detection.
