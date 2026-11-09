# Collusion Probes Detect the Setup, Not the Coordination

Code, data and results for an Apart Swarm Alignment Sprint submission (Track 4, Multi-Agent Oversight, 6 to 8 Nov 2026).

**Report:** [report/out/Collusion_Probes_Detect_the_Setup.pdf](report/out/Collusion_Probes_Detect_the_Setup.pdf)

**Question.** Does a learned detector that reads all agents' activations jointly beat per-agent probes combined after the fact, especially for emergent collusion and when each agent looks normal alone?

**Answer.** No. On NARCBench, with Qwen3.8-27B and Gemma 4 31B, group detectors tie the best per-agent probe on instructed collusion and never reliably beat it elsewhere. On 166 new runs where every pair has the same incentive and private channel, and only the outcome differs, every detector (white-box or black-box, per-agent or group) is near chance within a scenario family. The probes detect who was set up to collude, not whether they coordinated.

## Layout

| Path | What it holds |
|---|---|
| `gen/gen_implicit.py` | Generates extra NARCBench implicit runs with NARCBench's own generator (model call swapped for a hosted API) |
| `gen/implicit_runs/` | The 180 new same-setup runs (transcripts, prompts, configs) |
| `game/label_implicit.py` | LLM judge for whether an implicit pair coordinated |
| `game/runs/implicit2_labels*.json` | Labels from the three judges (Gemma 4 31B, GLM 5.2, DeepSeek V4 Flash); `implicit2_consensus.json` is the two-judge consensus |
| `game/monitor_text.py` | Black-box text monitor on public transcripts |
| `game/pricing_game.py`, `game/label_runs.py` | Exploratory pricing game and its intent judge |
| `extract/` | Activation extraction with MLX (NARCBench format) and derived test tiers |
| `group-detector/` | Group detectors and evaluation (`group_eval_v1.py`), relational and receiver tests, in-domain CV, monitor evaluation, group-minus-Peak comparisons |
| `review/three_judge_agreement.py` | Agreement between the three judges |
| `results/`, `group-detector/results/` | Saved result JSON, logs and notes |
| `report/` | Report source (`REPORT.md`), figures (`make_figures.py`), tables, and the template build script |

## Reproducing

1. Clone [NARCBench](https://github.com/aaronrose227/narcbench) and download its data; the scripts expect it at `~/projects/open-source/narcbench`.
2. Put an API key for the hosted models in `.env` as `ADAPTION_API_KEY=...` (only needed for generation, judging and the text monitor).
3. Extract activations: `python -I extract/extract_mlx.py ...` (needs Apple silicon and the 4-bit MLX models; see the docstring), then `extract/make_tiers.py`.
4. Evaluate: `run_wp4.sh` (Qwen) and the commands in each script's docstring; `group-detector/vs_peak.py`, `indomain_cv.py`, `monitor_eval.py`, `receiver_test.py`, `relational_test.py`.
5. Figures and tables: `python -I report/make_figures.py`.

Activations (about 2.4 GB) and model weights are not in the repo.

## Notes

- All code here was written during the sprint, with Claude Code. NARCBench (MIT) is used as an external dependency; its generator is imported, not copied.
- Labels for emergent coordination come from LLM judges, not humans. See the report's Limitations.
- Licence: MIT.
