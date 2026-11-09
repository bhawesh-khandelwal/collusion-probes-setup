#!/usr/bin/env bash
# Main evaluation on Qwen3.8-27B activations: train on Core, test zero-shot.
set -euo pipefail
cd "$(dirname "$0")"
NB=~/projects/open-source/narcbench
PY=$NB/.venv/bin/python
TIERS=(--tier transfer=acts/transfer --tier implicit=acts/implicit --tier implicit_strict=acts/implicit_strict)
if [ -d acts/game_tier ]; then TIERS+=(--tier game=acts/game_tier); fi

for phase in discussion final; do
  $PY -I -W ignore group-detector/group_eval_v1.py "$NB" \
    --core acts/core --core-runs-from "$NB/data/activations/qwen3_32b/core" \
    "${TIERS[@]}" --phase "$phase" --out "results/wp4_27b_$phase.json" \
    > "results/wp4_27b_$phase.log" 2>&1
done
echo done
