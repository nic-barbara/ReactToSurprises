#!/usr/bin/env bash
# Train PPO policies (one per seed) in parallel on the GPU.
# Usage: ./run_sweep.sh <lstm|mlp> [extra train_ppo.py args...]
set -euo pipefail
cd "$(dirname "$0")"
policy=$1
shift
mkdir -p results

for seed in 0 1 2 3 4 5; do
    uv run python train_ppo.py --policy $policy --seed $seed "$@" > results/ppo_${policy}_s${seed}.log 2>&1 &
done
wait
echo "All done."
