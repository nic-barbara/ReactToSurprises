#!/usr/bin/env bash
# Train the PPO LSTM policies (one per seed) in parallel on the GPU.
# Usage: ./run_sweep.sh [extra train_ppo.py args...]
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results

for seed in 0 1 2 3 4 5; do
    uv run python train_ppo.py --seed $seed "$@" > results/ppo_lstm_s${seed}.log 2>&1 &
done
wait
echo "All done."
