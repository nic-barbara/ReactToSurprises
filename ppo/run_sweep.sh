#!/usr/bin/env bash
# Train PPO LSTM runs (one per seed) in parallel on the GPU.
# Usage: ./run_sweep.sh <tag> <epochs> "<seeds>" [extra train_ppo.py args...]
# Example: ./run_sweep.sh g999 6400 "0 1 2" --gamma 0.999 --lr-decay
set -euo pipefail
cd "$(dirname "$0")"

tag=$1
epochs=$2
seeds=$3
shift 3
mkdir -p results/$tag

for seed in $seeds; do
    uv run python train_ppo.py --policy lstm --seed $seed --epochs $epochs --tag $tag "$@" \
        > results/$tag/ppo_lstm_s${seed}.log 2>&1 &
done
wait
echo "All done."
