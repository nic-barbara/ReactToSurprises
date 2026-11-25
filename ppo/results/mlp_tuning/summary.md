# PPO-MLP tuning

## stage1 (1600 epochs, score from epoch 1200)

| Setting | Score (mean log10 test cost) | Per seed |
|---|---|---|
| g0.99_ent0.01_lradaptive | 23.32 | 21.6, 25.0 |
| g0.99_ent0_lr0.0001 | 27.74 | 27.7, 27.8 |
| g0.99_ent0_lr0.0003 | 28.83 | 29.5, 28.1 |
| g0.99_ent0.001_lr0.0001 | 28.92 | 29.0, 28.8 |
| g0.99_ent0.01_lr0.0003 | 29.17 | 27.4, 31.0 |
| g0.99_ent0.001_lr0.0003 | 29.34 | 29.0, 29.7 |
| g0.99_ent0.01_lr0.0001 | 29.48 | 30.6, 28.4 |
| g0.99_ent0.001_lradaptive | 29.54 | 29.2, 29.9 |
| g0.999_ent0.001_lradaptive | 29.89 | 30.0, 29.7 |
| g0.99_ent0_lradaptive | 29.98 | 29.8, 30.2 |
| g0.999_ent0_lradaptive | 30.43 | 29.8, 31.0 |
| g0.999_ent0_lr0.0001 | 30.51 | 29.7, 31.3 |
| g0.999_ent0.01_lr0.0001 | 31.25 | 32.1, 30.4 |
| g0.999_ent0.001_lr0.0001 | 31.32 | 31.7, 30.9 |
| g0.999_ent0.001_lr0.0003 | 32.10 | 32.2, 32.0 |
| g0.999_ent0_lr0.0003 | 32.20 | 32.5, 31.9 |
| g0.999_ent0.01_lr0.0003 | 36.24 | 37.0, 35.4 |
| g0.999_ent0.01_lradaptive | 85.16 | 135.5, 34.8 |

## stage2 (6400 epochs, score from epoch 5900)

| Setting | Score (mean log10 test cost) | Per seed |
|---|---|---|
| g0.99_ent0.01_lradaptive | 26.72 | 22.5, 28.8, 28.0, 26.8, 23.3, 30.9 |
| g0.99_ent0_lr0.0001 | 28.42 | 27.3, 28.2, 28.5, 29.0, 28.2, 29.5 |
| g0.99_ent0_lr0.0003 | 29.09 | 30.6, 27.8, 29.6, 28.5, 30.0, 28.0 |

Best setting: **g0.99_ent0.01_lradaptive** (copied to `results/ppo_mlp_s*`)
