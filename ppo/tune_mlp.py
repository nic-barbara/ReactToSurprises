"""
Tune PPO for the black-box MLP policies, then train the best settings in full.

Stage 1 (screening): 18 PPO settings x 2 seeds x 1600 epochs, test cost every
25 epochs, no lr cut. Score = mean over seeds of the mean log10 test cost over
the last 25% of epochs (NaN/Inf counts as 1e308). The top 3 settings go to
Stage 2: 6 seeds x 6400 epochs with the full setup (lr cut, test every 5
epochs). The best setting (same score, last 500 epochs) is copied to
results/ppo_mlp_s*, and the figures are re-plotted.

Runs are spread over the GPU and CPU (one thread each). Finished runs are
skipped, so the script can be restarted.

    nohup uv run python tune_mlp.py > results/mlp_tuning/tune.out 2>&1 &
"""
import itertools
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "results" / "mlp_tuning"

gammas = [0.99, 0.999]
entropy_coefs = [0.01, 0.001, 0.0]
lrs = [None, 3e-4, 1e-4]          # None = adaptive (RSL-RL default)
settings = list(itertools.product(gammas, entropy_coefs, lrs))

stage1 = dict(seeds=[0, 1], epochs=1600, log_every=25, lr_decay=False, score_from=1200)
stage2 = dict(seeds=[0, 1, 2, 3, 4, 5], epochs=6400, log_every=5, lr_decay=True, score_from=5900)
n_best = 3
n_gpu, n_cpu = 3, 29              # 32 hardware threads


def name(setting):
    g, e, lr = setting
    return f"g{g:g}_ent{e:g}_lr{'adaptive' if lr is None else f'{lr:g}'}"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(OUT / "tune.log", "a") as f:
        f.write(line + "\n")


def run_dir(stage, setting):
    return OUT / stage / name(setting)


def is_done(stage, setting, seed):
    logfile = run_dir(stage, setting) / f"ppo_mlp_s{seed}.log"
    return logfile.exists() and "Done" in logfile.read_text()


def run_job(job, device):
    stage, setting, seed, cfg = job
    g, e, lr = setting
    outdir = run_dir(stage, setting)
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = ["uv", "run", "python", "train_ppo.py", "--policy", "mlp", "--seed", str(seed),
           "--epochs", str(cfg["epochs"]), "--device", device, "--gamma", str(g),
           "--entropy-coef", str(e), "--log-every", str(cfg["log_every"]),
           "--outdir", str(outdir.relative_to(HERE / "results"))]
    if lr is not None:
        cmd += ["--lr", str(lr)]
    if not cfg["lr_decay"]:
        cmd += ["--no-lr-decay"]
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    with open(outdir / f"ppo_mlp_s{seed}.log", "w") as f:
        subprocess.run(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT, env=env)
    log(f"finished {stage}/{name(setting)}/s{seed} on {device}")


def run_all(jobs):
    """Run jobs on n_gpu GPU slots and n_cpu CPU slots."""
    q = queue.Queue()
    for job in jobs:
        q.put(job)

    def worker(device):
        while True:
            try:
                job = q.get_nowait()
            except queue.Empty:
                return
            run_job(job, device)

    threads = [threading.Thread(target=worker, args=(d,))
               for d in ["cuda"] * n_gpu + ["cpu"] * n_cpu]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def score(stage, setting, cfg):
    """Mean over seeds of the mean log10 test cost from epoch `score_from` on."""
    per_seed = []
    for seed in cfg["seeds"]:
        d = np.loadtxt(run_dir(stage, setting) / f"ppo_mlp_s{seed}.csv",
                       delimiter=",", skiprows=1, ndmin=2)
        J = d[d[:, 0] >= cfg["score_from"], 1]
        J = np.where(np.isfinite(J), J, 1e308)
        per_seed.append(np.mean(np.log10(J)))
    return np.mean(per_seed), per_seed


def run_stage(stage, cfg, stage_settings):
    jobs = [(stage, s, seed, cfg) for s in stage_settings for seed in cfg["seeds"]
            if not is_done(stage, s, seed)]
    log(f"{stage}: {len(jobs)} runs to do")
    run_all(jobs)
    results = sorted((score(stage, s, cfg) + (s,) for s in stage_settings), key=lambda r: r[0])
    with open(OUT / "summary.md", "a") as f:
        f.write(f"\n## {stage} ({cfg['epochs']} epochs, score from epoch {cfg['score_from']})\n\n"
                "| Setting | Score (mean log10 test cost) | Per seed |\n|---|---|---|\n")
        for mean, per_seed, s in results:
            f.write(f"| {name(s)} | {mean:.2f} | {', '.join(f'{x:.1f}' for x in per_seed)} |\n")
    log(f"{stage}: best {[name(r[2]) for r in results[:n_best]]}")
    return [r[2] for r in results]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text("# PPO-MLP tuning\n")
    log(f"Starting: {len(settings)} settings")

    ranked = run_stage("stage1", stage1, settings)
    ranked = run_stage("stage2", stage2, ranked[:n_best])

    # Copy the best setting's runs to results/ (used by plot_ppo.jl)
    best = run_dir("stage2", ranked[0])
    for f in best.glob("ppo_mlp_s*"):
        shutil.copy(f, HERE / "results" / f.name)
    log(f"Best setting: {name(ranked[0])}, copied to results/")
    with open(OUT / "summary.md", "a") as f:
        f.write(f"\nBest setting: **{name(ranked[0])}** (copied to `results/ppo_mlp_s*`)\n")

    subprocess.run(["julia", "+1.10", "--project=.", "ppo/plot_ppo.jl"], cwd=HERE.parent)
    log("Figures re-plotted. All done.")


if __name__ == "__main__":
    main()
