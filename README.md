# React to Surprises with the Youla-REN

This repository contains all the code for the paper [*React to Surprises: Stable-by-Design Neural Feedback Control and the Youla-REN*](https://arxiv.org/abs/2506.01226) (Barbara, Wang, Megretski, & Manchester, 2025). 

![animation-ezgif com-crop](https://github.com/user-attachments/assets/534c2fbf-19ba-420e-b531-6c5d69a28286)


## Installation and Setup

The experiments in this repository are written in Julia `v1.10.0`. An easy way to install Julia is via [`juliaup`](https://github.com/JuliaLang/juliaup). For Mac or Linux, install with

    curl -fsSL https://install.julialang.org | sh
and follow the prompts in your terminal. For Windows, use

    winget install julia -s msstore


Once Julia is installed, open a terminal and navigate to the root directory of this repository. Type `julia` in the terminal to start Julia, and install all dependencies with:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Organisation of this Repository

This repositroy is structured as follows.

`src/`: contains all source code used to run experiments, process results, and generate plots.

`results/`: contains all plots and saved model weights used to produce the main results figures in the paper.

`matlab/`: contains a few MATLAB scripts used to design a robust base controller for the third example in this paper.

There is a lot of code repetition in this repository. The intention is for the code used to study each example in the paper to be completely independent of the other examples.

## Reproducing the Results

### Getting Started

Let's first walk through how to run the code in Julia. Open up a terminal, navigate to the root directory of this repository, and type `julia` into the command line. We'll start a session with
```julia
using Pkg; Pkg.activate(".")
``` 
To generate the results from Example 1 of the paper, run the two scripts simulating the dynamical system with process and measurement noise (respectively).
```julia
include("src/counter_example_dx.jl")
include("src/counter_example_dy.jl")
```

### Plotting the Results

To plot the results in the paper from existing data, each sub-folder in `src/` has one or more plotting scripts:

- `src/nonlinear-system/plot_paperfig.jl`: produces Figure 7 in the paper.
- `src/stability-guarantees/plot_paperfig.jl`: produces Figure 8 in the paper.
- `src/model-uncertainty/plot_youla_residual_results.jl`: produces Figures 10-11 in the paper.
- `src/model-uncertainty/plot_ablation_results.jl`: produces Figure 12 in the paper.

Note that Figure 9 was plotted in MATLAB with the script `matlab/lcp_lqg_paperplots.m`.

### Re-training from Scratch

If you are interested in re-training the networks used to produce the results of this paper from scratch, each folder under `src/` has a `train_batch.jl` script. These scripts are set up to run 10 random model initializations in parallel. To begin, open a distributed Julia REPL by typing the following into a terminal:
```
julia -t 1 -p 10
```
This will create a Julia session with 10 parallel workers, each with a single thread. If your CPU can handle more than 10 threads, feel free to increas to `-t 2` or `-t 3` for faster training. All results will be saved in a `results/<experiment_name>/batch/` directory, which will be created automatically if it does not exist.

To reproduce the results in Figure 10 (c) for the uncertain linear cartpole example, be sure to run `src/model-uncertainty/choose_best_models.jl` and `src/model-uncertainty/eval_adaptation.jl` (in that order) before plotting anything.

### A Note on Terminology

Many of the variable names in the code refer to `Feedback-REN` or `Feedback-LSTM`. The "Feedback" architecture is called Residual-RL in the paper. Similarly for "Vanilla-MLP" and "Vanilla-LSTM", which are called "Black-box MLP" and "Black-box LSTM".


## Experimental Details

The following tables summarise hyperparameters used to train the policies in Sections VII.B–VII.D of the paper. All policies were trained with analytic policy gradients using Adam, with gradient norms clipped at 1. Learning rates follow a piecewise-constant schedule: starting at the specified value, reduced by 10× at the first fraction of total training epochs, and again by 100× at the second. All neural models were chosen to have a similar number of learnable parameters.

### Table 1: Main hyperparameters

| Hyperparameter | Section VII.B | Section VII.C | Section VII.D |
|---|---|---|---|
| Training batches $N$ | 32 | 50 | 64 |
| Maximum steps in training horizon $T$ | 200 | 10,000 | 800 |
| Rollouts per episode $k_{\max}$ | 1 | 50 | 4 |
| Training epochs | 2500 | 4000 | 1600 |
| Activation (RENs) | `tanh` | `tanh` | `ReLU` |
| No. states $n_x$ (RENs) | 16 | 2 | 84 |
| No. neurons $n_v$ (RENs) | 128 | 8 | 128 |
| No. states (Linear REN) | – | – | 138 |
| No. neurons (LSTM) | – | – | 154 |
| No. neurons (MLP) | – | – | 177 |
| lr (Youla-REN) | $3 \times 10^{-4}$ | $10^{-2}$ | $10^{-3}$ |
| lr (Residual-REN) | – | $3 \times 10^{-4}$ | $10^{-3}$ |
| lr (Residual-LSTM) | – | – | $10^{-2}$ |
| lr schedule | 1/2, 3/4 | 1/4, 3/4 | see Table 2 |

$k_{\max}$ is the number of consecutive training rollouts that share a state trajectory before the environment is reset (i.e., `max_steps / train_horizon` in the code). Splitting the trajectory is useful for long time horizons (similar to multiple shooting in trajectory optimisation). In Section VII.B, the time horizons are short so the state is reset every epoch.

### Table 2: Section VII.D policy-specific settings

The weighting filter $\mathcal{W}_2(s) = (s+3)^4 / \big(\nu\,(s+50)^4\big)$ is defined in Section VII.D of the paper, and $\gamma$ is the user-imposed Lipschitz bound on the learnable component $\mathcal{Q}$.

| Policy | lr schedule | $\nu$ in $\mathcal{W}_2$ | Lipschitz bound $\gamma$ |
|---|---|---|---|
| Youla-γREN | 1/2, 7/8 | $5 \times 10^{-4}$ | 1.7 |
| Youla-γREN (linear) | 1/2, 7/8 | $5 \times 10^{-4}$ | 1.7 |
| Youla-γREN (no filter) | 3/4, 7/8 | – | 120 |
| Residual-γREN | 1/2, 7/8 | $10^{-2}$ | 0.15 |
| Residual-LSTM | 3/4, 7/8 | – | – |

### Table 3: Black-box policies (Section VII.D.3, Figure 13)

The black-box MLP and LSTM policies in Figure 13 were trained for 4× longer than the stable architectures in Table 2 (6400 epochs), and results reported in the paper are the best-performing policy after sweeping the learning rate over $\{10^{-4}, 10^{-3}, 5 \times 10^{-3}, 10^{-2}\}$ with 3 random seeds per learning rate.

| Policy | No. neurons | Training epochs | lr (selected) | lr schedule |
|---|---|---|---|---|
| Black-box MLP | $n_v = 177$ | 6400 | $5 \times 10^{-3}$ | 3/4, 7/8 |
| Black-box LSTM | $n_v = 154$ | 6400 | $10^{-2}$ | 3/4, 7/8 |

All other settings (training horizon, batch size, noise, etc.) match those in Table 1, Section VII.D. Corresponding training scripts are in `src/model-uncertainty/test-vanilla/`.

## Contact

Please contact Nic Barbara (nicholas.barbara@sydney.edu.au) with any questions.
