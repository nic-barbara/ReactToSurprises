# React to Surprises with the Youla-REN

This repository contains all the code for our upcoming journal paper *React to Surprises: The Youla-REN Class of Stable-by-Design Neural Feedback Controllers*. 

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

Many of the variable names in the code refer to `Feedback-REN` or `Feedback-LSTM`. The "Feedback" architecture is called Residual-RL in the paper.


## Contact

Please contact Nic Barbara (nicholas.barbara@sydney.edu.au) with any questions.
