using BSON
using CairoMakie
using Random
using RobustNeuralNetworks
using Statistics

include(joinpath(@__DIR__, "../src/model-uncertainty/functions.jl"))
include(joinpath(@__DIR__, "../src/model-uncertainty/setup_mass.jl"))


#######################################################################
#
# Settings (must match src/model-uncertainty/test-vanilla/)
#
#######################################################################

# Best APG learning rates and LSTM seed (from choose_best_vanilla.jl)
best_lstm_apg = (lr=5e-3, batch_id=1)
best_mlp_apg = (lr=1e-2,)

# PPO runs
ppo_dir = joinpath(@__DIR__, "results/")    # all seeds found here are plotted


#######################################################################
#
# Load APG (black-box and Youla-REN) results
#
#######################################################################

load_data(fname, key) = BSON.load(fname)[key]

fpath_vanilla = joinpath(@__DIR__, "../results/model-uncertainty/batch-vanilla/")
fnames_vanilla = get_bson_files(fpath_vanilla)
fnames_lr(lr) = filter(f -> occursin("_lr$(lr)_", f), fnames_vanilla)
get_batch_id(fname) = parse(Int, match(r"_v(\d+)\.bson$", fname).captures[1])

costs_vl_apg = [(cost=load_data(f, "costs_vl"), batch_id=get_batch_id(f))
                for f in fnames_lr(best_lstm_apg.lr)]
costs_vm_apg = [load_data(f, "costs_vm") for f in fnames_lr(best_mlp_apg.lr)]

fpath_youla = joinpath(@__DIR__, "../results/model-uncertainty/batch/")
fnames_youla = get_bson_files(fpath_youla)
costs_yr = load_data.(fnames_youla, "costs_yr")
μ_yr, σ_yr, min_yr, max_yr = cost_stats(costs_yr)

J_base = mean(load_data.(fnames_youla, "J_base"))
J_opt = mean(load_data.(fnames_youla, "J_opt"))


#######################################################################
#
# Load PPO results
#
#######################################################################

# Read test costs from CSV (header "epoch,test_cost,ep_len_mean")
function read_ppo_costs(fname)
    lines = readlines(fname)[2:end]
    return [parse(Float64, split(line, ",")[2]) for line in lines]
end

function load_ppo(policy)
    files = filter(f -> startswith(basename(f), "ppo_$(policy)_") && endswith(f, ".csv"),
                   readdir(ppo_dir; join=true))
    println("PPO runs plotted: ", basename.(files))
    return read_ppo_costs.(files)
end
costs_vl_ppo = load_ppo("lstm")
costs_vm_ppo = load_ppo("mlp")


#######################################################################
#
# Plot: Youla-REN and black-box policies trained with APG and PPO
#
#######################################################################

colours = Makie.wong_colors()
colour_yr = colours[2]
colour_b = colours[4]
colour_o = colours[1]
colour_vl = :dodgerblue
colour_vm = :coral
α_seeds = 0.35     # opacity of the individual black-box curves

# Costs are logged every 5 epochs (except the first point)
epochs(c) = vcat(1, 5:5:((length(c) - 1) * 5))

# Drawn on top of all other lines (z = 1) so it is not hidden by the black-box curves
function plot_loss!(ax, xc, μ, cmax, cmin; linewidth=3, kwargs...)
    colour = kwargs[:color]
    b = band!(ax, xc, cmax, cmin, color=(colour, 0.4))
    l = lines!(ax, xc, μ; linewidth, kwargs...)
    translate!.((b, l), 0, 0, 1)
end

# z < 0 draws the curves underneath the others (without changing the legend order)
function plot_seeds!(ax, costs, colour, label; z=0)
    for (i, c) in enumerate(costs)
        l = lines!(ax, epochs(c), c; linewidth=1.25, color=(colour, α_seeds),
                   label=(i == 1 ? label : nothing))
        translate!(l, 0, 0, z)
    end
end

# Highlight the same best APG LSTM as in the original figure
i_best = findfirst(e -> e.batch_id == best_lstm_apg.batch_id, costs_vl_apg)
best_vl_apg = costs_vl_apg[i_best].cost
others_vl_apg = [e.cost for (i, e) in enumerate(costs_vl_apg) if i != i_best]

function plot_refs!(ax, n)
    xc = epochs(ones(n))
    lines!(ax, xc, J_base * ones(n), linestyle=:dash, color=colour_b, linewidth=2, label="Base")
    lines!(ax, xc, J_opt * ones(n), linestyle=:dash, color=colour_o, linewidth=2,
           label=L"LQG (known $m_p$)")
    xlims!(ax, 0, xc[end])
    ylims!(ax, J_opt * 0.8, 1e35)
end

# Single panel in the style of the original Fig. 13: policies trained with APG
# (blue) and PPO (coral), with the legend on the right. The APG policy `best`
# is highlighted if given.
function plot_apg_ppo(fname, apg, ppo, model; best=nothing)
    with_theme(theme_latexfonts()) do

        fig = Figure(size=(550, 320), fontsize=19)
        ax = Axis(fig[1,1], xlabel="Training epochs", ylabel="Time-averaged test cost",
                  yscale=Makie.log10, xticks=WilkinsonTicks(3; k_min=3, k_max=4))

        n = maximum(length, vcat(apg, ppo, [μ_yr], isnothing(best) ? [] : [best]))
        plot_loss!(ax, epochs(μ_yr), μ_yr, max_yr, min_yr; color=colour_yr, label="Youla-γREN")
        plot_seeds!(ax, apg, colour_vl, "Black-box $model")
        isnothing(best) || lines!(ax, epochs(best), best; linewidth=2, color=colour_vl)
        plot_seeds!(ax, ppo, colour_vm, "Black-box $model\n(PPO)"; z=-1)
        plot_refs!(ax, n)

        Legend(fig[1,2], ax, orientation=:vertical)

        resize_to_layout!(fig)
        save(joinpath(@__DIR__, "results", fname), fig)
    end
end

# Paper: LSTMs only, best APG LSTM highlighted as in the original Fig. 13
plot_apg_ppo("lcp_vanilla_ppo_combined_training.pdf", others_vl_apg, costs_vl_ppo, "LSTM";
             best=best_vl_apg)

# Response to reviewers: one figure each for the LSTMs and MLPs, no highlighted models
plot_apg_ppo("lcp_vanilla_ppo_response_lstm.pdf", [e.cost for e in costs_vl_apg], costs_vl_ppo, "LSTM")
plot_apg_ppo("lcp_vanilla_ppo_response_mlp.pdf", costs_vm_apg, costs_vm_ppo, "MLP")
