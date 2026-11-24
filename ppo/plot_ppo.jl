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

# Best APG learning rate and seed (from choose_best_vanilla.jl)
best_lstm_apg = (lr=5e-3, batch_id=1)

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

ppo_files = filter(f -> startswith(basename(f), "ppo_lstm_") && endswith(f, ".csv"),
                   readdir(ppo_dir; join=true))
costs_vl_ppo = read_ppo_costs.(ppo_files)
println("PPO runs plotted: ", basename.(ppo_files))


#######################################################################
#
# Plot: Youla-REN and black-box LSTMs trained with APG and PPO
#
#######################################################################

colours = Makie.wong_colors()
colour_yr = colours[2]
colour_b = colours[4]
colour_o = colours[1]
colour_vl = :dodgerblue

# Costs are logged every 5 epochs (except the first point)
epochs(c) = vcat(1, 5:5:((length(c) - 1) * 5))

function plot_loss!(ax, xc, μ, cmax, cmin; linewidth=2, kwargs...)
    colour = kwargs[:color]
    band!(ax, xc, cmax, cmin, color=(colour, 0.3))
    lines!(ax, xc, μ; linewidth, kwargs...)
end

function plot_seeds!(ax, costs, colour, label)
    for (i, c) in enumerate(costs)
        lines!(ax, epochs(c), c; linewidth=1.25, color=(colour, 0.5),
               label=(i == 1 ? label : nothing))
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

# Paper figure: same size as the original Fig. 13, APG (left) and PPO (right)
# with a shared y-axis and a shared legend along the bottom.
with_theme(theme_latexfonts()) do

    fig = Figure(size=(550, 320), fontsize=19)
    ga = fig[1,1] = GridLayout()

    xticks = [0, 3000, 6000]
    ax1 = Axis(ga[1,1], title="Analytic Gradients", titlefont=:regular,
               ylabel="Time-averaged test cost", ylabelsize=15, yscale=Makie.log10, xticks=xticks)
    ax2 = Axis(ga[1,2], title="PPO", titlefont=:regular,
               yscale=Makie.log10, yticklabelsvisible=false, xticks=xticks)
    Label(ga[2,1:2], text="Training epochs")
    linkyaxes!(ax1, ax2)
    colgap!(ga, 15)
    rowgap!(ga, 5)

    n = maximum(length, vcat(costs_vl_ppo, others_vl_apg, [best_vl_apg, μ_yr]))
    for ax in (ax1, ax2)
        plot_loss!(ax, epochs(μ_yr), μ_yr, max_yr, min_yr; color=colour_yr, label="Youla-γREN")
    end
    plot_seeds!(ax1, others_vl_apg, colour_vl, "Black-box LSTM")
    lines!(ax1, epochs(best_vl_apg), best_vl_apg; linewidth=2, color=colour_vl)
    plot_seeds!(ax2, costs_vl_ppo, colour_vl, nothing)
    plot_refs!(ax1, n)
    plot_refs!(ax2, n)

    Legend(fig[2,1], ax1, orientation=:horizontal, nbanks=2)

    resize_to_layout!(fig)
    save(joinpath(@__DIR__, "results/lcp_vanilla_ppo_combined_training.pdf"), fig)
end

