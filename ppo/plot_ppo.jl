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
colour_ppo = :coral

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

with_theme(theme_latexfonts()) do

    fig = Figure(size=(550, 320), fontsize=19)
    ga = fig[1,1] = GridLayout()
    gb = fig[1,2] = GridLayout()

    ax = Axis(ga[1,1],
        xlabel="Training epochs",
        ylabel="Time-averaged test cost",
        yscale=Makie.log10,
        xticks=WilkinsonTicks(3; k_min=3, k_max=4)
    )

    plot_loss!(ax, epochs(μ_yr), μ_yr, max_yr, min_yr; color=colour_yr, label="Youla-γREN")
    plot_seeds!(ax, others_vl_apg, colour_vl, "Black-box LSTM")
    plot_seeds!(ax, costs_vl_ppo, colour_ppo, "Black-box LSTM\n(PPO)")
    lines!(ax, epochs(best_vl_apg), best_vl_apg; linewidth=2, color=colour_vl)

    n = maximum(length, vcat(costs_vl_ppo, others_vl_apg, [best_vl_apg, μ_yr]))
    xc = epochs(ones(n))
    lines!(ax, xc, J_base * ones(n), linestyle=:dash, color=colour_b, label="Base", linewidth=2)
    lines!(ax, xc, J_opt * ones(n), linestyle=:dash, color=colour_o,
           label=L"LQG (known $m_p$)", linewidth=2)

    xlims!(ax, 0, xc[end])
    ylims!(ax, J_opt * 0.8, 1e35)

    Legend(gb[1,1], ax, orientation=:vertical)

    resize_to_layout!(fig)
    save(joinpath(@__DIR__, "results/lcp_vanilla_ppo_combined_training.pdf"), fig)
end


#######################################################################
#
# Side-by-side version (APG left, PPO right) for the response to reviewers
#
#######################################################################

function plot_refs!(ax, n)
    xc = epochs(ones(n))
    lines!(ax, xc, J_base * ones(n), linestyle=:dash, color=colour_b, linewidth=2)
    lines!(ax, xc, J_opt * ones(n), linestyle=:dash, color=colour_o, linewidth=2)
    xlims!(ax, 0, xc[end])
    ylims!(ax, J_opt * 0.8, 1e35)
end

with_theme(theme_latexfonts()) do

    fig = Figure(size=(850, 320), fontsize=19)
    ga = fig[1,1] = GridLayout()
    gb = fig[1,2] = GridLayout()

    xticks = WilkinsonTicks(3; k_min=3, k_max=4)
    ax1 = Axis(ga[1,1], title="APG", xlabel="Training epochs",
               ylabel="Time-averaged test cost", yscale=Makie.log10, xticks=xticks)
    ax2 = Axis(ga[1,2], title="PPO", xlabel="Training epochs",
               yscale=Makie.log10, xticks=xticks)
    linkyaxes!(ax1, ax2)
    hideydecorations!(ax2, grid=false)
    colgap!(ga, 15)

    # APG (all seeds drawn the same, no highlighted model)
    all_vl_apg = [e.cost for e in costs_vl_apg]
    plot_loss!(ax1, epochs(μ_yr), μ_yr, max_yr, min_yr; color=colour_yr)
    plot_seeds!(ax1, all_vl_apg, colour_vl, nothing)
    plot_refs!(ax1, maximum(length, vcat(all_vl_apg, [μ_yr])))

    # PPO
    plot_loss!(ax2, epochs(μ_yr), μ_yr, max_yr, min_yr; color=colour_yr)
    plot_seeds!(ax2, costs_vl_ppo, colour_ppo, nothing)
    plot_refs!(ax2, maximum(length, vcat(costs_vl_ppo, [μ_yr])))

    # Shared legend
    elements = [
        LineElement(color=colour_yr, linewidth=2),
        LineElement(color=(colour_vl, 0.5), linewidth=1.25),
        LineElement(color=(colour_ppo, 0.5), linewidth=1.25),
        LineElement(color=colour_b, linestyle=:dash, linewidth=2),
        LineElement(color=colour_o, linestyle=:dash, linewidth=2),
    ]
    labels = ["Youla-γREN", "Black-box LSTM", "Black-box LSTM\n(PPO)", "Base", L"LQG (known $m_p$)"]
    Legend(gb[1,1], elements, labels, orientation=:vertical)

    resize_to_layout!(fig)
    save(joinpath(@__DIR__, "results/lcp_vanilla_ppo_side_by_side.pdf"), fig)
end
