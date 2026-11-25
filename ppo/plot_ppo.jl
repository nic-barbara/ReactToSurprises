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
# Plot: Youla-REN and black-box LSTMs trained with APG and PPO
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

function plot_seeds!(ax, costs, colour, label)
    for (i, c) in enumerate(costs)
        lines!(ax, epochs(c), c; linewidth=1.25, color=(colour, α_seeds),
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


#######################################################################
#
# Response to reviewers: 2x2 grid of LSTM (top) and MLP (bottom) policies
# trained with APG (left) and PPO (right). No highlighted models.
#
#######################################################################

with_theme(theme_latexfonts()) do

    fig = Figure(size=(800, 560), fontsize=19)
    ga = fig[1,1] = GridLayout()

    xticks = [0, 3000, 6000]
    costs_vl_apg_all = [e.cost for e in costs_vl_apg]
    panels = [(costs_vl_apg_all, colour_vl),
              (costs_vl_ppo, colour_vl),
              (costs_vm_apg, colour_vm),
              (costs_vm_ppo, colour_vm)]
    n = maximum(length, vcat(costs_vl_apg_all, costs_vl_ppo, costs_vm_apg, costs_vm_ppo, [μ_yr]))

    axs = [Axis(ga[i,j], yscale=Makie.log10, xticks=xticks, titlefont=:regular,
                title=(i == 1 ? ["Analytic Gradients", "PPO"][j] : ""),
                xticklabelsvisible=(i == 2), yticklabelsvisible=(j == 1))
           for i in 1:2, j in 1:2]
    linkaxes!(axs...)
    Label(ga[3,1:2], text="Training epochs")
    Label(ga[1:2,0], text="Time-averaged test cost", rotation=π/2)
    colgap!(ga, 1, 5)
    colgap!(ga, 2, 15)
    rowgap!(ga, 1, 15)
    rowgap!(ga, 2, 5)

    for (ax, (costs, colour)) in zip(permutedims(axs)[:], panels)
        plot_loss!(ax, epochs(μ_yr), μ_yr, max_yr, min_yr; color=colour_yr, label="Youla-γREN")
        plot_seeds!(ax, costs, colour, nothing)
        plot_refs!(ax, n)
    end

    elements = [
        LineElement(color=colour_yr, linewidth=3),
        LineElement(color=(colour_vl, α_seeds), linewidth=1.25),
        LineElement(color=(colour_vm, α_seeds), linewidth=1.25),
        LineElement(color=colour_b, linestyle=:dash, linewidth=2),
        LineElement(color=colour_o, linestyle=:dash, linewidth=2),
    ]
    labels = ["Youla-γREN", "Black-box LSTM", "Black-box MLP", "Base", L"LQG (known $m_p$)"]
    Legend(fig[2,1], elements, labels, orientation=:horizontal, nbanks=2)

    resize_to_layout!(fig)
    save(joinpath(@__DIR__, "results/lcp_vanilla_ppo_response.pdf"), fig)
end
