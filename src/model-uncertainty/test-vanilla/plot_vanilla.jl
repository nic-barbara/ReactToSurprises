using BSON
using CairoMakie
using Random
using RobustNeuralNetworks
using Statistics

include(joinpath(@__DIR__, "../models.jl"))
include(joinpath(@__DIR__, "../functions.jl"))
include(joinpath(@__DIR__, "../setup_mass.jl"))


#######################################################################
#
# Hyperparameters (must match training script)
#
#######################################################################

batch_ids = [0, 1, 2]
learning_rates = [1e-5, 1e-4, 1e-3, 5e-3, 1e-2]
nlr = length(learning_rates)


#######################################################################
#
# Load vanilla (MLP/LSTM) cost data
#
#######################################################################

fpath_vanilla = joinpath(@__DIR__, "../../../results/model-uncertainty/batch-vanilla/")
fnames_vanilla = get_bson_files(fpath_vanilla)

load_data(fname, key) = BSON.load(fname)[key]

function load_costs_by_lr(fnames, key)
    costs_by_lr = [[] for _ in 1:nlr]
    for fname in fnames
        for (li, lr) in enumerate(learning_rates)
            if occursin("_lr$(lr)_", fname)
                push!(costs_by_lr[li], load_data(fname, key))
            end
        end
    end
    return costs_by_lr
end

costs_vl_by_lr = load_costs_by_lr(fnames_vanilla, "costs_vl")
costs_vm_by_lr = load_costs_by_lr(fnames_vanilla, "costs_vm")

function cost_stats_by_lr(costs_by_lr)
    μs, mins, maxs = [], [], []
    for costs in costs_by_lr
        if isempty(costs)
            push!(μs, nothing)
            push!(mins, nothing)
            push!(maxs, nothing)
        else
            μ, _, cmin, cmax = cost_stats(costs)
            push!(μs, μ)
            push!(mins, cmin)
            push!(maxs, cmax)
        end
    end
    return μs, mins, maxs
end

μ_vl, min_vl, max_vl = cost_stats_by_lr(costs_vl_by_lr)
μ_vm, min_vm, max_vm = cost_stats_by_lr(costs_vm_by_lr)


#######################################################################
#
# Load Youla-REN cost data
#
#######################################################################

fpath_youla = joinpath(@__DIR__, "../../../results/model-uncertainty/batch/")
fnames_youla = get_bson_files(fpath_youla)

costs_yr = load_data.(fnames_youla, "costs_yr")
μ_yr, σ_yr, min_yr, max_yr = cost_stats(costs_yr)

J_base = mean(load_data.(fnames_youla, "J_base"))
J_opt = mean(load_data.(fnames_youla, "J_opt"))


#######################################################################
#
# Plot cost curves
#
#######################################################################

# xc vectors (we only log costs every 5 steps, except first point)
n_v = length(μ_vl[findfirst(!isnothing, μ_vl)])
n_yr = length(μ_yr)
xc_v = vcat(1, 5:5:((n_v - 1) * 5))
xc_yr = vcat(1, 5:5:((n_yr - 1) * 5))
n_max = max(n_v, n_yr)
xc_ref = vcat(1, 5:5:((n_max - 1) * 5))

function plot_loss!(ax, xc, μ, cmax, cmin; linewidth=2, kwargs...)
    colour = kwargs[:color]
    band!(ax, xc, cmax, cmin, color=(colour, 0.3))
    lines!(ax, xc, μ; linewidth, kwargs...)
end

# Colours: Wong (2011) palette
colours = Makie.wong_colors()
colour_yr = colours[2]
colour_b = colours[4]
colour_o = colours[1]
lr_colours = [colours[3], colours[5], colours[6], colours[7], :grey]

with_theme(theme_latexfonts()) do

    fig = Figure(size=(800, 450), fontsize=19)
    ga = fig[1,1] = GridLayout()
    gb = fig[2,1] = GridLayout()

    ax1 = Axis(
        ga[1,1], xlabel="Training epochs", ylabel="Time-averaged test cost",
        title="LSTM", xticks=WilkinsonTicks(4; k_min=4, k_max=8),
        yscale=Makie.log10
    )
    ax2 = Axis(
        ga[1,2], xlabel="Training epochs", yticklabelsvisible=false,
        title="MLP", xticks=WilkinsonTicks(4; k_min=4, k_max=8),
        yscale=Makie.log10
    )

    # LSTM curves by learning rate
    for (li, lr) in enumerate(learning_rates)
        if !isnothing(μ_vl[li])
            plot_loss!(ax1, xc_v, μ_vl[li], max_vl[li], min_vl[li];
                       color=lr_colours[li], label="lr = $lr")
        end
    end

    # MLP curves by learning rate
    for (li, lr) in enumerate(learning_rates)
        if !isnothing(μ_vm[li])
            plot_loss!(ax2, xc_v, μ_vm[li], max_vm[li], min_vm[li];
                       color=lr_colours[li], label="lr = $lr")
        end
    end

    # Add Youla-REN and reference lines to both panels
    for ax in [ax1, ax2]
        plot_loss!(ax, xc_yr, μ_yr, max_yr, min_yr;
                   color=colour_yr, label="Youla-γREN")
        lines!(ax, xc_ref, J_base * ones(n_max), linestyle=:dash, color=colour_b,
               label="Base", linewidth=2)
        lines!(ax, xc_ref, J_opt * ones(n_max), linestyle=:dash, color=colour_o,
               label=L"LQG (known $m_p$)", linewidth=2)

        xlims!(ax, 1, max(xc_v[end], xc_yr[end]))
        # ylims!(ax, -2, 1.2 * J_base)
    end

    Legend(gb[1,1], ax1, orientation=:horizontal, nbanks=2)
    save(string(fpath_vanilla, "lcp_vanilla_costs.pdf"), fig)
end
