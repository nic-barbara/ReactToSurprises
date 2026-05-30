using BSON
using CairoMakie
using Random
using RobustNeuralNetworks
using Statistics

include(joinpath(@__DIR__, "../models.jl"))
include(joinpath(@__DIR__, "../functions.jl"))
include(joinpath(@__DIR__, "../setup_mass.jl"))


# (plotting code generated with the help of claude.ai)


#######################################################################
#
# Hyperparameters (must match training script)
#
#######################################################################

batch_ids = [0, 1, 2]
learning_rates = [1e-5, 1e-4, 1e-3, 5e-3, 1e-2]
nlr = length(learning_rates)
nv_lstm = 154
nv_mlp = 177

# Best models (must match showcase selection script)
best_lstm = (lr=5e-3, batch_id=1)
best_mlp = (lr=1e-2, batch_id=1)


#######################################################################
#
# Load vanilla (MLP/LSTM) cost data
#
#######################################################################

fpath_vanilla = joinpath(@__DIR__, "../../../results/model-uncertainty/batch-vanilla/")
fnames_vanilla = get_bson_files(fpath_vanilla)

load_data(fname, key) = BSON.load(fname)[key]

function load_costs_by_lr(fnames, key)
    # Store (cost, batch_id) tuples so we can identify the best model
    costs_by_lr = [[] for _ in 1:nlr]
    for fname in fnames
        for (li, lr) in enumerate(learning_rates)
            if occursin("_lr$(lr)_", fname)
                # Extract batch_id from filename: ..._v<id>.bson
                m = match(r"_v(\d+)\.bson$", fname)
                bid = parse(Int, m.captures[1])
                push!(costs_by_lr[li], (cost=load_data(fname, key), batch_id=bid))
            end
        end
    end
    return costs_by_lr
end

costs_vl_by_lr = load_costs_by_lr(fnames_vanilla, "costs_vl")
costs_vm_by_lr = load_costs_by_lr(fnames_vanilla, "costs_vm")


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
# Generate cost trajectory rollouts
#
#######################################################################

fname_youla_best = string(fpath_youla, "showcase/lcp_$(label)_84nx_128nv_best.bson")
fname_vanilla_best = string(fpath_vanilla, "showcase/lcp_vanilla_$(label)_$(nv_lstm)nvl_$(nv_mlp)nvm_best.bson")

youla_ren = REN(load_data(fname_youla_best, "youla_ren"))
vanilla_lstm = load_data(fname_vanilla_best, "vanilla_lstm")
vanilla_mlp = load_data(fname_vanilla_best, "vanilla_mlp")

base_ren = deepcopy(youla_ren)
set_output_zero!(base_ren)

rng_() = Xoshiro(1)
test_batches = 64
train_horizon = 800
test_horizon = 12 * train_horizon

nf = K_base.nx_filter
test_states_ren = init_states!(G, base_ren.nx, nf, test_batches; rng=rng_())
test_states_vl = init_states!(G, vanilla_lstm.nx, nf, test_batches; rng=rng_())
test_states_vm = init_states!(G, vanilla_mlp.nx, nf, test_batches; rng=rng_())
test_states_opt = init_states!(Gopt, 0, 0, test_batches; rng=rng_())

w_test = procnoise(G, test_batches, test_horizon; rng=rng_())
v_test = measnoise(G, test_batches, test_horizon; rng=rng_())

function sim_test(model, test_states, K; youla=true)
    _, _, traj = simulate(G, model, K, cost, test_states; youla,
                          w=w_test, v=v_test, horizon=test_horizon,
                          log_states=true)
    costs = cost.(traj[1], traj[2])
    return cumsum(costs) ./ (1:test_horizon), traj
end

J_bs, traj_b = sim_test(base_ren, test_states_ren, K_base)
J_yr, traj_yr = sim_test(youla_ren, test_states_ren, K_base)
J_vl, traj_vl = sim_test(vanilla_lstm, test_states_vl, K_vanilla; youla=false)
J_vm, traj_vm = sim_test(vanilla_mlp, test_states_vm, K_vanilla; youla=false)

_, _, traj_o = simulate(Gopt, cost, test_states_opt; horizon=test_horizon,
                        w=w_test, v=v_test, log_states=true)
J_os = cumsum(cost.(traj_o[1], traj_o[2])) ./ (1:test_horizon)


#######################################################################
#
# Plotting setup
#
#######################################################################

colours = Makie.wong_colors()
colour_yr = colours[2]
colour_b = colours[4]
colour_o = colours[1]
colour_vl = :dodgerblue
colour_vm = :coral

n_yr = length(μ_yr)
xc_yr = vcat(1, 5:5:((n_yr - 1) * 5))

function plot_loss!(ax, xc, μ, cmax, cmin; linewidth=2, kwargs...)
    colour = kwargs[:color]
    band!(ax, xc, cmax, cmin, color=(colour, 0.3))
    lines!(ax, xc, μ; linewidth, kwargs...)
end

function lr_label_str(lr)
    e = Int(floor(log10(lr)))
    c = lr / 10.0^e
    if c ≈ 1.0
        return L"10^{%$e}"
    else
        ci = Int(round(c))
        return L"%$ci \times 10^{%$e}"
    end
end


#######################################################################
#
# Plot 1: Training curves (one figure per model × learning rate)
#
#######################################################################

with_theme(theme_latexfonts()) do
    for (model_name, costs_by_lr, best) in [
        ("LSTM", costs_vl_by_lr, best_lstm),
        ("MLP",  costs_vm_by_lr, best_mlp)
    ]
        for (li, lr) in enumerate(learning_rates)

            costs_li = costs_by_lr[li]
            isempty(costs_li) && continue

            n_v = length(costs_li[1].cost)
            xc_v = vcat(1, 5:5:((n_v - 1) * 5))
            n_max = max(n_v, n_yr)
            xc_ref = vcat(1, 5:5:((n_max - 1) * 5))

            fig = Figure(size=(450, 400), fontsize=19)
            ga = fig[1,1] = GridLayout()
            gb = fig[2,1] = GridLayout()
            rowgap!(fig.layout, 5)

            lr_str = lr_label_str(lr)
            ax = Axis(ga[1,1],
                xlabel="Training epochs",
                ylabel="Time-averaged test cost",
                title=L"%$model_name ($\mathrm{lr}$ = %$lr_str)",
                yscale=Makie.log10
            )

            # Draw non-best seed curves first (thin, transparent)
            first_regular = true
            for entry in costs_li
                entry.batch_id == best.batch_id && lr == best.lr && continue
                xc_si = vcat(1, 5:5:((length(entry.cost) - 1) * 5))
                label = first_regular ? model_name : nothing
                lines!(ax, xc_si, entry.cost; linewidth=1.25, color=(:grey, 0.5), label)
                first_regular = false
            end

            # Draw best model on top (thicker, solid grey) if it's in this lr group
            if lr == best.lr
                best_entry = findfirst(e -> e.batch_id == best.batch_id, costs_li)
                if !isnothing(best_entry)
                    c = costs_li[best_entry].cost
                    xc_si = vcat(1, 5:5:((length(c) - 1) * 5))
                    lbl = first_regular ? model_name : nothing
                    lines!(ax, xc_si, c; linewidth=2, color=:grey, label=lbl)
                end
            end

            # Youla-REN aggregated
            plot_loss!(ax, xc_yr, μ_yr, max_yr, min_yr;
                       color=colour_yr, label="Youla-γREN")

            # Reference lines
            lines!(ax, xc_ref, J_base * ones(n_max), linestyle=:dash, color=colour_b,
                   label="Base", linewidth=2)
            lines!(ax, xc_ref, J_opt * ones(n_max), linestyle=:dash, color=colour_o,
                   label=L"LQG (known $m_p$)", linewidth=2)

            xlims!(ax, 0, max(xc_v[end], xc_yr[end]))
            ylims!(ax, J_opt * 0.8, 1e35)

            Legend(gb[1,1], ax, orientation=:horizontal, nbanks=2)

            resize_to_layout!(fig)
            lr_save = replace(string(lr), "." => "p")
            save(string(fpath_vanilla, "lcp_vanilla_$(lowercase(model_name))_lr$(lr_save).pdf"), fig)
        end
    end
end


#######################################################################
#
# Plot 2: Cost rollouts
#
#######################################################################

t = LinRange(0, length(J_bs) / train_horizon, length(J_bs))

with_theme(theme_latexfonts()) do

    fig = Figure(size=(500, 450), fontsize=19)
    ga = fig[1,1] = GridLayout()
    gb = fig[2,1] = GridLayout()
    rowgap!(fig.layout, 5)

    ax = Axis(ga[1,1],
        xlabel="Time (test horizon/train horizon)",
        ylabel="Time-averaged test cost"
    )

    linewidth = 2
    lines!(ax, t, J_yr; linewidth, color=colour_yr, label="Youla-γREN")
    lines!(ax, t, J_vl; linewidth, color=colour_vl, label="LSTM")
    lines!(ax, t, J_vm; linewidth, color=colour_vm, label="MLP")
    lines!(ax, t, J_bs; linewidth, color=colour_b, linestyle=:dash, label="Base")
    lines!(ax, t, J_os; linewidth, color=colour_o, linestyle=:dash, label=L"LQG (known $m_p$)")

    xlims!(ax, t[1], t[end])
    ylims!(ax, -2, 1.0 * J_base)

    Legend(gb[1,1], ax, orientation=:horizontal, nbanks=2)

    resize_to_layout!(fig)
    save(string(fpath_vanilla, "lcp_vanilla_rollouts.pdf"), fig)
end


#######################################################################
#
# Plot 3: Best LR training curves for both models
#
#######################################################################

# Get the costs at the best learning rate for each model
li_lstm = findfirst(==(best_lstm.lr), learning_rates)
li_mlp = findfirst(==(best_mlp.lr), learning_rates)
costs_vl_best_lr = costs_vl_by_lr[li_lstm]
costs_vm_best_lr = costs_vm_by_lr[li_mlp]

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

    # Youla-REN aggregated
    plot_loss!(ax, xc_yr, μ_yr, max_yr, min_yr;
               color=colour_yr, label="Youla-γREN")

    # MLP curves (thin, all same colour)
    first_mlp = true
    for entry in costs_vm_best_lr
        xc_si = vcat(1, 5:5:((length(entry.cost) - 1) * 5))
        label = first_mlp ? "Black-box MLP" : nothing
        lines!(ax, xc_si, entry.cost; linewidth=1.25, color=(colour_vm, 0.5), label)
        first_mlp = false
    end

    # LSTM curves (thin, all same colour), then highlight best
    first_lstm = true
    for entry in costs_vl_best_lr
        entry.batch_id == best_lstm.batch_id && continue
        xc_si = vcat(1, 5:5:((length(entry.cost) - 1) * 5))
        label = first_lstm ? "Black-box LSTM" : nothing
        lines!(ax, xc_si, entry.cost; linewidth=1.25, color=(colour_vl, 0.5), label)
        first_lstm = false
    end

    # Best LSTM on top
    best_entry = findfirst(e -> e.batch_id == best_lstm.batch_id, costs_vl_best_lr)
    if !isnothing(best_entry)
        c = costs_vl_best_lr[best_entry].cost
        xc_si = vcat(1, 5:5:((length(c) - 1) * 5))
        lbl = first_lstm ? "Black-box LSTM" : nothing
        lines!(ax, xc_si, c; linewidth=2, color=colour_vl, label=lbl)
    end

    # Reference lines
    n_max_all = max(
        maximum(length(e.cost) for e in costs_vl_best_lr),
        maximum(length(e.cost) for e in costs_vm_best_lr),
        n_yr
    )
    xc_ref = vcat(1, 5:5:((n_max_all - 1) * 5))
    lines!(ax, xc_ref, J_base * ones(n_max_all), linestyle=:dash, color=colour_b,
           label="Base", linewidth=2)
    lines!(ax, xc_ref, J_opt * ones(n_max_all), linestyle=:dash, color=colour_o,
           label=L"LQG (known $m_p$)", linewidth=2)

    xlims!(ax, 0, xc_ref[end])
    ylims!(ax, J_opt * 0.8, 1e35)

    Legend(gb[1,1], ax, orientation=:vertical)

    resize_to_layout!(fig)
    save(string(fpath_vanilla, "lcp_vanilla_combined_training.pdf"), fig)
end
