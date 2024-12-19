using BSON
using CairoMakie
using Random
using RobustNeuralNetworks
using Statistics

include(joinpath(@__DIR__, "models.jl"))
include(joinpath(@__DIR__, "functions.jl"))
include(joinpath(@__DIR__, "setup_mass.jl"))


#######################################################################
#
# Get cost data from training
#
#######################################################################

# Load the costs
fpath = joinpath(@__DIR__, "../../results/model-uncertainty/batch-inputfilter/")
fnames = get_bson_files(fpath)

load_data(fname, key) = BSON.load(fname)[key]
costs_yr = load_data.(fnames, ("costs_yr"))
costs_lr = load_data.(fnames, ("costs_lr"))
costs_nf = load_data.(fnames, ("costs_yr_nf"))
costs_fr = load_data.(fnames, ("costs_fr"))
J_base = mean(load_data.(fnames, ("J_base")))
J_opt = mean(load_data.(fnames, ("J_opt")))

# Process costs
μ_yr, σ_yr, min_yr, max_yr = cost_stats(costs_yr)
μ_lr, σ_lr, min_lr, max_lr = cost_stats(costs_lr)
μ_nf, σ_nf, min_nf, max_nf = cost_stats(costs_nf)
μ_fr, σ_fr, min_fr, max_fr = cost_stats(costs_fr)


#######################################################################
#
# Generate cost trajectory rollouts
#
#######################################################################

# Load the models
name = "showcase/lcp_$(label)_84nx_128nv_best.bson"
fname = string(fpath, name)

youla_ren  = REN(load_data(fname, "youla_ren"))
youla_lren = REN(load_data(fname, "youla_lren"))
youla_ren_nf = REN(load_data(fname, "youla_ren_nf"))
fdbak_ren = REN(load_data(fname, "fdbak_ren"))

base_ren = deepcopy(youla_ren)
set_output_zero!(base_ren)

# Choose test data
rng_() = Xoshiro(1)
test_batches = 64
train_horizon = 800
test_horizon = 12 * train_horizon

nf = K_base.nx_filter
test_states_ren = init_states!(G, base_ren.nx, nf, test_batches; rng=rng_())
test_states_ren_nf = init_states!(G, base_ren.nx, 0, test_batches; rng=rng_())
test_states_lren = init_states!(G, youla_lren.nx, nf, test_batches; rng=rng_())
test_states_opt = init_states!(Gopt, 0, 0, test_batches; rng=rng_())

w_test = procnoise(G, test_batches, test_horizon; rng=rng_())
v_test = measnoise(G, test_batches, test_horizon; rng=rng_())

# Roll out the policies
function sim_test(model, test_states, K; youla=true)
    _, _, traj = simulate(G, model, K, cost, test_states; youla, 
                          w=w_test, v=v_test, horizon=test_horizon, 
                          log_states=true)
    costs = cost.(traj[1], traj[2])
    return cumsum(costs) ./ (1:test_horizon), traj
end

J_bs, traj_b = sim_test(base_ren, test_states_ren, K_base)
J_yr, traj_yr = sim_test(youla_ren, test_states_ren, K_base)
J_lr, traj_lr = sim_test(youla_lren, test_states_lren, K_base)
J_nf, traj_nf = sim_test(youla_ren_nf, test_states_ren_nf, K_base_nofilter)
J_fr, traj_fr = sim_test(fdbak_ren, test_states_ren, K_base; youla=false)

# Compute the optimal costs too
_, _, traj_o = simulate(Gopt, cost, test_states_opt; horizon=test_horizon, 
                        w=w_test, v=v_test, log_states=true)
J_os = cumsum(cost.(traj_o[1], traj_o[2])) ./ (1:test_horizon)
J_test = get_optimal_lqg_cost(Gopt, test_horizon) / test_horizon


#######################################################################
#
# Plot cost curves and cost rollouts
#
#######################################################################

# Useful for plotting
xc = vcat(1, 5:5:((length(μ_yr) - 1) * 5))      # (we only log costs every 5 points)
t = LinRange(0, length(J_bs) / train_horizon, length(J_bs))

function plot_loss!(ax, μ, cmax, cmin; linewidth=2, kwargs...)
    colour = kwargs[:color]
    band!(ax, xc, cmax, cmin, color = (colour, 0.3))
    lines!(ax, xc, μ; linewidth=linewidth, kwargs...)
end

# Use the Wong (2011) colour pallette
colours = Makie.wong_colors()
colour_yr = colours[2]
colour_lr = colours[6]
colour_nf = colours[5]
colour_fr = :grey
colour_fl = colours[3]
colour_b = colours[4]
colour_o = colours[1]
colour_n = :grey64

with_theme(theme_latexfonts()) do

    # Figure setup
    fig = Figure(size=(550,370), fontsize=19)
    ga1 = fig[1,1] = GridLayout()
    ga2 = fig[2,1] = GridLayout()

    ax1 = Axis(ga1[1,1], xlabel="Training epochs", ylabel="Time-averaged test cost", xticks=WilkinsonTicks(4; k_min=4, k_max=8))
    ax2 = Axis(ga1[1,2], xlabel="Test horizon/Train horizon", yticklabelsvisible=false)

    # Panel 1: loss curves
    n = length(costs_yr[1])
    plot_loss!(ax1, μ_yr, max_yr, min_yr; color=colour_yr, label="Youla-γREN")
    plot_loss!(ax1, μ_lr, max_lr, min_lr; color=colour_lr, label="Youla-γREN (linear)")
    plot_loss!(ax1, μ_nf, max_nf, min_nf; color=colour_nf, label="Youla-γREN (no filter)")
    plot_loss!(ax1, μ_fr, max_fr, min_fr; color=colour_fr, label="Residual-γREN")
    
    lines!(ax1, xc, J_base*ones(n), linestyle=:dash, color=colour_b, label="Base", linewidth=2)
    lines!(ax1, xc, J_opt*ones(n) , linestyle=:dash, color=colour_o, label=L"LQG (known $m_p$)", linewidth=2)
    
    xlims!(ax1, 0, xc[end])
    ylims!(ax1, -2, 1.2*J_base)

    # Panel 2: cost rollouts
    linewidth=2
    lines!(ax2, t, J_yr; linewidth, color=colour_yr)
    lines!(ax2, t, J_lr; linewidth, color=colour_lr, label="Youla-γREN (linear)")
    lines!(ax2, t, J_nf; linewidth, color=colour_nf, label="Youla-γREN (no filter)")
    lines!(ax2, t, J_fr; linewidth, color=colour_fr, label="Residual-γREN")
    lines!(ax2, t, J_bs; linewidth, color=colour_b, linestyle=:dash)
    lines!(ax2, t, J_os; linewidth, color=colour_o, linestyle=:dash)
   
    xlims!(ax2, t[1], t[end])
    ylims!(ax2, -2, 1.2*J_base)
    
    # Add legend and save
    Legend(ga2[1,1], ax1, orientation=:horizontal, nbanks=3)
    save(string(
        @__DIR__, "/../../results/model-uncertainty/lcp_ablation_costs.pdf"
        ), fig
    )
end
