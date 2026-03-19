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
fpath = joinpath(@__DIR__, "../../results/model-uncertainty/batch/")
fnames = get_bson_files(fpath)
    
load_data(fname, key) = BSON.load(fname)[key]
costs_yr = load_data.(fnames, ("costs_yr"))
costs_fl = load_data.(fnames, ("costs_fl"))
costs_lr = load_data.(fnames, ("costs_lr"))
costs_fr = load_data.(fnames, ("costs_fr"))
J_base = mean(load_data.(fnames, ("J_base")))
J_opt = mean(load_data.(fnames, ("J_opt")))

# Process costs
μ_yr, σ_yr, min_yr, max_yr = cost_stats(costs_yr)
μ_fl, σ_fl, min_fl, max_fl = cost_stats(costs_fl)
μ_lr, σ_lr, min_lr, max_lr = cost_stats(costs_lr)
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
fdbak_lstm = load_data(fname, "fdbak_lstm")

base_ren = deepcopy(youla_ren)
set_output_zero!(base_ren)

# Choose test data
rng_() = Xoshiro(1)
test_batches = 64
train_horizon = 800
test_horizon = 8 * train_horizon

nf = K_base.nx_filter
test_states_ren = init_states!(G, base_ren.nx, nf, test_batches; rng=rng_())
test_states_lstm = init_states!(G, fdbak_lstm.nx, 0, test_batches; rng=rng_())
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
J_fl, traj_fl = sim_test(fdbak_lstm, test_states_lstm, K_base_nofilter; youla=false)

# Compute the optimal costs too
_, _, traj_o = simulate(Gopt, cost, test_states_opt; horizon=test_horizon, 
                        w=w_test, v=v_test, log_states=true)
J_os = cumsum(cost.(traj_o[1], traj_o[2])) ./ (1:test_horizon)
J_test = get_optimal_lqg_cost(Gopt, test_horizon) / test_horizon


#######################################################################
#
# Generate cart/control trajectories for plotting
#
#######################################################################

# Choose new test data
test_batches = 2
train_horizon = 800
test_horizon = 8 * train_horizon

nf = K_base.nx_filter
test_states_ren = init_states!(G, base_ren.nx, nf, test_batches; rand_init=false)
test_states_lstm = init_states!(G, fdbak_lstm.nx, 0, test_batches; rand_init=false)
test_states_opt = init_states!(Gopt, 0, 0, test_batches; rand_init=false)

w_test = procnoise(G, test_batches, test_horizon; rng=Xoshiro(1))
v_test = measnoise(G, test_batches, test_horizon; rng=Xoshiro(1))

# Fix the params of interest
ρs = [0.145, 0.28]
G.A = G.Afunc.(ρs)
Gopt.sys.A = Gopt.sys.Afunc.(ρs)

# Roll out the policies
_, traj_b = sim_test(base_ren, test_states_ren, K_base)
_, traj_yr = sim_test(youla_ren, test_states_ren, K_base)
_, traj_fl = sim_test(fdbak_lstm, test_states_lstm, K_base_nofilter; youla=false)

# x-axis for plotting
npoints = length(traj_b[1])
t_traj = LinRange(0, npoints / train_horizon, npoints)

"""
Picks out either the cart position (`indx=1`) or the control
effort (`indx=2`) and sorts data into a nice order for plotting.
"""
function get_cartpole_variable(traj, indx)
    xs = stack(traj[indx])
    xs = permutedims(xs, (1,3,2)) # put batches last
    return xs[1,:,:]
end

"""
Plot trajectories on a specified axis. Same use of `indx`
as for `get_cartpole_variable`.
"""
function plot_trajs!(ax, traj, indx, ylim)
    x = get_cartpole_variable(traj, indx)
    colours = [:black, :red, :grey]
    labels = [L"small $m_p$", L"nominal $m_p$"]
    for k in axes(x,2)
        label = labels[k]
        color = colours[k]
        lines!(ax, t_traj, x[:,k]; linewidth=1.2, label, color, alpha=(1.0 - (k-1)*0.1))
    end
    xlims!(ax, t_traj[1], t_traj[end])
    ylims!(ax, ylim...)
end

# Axes limits should be consistent
xb_pos = get_cartpole_variable(traj_b, 1)
ylim_pos = (-3.2, 5.5)


#######################################################################
#
# Adaptation results
#
#######################################################################

# Load previously-computed data just for plotting
name = "showcase/param_variation_costs.bson"
fname = string(fpath, name)
data = BSON.load(fname)

ρs_adap = data["ρs"]
Js_bs = data["J_bs"]
Js_nom = data["J_nom"]
Js_yr = data["J_yr"]
Js_fl = data["J_fl"]
Js_os = data["J_os"]
Js_os_lti = data["J_os_lti"]
lo_range_max = data["lo_range_max"]
hi_range_min = data["hi_range_min"]

# To scale axes
scale_array = [0.135, 0.355] # For plot scaling only
normalize(x, y=scale_array) = (x .- minimum(y)) ./ (maximum(y) - minimum(y))


#######################################################################
#
# Plot final results
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
colour_lr = colours[5]
colour_nf = :grey
colour_fr = colours[6]
colour_fl = colours[3]
colour_b = colours[4]
colour_o = colours[1]
colour_n = :grey64

# Make the plot
with_theme(theme_latexfonts()) do

    # Figure setup
    linewidth = 2
    fig = Figure(size=(850, 670), fontsize=16)
    ga1 = fig[1,1] = GridLayout()
    ga2 = fig[1,2] = GridLayout()
    ga3 = fig[2,1] = GridLayout()
    ga4 = fig[2,2] = GridLayout()
    ga5 = fig[3,1:2] = GridLayout()
    colgap!(fig.layout, 50)

    # Training and rollout costs
    ax1 = Axis(ga1[1,1], xlabel="Training epochs", ylabel="Test cost", xticks=WilkinsonTicks(4; k_min=4, k_max=8))
    ax2 = Axis(ga2[1,1], xlabel="Test horizon/Train horizon", ylabel="Test cost")

    # Panel 1: loss curves
    n = length(costs_yr[1])
    plot_loss!(ax1, μ_yr, max_yr, min_yr; color=colour_yr, label="Youla-REN")
    plot_loss!(ax1, μ_lr, max_lr, min_lr; color=colour_lr, label="Youla-REN (linear)")
    plot_loss!(ax1, μ_fr, max_fr, min_fr; color=colour_fr, label="Residual-REN")
    plot_loss!(ax1, μ_fl, max_fl, min_fl; color=colour_fl, label="Residual-LSTM")
    lines!(ax1, xc, J_base*ones(n), linestyle=:dash, color=colour_b, label="Base controller", linewidth=2)
    lines!(ax1, xc, J_opt*ones(n) , linestyle=:dash, color=colour_o, label=L"LQG (LTV, known $m_p$)", linewidth=2)    
    xlims!(ax1, 0, xc[end])
    ylims!(ax1, -4, 1.2*J_base)
    
    # To trick the legend
    lines!(ax1, [0.1, 0.1], [0.1, 0.1]; color=:purple, linewidth, linestyle=:dash, alpha=0.8, label=L"LQG (LTI, known $m_p$)")
    lines!(ax1, [0.1, 0.1], [0.1, 0.1]; color=colour_n, linewidth, linestyle=:dash, label=L"LQG (LTI, nominal $m_p$)")

    # Panel 2: cost rollouts
    lines!(ax2, t, J_yr, linewidth=2, color=colour_yr, label="Youla-REN")
    lines!(ax2, t, J_fl, linewidth=2, color=colour_fl, label="Residual-LSTM")
   
    xlims!(ax2, t[1], t[end])
    ylims!(ax2, 0, 150)

    # Third panel: cost vs. pole mass
    xlabs = [0.15, 0.2, 0.34]
    ax3 = Axis(
        ga3[1,1], xminorticksvisible=true, xminorgridvisible=true, 
        xscale=Makie.logit, ylabel="Test cost", titlefont=:regular, 
        xlabel=L"Pole mass $m_p$ (kg)",
        xminorticks = IntervalsBetween(4),
        xticks = (normalize(xlabs), string.(xlabs)),
        yscale = Makie.log10
    )

    # Change x-scale for nice plotting
    x = normalize(ρs_adap)

    # Plot
    lines!(ax3, x, Js_yr; color=colour_yr, linewidth, label="Youla-REN")
    lines!(ax3, x, Js_fl; color=colour_fl, linewidth, label="Residual-LSTM")
    lines!(ax3, x, Js_os_lti; color=:purple, linewidth, linestyle=:dash, alpha=0.8)
    lines!(ax3, x, Js_nom; color=colour_n, linewidth, linestyle=:dash)

    xlims!(ax3, minimum(x), maximum(x))
    ylims!(ax3, 6, 10^3)

    # Fourth panel: trajectories
    ax41 = Axis(ga4[1,1], ylabel=L"x \ (\text{m})",  title="Youla-REN", xticklabelsvisible=false, titlefont=:regular, xticks=WilkinsonTicks(6), yticks=WilkinsonTicks(3))
    ax42 = Axis(ga4[2,1], ylabel=L"x \ (\text{m})", title="Residual-LSTM", xlabel="Test horizon/Train horizon", titlefont=:regular, xticks=WilkinsonTicks(6), yticks=WilkinsonTicks(3))

    plot_trajs!(ax41, traj_yr, 1, ylim_pos)
    plot_trajs!(ax42, traj_fl, 1, ylim_pos)
    axislegend(ax41, position=:rt, linewidth=10, orientation=:horizontal)

    # Format
    ax1.width = 270
    ax2.width = 270
    ax3.width = 270
    Legend(ga5[1,1], ax1, orientation=:horizontal, nbanks=2)

    # Add legend and save
    save(string(
        @__DIR__, "/../../results/model-uncertainty/youla_uncertain_cartpole.pdf"
        ), fig
    )
end
