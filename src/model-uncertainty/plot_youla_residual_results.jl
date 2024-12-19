# using BSON
# using CairoMakie
# using Random
# using RobustNeuralNetworks
# using Statistics

# include(joinpath(@__DIR__, "models.jl"))
# include(joinpath(@__DIR__, "functions.jl"))
# include(joinpath(@__DIR__, "setup_mass.jl"))


# #######################################################################
# #
# # Get cost data from training
# #
# #######################################################################

# # Load the costs
# fpath = joinpath(@__DIR__, "../../results/model-uncertainty/batch-inputfilter/")
# fnames = get_bson_files(fpath)
    
# load_data(fname, key) = BSON.load(fname)[key]
# costs_yr = load_data.(fnames, ("costs_yr"))
# costs_fl = load_data.(fnames, ("costs_fl"))
# J_base = mean(load_data.(fnames, ("J_base")))
# J_opt = mean(load_data.(fnames, ("J_opt")))

# # Process costs
# μ_yr, σ_yr, min_yr, max_yr = cost_stats(costs_yr)
# μ_fl, σ_fl, min_fl, max_fl = cost_stats(costs_fl)


# #######################################################################
# #
# # Generate cost trajectory rollouts
# #
# #######################################################################

# # Load the models
# name = "showcase/lcp_$(label)_84nx_128nv_best.bson"
# fname = string(fpath, name)

# youla_ren  = REN(load_data(fname, "youla_ren"))
# fdbak_lstm = load_data(fname, "fdbak_lstm")

# base_ren = deepcopy(youla_ren)
# set_output_zero!(base_ren)

# # Choose test data
# rng_() = Xoshiro(1)
# test_batches = 64
# train_horizon = 800
# test_horizon = 12 * train_horizon

# nf = K_base.nx_filter
# test_states_ren = init_states!(G, base_ren.nx, nf, test_batches; rng=rng_())
# test_states_lstm = init_states!(G, fdbak_lstm.nx, 0, test_batches; rng=rng_())
# test_states_opt = init_states!(Gopt, 0, 0, test_batches; rng=rng_())

# w_test = procnoise(G, test_batches, test_horizon; rng=rng_())
# v_test = measnoise(G, test_batches, test_horizon; rng=rng_())

# # Roll out the policies
# function sim_test(model, test_states, K; youla=true)
#     _, _, traj = simulate(G, model, K, cost, test_states; youla, 
#                           w=w_test, v=v_test, horizon=test_horizon, 
#                           log_states=true)
#     costs = cost.(traj[1], traj[2])
#     return cumsum(costs) ./ (1:test_horizon), traj
# end

# J_bs, traj_b = sim_test(base_ren, test_states_ren, K_base)
# J_yr, traj_yr = sim_test(youla_ren, test_states_ren, K_base)
# J_fl, traj_fl = sim_test(fdbak_lstm, test_states_lstm, K_base_nofilter; youla=false)

# # Compute the optimal costs too
# _, _, traj_o = simulate(Gopt, cost, test_states_opt; horizon=test_horizon, 
#                         w=w_test, v=v_test, log_states=true)
# J_os = cumsum(cost.(traj_o[1], traj_o[2])) ./ (1:test_horizon)
# J_test = get_optimal_lqg_cost(Gopt, test_horizon) / test_horizon


# #######################################################################
# #
# # Plot cost curves and cost rollouts
# #
# #######################################################################

# # Useful for plotting
# xc = vcat(1, 5:5:((length(μ_yr) - 1) * 5))      # (we only log costs every 5 points)
# t = LinRange(0, length(J_bs) / train_horizon, length(J_bs))

# function plot_loss!(ax, μ, cmax, cmin; linewidth=2, kwargs...)
#     colour = kwargs[:color]
#     band!(ax, xc, cmax, cmin, color = (colour, 0.3))
#     lines!(ax, xc, μ; linewidth=linewidth, kwargs...)
# end

# # Use the Wong (2011) colour pallette
# colours = Makie.wong_colors()
# colour_yr = colours[2]
# colour_lr = colours[6]
# colour_nf = colours[5]
# colour_fr = :grey
# colour_fl = colours[3]
# colour_b = colours[4]
# colour_o = colours[1]
# colour_n = :grey64

# with_theme(theme_latexfonts()) do

#     # Figure setup
#     fig = Figure(size=(700,300), fontsize=18)
#     ga1 = fig[1,1] = GridLayout()
#     ga2 = fig[1,2] = GridLayout()

#     colsize!(fig.layout, 1, Relative(1/2))
#     colgap!(fig.layout, 1, Relative(0.08))

#     ax1 = Axis(ga1[1,1], xlabel="Training epochs", ylabel="Time-averaged test cost", xticks=WilkinsonTicks(4; k_min=4, k_max=8))
#     ax2 = Axis(ga2[1,1], xlabel="Test horizon/Train horizon", yticklabelsvisible=false)

#     # Panel 1: loss curves
#     n = length(costs_yr[1])
#     plot_loss!(ax1, μ_yr, max_yr, min_yr; color=colour_yr, label="Youla-γREN")
#     plot_loss!(ax1, μ_fl, max_fl, min_fl; color=colour_fl, label="Residual-LSTM")
    
#     lines!(ax1, xc, J_base*ones(n), linestyle=:dash, color=colour_b, label="Base", linewidth=2)
#     lines!(ax1, xc, J_opt*ones(n) , linestyle=:dash, color=colour_o, label=L"LQG (known $m_p$)", linewidth=2)    
#     xlims!(ax1, 0, xc[end])
#     ylims!(ax1, -4, 1.2*J_base)

#     # Panel 2: cost rollouts
#     lines!(ax2, t, J_yr, linewidth=2, color=colour_yr, label="Youla-γREN")
#     lines!(ax2, t, J_fl, linewidth=2, color=colour_fl, label="Residual-LSTM")
#     lines!(ax2, t, J_bs, linewidth=2, color=colour_b, linestyle=:dash, label="Base")
#     lines!(ax2, t, J_os, linewidth=2, color=colour_o, linestyle=:dash, label=L"LQG (known $m_p$)")
   
#     xlims!(ax2, t[1], t[end])
#     ylims!(ax2, -4, 1.2*J_base)
    
#     # Fudge sizes
#     ax1.width = 270
#     ax2.width = 270

#     # Add legend and save
#     axislegend(ax2, position=:rt)
#     save(string(
#         @__DIR__, "/../../results/model-uncertainty/lcp_youla_residual_costs.pdf"
#         ), fig
#     )
# end


# #######################################################################
# #
# # Generate cart/control trajectories for plotting
# #
# #######################################################################

# # Choose new test data
# test_batches = 3
# train_horizon = 800
# test_horizon = 8 * train_horizon

# nf = K_base.nx_filter
# test_states_ren = init_states!(G, base_ren.nx, nf, test_batches; rand_init=false)
# test_states_lstm = init_states!(G, fdbak_lstm.nx, 0, test_batches; rand_init=false)
# test_states_opt = init_states!(Gopt, 0, 0, test_batches; rand_init=false)

# w_test = procnoise(G, test_batches, test_horizon; rng=Xoshiro(1))
# v_test = measnoise(G, test_batches, test_horizon; rng=Xoshiro(1))

# # Fix the params of interest
# ρs = [0.145, 0.2, 0.345]
# G.A = G.Afunc.(ρs)
# Gopt.sys.A = Gopt.sys.Afunc.(ρs)

# # Roll out the policies
# _, traj_b = sim_test(base_ren, test_states_ren, K_base)
# _, traj_yr = sim_test(youla_ren, test_states_ren, K_base)
# _, traj_fl = sim_test(fdbak_lstm, test_states_lstm, K_base_nofilter; youla=false)


# #######################################################################
# #
# # Plot trajectories
# #
# #######################################################################

# # x-axis for plotting
# npoints = length(traj_b[1])
# t = LinRange(0, npoints / train_horizon, npoints)


# """
# Picks out either the cart position (`indx=1`) or the control
# effort (`indx=2`) and sorts data into a nice order for plotting.
# """
# function get_cartpole_variable(traj, indx)
#     xs = stack(traj[indx])
#     xs = permutedims(xs, (1,3,2)) # put batches last
#     return xs[1,:,:]
# end

# """
# Plot trajectories on a specified axis. Same use of `indx`
# as for `get_cartpole_variable`.
# """
# function plot_trajs!(ax, traj, indx, ylim)
#     x = get_cartpole_variable(traj, indx)
#     colours = [:black, :red, :grey]
#     for k in axes(x,2)
#         label = L"$m_p = %$(ρs[k])$\,kg"
#         color = colours[k]
#         lines!(ax, t, x[:,k]; linewidth=1.2, label, color, alpha=(1.0 - (k-1)*0.1))
#     end
#     xlims!(ax, t[1], t[end])
#     ylims!(ax, ylim...)
# end

# # Axes limits should be consistent
# xb_pos = get_cartpole_variable(traj_b, 1)
# xb_ctrl = get_cartpole_variable(traj_b, 2)
# ylim_pos = 0.75 .* (-maximum(xb_pos), maximum(xb_pos))
# ylim_ctrl = 0.4 .* (-maximum(xb_ctrl), maximum(xb_ctrl))

# # Make the plot
# with_theme(theme_latexfonts()) do

#     # Set up figure
#     fig = Figure(size=(550,370), fontsize=19)

#     # Youla-REN (linear)
#     ga1 = fig[1,1] = GridLayout()
#     ax1 = Axis(ga1[1,1], ylabel=L"x_c \ (\text{m})", title="Youla-γREN", xticklabelsvisible=false, titlefont=:regular, xticks=WilkinsonTicks(6), yticks=WilkinsonTicks(3))
#     ax2 = Axis(ga1[2,1], ylabel=L"u \ (\text{N})", titlefont=:regular, xticks=WilkinsonTicks(6), yticks=WilkinsonTicks(3))
#     plot_trajs!(ax1, traj_yr, 1, ylim_pos)
#     plot_trajs!(ax2, traj_yr, 2, ylim_ctrl)

#     # Feedback-LSTM
#     ga2 = fig[1,2] = GridLayout()
#     ax1 = Axis(ga2[1,1], title="Residual-LSTM", xticklabelsvisible=false, yticklabelsvisible=false, titlefont=:regular, xticks=WilkinsonTicks(6), yticks=WilkinsonTicks(3))
#     ax2 = Axis(ga2[2,1], yticklabelsvisible=false, titlefont=:regular, xticks=WilkinsonTicks(6), yticks=WilkinsonTicks(3))
#     plot_trajs!(ax1, traj_fl, 1, ylim_pos)
#     plot_trajs!(ax2, traj_fl, 2, ylim_ctrl)

#     # x-axis label
#     ga3 = fig[2,1:2] = GridLayout()
#     Label(ga3[1,1], text="Test horizon/Train horizon")
#     Legend(ga3[2,1], ax1, orientation=:horizontal, linewidth=10)

#     save(string(
#         @__DIR__, "/../../results/model-uncertainty/lcp_youla_residual_trajectories.pdf"
#         ), fig
#     )
# end


#######################################################################
#
# Plot adaptation results
#
#######################################################################

# Load previously-computed data just for plotting
name = "showcase/param_variation_costs.bson"
fname = string(fpath, name)
data = BSON.load(fname)

ρs = data["ρs"]
J_bs = data["J_bs"]
J_nom = data["J_nom"]
J_yr = data["J_yr"]
J_fl = data["J_fl"]
J_os = data["J_os"]
J_os_lti = data["J_os_lti"]
lo_range_max = data["lo_range_max"]
hi_range_min = data["hi_range_min"]

# To scale axes
scale_array = [0.135, 0.355] # For plot scaling only
normalize(x, y=scale_array) = (x .- minimum(y)) ./ (maximum(y) - minimum(y))

# Make the plot
with_theme(theme_latexfonts()) do

    # Figure setup
    fig = Figure(size=(470,400), fontsize=19)
    ga1 = fig[1,1] = GridLayout()
    ga2 = fig[2,1] = GridLayout()

    # Main panel: cost vs. pole mass
    xlabs = [0.15, 0.2, 0.34]
    ax = Axis(
        ga1[1,1], xminorticksvisible=true, xminorgridvisible=true, 
        xscale=Makie.logit, ylabel="Test cost", titlefont=:regular, 
        xlabel=L"Pole mass $m_p$ (kg)",
        xminorticks = IntervalsBetween(4),
        xticks = (normalize(xlabs), string.(xlabs)),
        yscale = Makie.log10
    )

    # Change x-scale for nice plotting
    x = normalize(ρs)

    # Plot
    linewidth = 2
    lines!(ax, x, J_yr; color=colour_yr, linewidth, label="Youla-γREN")
    lines!(ax, x, J_fl; color=colour_fl, linewidth, label="Residual-LSTM")
    lines!(ax, x, J_bs; color=colour_b, linewidth, linestyle=:dash, label="Base")
    lines!(ax, x, J_os; color=colour_o, linewidth, linestyle=:dash, label=L"LQG (known $m_p$)")
    lines!(ax, x, J_os_lti; color=:purple, linewidth, linestyle=:dash, alpha=0.8, label=L"LTI LQG (known $m_p$)")
    lines!(ax, x, J_nom; color=colour_n, linewidth, linestyle=:dash, label=L"LTI LQG (nominal $m_p$)")

    # Format
    xlims!(ax, minimum(x), maximum(x))
    # ylims!(ax, -5, maximum(J_bs))
    ylims!(ax, 0.6, 10^4.5)
    Legend(ga2[1,1], ax, orientation=:horizontal, nbanks=3)

    # Save figure
    ax.width = 250 # to stop labels being chopped off
    save(string(
        @__DIR__, "/../../results/model-uncertainty/lcp_youla_residual_sensitivity.pdf"
        ), fig
    )
end