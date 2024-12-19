using BSON
using CairoMakie
using LinearAlgebra
using Random
using RobustNeuralNetworks
using Statistics

include(joinpath(@__DIR__, "models.jl"))
include(joinpath(@__DIR__, "functions.jl"))


# Problem setup
umax = 15.0
G = NonlinearExample()
cost(x::Matrix, u::Matrix) = 0 # (not needed in this script)


#######################################################################
#
# Load cost data and simulate trajectories
#
#######################################################################

# Load the costs
fpath = joinpath(@__DIR__, "../../results/nonlinear-system/batch/")
fnames = get_bson_files(fpath)

load_data(fname, key) = BSON.load(fname)[key]
costs_yr = load_data.(fnames, ("costs_yr"))
J_base = mean(load_data.(fnames, ("J_base")))

# Process costs
n = length(costs_yr[1])
μ_yr, σ_yr, min_yr, max_yr = cost_stats(costs_yr)
xc = collect(1:n)

# Load a single model for eval
batch_id = 0
fname = fnames[batch_id+1]
youla_ren = REN(load_data(fname,"youla_ren"))

# Model to test the base controller
base_ren = deepcopy(youla_ren)
set_output_zero!(base_ren)

# Some dummy test data
horizon = Int(10 / G.dt)
batches = 64
w_test = procnoise(G, batches, horizon; rng=Xoshiro(batch_id))
v_test = measnoise(G, batches, horizon; rng=Xoshiro(batch_id))

# Special state initialisation to make plots look nicer
x0 = scale.(rand(Xoshiro(batch_id), G.nx, batches), [-10,8,-10], [10,10,-8])
ξ0 = zeros(youla_ren.nx + G.nx, batches)
states = [x0, ξ0]

# Simulate the trained controller and the base controller
J_y, traj_y = simulate(G, youla_ren, cost, states; horizon, 
                       w=w_test, v=v_test, log_states=true)
J_b, traj_b = simulate(G, base_ren, cost, states; horizon,
                       w=w_test, v=v_test, log_states=true)

# Split trajectories into states and controls
xs_y, us_y = traj_y
xs_b, us_b = traj_b
ts = (collect(1:horizon) .- 1) * G.dt


#######################################################################
#
# Plotting
#
#######################################################################

function plot_trajectories!(axs, xs, us; label=nothing, color=:grey, 
                            alpha=0.3, linewidth=1)

    # Re-order for plotting (batches last)
    xs = permutedims(stack(xs), (1,3,2))
    us = permutedims(stack(us), (1,3,2))

    # Just pick out x2 for plotting
    xs = xs[2:2, :, :]

    ax1, ax2 = axs

    # Plot trajectories
    for k in axes(xs, 3)
        lines!(ax1, ts, xs[1,:,k]; color, alpha, label, linewidth)
        lines!(ax2, ts, us[1,:,k]; color, alpha, label, linewidth)
         (k == 1) && (label=nothing)
    end
end

# Use the Wong (2011) colour pallette
colours = Makie.wong_colors()
colour_b = colours[2]
colour_y = colours[3]
colour_ref = colours[1]
colour_max = :brown

# Make the plot
with_theme(theme_latexfonts()) do

    # Set up figure
    fig = Figure(size=(700,300), fontsize=18)
    ga1 = fig[1,1] = GridLayout()
    ga2 = fig[1,2] = GridLayout()

    # Set up axes
    ax1 = Axis(ga1[1,1], xlabel="Training epochs", ylabel="Test cost")
    ax3 = Axis(ga2[1,1], xticklabelsvisible=false, ylabel=L"u(t)")
    ax2 = Axis(ga2[2,1], xlabel="Time (s)", ylabel=L"x_2(t)")

    # Panel 1: cost curve
    band!(ax1, xc, max_yr, min_yr, color = (colour_y, 0.3))
    lines!(ax1, xc, μ_yr; color=colour_y, linewidth=2, label="Youla")
    lines!(ax1, xc, J_base*ones(n); color=colour_b, linewidth=2,   
           linestyle=:dash,label="Base")

    # Panels 2 and 3: trajectory rollouts
    axs = [ax2, ax3]
    plot_trajectories!(axs, xs_b, us_b; color=colour_b, label="Base")
    plot_trajectories!(axs, xs_y, us_y; color=colour_y, label="Youla")

    # Plot reference states and limits
    zs = zeros(size(ts))
    os = ones(size(ts))
    lines!(ax3, ts, umax .* os, color=colour_max, linestyle=:dot)
    lines!(ax3, ts, -umax .* os, color=colour_max, linestyle=:dot, label=L"\pm u_\text{max}")

    # Set axes limits
    xlims!(ax1, 0, 2000)
    ylims!(ax1, 0, 1.1*J_base)

    xlims!(ax2, minimum(ts), 6) #maximum(ts))
    xlims!(ax3, minimum(ts), 6) #maximum(ts))

    # Add a legend and format
    axislegend(ax1, position=:rt)

    save(string(
        @__DIR__, "/../../results/nonlinear-system/nonlinear_system_results.pdf"
        ), fig
    )
end
