using BSON
using CairoMakie
using LinearAlgebra
using Peaks
using Statistics

include(joinpath(@__DIR__, "models.jl"))
include(joinpath(@__DIR__, "functions.jl"))


###########################################################
#
# Experimental setup
#
###########################################################

# Linear system from Doyle (1978), discretise it
# Set a fixed simulation horizon too (the dynamics are slow so it's long)
dt = 0.01
A  = I + dt*[1 1; 0 1]
B  = dt*reshape([0,1],2,1)
Bw = dt*reshape([1,1],2,1)
C  = [1 0]
σw = 1e3
G  = LinearSystem(A, B, Bw, C; σw, max_steps = Int(100 / dt))

# True cost func weights and optimal (time-invariant) LQG controller
q = 1e3
Q_opt = q * ones(2,2)
R_opt = ones(1,1)
Σw_opt = Bw * σw * Bw'
Σv_opt = ones(1,1)
K_opt = LinearCtrl(G, Q_opt, R_opt, Σw_opt, Σv_opt)

# Design base LQG controller (slightly sub-optimal)
Q = q * [1 0;0 1]
R = ones(1,1)
Σw = Bw * (0.01σw) * Bw'
Σv = ones(1,1)
K_base = LinearCtrl(G, Q, R, Σw, Σv)

# Cost function
cost(x::Vector, u::Vector) = (x' * Q_opt * x) + (u' * R_opt * u)
cost(x::Matrix, u::Matrix) = mean(sum((Q_opt * x) .* x; dims=1) + 
                                  sum((R_opt * u) .* u; dims=1))


#######################################################################
#
# Load cost curve data
#
#######################################################################

load_youla_costs(fname) = BSON.load(fname)["costs_y"]
load_fdbak_costs(fname) = BSON.load(fname)["costs_f"]
load_base_costs(fname) = BSON.load(fname)["J_base"]
load_opt_costs(fname) = BSON.load(fname)["J_opt"]

# Load the costs
fpath = joinpath(@__DIR__, "../../results/stability-guarantees/batch/")
fnames = get_bson_files(fpath)

raw_costs_y = load_youla_costs.(fnames)
raw_costs_f = load_fdbak_costs.(fnames)
J_base = mean(load_base_costs.(fnames))
J_opt = mean(load_opt_costs.(fnames))

# Process costs
costs_y = movingaverage.(raw_costs_y, 100)
costs_f = movingaverage.(raw_costs_f, 100)
μ_y, σ_y, min_y, max_y = cost_stats(costs_y)
μ_f, σ_f, min_f, max_f = cost_stats(costs_f)


#######################################################################
#
# Rollouts
#
#######################################################################

max_epochs = length(μ_y)

function rollout_checkpoint_model(batch_id=1, epoch=500)

    # Load models
    if epoch < max_epochs
        youla_nn = BSON.load(string(fpath, 
            "checkpoints/doyle_2nx_8nv_tanh_v$(batch_id)_youla_e$(epoch).bson"))["model"]
        fdbak_nn = BSON.load(string(fpath, 
            "checkpoints/doyle_2nx_8nv_tanh_v$(batch_id)_fdbak_e$(epoch).bson"))["model"]
    else
        data = BSON.load(string(fpath, "doyle_2nx_8nv_tanh_v$(batch_id)_models.bson"))
        youla_nn = data["youla_nn"]
        fdbak_nn = data["fdbak_nn"]
    end

    # Construct a model to use for base/optimal controller, process others
    base_ren = REN(deepcopy(youla_nn))
    youla_nn = setup_model(youla_nn)
    fdbak_nn = setup_model(fdbak_nn)
    set_output_zero!(base_ren)

    # Test data
    rng = Xoshiro(0)
    test_rng1 = Xoshiro(1)
    test_rng2 = Xoshiro(1)

    test_batches = 100
    test_horizon = 3 * G.max_steps
    youla_test_states = init_states(G.nx, youla_nn.nx, test_batches; rng=test_rng1)
    fdbak_test_states = init_states(G.nx, fdbak_nn.nx, test_batches; rng=test_rng2)
    w_test = procnoise(G, test_batches, test_horizon; rng)
    v_test = measnoise(G, test_batches, test_horizon; rng)

    # Evaluate each model
    Jb, _ = sim_test(G, base_ren, K_base, cost, youla_test_states; w=w_test, v=v_test, horizon=test_horizon)
    Jo, _ = sim_test(G, base_ren, K_opt, cost, youla_test_states; w=w_test, v=v_test, horizon=test_horizon)
    Jy, _ = sim_test(G, youla_nn, K_base, cost, youla_test_states; w=w_test, v=v_test, horizon=test_horizon)
    Jf, _ = sim_test(G, fdbak_nn, K_base, cost, fdbak_test_states; w=w_test, v=v_test, horizon=test_horizon, youla=false)

    return Jb, Jo, Jy, Jf, epoch
end

# Get cost rollout data for a couple of epochs
# rollout_data = rollout_checkpoint_model.(3, [747, 1500, max_epochs])
rollout_data = rollout_checkpoint_model.(7, [687, 1500, max_epochs])


#######################################################################
#
# Plotting
#
#######################################################################

# Useful for plotting
function plot_loss(ax, μ, cmax, cmin, label, colour; linewidth=2, linestyle=:solid)
    x = collect(1:length(μ))
    band!(ax, x, cmax, cmin, color = (colour, 0.2))
    lines!(ax, x, μ, label=label, linewidth=linewidth, color=colour, linestyle=linestyle)
end

# Use the Wong (2011) colour pallette
colours = Makie.wong_colors()
colour_y = colours[2]
colour_f = colours[3]
colour_b = colours[4]
colour_o = colours[1]

# Set up figure
with_theme(theme_latexfonts()) do
    fig = Figure(size=(800,380), fontsize=19)
    ga = fig[1, 1] = GridLayout()
    gb = fig[1, 2] = GridLayout()
    colsize!(fig.layout, 1, Relative(4/7))
    colgap!(fig.layout, 1, Relative(0.08))

    # Main chart: training cost vs epochs in bands
    ax = Axis(ga[1,1], xlabel="Training epochs", ylabel="Training cost", yscale=Makie.log10)
    plot_loss(ax, μ_y, max_y, min_y, "Youla", colour_y)
    plot_loss(ax, μ_f, max_f, min_f, "Residual", colour_f)

    n = length(costs_y[1])
    lines!(ax, J_base*ones(n), linestyle=:dash, color=colour_b, label="Base", linewidth=2)
    lines!(ax, J_opt*ones(n),  linestyle=:dash, color=colour_o, label="Optimal", linewidth=2)

    axislegend(ax, position=:rt)
    xlims!(ax, 0, n)
    ylims!(ax, 10^3.5, 10^5.5)

    # Plot rollouts at the checkpoints
    for i in eachindex(rollout_data)
        
        # Get data
        Jb, Jo, Jy, Jf, epoch = rollout_data[i]
        t = LinRange(0, length(Jb) / G.max_steps, length(Jb))

        # Set up labels and axis
        xlab = i == 3 ? "Test horizon/Train horizon" : ""
        ylab = i == 2 ? "Time-averaged test cost" : ""
        xvis = i == 3 ? true : false
        title = "Training epoch: $(epoch)"
        ax_i = Axis(gb[i,1], xlabel=xlab, ylabel=ylab, yscale=Makie.log10, 
                    xticklabelsvisible=xvis,  yticks=LogTicks(WilkinsonTicks(3)),
                    title=title, titlefont=:regular)

        # Make the plots
        lines!(ax_i, t, Jy, linewidth=2, color=colour_y)
        lines!(ax_i, t, Jf, linewidth=2, color=colour_f)
        lines!(ax_i, t, Jb, linewidth=2, color=colour_b, linestyle=:dash)
        lines!(ax_i, t, Jo, linewidth=2, color=colour_o, linestyle=:dash)
        xlims!(ax_i, t[1], t[end])
        ylims!(ax_i, 10^3.5, 10^4.5)
    end

    save(string(
        @__DIR__, "/../../results/stability-guarantees/stability_guarantees_learning.pdf"
        ), fig
    )
end

# Useful to find peaks for fine-tuning the plot
function find_n_spikes(x; p=100, n=4)
    dx = diff(x)
    dx[dx .< 0] .= 0

    pks, vals = findmaxima(dx)
    sort_index = sortperm(vals, rev=true)[1:n]
    pks_keep = pks[sort_index]
    pks_keep = pks_keep[pks_keep .> p]
    return pks_keep, x[pks_keep]
end

function get_batch_id_from_fname(fname)
    data = split(fname, "_")
    return data[end-1]
end

batchid = "v7"
ids = get_batch_id_from_fname.(fnames)
costs_fid = costs_f[ids .== batchid][1]
pks, vals = find_n_spikes(costs_fid; n=3)

# Make note of which spikes came from which versions
println("Spike IDs for batch $batchid: ")
for i in eachindex(pks)
    println("Spike $i: index $(pks[i]), value $(round(vals[i],digits=2))")
end
