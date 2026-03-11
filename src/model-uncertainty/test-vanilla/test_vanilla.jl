using BSON
using CairoMakie
using LinearAlgebra
using Random
using RobustNeuralNetworks
using Statistics

include(joinpath(@__DIR__, "../models.jl"))
include(joinpath(@__DIR__, "../functions.jl"))

# Make a results folder
savedir = string(@__DIR__, "/../../../results/model-uncertainty/batch-vanilla/")
if !isdir(savedir)
    mkdir(savedir)
end


###########################################################
#
# Experimental setup
#
###########################################################

include(joinpath(@__DIR__, "../setup_mass.jl"))


###########################################################
#
# Training
#
###########################################################

function run_experiment_and_plot(batch_id::Int; lr=1e-2, verbose=true)

    # Set random seeds
    rng = Xoshiro(batch_id)

    # Choose network sizes
    nu, nx, nv, ny = G.ny, 84, 128, G.nu
    nv_lstm = 154
    nv_mlp = 177

    # Save paths
    save_name = "lcp_vanilla_$(label)_$(nv_lstm)nvl_$(nv_mlp)nvm_lr$(lr)_v$(batch_id)"
    modelpath = string(savedir, save_name, ".bson")
    println("Starting experiment ", save_name)

    # Construct the models
    init = :cholesky
    nonlinearity = relu
    γ_yren_filt = 0.95 / γ_dy_filt

    youla_ren = LipschitzRENParams{Float32}(nu, nx, nv, ny, γ_yren_filt; rng, nl=nonlinearity, init)
    vanilla_lstm = LSTMNetwork(nu, nv_lstm, ny; rng, T=Float32)
    vanilla_mlp = MLPNetwork(nu, nv_mlp, ny; rng, T=Float32)

    # Don't set output to zero on init for vanilla policies
    # set_output_zero!(vanilla_lstm)
    # set_output_zero!(vanilla_mlp)

    # Hyperparams and testing params
    max_steps = 800
    train_horizon = 200
    train_batches = 64

    domain_rand = true
    test_batches = 64
    test_horizon = 2max_steps
    test_seed = 1

    # Train models
    nepochs = 4*1600
    costs_vl = train_model!(
        vanilla_lstm, G, K_vanilla, cost; rng, lr, nepochs, train_batches,
        max_steps, train_horizon, test_horizon, test_batches, youla=false, domain_rand,
        verbose, test_seed, lr_decay=[3/4, 7/8]
    )
    costs_vm = train_model!(
        vanilla_mlp, G, K_vanilla, cost; rng, lr, nepochs, train_batches,
        max_steps, train_horizon, test_horizon, test_batches, youla=false, domain_rand,
        verbose, test_seed, lr_decay=[3/4, 7/8]
    )

    # To test the base controller (zero-output REN)
    base_ren = REN(deepcopy(youla_ren))
    set_output_zero!(base_ren)

    # Get the base and optimal costs
    rng_() = Xoshiro(test_seed)
    test_states = init_states!(G, base_ren.nx, K_base.nx_filter, test_batches; rng=rng_())
    test_states_opt = init_states!(Gopt, 0, 0, test_batches; rng=rng_())
    w_test = procnoise(G, test_batches, test_horizon; rng=rng_())
    v_test = measnoise(G, test_batches, test_horizon; rng=rng_())

    J_base, _ = simulate(G, base_ren, K_base, cost, test_states; 
                            w=w_test, v=v_test, horizon=test_horizon)
    J_opt, _ = simulate(Gopt, cost, test_states_opt; 
                        w=w_test, v=v_test, horizon=test_horizon)

    # Save the model params, costs, etc.
    bson(modelpath, Dict(
        "J_base" => J_base,
        "J_opt" => J_opt,
        
        "vanilla_lstm" => vanilla_lstm,
        "vanilla_mlp" => vanilla_mlp,

        "costs_vl" => costs_vl,
        "costs_vm" => costs_vm
    ))


    #######################################################################
    #
    # Plotting
    #
    #######################################################################

    # Use the Wong (2011) colour pallette
    colours = Makie.wong_colors()
    colour_vl = colours[6]
    colour_vm = colours[3]
    colour_b = colours[4]
    colour_o = colours[1]
    n = length(costs_vm)

    # We only log costs every 5 steps (except first point)
    xc = vcat(1, 5:5:((length(costs_vm) - 1) * 5))

    # Plot learning curves, with optimal test cost as a reference
    fig = Figure(size=(700,450), fontsize=18)
    ga = fig[1,1] = GridLayout()
    ax = Axis(ga[1,1], xlabel="Epochs", ylabel="Test cost", yscale=Makie.log10)

    lines!(ax, xc, costs_vl, label="LSTM", linewidth=2, color=colour_vl)
    lines!(ax, xc, costs_vm, label="MLP", linewidth=2, color=colour_vm)

    lines!(ax, xc, J_base*ones(n), linestyle=:dash, color=colour_b, label="Base", linewidth=2)
    lines!(ax, xc, J_opt*ones(n) , linestyle=:dash, color=colour_o, label="Optimal (known mass)", linewidth=2)

    xlims!(ax, 1, xc[end])
    # ylims!(ax, J_opt, 1.15*J_base)
    Legend(ga[1,2], ax, orientation=:vertical)
    save(string(savedir, save_name, "_losscurve.pdf"), fig)

    println("Done.")
end

# batch_ids = [0,1,2,5,6,7]
learning_rates = [1e-5, 1e-4, 1e-3, 5e-3, 1e-2]
for lr in learning_rates
    run_experiment_and_plot(batch_ids[1]; lr)
end
