using Distributed
# Open Julia with: julia -t 4 -p 4 for 4 workers, each with 4 threads

@everywhere begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, "../.."))
end

@everywhere begin
    
    using BSON
    using CairoMakie
    using LinearAlgebra
    using Random
    using RobustNeuralNetworks
    using Statistics

    include(joinpath(@__DIR__, "models.jl"))
    include(joinpath(@__DIR__, "functions.jl"))

    # batch_ids = 0:9
    batch_ids = [0,1,2,5,6,7]

    # Make a results folder
    savedir = string(@__DIR__, "/../../results/model-uncertainty/batch-inputfilter/")
    if !isdir(savedir)
        mkdir(savedir)
    end


    ###########################################################
    #
    # Experimental setup
    #
    ###########################################################

    include(joinpath(@__DIR__, "setup_mass.jl"))


    ###########################################################
    #
    # Training
    #
    ###########################################################

    function run_experiment_and_plot(batch_id::Int)

        # Set random seeds
        rng = Xoshiro(batch_id)

        # Choose network sizes
        nu, nx, nv, ny = G.ny, 84, 128, G.nu
        lx, lv = 138, 0 # For linear REN
        nv_lstm = 154

        # Save paths
        save_name = "lcp_$(label)_$(nx)nx_$(nv)nv_v$(batch_id)"
        modelpath = string(savedir, save_name, ".bson")
        println("Starting experiment ", save_name)

        # Construct the models
        init = :cholesky
        nonlinearity = relu
        γ_yren_filt = 0.95 / γ_dy_filt      # 0.95 to give a bit of wiggle room
        γ_fren_filt = 0.95 / γ_y_filt
        γ_yren_nofilt = 0.95 / γ_dy_nofilt

        youla_ren = LipschitzRENParams{Float32}(nu, nx, nv, ny, γ_yren_filt; rng, nl=nonlinearity, init)
        youla_lren = LipschitzRENParams{Float32}(nu, lx, lv, ny, γ_yren_filt; rng, nl=nonlinearity, init)
        youla_ren_nf = LipschitzRENParams{Float32}(nu, nx, nv, ny, γ_yren_nofilt; rng, nl=nonlinearity, init)
        fdbak_ren = LipschitzRENParams{Float32}(nu, nx, nv, ny, γ_fren_filt; rng, nl=nonlinearity, init)
        fdbak_lstm = LSTMNetwork(nu, nv_lstm, ny; rng, T=Float32)

        # Set output to zero on init
        set_output_zero!(youla_ren)
        set_output_zero!(youla_lren)
        set_output_zero!(youla_ren_nf)
        set_output_zero!(fdbak_ren)
        set_output_zero!(fdbak_lstm)

        # Hyperparams and testing params
        max_steps = 800
        train_horizon = 200
        train_batches = 64

        domain_rand = true
        test_batches = 64
        test_horizon = 2max_steps
        test_seed = 1

        # Train models
        verbose=false
        costs_yr = train_model!(
            youla_ren, G, K_base, cost; rng, lr=1e-3, nepochs=1600, train_batches,
            max_steps, train_horizon, test_horizon, test_batches, youla=true, domain_rand,
            verbose, test_seed
        )
        costs_lr = train_model!(
            youla_lren, G, K_base, cost; rng, lr=1e-3, nepochs=1600, train_batches,
            max_steps, train_horizon, test_horizon, test_batches, youla=true, domain_rand,
            verbose, test_seed
        )
        costs_yr_nf = train_model!(
            youla_ren_nf, G, K_base_nofilter, cost; rng, lr=1e-3, nepochs=1600, train_batches,
            max_steps, train_horizon, test_horizon, test_batches, youla=true, domain_rand,
            verbose, test_seed, lr_decay=[3/4, 7/8]
        )
        costs_fr = train_model!(
            fdbak_ren, G, K_base, cost; rng, lr=1e-3, nepochs=1600, train_batches,
            max_steps, train_horizon, test_horizon, test_batches, youla=false, domain_rand,
            verbose, test_seed
        )
        costs_fl = train_model!(
            fdbak_lstm, G, K_base_nofilter, cost; rng, lr=1e-2, nepochs=1600, train_batches,
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
            "youla_ren" => youla_ren,
            "youla_lren" => youla_lren,
            "youla_ren_nf" => youla_ren_nf,
            "fdbak_ren" => fdbak_ren,
            "fdbak_lstm" => fdbak_lstm,
            "costs_yr" => costs_yr,
            "costs_lr" => costs_lr,
            "costs_yr_nf" => costs_yr_nf,
            "costs_fr" => costs_fr,
            "costs_fl" => costs_fl,
            "J_base" => J_base,
            "J_opt" => J_opt,
        ))


        #######################################################################
        #
        # Plotting
        #
        #######################################################################

        # Use the Wong (2011) colour pallette
        colours = Makie.wong_colors()
        colour_yr = colours[2]
        colour_lr = colours[6]
        colour_yr_nf = colours[3]
        colour_fr = :grey
        colour_fl = colours[5]
        colour_b = colours[4]
        colour_o = colours[1]
        n = length(costs_yr)

        # We only log costs every 5 steps (except first point)
        xc = vcat(1, 5:5:((length(costs_yr) - 1) * 5))

        # Plot learning curves, with optimal test cost as a reference
        fig = Figure(size=(700,450), fontsize=18)
        ga = fig[1,1] = GridLayout()
        ax = Axis(ga[1,1], xlabel="Epochs", ylabel="Test cost")

        lines!(ax, xc, costs_yr, label="Youla-REN", linewidth=2, color=colour_yr)
        lines!(ax, xc, costs_lr, label="Youla-REN (linear)", linewidth=2, color=colour_lr)
        lines!(ax, xc, costs_yr_nf, label="Youla-REN (no filter)", linewidth=2, color=colour_yr_nf)
        lines!(ax, xc, costs_fr, label="Residual-REN (stable)", linewidth=2, color=colour_fr)
        lines!(ax, xc, costs_fl, label="Residual-LSTM", linewidth=2, color=colour_fl)
        lines!(ax, xc, J_base*ones(n), linestyle=:dash, color=colour_b, label="Base", linewidth=2)
        lines!(ax, xc, J_opt*ones(n) , linestyle=:dash, color=colour_o, label="Optimal (known mass)", linewidth=2)

        xlims!(ax, 1, xc[end])
        ylims!(ax, 0, 1.15*J_base)
        Legend(ga[1,2], ax, orientation=:vertical)
        save(string(savedir, save_name, "_losscurve.pdf"), fig)

        println("Done.")
        return nothing
    end
end

println("Starting $(length(batch_ids)) experiments...")
pmap(run_experiment_and_plot, batch_ids)
println("All done.")
