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
    using Statistics

    include(joinpath(@__DIR__, "models.jl"))
    include(joinpath(@__DIR__, "functions.jl"))

    batch_ids = 0:9

    # Make a results folder
    savedir = joinpath(@__DIR__, "../../results/nonlinear-system/batch/")
    if !isdir(savedir)
        mkdir(savedir)
    end


    ###########################################################
    #
    # Experimental setup
    #
    ###########################################################

    # Problem setup: heavily penalise control effort > 15
    umax = 15.0
    G = NonlinearExample()
    cost(x::Matrix, u::Matrix) = mean(abs.(u) + 500 * max.(abs.(u) .- umax, 0))


    ###########################################################
    #
    # Training
    #
    ###########################################################

    function run_experiment_and_plot(batch_id::Int)

        # Set random seeds
        rng = Xoshiro(batch_id)

        # Choose network sizes
        nu, nx, nv, ny = G.ny, 16, 128, G.nu

        # File saving
        save_name = "nl_$(nx)nx_$(nv)nv_v$(batch_id)"
        modelpath = string(savedir, save_name, ".bson")
        println("Starting experiment ", save_name)

        # Construct the model (with zero output on init)
        init = :cholesky
        nonlinearity = tanh
        youla_ren = ContractingRENParams{Float32}(nu, nx, nv, ny; init, rng, nl=nonlinearity)
        set_output_zero!(youla_ren)

        # To test the base controller
        base_ren = REN(deepcopy(youla_ren))

        # Hyperparams and testing params
        max_steps = Int(10 / G.dt)
        train_batches = 32
        test_batches = 64
        test_seed = 24

        # Roll out the base controller to check base cost
        rng_() = Xoshiro(test_seed)
        states = init_states(G.nx, youla_ren.nx, test_batches; rng=rng_())
        w_test = procnoise(G, test_batches, max_steps; rng=rng_())
        v_test = measnoise(G, test_batches, max_steps; rng=rng_())
        J_base, _ = simulate(G, base_ren, cost, states; 
                            horizon=max_steps, w=w_test, v=v_test, rng)

        # Train the Youla-REN
        verbose=false
        costs_yr = train_model!(youla_ren, G, cost; rng, lr=3e-4, nepochs=2500, max_steps,
                                train_batches, test_batches, test_seed, verbose)

        # Save model params and costs
        bson(modelpath, Dict(
            "youla_ren" => youla_ren,
            "costs_yr" => costs_yr,
            "J_base" => J_base,
        ))


        #######################################################################
        #
        # Plotting
        #
        #######################################################################

        # Use the Wong (2011) colour pallette
        colours = Makie.wong_colors()
        colour_y = colours[2]
        colour_b = colours[1]
        n = length(costs_yr)

        # Plot learning curves, with optimal test cost as a reference
        fig = Figure(size=(600,450), fontsize=18)
        ax = Axis(fig[1,1], xlabel="Epochs", ylabel="Training cost")

        lines!(ax, costs_yr, label="Youla", linewidth=2, color=colour_y)
        lines!(ax, J_base*ones(n), linestyle=:dash, color=colour_b, label="Base", linewidth=2)

        axislegend(ax, position=:rt)
        xlims!(ax, 0, n)
        ylims!(ax, 0, 1.1*J_base)
        ax.width = 480
        save(string(savedir, save_name, "_losscurve.pdf"), fig)

        return nothing
    end
end

println("Starting $(length(batch_ids)) experiments...")
pmap(run_experiment_and_plot, batch_ids)
println("All done.")
