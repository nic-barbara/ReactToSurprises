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

    batch_ids = 0:9
    name = "doyle_2nx_8nv_tanh"

    # Make a results folder
    savedir = string(@__DIR__, "/../../results/stability-guarantees/batch/")
    if !isdir(savedir)
        mkdir(savedir)
    end
    if !isdir(string(savedir,"checkpoints/"))
        mkdir(string(savedir,"checkpoints/"))
    end
    

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


    ###########################################################
    #
    # Training
    #
    ###########################################################

    function run_experiment_and_plot(batch_id::Int)

        # Set random seeds
        rng = Xoshiro(batch_id)

        # Save paths
        save_name = string(name, "_v$(batch_id)")
        modelpath = string(savedir, save_name, "_models.bson")
        println("Starting experiment ", save_name)

        # Construct the models
        nu, nx, nv, ny = G.ny, 2, 8, G.nu
        nonlinearity = tanh
        youla_model = ContractingRENParams{Float32}(nu, nx, nv, ny; rng, nl=nonlinearity)
        fdbak_model = deepcopy(youla_model)

        # Set output to zero on initialisation
        set_output_zero!(youla_model)
        set_output_zero!(fdbak_model)

        # Choose a few checkpoints to save params at. These are hard-coded
        # for nice plotting, but any choice shows similar results
        checkpoints = [500, 687, 747, 1500]
        # checkpoints = collect(500:500:3500)

        # Train the models
        # Note on learning rate: for feedback, 1e-4 means no learning, >=5e-4 is unstable.
        # Using 3e-4 is slightly more stable and a happy medium. Can choose basically
        # anything for Youla and it will work fine.
        verbose = false
        costs_f = train_model!(
            fdbak_model, G, K_base, cost; rng, lr=3e-4, nepochs=4000, batches=50,
            horizon=200, youla=false, verbose,
            save_params=(string(savedir, "checkpoints/", save_name, "_fdbak"), checkpoints)
        )
        costs_y = train_model!(
            youla_model, G, K_base, cost; rng, lr=1e-2, nepochs=4000, batches=50,
            horizon=200, youla=true, verbose,
            save_params=(string(savedir, "checkpoints/", save_name, "_youla"), checkpoints)
        )

        # To test the base controller (zero-output model)
        base_ren = REN(deepcopy(youla_model))
        set_output_zero!(base_ren)

        # Compute costs for the base/optimal controllers
        test_batches = 100
        test_states = init_states(G.nx, base_ren.nx, test_batches; rng)
        w_test = procnoise(G, test_batches, G.max_steps; rng)
        v_test = measnoise(G, test_batches, G.max_steps; rng)
        J_base, _  = simulate(G, base_ren, K_base, cost, test_states;  
                              w=w_test, v=v_test, horizon=G.max_steps)
        J_opt, _  = simulate(G, base_ren, K_opt, cost, test_states;  
                             w=w_test, v=v_test, horizon=G.max_steps)

        # Save the model parameters, costs, etc.
        bson(modelpath, Dict(
            "youla_nn" => youla_model,
            "fdbak_nn" => fdbak_model,
            "costs_y" => costs_y,
            "costs_f" => costs_f,
            "J_base" => J_base,
            "J_opt" => J_opt,
        ))

        # Use the Wong (2011) colour pallette
        colours = Makie.wong_colors()
        colour_y = colours[2]
        colour_f = colours[3]
        colour_b = colours[1]
        colour_o = colours[4]

        # Process costs so they look nice
        c_y = movingaverage(costs_y, 100)
        c_f = movingaverage(costs_f, 100)

        # Plot learning curves, with base and optimal test costs as a reference
        fig = Figure(size=(500,400), fontsize=18)
        ax = Axis(fig[1,1], xlabel="Epochs", ylabel="Training cost", yscale=Makie.log10)

        lines!(ax, c_y, label="Youla", linewidth=2, color=colour_y)
        lines!(ax, c_f, label="Feedback", linewidth=2, color=colour_f)
        lines!(ax, J_base*ones(length(c_y)), linestyle=:dash, color=colour_b, label="Base", linewidth=2)
        lines!(ax, J_opt*ones(length(c_y)) , linestyle=:dash, color=colour_o, label="Optimal", linewidth=2)

        axislegend(ax, position=:rt)
        xlims!(ax, 0, length(c_y))
        ax.width = 380
        display(fig)
        save(string(savedir, save_name, "_losscurve.pdf"), fig)

        return nothing
    end
end

println("Starting $(length(batch_ids)) experiments...")
pmap(run_experiment_and_plot, batch_ids)
println("All done.")
