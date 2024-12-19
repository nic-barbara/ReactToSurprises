using BSON
using ChainRulesCore: @ignore_derivatives
using Flux
using Printf
using Random
using RobustNeuralNetworks
using Statistics

include(joinpath(@__DIR__, "models.jl"))


########### Simulation functions ###########

get_internal_states(ξ::Vector, n::Int) = ξ[1:n], ξ[(n+1):end]
get_internal_states(ξ::Matrix, n::Int) = ξ[1:n, :], ξ[(n+1):end, :]

"""
Initial states: time, plant state, model state
"""
function init_states(nx, nξ, batches; rng=Random.GLOBAL_RNG)
    t0 = 1
    x0 = 0.5*randn(rng, nx, batches)
    ξ0 = zeros(nξ+nx, batches)
    return [t0, x0, ξ0]
end

"""
Evaluate network in Feedback or Youla frameworks
"""
function feedback_augmentation(model, basectrl::LinearCtrl, ξ, y; youla=true)
    x̂, x̄ = get_internal_states(ξ, basectrl.nx)
    ỹ = y - measurement(basectrl.G, x̂)
    if youla
        x̄, ũ = model(x̄, ỹ)
    else
        x̄, ũ = model(x̄, y)
    end
    u = -basectrl.K * x̂ + ũ
    x̂ = dynamics(basectrl.G, x̂, u) + basectrl.L * ỹ
    return vcat(x̂, x̄), u
end

"""
Simulate over some time horizon and return final cost & state.
"""
function simulate(
    plant::LinearSystem,
    model,
    basectrl::LinearCtrl,
    costfunc::Function,
    states;
    w=[],
    v=[],
    horizon=1,
    youla=true,
    rng=Random.GLOBAL_RNG
)

    # Initialisation
    J = 0
    t0, x, ξ = states                                           # Unpack states/time
    N = size(x, 2)                                              # Batches
    w = isempty(w) ? procnoise(plant, N, horizon; rng) : w      # Process noise
    v = isempty(v) ? measnoise(plant, N, horizon; rng) : v      # Measurement noise
    ts = t0:(t0+horizon-1)

    # Simulate over horizon until maximum time is reached,
    # then break early if required.
    for t in ts
        i = t - t0 + 1
        vt = v[:,:,i]
        wt = w[:,:,i]

        y = measurement(plant, x, vt)
        ξ, u = feedback_augmentation(model, basectrl, ξ, y; youla)
        J += costfunc(x, u)
        x = dynamics(plant, x, u, wt)

        (t >= plant.max_steps) && (horizon = t; break)
    end
    return J / horizon, [t0 + horizon, x, ξ]                    # Mean cost and pack states
end


"""
Same as `simulate()` but it logs the costs cumulatively,
returns the state trajectories, and doesn't break at `max_steps`.
"""
function sim_test(
    plant::LinearSystem,
    model,
    basectrl::LinearCtrl,
    costfunc::Function,
    states;
    w=[],
    v=[],
    horizon=1,
    youla=true,
    rng=Random.GLOBAL_RNG
)
    t0, x, ξ = states                                           # Unpack states/time
    N = size(x, 2)                                              # Batches
    w = isempty(w) ? procnoise(plant, N, horizon; rng) : w      # Process noise
    v = isempty(v) ? measnoise(plant, N, horizon; rng) : v      # Measurement noise
    ts = t0:(t0+horizon-1)
    js = zeros(horizon)
    xs = zeros(plant.nx, horizon)

    for t in ts
        i = t - t0 + 1
        vt = v[:,:,i]
        wt = w[:,:,i]
        xs[:,i] .= x[:,1]

        y = measurement(plant, x, vt)
        ξ, u = feedback_augmentation(model, basectrl, ξ, y; youla)
        js[i] = costfunc(x, u)
        x = dynamics(plant, x, u, wt)
    end
    return cumsum(js) ./ (1:horizon), xs
end


########### Training functions ###########

windowaverage(x, n) = length(x) < n ? mean(x) : mean(x[end-n+1:end])
movingaverage(x, n) = [i < n ? mean(x[1:i]) : mean(x[i-n+1:i]) for i in 1:length(x)]

setup_model(m) = m
setup_model(m::AbstractRENParams) = REN(m)

"""
Train a controller augmentation network using a version of analytic policy grads.

# Arguments

- `model`: the trainable model.
- `plant::LinearSystem`: the plant to control.
- `basectrl::LinearCtrl`: linear base controller.
- `costfunc::Function`: cost function to minimise.

# Keyword Arguments

- `lr=1e-3`: learning rate, decreased by factor of 10 at intervals.
- `nepochs=10`: number of training epochs.
- `horizon=200`: number of samples in time for a single update step.
- `batches=10`: number of parallel environments in training batch.
- `youla=true`: whether to use the Feedback or Youla augmentation architecture.
- `verbose=true`: whether to print losses along the way.
- `rng=Random.GLOBAL_RNG`: random seed.
- `save_params=("", [])`: optional argument to save the model parameters at intermediate epochs in a specified fielpath. Don't specify ".bson" in the filename, we do that:
    save_params=("<save_path/name>", [epochs_to_save_at])
"""
function train_model!(
    model,
    plant::LinearSystem,
    basectrl::LinearCtrl,
    costfunc::Function;
    lr=1e-3,
    nepochs=10,
    horizon=200,
    batches=10,
    youla=true,
    verbose=true,
    rng=Random.GLOBAL_RNG,
    save_params=("", [])
)

    # Setup
    costs = Vector{Float64}()
    opts = OptimiserChain(ClipNorm(1), Adam(lr))
    opt_state = Flux.setup(opts, model)
    states = init_states(plant.nx, model.nx, batches; rng)
    save_intermediate = !isempty(save_params[2])

    # Loss function: updates states [x, ξ] internally
    function loss!(states, model)
        eval_model = setup_model(model)
        J, x_new = simulate(plant, eval_model, basectrl, costfunc, states;  
                            horizon, youla, rng)
        @ignore_derivatives states .= x_new
        return J
    end

    for k in 1:nepochs

        # Update model params
        train_cost, ∇J = Flux.withgradient(loss!, states, model)
        Flux.update!(opt_state, model, ∇J[2])

        # Log costs and print every little while
        push!(costs, train_cost)
        if k % 10 == 0
            verbose && (@printf "Iter %d loss: %.2f\n" k windowaverage(costs, 100))
        end

        # Drop the learning rate at 1/4, 3/4 way
        (k == Int(nepochs/4)) && Flux.adjust!(opt_state, 0.1lr)
        (k == Int(3nepochs/4)) && Flux.adjust!(opt_state, 0.01lr)

        # Save model params if required
        if save_intermediate && (k in save_params[2])
            save_cost = windowaverage(costs, 100)
            save_model_params(save_params[1], model, k, save_cost)
        end

        # Reset states only if we exceed the timer
        if states[1] >= plant.max_steps
            states .= init_states(plant.nx, model.nx, batches; rng)
        end
    end
    return costs
end

function save_model_params(fname, model, k, cost)
    fname = string(fname, "_e$(k).bson")
    bson(fname, Dict("model" => model, "epoch" => k, "cost" => cost))
end


########### Plotting functions ###########

function get_bson_files(fpath)
    fnames = String[]
    for fname in readdir(fpath)
        if fname[end-4:end] == ".bson"
            push!(fnames, string(fpath, fname))
        end
    end
    return fnames
end

function cost_stats(costs)
    μ = mean(costs)
    σ = std(costs)

    cs = zeros(length(costs), length(μ))
    for k in eachindex(costs)
        cs[k,:] = costs[k]
    end
    cmin = vec(minimum(cs;dims=1))
    cmax = vec(maximum(cs;dims=1))

    return μ, σ, cmin, cmax
end
