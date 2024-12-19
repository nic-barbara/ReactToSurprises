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

scale(x, a, b) = (b - a) .* x .+ a

"""
Initial states: plant state, observer state, policy state
"""
function init_states(nx, nξ, batches; rng=Random.GLOBAL_RNG)
    x0 = scale.(rand(rng, nx, batches), -10, 10)
    ξ0 = zeros(nξ+nx, batches) # observer and policy states
    return [x0, ξ0]
end

"""
Evaluate base controller and network in Youla framework.
"""
function youla_augmentation(G::NonlinearExample, model, ξ, y)
    x̂, x̄ = get_internal_states(ξ, G.nx)
    
    # Network
    ỹ = y - measurement(x̂)
    x̄, ũ = model(x̄, ỹ)

    # Base controller and observer
    u = basectrl(y) + ũ
    x̂ = observer(G, x̂, y, u)

    return vcat(x̂, x̄), u
end

"""
Simulate over some time horizon and return cost & states
"""
function simulate(
    G::NonlinearExample,
    model,
    costfunc::Function,
    states;
    w=[],
    v=[],
    horizon=1, # number of steps to simulate
    log_states=false,
    rng=Random.GLOBAL_RNG
)

    # Setup
    J = 0
    x, ξ = states                                               # Unpack states/time
    N = size(x, 2)                                              # Batches
    w = isempty(w) ? procnoise(G, N, horizon; rng) : w      # Process noise
    v = isempty(v) ? measnoise(G, N, horizon; rng) : v      # Measurement noise
    ts = 1:horizon

    # For logging
    xs = [zeros(G.nx, N) for _ in 1:horizon]
    us = [zeros(G.nu, N) for _ in 1:horizon]

    for t in ts
        y = measurement(x, v[:,:,t])
        ξ, u = youla_augmentation(G, model, ξ, y)
        J += costfunc(x, u)
        @ignore_derivatives log_states && (
            xs[t] = x;
            us[t] = u
        )
        x = dynamics(G, x, u, w[:,:,t])
    end

    return J / horizon, (xs, us)
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
- `G::NonlinearExample`: the plant to control.
- `costfunc::Function`: cost function to minimise.

# Keyword Arguments

- `lr=1e-3`: learning rate, decreased by factor of 10 at intervals.
- `nepochs=10`: number of training epochs.
- `max_steps=100`: maximum number of plant steps to take before resetting state.
- `train_batches=10`: number of parallel environments in training batch.
- `test_batches=100`: number of parallel environments in test batch.
- `test_seed=0`: seed number for generating test data.
- `verbose=true`: whether to print losses along the way.
- `rng=Random.GLOBAL_RNG`: random seed.
"""
function train_model!(
    model,
    G::NonlinearExample,
    costfunc::Function;
    lr=1e-3,
    nepochs=10,
    max_steps=100,
    train_batches=10,
    test_batches=50,
    test_seed=0,
    verbose=true,
    rng=Random.GLOBAL_RNG,
)

    # Setup
    opts = OptimiserChain(ClipNorm(1), Adam(lr))
    opt_state = Flux.setup(opts, model)

    # Loss function
    function loss(states, model)
        eval_model = setup_model(model)
        J, _ = simulate(G, eval_model, costfunc, states; horizon=max_steps, rng)
        return J
    end

    # Evaluate test cost (always with the same data)
    G1 = deepcopy(G)
    function get_test_cost(model)
        rng_() = Xoshiro(test_seed)
        test_model = setup_model(model)
        test_states = init_states(G1.nx, model.nx, test_batches; rng=rng_())

        w_test = procnoise(G1, test_batches, max_steps; rng=rng_())
        v_test = measnoise(G1, test_batches, max_steps; rng=rng_())

        J_test, _ = simulate(G1, test_model, costfunc, test_states; 
                             w=w_test, v=v_test, horizon=max_steps)
        return J_test
    end

    costs = [get_test_cost(model)]
    verbose && (@printf "Iter %d loss: %.2f\n" 0 costs[1])
    for k in 1:nepochs
        
        # Log costs and occasionally print
        push!(costs, get_test_cost(model))
        if k % 5 == 0
            verbose && (@printf "Iter %d loss: %.2f\n" k windowaverage(costs, 5))
        end

        # Update model params with new data
        states = init_states(G.nx, model.nx, train_batches; rng)
        ∇J = Flux.gradient(loss, states, model)
        Flux.update!(opt_state, model, ∇J[2])

        # Drop the learning rate at 1/2 way, and again at 3/4 way
        (k == Int(nepochs/2)) && Flux.adjust!(opt_state, 0.1lr)
        (k == Int(3nepochs/4)) && Flux.adjust!(opt_state, 0.01lr)
    end
    return costs
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
