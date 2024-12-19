using ChainRulesCore: @ignore_derivatives
using Flux
using Printf
using Random
using Statistics

include(joinpath(@__DIR__, "models.jl"))


########### Simulation functions ###########

get_internal_states(ξ::Vector, n::Int) = ξ[1:n], ξ[(n+1):end]
get_internal_states(ξ::Matrix, n::Int) = ξ[1:n, :], ξ[(n+1):end, :]


"""
Initial states: plant state, model and policy state.

Assumes observer has the same number of states as the plant.

`nξ`: number of states for the neural network policy
`nf`: number of states for the output filter
"""
function init_states!(G, nξ, nf, batches; skew=false, rand_init=true, rng=Random.GLOBAL_RNG)
    t0 = 1
    x0 = init_uncertain_sys!(G, batches; skew, rand_init, rng)
    ξ0 = zeros(nξ + G.nx + nf, batches)
    return [t0, x0, ξ0]
end


"""
Evaluate network in Feedback or Youla frameworks.
"""
function feedback_augmentation(
    model, 
    basectrl, 
    plant::UncertainLsys,
    ξ, 
    y; 
    youla=true
)
    x̂, x̄ = get_internal_states(ξ, plant.nx + basectrl.nx_filter)
    x̂, s = get_internal_states(x̂, plant.nx)
    ỹ = y - measurement(plant, x̂)
    if youla
        s, ỹ_in = apply_filter(basectrl.youla_filter, s, ỹ)
        x̄, ũ = model(x̄, ỹ_in)
    else
        s, y_in = apply_filter(basectrl.fdbak_filter, s, y)
        x̄, ũ = model(x̄, y_in)
    end
    u = -basectrl.K * x̂ + ũ
    x̂ = nominal_dynamics(plant, x̂, u) + basectrl.L * ỹ
    return vcat(x̂, s, x̄), u
end


"""
Simulate over some time horizon and return final cost & state.
"""
function simulate(
    plant,
    model,
    basectrl,
    costfunc::Function,
    states;
    w=[],
    v=[],
    horizon=1,
    max_steps=Inf,
    youla=true,
    domain_rand=true,
    log_states=false,
    rng=Random.GLOBAL_RNG
)

    J = 0
    t0, x, ξ = states
    f = domain_rand ? dynamics : nominal_dynamics
    xs, us = [], []
    N = size(x, 2)                                              # Batches
    w = isempty(w) ? procnoise(plant, N, horizon; rng) : w      # Process noise
    v = isempty(v) ? measnoise(plant, N, horizon; rng) : v      # Measurement noise
    ts = t0:(t0+horizon-1)

    for t in ts
        i = t - t0 + 1
        vt = v[:,:,i]
        wt = w[:,:,i]

        y = measurement(plant, x, vt)
        ξ, u = feedback_augmentation(model, basectrl, plant, ξ, y; youla)
        J += costfunc(x, u)
        @ignore_derivatives log_states && (push!(xs, x); push!(us, u))

        x = f(plant, x, u, wt)
        (t >= max_steps) && (horizon = t; break)
    end
    return J / horizon, [t0 + horizon, x, ξ], (xs, us)
end


"""
Simulate the optimal (time-varying) LQG controller
"""
function simulate(
    plant::UncertainLqgSystem, 
    costfunc::Function, 
    states;
    w=[],
    v=[],
    horizon=1,
    domain_rand=true,
    log_states=false,
    rng=Random.GLOBAL_RNG,
    time_varying=true,
)

    J = 0
    _, x, x̂ = states
    x̂, _ = get_internal_states(x̂, plant.nx)
    xs, us = [], []
    N = size(x, 2)                                              # Batches
    w = isempty(w) ? procnoise(plant, N, horizon; rng) : w      # Process noise
    v = isempty(v) ? measnoise(plant, N, horizon; rng) : v      # Measurement noise

    # Choose LQG design functions (hacky approach to make the plots, fix later)
    if time_varying
        get_lqr_gain = get_tv_lqr_gain
        get_kalman_gain = get_tv_kalman_gain
    else
        get_lqr_gain = get_lti_lqr_gain
        get_kalman_gain = get_lti_kalman_gain
    end

    # Sort out dynamics and control/observer gains
    if domain_rand
        f = dynamics

        Ks = [get_lqr_gain(plant, Ai, horizon) for Ai in plant.sys.A]
        Ls = [get_kalman_gain(plant, Ai, horizon) for Ai in plant.sys.A]

        # TODO: This is ugly code. Fix it later.
        if length(Ks) > 1
            K = (t, x) -> stack([-Ks[i][t] * x[:,i] for i in axes(x,2)])
            L = (t, ỹ) -> stack([Ls[i][t] * ỹ[:,i] for i in axes(ỹ,2)])
        else
            K = (t, x) -> stack([-Ks[1][t] * x[:,i] for i in axes(x,2)])
            L = (t, ỹ) -> stack([Ls[1][t] * ỹ[:,i] for i in axes(ỹ,2)])
        end

    else
        f = nominal_dynamics

        Knom = get_lqr_gain(plant, plant.sys.Anom, horizon)
        Lnom = get_kalman_gain(plant, plant.sys.Anom, horizon)

        K = (t, x) -> (-Knom[t] * x)
        L = (t, ỹ) -> (Lnom[t] * ỹ)
    end

    # Simulate
    for t in 1:horizon
        vt = v[:,:,t]
        wt = w[:,:,t]

        # Control and logging
        y = measurement(plant, x, vt)
        u = K(t, x̂)
        J += costfunc(x, u)
        @ignore_derivatives log_states && (push!(xs, x); push!(us, u))

        # Plant and prediction updates
        x = f(plant, x, u, wt)
        x̂ = f(plant, x̂, u) + L(t, y - measurement(plant, x̂))
    end
    return J / horizon, [], (xs, us)
end

function simulate(
    plant::UncertainLqgSystem, model, basectrl, costfunc, states;
    w, v, horizon, domain_rand, log_states
)
    return simulate(plant, costfunc, states; w, v, horizon, domain_rand, log_states)
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
- `plant::UncertainLsys`: the plant to control.
- `basectrl::StaticLQG`: linear base controller.
- `costfunc::Function`: cost function to minimise.

# Keyword Arguments

- `lr=1e-3`: learning rate, decreased by factor of 10 at intervals.
- `lr_decay=[1/2, 7/8]`: fraction of `nepochs` at which to drop `lr` by `0.1`.
- `nepochs=10`: number of training epochs.
- `max_steps=100`: maximum number of plant steps to take before resetting state.
- `train_horizon=100`: number of samples in time for a single update step.
- `train_batches=10`: number of parallel environments in training batch.
- `domain_rand=true`: whether to use domain randomization during training.
- `youla=true`: whether to use the Feedback or Youla augmentation architecture.
- `verbose=true`: whether to print losses along the way.
- `test_batches=100`: number of parallel environments in test batch.
- `test_horizon=100`: number of samples in time for a test rollout.
- `test_seed=0`: seed number for generating test data.
- `rng=Random.GLOBAL_RNG`: random seed.
"""
function train_model!(
    model,
    plant::UncertainLsys,
    basectrl::StaticLQG,
    costfunc::Function;
    lr=1e-3,
    lr_decay=[1/2, 7/8], # Hard-coded to only accept 2 at the moment
    nepochs=10,
    max_steps=100,
    train_horizon=100,
    train_batches=10,
    domain_rand=true,
    youla=true,
    verbose=true,
    test_batches=50,
    test_horizon=100,
    test_seed=0,
    rng=Random.GLOBAL_RNG,
)

    # Setup
    opts = OptimiserChain(ClipNorm(1), Adam(lr))
    opt_state = Flux.setup(opts, model)
    n_filter = basectrl.nx_filter
    states = init_states!(plant, model.nx, n_filter, train_batches; rng)

    # Loss function
    function loss!(states, model)
        eval_model = setup_model(model)
        J, x_new, _ = simulate(plant, eval_model, basectrl, costfunc, states; rng,
                               horizon=train_horizon, max_steps, youla, domain_rand)
        @ignore_derivatives states .= x_new
        return J
    end

    # Test costs
    eval_plant = deepcopy(plant)
    function get_test_cost(model)
        rng_() = Xoshiro(test_seed)
        test_model = setup_model(model)
        test_states = init_states!(eval_plant, model.nx, n_filter, test_batches; rng=rng_())
        w_test = procnoise(eval_plant, test_batches, test_horizon; rng=rng_())
        v_test = measnoise(eval_plant, test_batches, test_horizon; rng=rng_())

        test_cost, _, _ = simulate(eval_plant, test_model, basectrl, costfunc, test_states; 
                                   w=w_test, v=v_test, horizon=test_horizon, youla)
        return test_cost
    end

    costs = [get_test_cost(model)]
    verbose && (@printf "Iter %d loss: %.2f\n" 0 costs[1])
    for k in 1:nepochs
        
        # Occasionally log costs and print
        if k % 5 == 0
            push!(costs, get_test_cost(model))
            verbose && (@printf "Iter %d loss: %.2f\n" k windowaverage(costs, 5))
        end

        # Update model params with new data
        ∇J = Flux.gradient(loss!, states, model)
        Flux.update!(opt_state, model, ∇J[2])

        # Drop the learning rate at 1/2 way, and again at 7/8 way (default)
        (k == Int(nepochs * lr_decay[1])) && Flux.adjust!(opt_state, 0.1lr)
        (k == Int(nepochs * lr_decay[2])) && Flux.adjust!(opt_state, 0.01lr)

        # Reset states only if we exceed the timer
        if states[1] >= max_steps
            states .= init_states!(plant, model.nx, n_filter, train_batches; rng)
        end
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
