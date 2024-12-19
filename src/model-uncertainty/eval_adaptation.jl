using BSON
using Random
using RobustNeuralNetworks
using Statistics

# Load tools and run experimental setup
include(joinpath(@__DIR__, "models.jl"))
include(joinpath(@__DIR__, "functions.jl"))
include(joinpath(@__DIR__, "setup_mass.jl"))

# Path to demo models
name = "showcase/lcp_$(label)_84nx_128nv_best.bson"
fpath = joinpath(@__DIR__, "../../results/model-uncertainty/batch-inputfilter/")
fname = string(fpath, name)

# Load all the models
load_data(fname, key) = BSON.load(fname)[key]
youla_ren  = REN(load_data(fname, "youla_ren"))
youla_lren = REN(load_data(fname, "youla_lren"))
youla_ren_nf = REN(load_data(fname, "youla_ren_nf"))
fdbak_ren = REN(load_data(fname, "fdbak_ren"))
fdbak_lstm = load_data(fname, "fdbak_lstm")

base_ren = deepcopy(youla_ren)
set_output_zero!(base_ren)

# Choose parameter range and sampling
n_params = 50
n_params_lims = 50
hi_range_min = (G.param_range[2] - mean(G.param_range)) * 0.8 + mean(G.param_range)
lo_range_max = (G.param_range[1] - mean(G.param_range)) * 0.8 + mean(G.param_range)
ρs = [
    LinRange(G.param_range[1], lo_range_max, n_params_lims)...,
    LinRange(lo_range_max, hi_range_min, n_params)...,
    LinRange(hi_range_min, G.param_range[2], n_params_lims)...,
]

# Choose test data
rng_() = Xoshiro(1)
test_batches = 50
train_horizon = 800
test_horizon = 8 * train_horizon

nf = K_base.nx_filter
test_states_ren = init_states!(G, base_ren.nx, nf, test_batches; rng=rng_())
test_states_ren_nf = init_states!(G, base_ren.nx, 0, test_batches; rng=rng_())
test_states_lren = init_states!(G, youla_lren.nx, nf, test_batches; rng=rng_())
test_states_lstm = init_states!(G, fdbak_lstm.nx, 0, test_batches; rng=rng_())
test_states_opt = init_states!(Gopt, 0, 0, test_batches; rng=rng_())

w_test = procnoise(G, test_batches, test_horizon; rng=rng_())
v_test = measnoise(G, test_batches, test_horizon; rng=rng_())

# Roll out the policies for each model param in a range
function sim_test(model, test_states, K; youla=true)
    costs = zeros(length(ρs))
    for k in eachindex(ρs)
        G.A = [G.Afunc(ρs[k])]
        J, _ = simulate(G, model, K, cost, test_states; horizon=test_horizon, 
                        youla, w=w_test, v=v_test)
        costs[k] = J
    end
    return costs
end

J_bs = sim_test(base_ren, test_states_ren, K_base)
J_nom = sim_test(base_ren, test_states_ren, K_nom)
J_yr = sim_test(youla_ren, test_states_ren, K_base)
J_lr = sim_test(youla_lren, test_states_lren, K_base)
J_nf = sim_test(youla_ren_nf, test_states_ren_nf, K_base_nofilter)
J_fr = sim_test(fdbak_ren, test_states_ren, K_base; youla=false)
J_fl = sim_test(fdbak_lstm, test_states_lstm, K_base_nofilter; youla=false)

# Handle the optimal policy separately
function sim_test(Gopt::UncertainLqgSystem, test_states; time_varying=true)
    costs = zeros(length(ρs))
    for k in eachindex(ρs)
        Gopt.sys.A = [Gopt.sys.Afunc(ρs[k])]
        J, _ = simulate(Gopt, cost, test_states; horizon=test_horizon, 
                        w=w_test, v=v_test, time_varying)
        costs[k] = J
    end
    return costs
end
J_os = sim_test(Gopt, test_states_opt)
J_os_lti = sim_test(Gopt, test_states_opt; time_varying=false)

# Save the cost curves for plotting later (they take a while to compute)
bson(
    joinpath(fpath, "showcase/param_variation_costs.bson"),
    Dict(
        "ρs" => ρs,
        "J_bs" => J_bs,
        "J_nom" => J_nom,
        "J_yr" => J_yr,
        "J_lr" => J_lr,
        "J_nf" => J_nf,
        "J_fr" => J_fr,
        "J_fl" => J_fl,
        "J_os" => J_os,
        "J_os_lti" => J_os_lti,
        "lo_range_max" => lo_range_max,
        "hi_range_min" => hi_range_min,
    )
)