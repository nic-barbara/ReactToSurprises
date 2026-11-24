using BSON
using Random
using RobustNeuralNetworks
using Statistics

include(joinpath(@__DIR__, "../src/model-uncertainty/functions.jl"))
include(joinpath(@__DIR__, "../src/model-uncertainty/setup_mass.jl"))


#######################################################################
#
# Export the test set used in `train_model!` (functions.jl) so that
# PPO policies are evaluated on identical initial states and noise.
#
#######################################################################

test_batches = 64
test_horizon = 1600         # 2 * max_steps in train_vanilla_batch.jl
test_seed = 1

savedir = joinpath(@__DIR__, "data")
!isdir(savedir) && mkdir(savedir)

# Write a matrix as CSV (Base only, full-precision floats)
function writecsv(fname, M)
    open(joinpath(savedir, fname), "w") do io
        for row in eachrow(M)
            println(io, join(row, ","))
        end
    end
end

# Same calls (and RNG order) as get_test_cost in train_model!
rng_() = Xoshiro(test_seed)
_, x0, _ = init_states!(G, 0, K_base.nx_filter, test_batches; rng=rng_())
w = procnoise(G, test_batches, test_horizon; rng=rng_())
v = measnoise(G, test_batches, test_horizon; rng=rng_())

# Rows are (time, batch) pairs with batch varying fastest
writecsv("x0.csv", x0')
writecsv("w.csv", reshape(w, G.nx, :)')
writecsv("v.csv", reshape(v, G.ny, :)')
writecsv("A.csv", stack(vec(Ai') for Ai in G.A)')

# Long test set for rollouts, as in plot_vanilla.jl (12 x 800 steps, same seed).
# Written as raw Float64 (column-major) since CSV would be ~55 MB.
long_horizon = 12 * 800
w_long = procnoise(G, test_batches, long_horizon; rng=rng_())
v_long = measnoise(G, test_batches, long_horizon; rng=rng_())
@assert w_long[:, :, 1:test_horizon] == w && v_long[:, :, 1:test_horizon] == v
write(joinpath(savedir, "w_long.bin"), w_long)
write(joinpath(savedir, "v_long.bin"), v_long)

# Base controller gains (for validating the Python environment)
writecsv("K.csv", K_base.K)
writecsv("L.csv", K_base.L')
writecsv("Anom.csv", G.Anom)

# Reference costs, computed on this test set during the APG experiments
fpath_vanilla = joinpath(@__DIR__, "../results/model-uncertainty/batch-vanilla/")
fpath_youla = joinpath(@__DIR__, "../results/model-uncertainty/batch/")
J_base_vanilla = BSON.load(get_bson_files(fpath_vanilla)[1])["J_base"]
J_base = mean(BSON.load(f)["J_base"] for f in get_bson_files(fpath_youla))
J_opt = mean(BSON.load(f)["J_opt"] for f in get_bson_files(fpath_youla))
println("J_base (vanilla runs): ", J_base_vanilla)
println("J_base (Youla runs):   ", J_base)
println("J_opt  (Youla runs):   ", J_opt)
writecsv("ref_costs.csv", [J_base_vanilla J_base J_opt])
