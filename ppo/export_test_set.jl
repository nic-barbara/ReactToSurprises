using Random
using RobustNeuralNetworks

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
