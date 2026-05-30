using CairoMakie
using Random
using RobustNeuralNetworks

rng = MersenneTwister(42)

# Create a contracting REN with just its state as an output 
nu, nx, nv, ny = 1, 1, 10, 1
ren_ps = ContractingRENParams{Float64}(nu, nx, nv, ny; output_map=false, rng, init=:cholesky)
ren = REN(ren_ps)

# Make it converge a little faster...
ren.explicit.A .-= 4e-3

# Simulate it from different initial conditions
function simulate()

    # Different initial conditions
    x1 = 10*ones(nx)
    x2 = 50*ones(nx)

    # Same inputs
    ts = 1:600
    u = 0.2sin.(0.01*ts)

    # Keep track of outputs
    y1 = zeros(length(ts))
    y2 = zeros(length(ts))

    # Simulate and return outputs
    for t in ts
        x1, ya = ren(x1, u[t:t]')
        x2, yb = ren(x2, u[t:t]')
        y1[t] = -ya[1]
        y2[t] = -yb[1]
    end
    return ts, y1, y2
end
ts, y1, y2 = simulate()

# Plot trajectories
fig = Figure(size = (500, 150))
ax = Axis(fig[1,1],
          xgridvisible=false, ygridvisible=false,
          xticksvisible=false, yticksvisible=false,
          xticklabelsvisible=false, yticklabelsvisible=false)
hidespines!(ax)

lines!(ax, y1, color=:black)
lines!(ax, y2, color=:grey)
save(joinpath(@__DIR__,"../results/contracting_ren_sketch.svg"), fig)
