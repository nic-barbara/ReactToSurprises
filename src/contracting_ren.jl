using CairoMakie
using Random
using RobustNeuralNetworks

rng = MersenneTwister(42)

# Create a contracting REN with just its state as an output 
nu, nx, nv, ny = 1, 1, 10, 1
ren_ps = ContractingRENParams{Float64}(nu, nx, nv, ny; output_map=false, rng, init=:cholesky)
ren = REN(ren_ps)

# Make it converge a little faster...
# ren.explicit.A .-= 4.5e-3
ren.explicit.A .-= 8e-3

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

colours = Makie.wong_colors()

# Plot trajectories
fig = Figure(size = (500, 180))
ax = Axis(fig[1,1], xlabel="Time", ylabel="Internal State",
          xgridvisible=false, ygridvisible=false,
          xticklabelsvisible=false, yticklabelsvisible=false)

lines!(ax, y1, label="Initial state 1", color=:black)
lines!(ax, y2, label="Initial state 2", color=:grey, linestyle=:dash)
axislegend(ax, position=:rb)
save(joinpath(@__DIR__,"../results/contracting_ren.pdf"), fig)

# # Create an animation
# fig = Figure(size = 2 .*(600, 400), fontsize=36)
# ax = Axis(fig[1,1], xlabel="Time samples", ylabel="Internal state")

# p1 = Observable(Point2f[(ts[1], y1[1])])
# p2 = Observable(Point2f[(ts[1], y2[1])])

# lines!(ax, p1, linewidth=4, label="Initial condition 1", color=:orange)
# lines!(ax, p2, linewidth=4, label="Initial condition 2", color=:blue)

# axislegend(ax, position=:rb)
# limits!(ax, 0, 625, -18.5, 12.5)

# time = 6
# framerate = 30
# dframe = Int(floor(length(ts) / time / framerate))
# frames = 1:dframe:length(ts)

# record(fig, joinpath(@__DIR__,"../results/contraction_animation.gif"), 1:length(frames); framerate) do i
#     if i > 1
#         indx = frames[(i-1)]:frames[i]
#         new_point1 = Point2f.(ts[indx], y1[indx])
#         new_point2 = Point2f.(ts[indx], y2[indx])
#         p1[] = append!(p1[], new_point1)
#         p2[] = append!(p2[], new_point2)
#     end
# end
# display(fig)