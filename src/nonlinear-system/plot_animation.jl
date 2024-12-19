using BSON
using GLMakie
using Random
using RobustNeuralNetworks
using Statistics

include(joinpath(@__DIR__, "models.jl"))
include(joinpath(@__DIR__, "functions.jl"))


#######################################################################
#
# Simulate trajectories
#
#######################################################################

# Problem setup
umax = 15.0
G = NonlinearExample()
cost(x::Matrix, u::Matrix) = 0 # (not needed in this script)

# Load a model
batch_id = 9
name = "nl_16nx_128nv_v$(batch_id).bson"
fpath = joinpath(@__DIR__, "../../results/nonlinear-system/batch/")
fname = string(fpath, name)
youla_ren = REN(BSON.load(fname)["youla_ren"])

# Model to test the base controller
base_ren = deepcopy(youla_ren)
set_output_zero!(base_ren)

# Some dummy test data
horizon = Int(10 / G.dt)
batches = 18
w_test = zeros(G.nx, batches, horizon)
v_test = zeros(G.ny, batches, horizon)

# Custom initial states for animations
minibatches = Int(batches/3)
rng = Xoshiro(batch_id)
x0 = hcat(
    scale.(rand(rng, G.nx, minibatches), -10, -9.5) .* [1, -1, 1],
    scale.(rand(rng, G.nx, minibatches), 9.5, 10) .* [1, -1, 1],
    scale.(rand(rng, G.nx, minibatches), 9.5, 10) .* [1, 1, 1],
)
ξ0 = zeros(G.nx + youla_ren.nx, batches)
states = [x0, ξ0]

# Simulate the trained controller and the base controller
J_y, traj_y = simulate(G, youla_ren, cost, states; horizon, 
                       w=w_test, v=v_test, log_states=true)
J_b, traj_b = simulate(G, base_ren, cost, states; horizon,
                       w=w_test, v=v_test, log_states=true)

# Split trajectories into states and controls
xs_y, us_y = traj_y
xs_b, us_b = traj_b
ts = (collect(1:horizon) .- 1) * G.dt

# Re-order for plotting
xs_b = permutedims(stack(xs_b), (1,3,2))
xs_y = permutedims(stack(xs_y), (1,3,2))


#######################################################################
#
# Plot trajectories
#
#######################################################################

# Colour choices
colormap = Makie.to_colormap(:hot)
colormap = colormap[12:length(colormap)]

# Set up figure
set_theme!(theme_black())
fig = Figure(size=(1050,750), fontsize=18)
ax = Axis3(
    fig[1,1],
    aspect=(1,1,1),
    protrusions=(0,0,0,0),
    limits=(-15, 15, -15, 15, -15, 15),
    viewmode=:fit,
    azimuth=1.275π,
    elevation=π/12
)
hidedecorations!(ax, grid=false)

# Set up multiple trajectory plots
n_traj = size(xs_y,3)
points = [Observable(Point3f[]) for _ in 1:n_traj]
colors = [Observable(Int[]) for _ in 1:n_traj]
ls = [
    lines!(
        ax, 
        points[i]; 
        color=colors[i], 
        colormap, 
        transparency=true, 
    )
    for i in 1:n_traj
]

# Add the origin
scatter!(ax, [0], [0], [0], color=:ghostwhite, markersize=15)

# Record the video
savename = string(@__DIR__, "/../../results/nonlinear-system/animation.mp4")
record(fig, savename, 1:(horizon-1)) do frame

    # Add points
    for i in [0,1]
        k = frame+i
        for n in 1:n_traj
            new_point = Point3f(xs_y[1,k,n], xs_y[2,k,n], xs_y[3,k,n])
            push!(points[n][], new_point)
            push!(colors[n][], frame)
        end
    end

    # Update figure
    amplitude = π/6
    ax.azimuth[] = 1.275pi + amplitude * sin(2pi * frame / (4*horizon))
    notify.(points)
    notify.(colors)
    for l in ls
        l.colorrange = (0, frame)
    end
end
