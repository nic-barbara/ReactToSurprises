using CairoMakie
using Random

rng = Xoshiro(24)

# Disturbance signals
dy1(t) = 0
dy2(t) = 1

# Dynamics: Youla param, plant, observer
Q(ỹ) = (ỹ >= 0.5) ? 1.5 * (0.5 - ỹ) : 0
f(x, x̂, dy) = 0.5*abs.(x) + Q.(x - x̂ .+ dy)
fo(x, x̂, dy) = 0.5*abs.(x .+ dy) + Q.(x - x̂ .+ dy)

# Rollout
function simulate(dy::Function)

    # Sample the system
    T = 25
    x0 = [0, -1]
    x̂0 = [-1.5, 0.5]

    # Setup
    xs = zeros(2, T)
    x̂s = zeros(2, T)
    xs[:,1] = x0
    x̂s[:,1] = x̂0

    # Roll out simulation
    for t in 1:T-1
        xs[:,t+1] = f(xs[:,t], x̂s[:,t], dy(t))
        x̂s[:,t+1] = fo(xs[:,t], x̂s[:,t], dy(t))
    end

    return xs, x̂s
end

# Plotting
function make_plot(xs, x̂s, ax, color, label; lstyles=[:solid, :dash])
    linewidth = 1.5
    lines!(ax, xs[1, :]; color, linewidth, linestyle=lstyles[1], label=label)
    lines!(ax, xs[2, :]; color, linewidth, linestyle=lstyles[2])
end

# Run the example
x1, x̂1 = simulate(dy1)
x2, x̂2 = simulate(dy2)

colours = Makie.wong_colors()

with_theme(theme_latexfonts()) do
    fig = Figure(size = (400, 200))
    ax = Axis(fig[1,1], xlabel="Time samples", ylabel="States")
    make_plot(x2, x̂2, ax, colours[3], L"v = 1")
    make_plot(x1, x̂1, ax, colours[2], L"v = 0")
    axislegend(position=:rb)
    xlims!(ax, (1,25))
    save(joinpath(@__DIR__, "../results/disturbance_example_dy.pdf"), fig)
end
