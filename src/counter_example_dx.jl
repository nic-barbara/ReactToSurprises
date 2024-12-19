using CairoMakie
using Random

rng = Xoshiro(24)

# Disturbance signals
dx1(t) = 0
dx2(t) = 2

# Dynamics: Youla param, plant, observer
a, k = 0.5, 5
Q(ỹ) = (ỹ >= a) ? k * (a - ỹ) : 0
f(x, x̂, dx) = 0.5*abs.(x) + Q.(x - x̂) .+ dx
fo(x, x̂) = 0.5*abs.(x̂) + Q.(x - x̂)


# Rollout
function simulate(dx::Function)

    # Sample the system
    T = 25
    x0 = [-1, 0]
    x̂0 = [0, -1]

    # Setup
    xs = zeros(2, T)
    x̂s = zeros(2, T)
    xs[:,1] = x0
    x̂s[:,1] = x̂0

    # Roll out simulation
    for t in 1:T-1
        xs[:,t+1] = f(xs[:,t], x̂s[:,t], dx(t))
        x̂s[:,t+1] = fo(xs[:,t], x̂s[:,t])
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
x1, x̂1 = simulate(dx1)
x2, x̂2 = simulate(dx2)

colours = Makie.wong_colors()

with_theme(theme_latexfonts()) do
    fig = Figure(size = (400, 200))
    ax = Axis(fig[1,1], xlabel="Time samples", ylabel="States")
    make_plot(x2, x̂2, ax, colours[3], L"w = 2")
    make_plot(x1, x̂1, ax, colours[2], L"w = 0")
    axislegend(position=:rb)
    ylims!(ax, (-14,10))
    xlims!(ax, (1,25))
    save(joinpath(@__DIR__, "../results/disturbance_example_dx.pdf"), fig)
end
