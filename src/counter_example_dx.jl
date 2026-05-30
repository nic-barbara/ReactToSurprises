using CairoMakie
# using Latexify
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
function make_plot(ax, xs, color, label; lstyles=[:solid, :dash])
    linewidth = 1.5
    lab1 = L"%$(label) (IC 1)"
    lab2 = L"%$(label) (IC 2)"
    lines!(ax, xs[1, :]; color, linewidth, linestyle=lstyles[1], label=lab1)
    lines!(ax, xs[2, :]; color, linewidth, linestyle=lstyles[2], label=lab2)
end

# Run the example
x1, x̂1 = simulate(dx1)
x2, x̂2 = simulate(dx2)

colours = Makie.wong_colors()

with_theme(theme_latexfonts()) do
    fig = Figure(size = (450, 240))
    ax = Axis(fig[1,1], xlabel="Time samples", ylabel="States")
    make_plot(ax, x2, colours[3], L"w = 2")
    make_plot(ax, x1, colours[2], L"w = 0")
    axislegend(position=:rb, orientation=:horizontal, nbanks=2, labelsize=12)
    ylims!(ax, (-20,8))
    xlims!(ax, (1,25))
    save(joinpath(@__DIR__, "../results/disturbance_example_dx.pdf"), fig)
end

with_theme(theme_latexfonts()) do
    fig = Figure(size = (450, 240))
    ax = Axis(fig[1,1], xlabel="Time samples", ylabel="States")
    make_plot(ax, x̂2, colours[3], L"w = 2")
    make_plot(ax, x̂1, colours[2], L"w = 0")
    axislegend(position=:rb, orientation=:horizontal, nbanks=2, labelsize=12)
    ylims!(ax, (-20,8))
    xlims!(ax, (1,25))
    save(joinpath(@__DIR__, "../results/disturbance_example_dx_observer.pdf"), fig)
end
