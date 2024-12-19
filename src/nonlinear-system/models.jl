using Random

"""
Discretise a continuous function with Euler integration.
"""
function discretise(f::Function, x, u, dt)
    return x + dt * f(x, u)
end


"""
Academic example for the paper - a simple 3-state nonlinear
system, where the last two states are measurable.
"""
Base.@kwdef mutable struct NonlinearExample
    nx = 3
    nu = 1
    ny = 2
    σw = 1e-2 * 0.05
    σv = 1e-3 / 0.05
    dt = 0.05
end

function cts_dynamics(x::AbstractMatrix, u::AbstractMatrix)

    x1 = @view x[1:1, :]
    x2 = @view x[2:2, :]
    x3 = @view x[3:3, :]

    ẋ1 = -x1 + x3
    ẋ2 = x1 .^ 2 - x2 - 2 * x1 .* x3 + x3
    ẋ3 = -x2 + u

    return [ẋ1; ẋ2; ẋ3]
end

dynamics(G::NonlinearExample, x, u, w=0) = discretise(cts_dynamics, x, u, G.dt) .+ w

measurement(x::AbstractMatrix, v=0) = (@view x[2:3, :]) .+ v

function procnoise(G::NonlinearExample, batches, steps; rng=Random.GLOBAL_RNG)
    return sqrt(G.σw) * randn(rng, G.nx, batches, steps)
end

function measnoise(G::NonlinearExample, batches, steps; rng=Random.GLOBAL_RNG)
    return sqrt(G.σv) * randn(rng, G.ny, batches, steps)
end


"""
Stabilising base controller regulates the system to `(x,u) = (0,0)`.
"""
function basectrl(y::AbstractMatrix; k=1.5)
    x2 = @view y[1:1, :]
    x3 = @view y[2:2, :]
    return x2 - k * x3
end


"""
State estimator (observer) used by the Youla controller.
"""
function cts_observer(x̂::AbstractMatrix, y::AbstractMatrix, u::AbstractMatrix)

    x̂1 = @view x̂[1:1, :]
    x̂2 = @view x̂[2:2, :]
    x̂3 = @view x̂[3:3, :]
    y1 = @view y[1:1, :]
    y2 = @view y[2:2, :]

    ẋ1 = -x̂1 + x̂3
    ẋ2 = x̂1 .^ 2 - x̂2 - 2 * x̂1 .* x̂3 + x̂3
    ẋ3 = -x̂2 + u + ((y2 - x̂3) - (y1 - x̂2))

    return [ẋ1; ẋ2; ẋ3]
end

function cts_observer(x̂, u)
    y = @view u[1:2, :]
    u = @view u[3:3, :]
    return cts_observer(x̂, y, u)
end

observer(G::NonlinearExample, x̂, y, u) = discretise(cts_observer, x̂, [y; u], G.dt)
