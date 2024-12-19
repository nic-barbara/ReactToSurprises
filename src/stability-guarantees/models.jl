using Flux
using MatrixEquations
using Random
using RobustNeuralNetworks


"""
Construct an LTI state-space model with a disturbance
channel and measurement noise.
"""
struct LinearSystem
    A
    B
    Bw
    C
    D
    nu::Int
    nx::Int
    ny::Int
    nw::Int
    σw # Process noise variance
    σv # Measurement noise variance
    max_steps::Union{Int, Nothing}
end

function LinearSystem(A, B, Bw, C, D=0; σw=1, σv=1, max_steps=nothing)
    nx = size(A,1)
    nu = size(B,2)
    ny = size(C,1)
    nw = size(Bw,2)
    (D == 0) && (D = zeros(ny, nu))

    return LinearSystem(A, B, Bw, C, D, nu, nx, ny, nw, σw, σv, max_steps)
end

dynamics(G::LinearSystem, x, u, w=0) = G.A * x .+ G.B * u .+ G.Bw * w
measurement(G::LinearSystem, x, v=0) = G.C * x .+ v

function procnoise(G::LinearSystem, batches, steps; rng=Random.GLOBAL_RNG)
    return sqrt(G.σw) * randn(rng, G.nw, batches, steps)
end

function measnoise(G::LinearSystem, batches, steps; rng=Random.GLOBAL_RNG)
    return sqrt(G.σv) * randn(rng, G.ny, batches, steps)
end


"""
Construct a linear feedback controller with an LQR and
a Kalman filter. Store the gains and a copy of the system.
"""
mutable struct LinearCtrl
    K
    L
    G::LinearSystem
    nu::Int
    nx::Int
    ny::Int
end

function LinearCtrl(G::LinearSystem, Q, R, Σw, Σv)
    _, _, K, _ = ared(G.A, G.B, R, Q, zeros(G.nx, G.nu))
    _, _, F, _ = ared(G.A', G.C', Σv, Σw, zeros(G.nx, G.ny))
    L = F'
    return LinearCtrl(K, L, G, G.ny, G.nx, G.nu)
end
