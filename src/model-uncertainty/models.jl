using ControlSystemsBase
using Flux
using MatrixEquations
using Random
using RobustNeuralNetworks


"""
Store gains for a static output-feedback controller and
I/O filter.
"""
mutable struct StaticLQG
    K
    L
    fdbak_filter::StateSpace
    youla_filter::StateSpace
    nx_filter
end

function StaticLQG(K, L, ff=ss(1), yf=ss(1))
    nx = size(yf.A, 1)
    return StaticLQG(K, L, ff, yf, nx)
end

function apply_filter(G::StateSpace, x, u)
    y = G.C * x + G.D * u
    x = G.A * x + G.B * u
    return x, y
end


"""
Construct an LPV system with full state knowledge.

Assumes parameter uncertainty in the A-matrix only.
"""
mutable struct UncertainLsys
    nx::Int
    nu::Int
    ny::Int
    Afunc::Function
    Anom # nominal A-matrix
    A
    B
    C
    σw # Process noise variance
    σv # Measurement noise variance
    param_range::Tuple
    x0_max
end

function UncertainLsys(
    Afunc::Function, 
    B,
    C;
    σw=1,
    σv=1,
    ρ_nominal=nothing,
    param_range::Tuple,
    x0_max
)
    nx, nu = size(B)
    ny = size(C,1)
    A = [zeros(nx, nx)]

    ρ_nominal === nothing && (ρ_nominal = mean(param_range))
    Anom = Afunc(ρ_nominal)

    return UncertainLsys(nx, nu, ny, Afunc, Anom, A, B, C, σw, σv, 
                         param_range, x0_max)
end

function init_uncertain_sys!(
    G::UncertainLsys,
    batches::Int; 
    rand_init=true,
    skew=false,
    rng=Random.GLOBAL_RNG,
)

    if skew
        # Preferentially bias sampling towards boundaries
        μ = mean(G.param_range)
        b1 = Int(floor(batches/2))
        b2 = Int(ceil(batches/2))
        ρs1 = scale(LinRange(0, 1, b1) .^2 , G.param_range[1], μ)
        ρs2 = scale(LinRange(0, 1, b2) .^2 , G.param_range[2], μ)
        ρs = vcat(ρs1, reverse(ρs2))
    else
        ρs = LinRange(G.param_range..., batches)
    end

    if rand_init
        x0 = scale(rand(rng, G.nx, batches), -G.x0_max, G.x0_max)
    else
        x0 = ones(G.nx, batches) .* G.x0_max
    end

    G.A = G.Afunc.(ρs)
    return x0
end

scale(x, a, b) = (b - a) .* x .+ a # map from [0,1] -> [a,b]

# Efficiently do Ax + Bu when A is a vector of `(nx, nx)` matrices.
function dynamics(G::UncertainLsys, x, u, w=0) 
    Ax = stack(G.A .* eachcol(x))
    return Ax .+ G.B * u .+ w
end

nominal_dynamics(G::UncertainLsys, x, u, w=0) = G.Anom * x + G.B * u .+ w
measurement(G::UncertainLsys, x, v=0) = G.C * x .+ v

function procnoise(G::UncertainLsys, batches, steps; rng=Random.GLOBAL_RNG)
    return sqrt.(G.σw) .* randn(rng, G.nx, batches, steps)
end

function measnoise(G::UncertainLsys, batches, steps; rng=Random.GLOBAL_RNG)
    return sqrt.(G.σv) .* randn(rng, G.ny, batches, steps)
end


"""
Close the loop of an `UncertainLsys` with an LQG controller
that always knows the true value of the uncertain parameter.

This controller is the optimal controller, use it for comparison.
"""
mutable struct UncertainLqgSystem
    sys::UncertainLsys
    nx::Int
    nu::Int
    ny::Int
    Q
    R
    Σw
    Σv
    K
    L
end

function UncertainLqgSystem(G::UncertainLsys, Q, R, Σw, Σv)
    K = [zeros(G.nu, G.nx)]
    L = [zeros(G.ny, G.nx)]
    return UncertainLqgSystem(G, G.nx, G.nu, G.ny, Q, R, Σw, Σv, K, L)
end

"""
Compute time-varying LQR gain
"""
function get_tv_lqr_gain(G::UncertainLqgSystem, A, T)

    B = G.sys.B
    Q = G.Q
    R = G.R

    P = Q
    Ks = Matrix{Float64}[]
    for _ in T:-1:1
        K = ((R + B' * P * B) \ (B' * P)) * A
        P = (A' * P * A) + Q - (A' * P * B) * K
        push!(Ks, K)
    end

    return reverse(Ks)
end


"""
Compute time-varying Kalman gain
"""
function get_tv_kalman_gain(G::UncertainLqgSystem, A, T)

    C = G.sys.C
    Σw = G.Σw
    Σv = G.Σv
    X = diagm(G.sys.x0_max .^ 2)

    Σt = X
    Ls = Matrix{Float64}[]
    for _ in 0:T
        L = (Σt * C') / (Σv + C * Σt * C')
        Σt = Σt - L * (C * Σt)
        Σt = A * Σt * A' + Σw
        push!(Ls, L)
    end

    return Ls
end


"""
Compute static LQR gain, and stack as if it was time-varying
so that we can use it the same as the time-varying gains.
"""
function get_lti_lqr_gain(G::UncertainLqgSystem, A, T)
    B = G.sys.B
    Q = G.Q
    R = G.R
    return [lqr(Discrete, A,B,Q,R) for _ in 1:T]
end


"""
Compute static Kalman gain, and stack as if it was time-varying
so that we can use it the same as the time-varying gains.
"""
function get_lti_kalman_gain(G::UncertainLqgSystem, A, T)
    C = G.sys.C
    Σw = G.Σw
    Σv = G.Σv
    return [lqr(Discrete, A', C', Σw, Σv)' for _ in 0:T]
end


"""
Compute optimal cost achievable with static LQG (infinite horizon)

Maths from Stephen Boyd's lecture series
"""
function get_optimal_lqg_static_cost(G::UncertainLqgSystem, A)

    B = G.sys.B
    C = G.sys.C
    Q = G.Q
    R = G.R
    Σw = G.Σw
    Σv = G.Σv

    P, _, _, _ = ared(A, B, R, Q, zeros(G.nx, G.nu))
    Σ̃, _, _, _ = ared(A', C', Σv, Σw, zeros(G.nx, G.ny))

    Σ = Σ̃ - Σ̃ * C' * ((C * Σ̃ * C' + Σv) \ (C * Σ̃))
    J = tr(Q * Σ) + tr(P * (Σ̃ - Σ))

    return J
end

function get_optimal_lqg_static_cost(G::UncertainLqgSystem)
    Js = get_optimal_lqg_static_cost.((G,), G.sys.A)
    return mean(Js)
end


"""
Compute optimal cost achievable with time-varying LQG (finite horizon)

Maths from Stephen Boyd's lecture series
"""
function get_optimal_lqg_cost(G::UncertainLqgSystem, A, T)

    B = G.sys.B
    C = G.sys.C
    Q = G.Q
    R = G.R
    Σw = G.Σw
    Σv = G.Σv
    X = diagm(G.sys.x0_max .^ 2)

    # Control calculation
    Ps = [Matrix{Float64}(Q)]
    for _ in T:-1:1
        P1 = Ps[end]
        P = A' * P1 * A + Q - A' * P1 * B * ((R + B' * P1 * B) \ (B' * P1)) * A
        push!(Ps, P)
    end
    Ps = reverse(Ps)

    # Covariance calculation
    Σ1 = X
    Σs = Matrix{Float64}[]
    for _ in 0:T
        Σt = Σ1 - Σ1 * C' * ((Σv + C * Σ1 * C') \ (C * Σ1))
        push!(Σs, Σt)
        Σ1 = A * Σt * A' + Σw
    end

    # Cost computation
    J_lqr = tr(Ps[1] * X) + sum([tr(Ps[t] * Σw) for t in 2:(T+1)])
    J_est = tr((Q - Ps[1]) * Σs[1]) + sum(
        [tr((Q - Ps[t]) * Σs[t]) + tr(Ps[t] * A * Σs[t-1] * A') for t in 2:(T+1)]
    )

    return J_lqr + J_est
end

function get_optimal_lqg_cost(G::UncertainLqgSystem, T)
    Js = get_optimal_lqg_cost.((G,), G.sys.A, T)
    return mean(Js)
end


function init_uncertain_sys!(
    G::UncertainLqgSystem, 
    batches::Int; 
    skew=false,
    rand_init=true,
    rng=Random.GLOBAL_RNG
)
    return init_uncertain_sys!(G.sys, batches; skew, rand_init, rng)
end

dynamics(G::UncertainLqgSystem, x, u, w=0) = dynamics(G.sys, x, u, w)

nominal_dynamics(G::UncertainLqgSystem, x, u, w=0) = nominal_dynamics(G.sys, x, u, w)

measurement(G::UncertainLqgSystem, x, v=0) = measurement(G.sys, x, v)

procnoise(G::UncertainLqgSystem, batches, steps; rng=Random.GLOBAL_RNG) = procnoise(G.sys, batches, steps; rng)

measnoise(G::UncertainLqgSystem, batches, steps; rng=Random.GLOBAL_RNG) = measnoise(G.sys, batches, steps; rng)


"""
    glorot_normal(n::Int, m::Int; T=Float64, rng=Random.GLOBAL_RNG)

Generate matrices or vectors from Glorot normal distribution
"""
glorot_normal(n::Int, m::Int; T=Float64, rng=Random.GLOBAL_RNG) = 
    convert.(T, randn(rng, n, m) / sqrt(n + m))
glorot_normal(n::Int; T=Float64, rng=Random.GLOBAL_RNG) = 
    convert.(T, randn(rng, n) / sqrt(n))

"""
Simple implementation of an LSTMNetwork
"""
mutable struct LSTMNetwork
    nu::Int
    nv::Int
    ny::Int
    nx::Int                 # Number of states = 2*nv for LSTM
    A
    B
    C
    bx
    by
end

"""
    LSTMNetwork(nu::Int, nv::Int, ny::Int)

Initialise LSTMNetwork network from given input, hidden, and output sizes.
"""
function LSTMNetwork(nu::Int, nv::Int, ny::Int; rng=Random.GLOBAL_RNG, T=Float32)
    nx = 4 * nv 
    A = glorot_normal(nx, nv; rng, T)
    B = glorot_normal(nx, nu; rng, T)
    C = glorot_normal(ny, nv; rng, T)
    bx = zeros(T, nx) / T(sqrt(nx))
    by = zeros(T, ny) / T(sqrt(ny))
    bx[nv+1:2*nv] .= T(1)

    return LSTMNetwork(nu, nv, ny, 2*nv, A, B, C, bx, by)
end

Flux.@functor LSTMNetwork
# Flux.trainable(m::LSTMNetwork) = (A=m.A, B=m.B, C=m.C, bx=m.bx, by=m.by, )

"""
    (m::LSTMNetwork)(ξ0, u)

Call LSTMNetwork given states and input
"""
function (m::LSTMNetwork)(ξ0, u)
    xt = m.A * _hr(ξ0,1:m.nv) + m.B * u .+ m.bx
    ft = Flux.sigmoid.(_hr(xt, 1:m.nv))
    it = Flux.sigmoid.(_hr(xt, m.nv+1:2*m.nv))
    ot = Flux.sigmoid.(_hr(xt, 2*m.nv+1:3*m.nv))
    ct = Flux.tanh.(_hr(xt, 3*m.nv+1:4*m.nv))
    c  = ft .* _hr(ξ0, m.nv+1:2*m.nv) .+ it .* ct
    h  = ot .* Flux.tanh.(c)
    y  = m.C * h .+ m.by 

    return vcat(h,c), y 
end

# Helper function to pick rows
_hr(x::AbstractVector,rows) = @view x[rows]
_hr(x::AbstractMatrix,rows) = @view x[rows,:]

"""
    set_output_zero!(m::LSTMNetwork)

Extend RobustNeuralNetworks.jl method to set
the LSTMNetwork output to 0
"""
function RobustNeuralNetworks.set_output_zero!(m::LSTMNetwork)
    m.C  .*= 0
    m.by .*= 0
    return nothing
end
