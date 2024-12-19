using ControlSystemsBase
using LinearAlgebra
using Statistics

label = "mass"

# Linearised (uncertain) cartpole system
mc = 1
l = 10
g = 9.81
mp_range = (0.14, 0.35)
mp_nom = 0.2
x0_max = [5, 0.5, 1.0, 0.5]

Ac(mp) = [0 1 0 0;
          0 0 -mp*g/mc 0;
          0 0 0 1;
          0 0 (mc+mp)*g / (mc*l) 0]
Bc = reshape([0, 1/mc, 0, -1/mc], 4, 1)
C = [1 0 0 0]
D = 0

# Discrete-time version
dt = 0.02
A(mp) = I(4) + dt * Ac(mp)
B = dt * Bc
Anom = A(mp_nom)

# Weights for nominal (robust) LQG controller
Q = diagm([1, 0, 1, 0])
R = I(1)
σw = [1, 1e3, 1, 1e3] * dt      # Process noise variance (not stdev)
σv = [1e-3] / dt                # Measurement noise variance (not stdev)
Σw = diagm(σw)
Σv = diagm(σv)

K = lqr(Discrete, Anom, B, Q, R)
L = transpose(lqr(Discrete, Anom', C', Σw, Σv))

# Different weights for known-mass controller
σw = 1e-3 * ones(4) * dt
σv = 1e-4 * ones(1) / dt
Σw = diagm(σw)
Σv = diagm(σv)

# Store the LTI model as an uncertain linear system
G = UncertainLsys(A, B, C; σw, σv, x0_max, param_range=mp_range, ρ_nominal=mp_nom)
Gopt = UncertainLqgSystem(deepcopy(G), Q, R, Σw, Σv)

# Quadratic cost function to minimise
cost(x::AbstractVector, u::AbstractVector) = (x' * Q * x) + (u' * R * u)
cost(x::Matrix, u::Matrix) = mean(sum((Q * x) .* x; dims=1) + sum((R * u) .* u; dims=1))

# Design frequency filter for Feedback output (hardcoded from matlab)
Af = [ 0.36788      0.12589     -0.28194     -0.40305
             0      0.36788            1            0
             0            0      0.36788     -0.48424
             0            0            0      0.36788]
Bf = [4.0556, 0, 4.8726, 6.3604]
Cf = [-6.3369 3.1042 -6.9518 -9.9381]
Df = [100]

# Design frequency filter for Youla output (hardcoded from matlab)
Ay = [ 0.36788      0.12589     -0.28194     -0.40305
             0      0.36788            1            0
             0            0      0.36788     -0.48424
             0            0            0      0.36788]
By = [32.445, 0, 38.981, 50.883]
Cy = [-15.842 7.7605 -17.38 -24.845]
Dy = [2000]

# Stick it together
Bf = reshape(Bf, length(Bf), 1)
By = reshape(By, length(By), 1)
Df = reshape(Df, 1, 1)
Dy = reshape(Dy, 1, 1)
Gf_filter = ss(Af, Bf, Cf, Df, dt)
Gy_filter = ss(Ay, By, Cy, Dy, dt)

Gf_nofilter = ss(1, dt)
Gy_nofilter = ss(1, dt)

# Store in struct
K_base = StaticLQG(K, L, Gf_filter, Gy_filter)
K_base_nofilter = StaticLQG(K, L, Gf_nofilter, Gy_nofilter)

# Nominal-mass optimal controller (for evaluation)
K_lqg_nom = lqr(Discrete, Anom, B, Q, R)
L_lqg_nom = transpose(lqr(Discrete, Anom', C', Σw, Σv))
K_nom = deepcopy(K_base)
K_nom.K = K_lqg_nom
K_nom.L = L_lqg_nom

# Gain bounds from ũ -> y and ũ -> ỹ (hardcoded from matlab),
# with and without the output weighting filters (respectively)
# It seems hinfnorm doesn't always work in ControlSystemsBase.jl
γ_y_filt = 6.4858
γ_dy_filt = 0.56251

γ_y_nofilt = 5.3212
γ_dy_nofilt = 0.0082191
