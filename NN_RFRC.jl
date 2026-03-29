# ─────────────────────────────────────────────────────────────────────────────
# NN_RFRC.jl  —  Recurrence-Free Reservoir Computer (RF-RC) model
# ─────────────────────────────────────────────────────────────────────────────
# Julia convention: matrices are (features × samples), i.e. column-major.
# All layer weights are (out_dim × in_dim) so  y = W * x + b  works directly.
# ─────────────────────────────────────────────────────────────────────────────

module NNRFRC

using LinearAlgebra, Random, Statistics

export ReservoirLayer, RFRC, get_features, fit_readout!, evaluate_model,
       l1_loss, l2_loss

# ── Activation functions ─────────────────────────────────────────────────────
const ACTIVATIONS = Dict{String, Function}(
    "tanh"    => tanh,
    "relu"    => x -> max(zero(x), x),
    "sigmoid" => x -> one(x) / (one(x) + exp(-x)),
    "elu"     => x -> x >= zero(x) ? x : exp(x) - one(x),
)

# ── Single reservoir layer ────────────────────────────────────────────────────
"""
    ReservoirLayer{T}

Fixed (non-trainable) random affine map followed by a nonlinear activation.
Weights are drawn once at construction and never updated.
"""
struct ReservoirLayer{T <: AbstractFloat}
    W   :: Matrix{T}   # (out_dim × in_dim)  frozen
    b   :: Vector{T}   # (out_dim,)          frozen zeros
    act :: Function
end

function ReservoirLayer(T::Type, in_dim::Int, out_dim::Int;
                        activation    = "tanh",
                        input_scaling = one(T),
                        rng           = Random.default_rng())
    haskey(ACTIVATIONS, activation) ||
        error("Unknown activation '$activation'. " *
              "Choose from: $(collect(keys(ACTIVATIONS)))")
    # Uniform draw on [-input_scaling, +input_scaling]
    W = rand(rng, T, out_dim, in_dim) .* T(2input_scaling) .- T(input_scaling)
    return ReservoirLayer{T}(W, zeros(T, out_dim), ACTIVATIONS[activation])
end

# Apply: x (in_dim × N)  →  act.(W*x .+ b)  (out_dim × N)
(l::ReservoirLayer)(x) = l.act.(l.W * x .+ l.b)


# ── RF-RC model ───────────────────────────────────────────────────────────────
"""
    RFRC{T}

Recurrence-Free Reservoir Computer.

Architecture
------------
    Input (N_in)
      → [Fixed Random Reservoir Layer] × num_layers    (no gradients)
      → Trainable Linear Readout  (output layer)

Training
--------
Only `W_out` and `b_out` are trainable.  Primary fit is one-shot ridge
regression via the augmented least-squares system:

    [H^T ; √α·I]  W_out^T  =  [Y^T ; 0]

where H = get_features(X_train).  No backprop, no epoch loop.
An optional Adam fine-tuning pass is available in train_rfrc.jl.

Fields
------
- `reservoir`    : Vector of frozen ReservoirLayer
- `W_out`        : (output_size × reservoir_dim)  trainable
- `b_out`        : (output_size,)                 trainable
- `ridge_alpha`  : ridge regularisation coefficient
"""
mutable struct RFRC{T <: AbstractFloat}
    reservoir     :: Vector{ReservoirLayer{T}}
    W_out         :: Matrix{T}
    b_out         :: Vector{T}
    ridge_alpha   :: T
    reservoir_dim :: Int
    output_size   :: Int
    input_size    :: Int
end

function RFRC(T::Type, input_size::Int, output_size::Int;
              reservoir_dim = 1024,
              num_layers    = 3,
              activation    = "tanh",
              input_scaling = 1.0,
              ridge_alpha   = 1e-4,
              rng           = Random.default_rng())
    layers   = Vector{ReservoirLayer{T}}(undef, num_layers)
    prev_dim = input_size
    for i = 1:num_layers
        layers[i] = ReservoirLayer(T, prev_dim, reservoir_dim;
                                   activation    = activation,
                                   input_scaling = T(input_scaling),
                                   rng           = rng)
        prev_dim = reservoir_dim
    end
    return RFRC{T}(layers,
                   zeros(T, output_size, reservoir_dim),
                   zeros(T, output_size),
                   T(ridge_alpha), reservoir_dim, output_size, input_size)
end

# ── Forward pass ─────────────────────────────────────────────────────────────
"""
    get_features(model, x)

Pass `x` through the frozen reservoir.
- x    : (input_size × N)
- returns (reservoir_dim × N)
"""
function get_features(m::RFRC, x)
    h = x
    for layer in m.reservoir
        h = layer(h)
    end
    return h
end

"""
    model(x)

Full forward pass: reservoir → linear readout.
- x       : (input_size × N)
- returns : (output_size × N)
"""
(m::RFRC)(x) = m.W_out * get_features(m, x) .+ m.b_out


# ── Loss functions ────────────────────────────────────────────────────────────
l1_loss(ŷ, y) = mean(abs.(ŷ .- y))
l2_loss(ŷ, y) = mean((ŷ .- y) .^ 2)

evaluate_model(m::RFRC, X, Y, crit) = crit(m(X), Y)


# ── One-shot ridge regression ─────────────────────────────────────────────────
"""
    fit_readout!(model, X_train, Y_train) → NamedTuple

Closed-form ridge regression fit of `W_out` and `b_out`.

Solves the numerically stable augmented system:

    [H^T ; √α·I_D] · W_out^T  =  [Y^T ; 0]

which is equivalent to (H H^T + αI)^{-1} H Y^T but avoids inverting a
potentially singular matrix.  Julia's `\\` dispatches to LAPACK gelsd/gelsy
for overdetermined systems.

- X_train : (input_size  × N_train)
- Y_train : (output_size × N_train)
"""
function fit_readout!(m::RFRC{T}, X_train::Matrix{T}, Y_train::Matrix{T}) where T
    H  = get_features(m, X_train)   # (D × N)
    Ht = Matrix(transpose(H))       # (N × D)  — copy needed for \
    Yt = Matrix(transpose(Y_train)) # (N × out)
    D  = m.reservoir_dim
    sa = sqrt(m.ridge_alpha)

    H_aug = vcat(Ht, sa .* Matrix{T}(I, D, D))              # ((N+D) × D)
    Y_aug = vcat(Yt, zeros(T, D, m.output_size))             # ((N+D) × out)

    W_T = H_aug \ Y_aug   # (D × out) — LAPACK least-squares, never singular
    m.W_out .= transpose(W_T)
    fill!(m.b_out, zero(T))

    return (H_shape     = size(Ht),
            Y_shape     = size(Yt),
            W_shape     = size(W_T),
            ridge_alpha = m.ridge_alpha)
end

end  # module NNRFRC
