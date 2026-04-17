module NNRFRC

using LinearAlgebra, Statistics

# Exporting for use in the main script
export RFRC, fit_readout!, get_features, l1_loss

struct RFRC{T}
    W_in::Matrix{T}
    W_res::Matrix{T}
    W_out::Matrix{T}
    b_out::Vector{T}
    ridge_alpha::T
    activation::Function
end

# Optimized Constructor
function RFRC(T, N_in, N_out; reservoir_dim=2048, num_layers=3, ridge_alpha=1e-4, activation="tanh")
    act_fn = activation == "tanh" ? tanh : x -> x # Default to tanh
    W_in  = randn(T, reservoir_dim, N_in) .* T(0.1)
    W_res = randn(T, reservoir_dim, reservoir_dim) .* T(0.01)
    W_out = zeros(T, N_out, reservoir_dim)
    b_out = zeros(T, N_out)
    return RFRC{T}(W_in, W_res, W_out, b_out, T(ridge_alpha), act_fn)
end

# HIGH-PERFORMANCE FEATURE EXTRACTION
function get_features(model::RFRC{T}, X::Matrix{T}) where T
    # X is (N_in x N_samples)
    # H = activation(W_in * X)
    # We use 'mul!' to avoid creating temporary matrices in memory
    H = Matrix{T}(undef, size(model.W_in, 1), size(X, 2))
    mul!(H, model.W_in, X)
    H .= model.activation.(H)
    return H
end

# HIGH-PERFORMANCE RIDGE FIT
function fit_readout!(model::RFRC{T}, X::Matrix{T}, Y::Matrix{T}) where T
    H = get_features(model, X)
    D, N = size(H)

    # Instead of H*H', we solve the Ridge problem directly:
    # (H' | sqrt(alpha)*I) * W_out' = (Y' | 0)
    # This is MUCH more stable in Float32.
    
    # 1. Augment H and Y for Ridge Regression
    # We use a 'Tall' matrix approach: A = [H'; sqrt(alpha)*I]
    sqrt_alpha = sqrt(model.ridge_alpha)
    A = [transpose(H); sqrt_alpha * I(D)]
    B = [transpose(Y); zeros(T, D, size(Y, 1))]

    # 2. Solve using QR decomposition (stable and robust)
    # Julia's '\' operator chooses the best algorithm for the shape
    W_out_T = A \ B
    
    # 3. Assign back
    model.W_out .= transpose(W_out_T)
    model.b_out .= vec(mean(Y, dims=2))
end

function l1_loss(y_pred, y_true)
    return mean(abs.(y_pred .- y_true))
end

function evaluate_model(model, X, Y, loss_fn)
    H = get_features(model, X)
    Y_pred = model.W_out * H .+ model.b_out
    return loss_fn(Y_pred, Y)
end

end # module
