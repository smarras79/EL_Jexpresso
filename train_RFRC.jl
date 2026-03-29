using Pkg
if !isfile("Project.toml")
    @info "No Project.toml found. Creating one with required dependencies..."
    Pkg.activate(".")
    Pkg.add(["LinearAlgebra", "Random", "Statistics", "Printf", 
             "CSV", "DataFrames", "MAT", "JLD2", "Plots"])
end

Pkg.activate(".")   # Points Julia to the current folder's environment
Pkg.instantiate()  # Downloads missing pkgs and PRECOMPILES everything

# ─────────────────────────────────────────────────────────────────────────────
# train_rfrc.jl  —  Recurrence-Free Reservoir Computing (RF-RC) trainer
# ─────────────────────────────────────────────────────────────────────────────
#
# Phase 1 – ONE-SHOT  : closed-form ridge regression — no epochs, no backprop
# Phase 2 – FINE-TUNE : Adam on readout only, warm-started from Phase 1
#            NOTE: reservoir features H are precomputed once (frozen reservoir),
#            making each epoch O(D × N) matrix-vector ops — faster than Python.
#
# Output files:
#   JX_RFRC_model.jld2       — last-epoch checkpoint
#   JX_RFRC_best_model.jld2  — best-test-loss checkpoint
#   JX_RFRC_error.mat        — loss histories (compatible with MATLAB/scipy)
#   JX_RFRC_loss.png         — loss plot
#
# ONNX note: Julia has no mature ONNX export path.  To produce an ONNX file
# for use with the existing ONNXRunTime-based inference, run the Python
# train_rfrc.py instead, or load the .jld2 weights and use the Julia RFRC
# struct directly by swapping elementLearning_infer! to call model(x).
# ─────────────────────────────────────────────────────────────────────────────

using LinearAlgebra, Random, Statistics, Printf
using CSV, DataFrames, MAT, JLD2, Plots

include("NN_RFRC.jl")
using .NNRFRC

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
inputfile  = "input_tensor.csv"
outputfile = "output_tensor.csv"
dataname   = "JX_RFRC"

# Reservoir hyper-parameters
reservoir_dim       = 2048      # width of each reservoir layer
num_layers          = 3         # depth of stacked reservoir
activation          = "tanh"    # "tanh" | "relu" | "sigmoid" | "elu"
input_scaling       = 1.0       # half-range of uniform weight init
ridge_alpha         = 1e-4      # L2 regularisation for ridge fit

# Optional fine-tuning (set finetune_epochs = 0 to disable)
finetune_epochs     = 200
finetune_lr         = 1e-3
grad_clip_norm      = 0.5
early_stop_patience = 50

T          = Float64
test_split = 0.2

modelname      = dataname * "_model.jld2"
best_modelname = dataname * "_best_model.jld2"
training_error = dataname * "_error.mat"

# ─────────────────────────────────────────────
# Data  (mirrors IO_EL.py / csv2pyt_fc)
# CSV layout: N_features rows × N_samples columns, one header row.
# Julia convention: matrices are (features × samples).
# ─────────────────────────────────────────────
function load_and_split(input_file, output_file, T, test_split; seed = 42)
    X = Matrix{T}(CSV.read(input_file,  DataFrame))   # (N_in  × N_samp)
    Y = Matrix{T}(CSV.read(output_file, DataFrame))   # (N_out × N_samp)
    N_samp = size(X, 2)
    idx    = randperm(MersenneTwister(seed), N_samp)
    n_test = floor(Int, N_samp * test_split)
    vi     = idx[1:n_test]
    ti     = idx[n_test+1:end]
    return X[:, ti], Y[:, ti], X[:, vi], Y[:, vi],
           size(X, 1), size(Y, 1), N_samp
end

X_train, Y_train, X_test, Y_test, N_in, N_out, N_samp =
    load_and_split(inputfile, outputfile, T, test_split)

println("\nDataset  : $N_samp samples | $N_in inputs | $N_out outputs")

# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
model = RFRC(T, N_in, N_out;
             reservoir_dim = reservoir_dim,
             num_layers    = num_layers,
             activation    = activation,
             input_scaling = input_scaling,
             ridge_alpha   = ridge_alpha)

n_reservoir = sum(prod(size(l.W)) for l in model.reservoir)
n_trainable = N_out * reservoir_dim + N_out
println("Model    : $(n_reservoir + n_trainable) total params | " *
        "$n_trainable trainable (readout only)")
println("Reservoir: dim=$reservoir_dim  layers=$num_layers  " *
        "activation=$activation  alpha=$ridge_alpha\n")

criterion = l1_loss

# ─────────────────────────────────────────────
# Phase 1 — One-shot ridge regression
# ─────────────────────────────────────────────
println("=" ^ 60)
println("Phase 1  — One-shot ridge regression fit")
println("=" ^ 60)

t0   = time_ns()
info = fit_readout!(model, X_train, Y_train)
t1   = time_ns()

train_loss_ridge = evaluate_model(model, X_train, Y_train, criterion)
test_loss_ridge  = evaluate_model(model, X_test,  Y_test,  criterion)

println("  H matrix : $(info.H_shape)  |  Y matrix : $(info.Y_shape)")
println("  W_out    : $(info.W_shape)  |  alpha=$(info.ridge_alpha)")
@printf("  Fit time : %.3f s\n",   (t1 - t0) / 1e9)
@printf("  Train L1 : %.10f\n",    train_loss_ridge)
@printf("  Test  L1 : %.10f\n\n",  test_loss_ridge)

train_loss_history = Float64[]
test_loss_history  = Float64[]
best_test_loss     = test_loss_ridge

# Save ridge solution as initial best checkpoint
@save best_modelname model best_test_loss train_loss_history test_loss_history

# ─────────────────────────────────────────────
# Phase 2 — Optional Adam fine-tuning on readout
# ─────────────────────────────────────────────
if finetune_epochs > 0
    println("=" ^ 60)
    println("Phase 2  — Fine-tuning readout for up to $finetune_epochs epochs")
    println("=" ^ 60)

    # Precompute reservoir features once — they are frozen, no need to recompute
    H_train = get_features(model, X_train)   # (D × N_train)
    H_test  = get_features(model, X_test)    # (D × N_test)

    # Adam state
    β1, β2, ε = T(0.9), T(0.999), T(1e-8)
    m_W = zeros(T, size(model.W_out));  v_W = zeros(T, size(model.W_out))
    m_b = zeros(T, N_out);              v_b = zeros(T, N_out)
    lr  = T(finetune_lr)

    lr_patience = 20;  lr_counter = 0
    patience_counter = 0

    for epoch = 1:finetune_epochs

        # ── L1 gradient w.r.t. W_out and b_out (analytic, no AD needed) ──────
        # ŷ = W_out * H_train + b_out,  loss = mean(|ŷ - Y|)
        # ∂loss/∂ŷ = sign(ŷ - Y) / N
        ŷ  = model.W_out * H_train .+ model.b_out   # (out × N_train)
        N  = T(size(Y_train, 2))
        δ  = sign.(ŷ .- Y_train) ./ N               # (out × N_train)
        ∇W = δ * transpose(H_train)                  # (out × D)
        ∇b = vec(sum(δ, dims = 2))                   # (out,)

        # ── Gradient clipping ─────────────────────────────────────────────────
        gnorm = sqrt(sum(abs2, ∇W) + sum(abs2, ∇b))
        if gnorm > grad_clip_norm
            s = T(grad_clip_norm) / gnorm;  ∇W .*= s;  ∇b .*= s
        end

        # ── Adam update ───────────────────────────────────────────────────────
        m_W .= β1 .* m_W .+ (1 - β1) .* ∇W
        v_W .= β2 .* v_W .+ (1 - β2) .* ∇W .^ 2
        m_b .= β1 .* m_b .+ (1 - β1) .* ∇b
        v_b .= β2 .* v_b .+ (1 - β2) .* ∇b .^ 2
        β1e  = β1 ^ epoch;  β2e = β2 ^ epoch
        model.W_out .-= lr .* (m_W ./ (1 - β1e)) ./ (sqrt.(v_W ./ (1 - β2e)) .+ ε)
        model.b_out .-= lr .* (m_b ./ (1 - β1e)) ./ (sqrt.(v_b ./ (1 - β2e)) .+ ε)

        # ── Losses (reuse H_train / H_test — no reservoir recompute) ──────────
        avg_train = mean(abs.(model.W_out * H_train .+ model.b_out .- Y_train))
        avg_test  = mean(abs.(model.W_out * H_test  .+ model.b_out .- Y_test))
        push!(train_loss_history, avg_train)
        push!(test_loss_history,  avg_test)

        # ── LR schedule (halve on plateau) ────────────────────────────────────
        lr_counter += 1
        if avg_test >= best_test_loss && lr_counter >= lr_patience
            lr = max(lr * T(0.5), T(1e-8));  lr_counter = 0
        end

        # ── Best checkpoint ───────────────────────────────────────────────────
        if avg_test < best_test_loss
            best_test_loss = avg_test;  patience_counter = 0
            @save best_modelname model best_test_loss train_loss_history test_loss_history
        else
            patience_counter += 1
        end

        @printf("  Epoch [%4d/%d] | Train: %.10f | Test: %.10f | " *
                "LR: %.2e | Patience: %d/%d\n",
                epoch, finetune_epochs, avg_train, avg_test,
                lr, patience_counter, early_stop_patience)

        # ── Early stopping ────────────────────────────────────────────────────
        if patience_counter >= early_stop_patience
            println("\n  Early stopping at epoch $epoch " *
                    "(no improvement for $early_stop_patience epochs).")
            break
        end
    end
    @printf("\n  Best test loss after fine-tuning: %.10f\n", best_test_loss)

else
    println("Fine-tuning disabled (finetune_epochs=0).")
end

# ─────────────────────────────────────────────
# Save last checkpoint + loss history
# ─────────────────────────────────────────────
@save modelname model best_test_loss train_loss_history test_loss_history
println("\nSaved last checkpoint : $modelname")
println("Saved best checkpoint : $best_modelname")

matwrite(training_error, Dict(
    "train_err"        => collect(train_loss_history),
    "test_err"         => collect(test_loss_history),
    "ridge_train_loss" => [train_loss_ridge],
    "ridge_test_loss"  => [test_loss_ridge],
))

# ─────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────
if !isempty(train_loss_history)
    p = plot(train_loss_history;
             label     = "Train Loss (fine-tune)",
             color     = :blue, linewidth = 1.5,
             yscale    = :log10,
             xlabel    = "Fine-tune Epochs",
             ylabel    = "Loss (Log Scale)",
             title     = "RF-RC Training Monitor: $dataname",
             legend    = :topright, grid = true, size = (800, 500))
    plot!(p, test_loss_history; label = "Test Loss (fine-tune)",
          color = :red, linewidth = 1.5)
    annotate!(p, [(0.98, maximum(train_loss_history),
        text("reservoir_dim=$(reservoir_dim)\nnum_layers=$(num_layers)\n" *
             "activation=$(activation)\nridge_alpha=$(ridge_alpha)\n" *
             "finetune_epochs=$(finetune_epochs)\n" *
             @sprintf("ridge train L1 = %.6f\nridge test  L1 = %.6f",
                      train_loss_ridge, test_loss_ridge),
        7, :right))])
    savefig(p, dataname * "_loss.png")
    display(p)
end

println("Done.")
