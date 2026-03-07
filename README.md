# JX_NN – Julia/Flux Translation of the Python FCNN Pipeline

## File mapping

| Python file         | Julia file            | Notes                                        |
|---------------------|-----------------------|----------------------------------------------|
| `NN_EL.py`          | `NN_EL.jl`            | `build_FCNN(N_in, N_out)` returns `Chain`    |
| `IO_EL.py`          | `IO_EL.jl`            | `csv2jl_fc(...)` → `MLUtils.DataLoader`      |
| `SLmodel_EL.py`     | `SLmodel_EL.jl`       | BSON checkpoint instead of `.pth`            |
| `train_common_EL.py`| `train_common_EL.jl`  | Modern Flux 0.14+ gradient/update API        |
| main script         | `main.jl`             | Identical hyperparameters                    |

## Key PyTorch → Flux equivalents

| PyTorch                       | Flux / Julia                             |
|-------------------------------|------------------------------------------|
| `nn.Sequential(…)`            | `Chain(…)`                               |
| `nn.Linear(in, out)`          | `Dense(in => out)`                       |
| `nn.ReLU()`                   | `relu` (plain function, no layer needed) |
| `torch.optim.Adam(lr=…)`      | `Adam(lr)` + `Flux.setup`               |
| `nn.L1Loss()`                 | `Flux.mae`                               |
| `model.to(device)`            | `model \|> device_fn`                    |
| `model.train()/.eval()`       | `Flux.trainmode!/testmode!(model)`       |
| `loss.backward()` + `step()`  | `Flux.withgradient` + `Flux.update!`     |
| `torch.save` / `.pth`         | `BSON.bson` / `.bson`                    |
| `scipy.io.savemat`            | `MAT.matwrite`                           |

## First-run setup

```julia
# From the jx_nn/ directory:
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Running

```julia
julia --project=. main.jl
```

## Continuing from a checkpoint

```julia
include("SLmodel_EL.jl"); using .SLmodel_EL
include("NN_EL.jl");      using .NN_EL

model = build_FCNN(N_in, N_out)
model, opt_state, start_epoch, train_loss, test_loss =
    load_checkpoint(model, "JX_NN_model.bson")

# Then call train_and_eval with start_epoch and the existing histories
```

## Switching activation / device

- **Activation**: edit the `act = relu` line in `NN_EL.jl` → `tanh`, `sigmoid`, `elu`
- **GPU (CUDA)**: `Pkg.add("CUDA")` then `Flux.gpu` is returned automatically
- **Apple Silicon**: uncomment the Metal block in `train_common_EL.jl`
