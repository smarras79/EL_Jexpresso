import torch
import torch.nn as nn


class ReservoirLayer(nn.Module):
    """
    Single recurrence-free reservoir layer.

    Applies a fixed (non-trainable) random affine map followed by a
    nonlinear activation.  Weights are drawn once at construction time
    and are never updated during training.
    """

    ACTIVATIONS = {
        'tanh':    nn.Tanh(),
        'relu':    nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'elu':     nn.ELU(),
    }

    def __init__(self, in_dim: int, out_dim: int,
                 activation: str = 'tanh',
                 input_scaling: float = 1.0):
        super().__init__()

        if activation not in self.ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Choose from {list(self.ACTIVATIONS.keys())}."
            )

        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        nn.init.uniform_(self.linear.weight, -input_scaling, input_scaling)
        nn.init.zeros_(self.linear.bias)

        # Freeze — reservoir weights are never updated
        for p in self.linear.parameters():
            p.requires_grad_(False)

        self.act = self.ACTIVATIONS[activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))


class RFRC(nn.Module):
    """
    Recurrence-Free Reservoir Computer (RF-RC).

    Architecture
    ------------
    Input (N_in)
      → [Fixed Random Reservoir Layer] × num_layers    (no gradients)
      → Trainable Linear Readout  (output layer)

    Training strategy
    -----------------
    Only the readout layer has trainable parameters.  Training is
    performed in **one shot** via closed-form ridge regression:

        W_out = (Hᵀ H + α I)⁻¹ Hᵀ Y

    where H is the matrix of reservoir states collected over the full
    training set.  No backpropagation or epoch loop is required for the
    primary fit.

    An optional gradient-descent fine-tuning phase (``fit_finetune``)
    can be run afterwards to squeeze out extra accuracy.

    Parameters
    ----------
    input_size   : number of input features
    output_size  : number of output targets
    reservoir_dim: width of each reservoir layer
    num_layers   : depth of the stacked reservoir
    activation   : nonlinearity used in every reservoir layer
    input_scaling: half-range of the uniform weight initialiser
    ridge_alpha  : L2 regularisation coefficient for ridge regression

      ridge_alpha:   float = 1e-4 or 1e-3, or 1e-2. A too-small value lets the regression overfit to near-zero singular directions.
    """

    def __init__(self,
                 input_size:    int,
                 output_size:   int,
                 reservoir_dim: int   = 1024,
                 num_layers:    int   = 3,
                 activation:    str   = 'tanh',
                 input_scaling: float = 1.0,
                 ridge_alpha:   float = 1e-4):
        super().__init__()

        self.ridge_alpha = ridge_alpha

        # ── Stacked reservoir (all weights frozen) ────────────────────────────
        layers = []
        prev_dim = input_size
        for _ in range(num_layers):
            layers.append(
                ReservoirLayer(prev_dim, reservoir_dim, activation, input_scaling)
            )
            prev_dim = reservoir_dim
        self.reservoir = nn.Sequential(*layers)

        # ── Trainable readout ─────────────────────────────────────────────────
        self.readout = nn.Linear(reservoir_dim, output_size, bias=True)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    # ── Forward pass ─────────────────────────────────────────────────────────
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Map input through the frozen reservoir."""
        return self.reservoir(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.get_features(x))

    # ── One-shot ridge regression fit ────────────────────────────────────────
    @torch.no_grad()
    def fit_readout(self, dataloader, device: torch.device) -> dict:
        """
        Closed-form ridge regression over the full training set.

        Collects all reservoir states H and targets Y, then solves:
            W_out = (Hᵀ H + α I)⁻¹ Hᵀ Y

        Returns a dict with diagnostic information.
        """
        self.eval()

        H_list, Y_list = [], []
        for inputs, targets in dataloader:
            H_list.append(self.get_features(inputs))
            Y_list.append(targets)

        H = torch.cat(H_list, dim=0)   # (N_train, reservoir_dim)
        Y = torch.cat(Y_list, dim=0)   # (N_train, output_size)

        # Ridge regression via the augmented least-squares system:
        #   [H ; sqrt(α)·I] @ W  =  [Y ; 0]
        # This is algebraically equivalent to (HᵀH + αI)⁻¹HᵀY but never
        # requires inverting a potentially singular matrix.
        sqrt_alpha = self.ridge_alpha ** 0.5
        D = H.shape[1]
        H_aug = torch.cat([H,
                           sqrt_alpha * torch.eye(D, device=device)], dim=0)
        Y_aug = torch.cat([Y,
                           torch.zeros(D, Y.shape[1], device=device)], dim=0)
        result = torch.linalg.lstsq(H_aug, Y_aug, driver='gelsd')
        W      = result.solution                                      # (D, out)
        
        self.readout.weight.data.copy_(W.T)
        self.readout.bias.data.zero_()

        info = dict(H_shape=tuple(H.shape), Y_shape=tuple(Y.shape),
                    W_shape=tuple(W.shape), ridge_alpha=self.ridge_alpha)
        return info

    # ── Evaluation helper ─────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(self, dataloader, criterion) -> float:
        """Return average loss over the dataloader."""
        self.eval()
        total, n = 0.0, 0
        for inputs, targets in dataloader:
            total += criterion(self(inputs), targets).item()
            n     += 1
        return total / n if n > 0 else float('nan')
