"""
train_rfrc.py  —  Recurrence-Free Reservoir Computing (RF-RC) trainer
======================================================================

RF-RC replaces the iterative backpropagation loop of a classical NN with:

  Phase 1 – ONE-SHOT  : closed-form ridge regression on the frozen reservoir
                        states.  Solves  W = (HᵀH + αI)⁻¹ HᵀY  analytically.
                        No epochs, no gradient computation — extremely fast.

  Phase 2 – FINE-TUNE : optional short gradient-descent pass on the readout
                        layer only, using the Phase-1 solution as a warm start.

The ONNX export at the end is identical to train_gpu.py so the saved model is
a drop-in replacement.
"""

import time
import torch
import torch.onnx
import matplotlib.pyplot as plt
from scipy.io import savemat

from IO_EL      import csv2pyt_fc
from NN_RFRC    import RFRC
from SLmodel_EL import save_checkpoint
from train_common_EL import get_device

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
inputfile  = 'input_tensor.csv'
outputfile = 'output_tensor.csv'
dataname   = 'JX_RFRC'

# ── Reservoir hyper-parameters ────────────────
reservoir_dim  = 2048   # width of each reservoir layer
num_layers     = 3      # depth of stacked reservoir
activation     = 'tanh' # 'tanh' | 'relu' | 'sigmoid' | 'elu'
input_scaling  = 1.0    # half-range of the uniform weight initialiser
ridge_alpha    = 1e-4   # L2 regularisation for the ridge regression fit

# ── Optional fine-tuning (set to 0 to disable) ────────────────────────────
finetune_epochs        = 200     # gradient-descent epochs on the readout only
finetune_lr            = 1e-3
finetune_b_size        = 256
grad_clip_norm         = 0.5
early_stop_patience    = 50

# ── Data / output files ───────────────────────
b_size         = 256
modelname      = dataname + '_model.pth'
best_modelname = dataname + '_best_model.pth'
training_error = dataname + '_error.mat'
onnx_name      = dataname + '_model.onnx'

# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────
device = get_device()

# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────
dataloader_train, dataloader_test, N_samp, N_in, N_out = \
    csv2pyt_fc(inputfile, outputfile, device, b_size, test_split=0.2)

print(f"\nDataset  : {N_samp} samples  |  {N_in} inputs  |  {N_out} outputs")

# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
model = RFRC(
    input_size    = N_in,
    output_size   = N_out,
    reservoir_dim = reservoir_dim,
    num_layers    = num_layers,
    activation    = activation,
    input_scaling = input_scaling,
    ridge_alpha   = ridge_alpha,
).to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Model    : {total:,} total params  |  {trainable:,} trainable (readout only)")
print(f"Reservoir: dim={reservoir_dim}  layers={num_layers}  "
      f"activation={activation}  alpha={ridge_alpha}\n")

criterion = torch.nn.L1Loss()

# ─────────────────────────────────────────────
# Real-time plot setup
# ─────────────────────────────────────────────
plt.ion()
fig, ax = plt.subplots(figsize=(10, 6))
line_train, = ax.plot([], [], 'b-', label='Train Loss (fine-tune)', linewidth=1.5)
line_test,  = ax.plot([], [], 'r-', label='Test Loss  (fine-tune)', linewidth=1.5)
ax.set_yscale('log')
ax.set_xlabel('Fine-tune Epochs')
ax.set_ylabel('Loss (Log Scale)')
ax.set_title(f'RF-RC Training Monitor: {dataname}')
ax.legend()
ax.grid(True, which="both", ls="-", alpha=0.3)
plt.tight_layout()

# ─────────────────────────────────────────────
# Phase 1 — One-shot ridge regression
# ─────────────────────────────────────────────
print("=" * 60)
print("Phase 1  — One-shot ridge regression fit")
print("=" * 60)

t0   = time.perf_counter()
info = model.fit_readout(dataloader_train, device)
t1   = time.perf_counter()

train_loss_ridge = model.evaluate(dataloader_train, criterion)
test_loss_ridge  = model.evaluate(dataloader_test,  criterion)

print(f"  H matrix : {info['H_shape']}  |  Y matrix : {info['Y_shape']}")
print(f"  W_out    : {info['W_shape']}  |  alpha={info['ridge_alpha']}")
print(f"  Fit time : {t1 - t0:.3f} s")
print(f"  Train L1 : {train_loss_ridge:.10f}")
print(f"  Test  L1 : {test_loss_ridge:.10f}\n")

train_loss_history = []
test_loss_history  = []

# ─────────────────────────────────────────────
# Phase 2 — Optional gradient-descent fine-tune
# ─────────────────────────────────────────────
if finetune_epochs > 0:
    print("=" * 60)
    print(f"Phase 2  — Fine-tuning readout for up to {finetune_epochs} epochs")
    print("=" * 60)

    # Only optimise the readout layer
    optimizer = torch.optim.Adam(model.readout.parameters(), lr=finetune_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-8, threshold=1e-6
    )

    best_test_loss   = test_loss_ridge
    patience_counter = 0

    # Save the ridge solution as the initial best checkpoint
    torch.save({
        'epoch':                0,
        'model_state_dict':     {k: v.cpu() for k, v in model.state_dict().items()},
        'optimizer_state_dict': optimizer.state_dict(),
        'best_test_loss':       best_test_loss,
    }, best_modelname)

    for epoch in range(1, finetune_epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        epoch_train_loss = 0.0
        for inputs, targets in dataloader_train:
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.readout.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            epoch_train_loss += loss.item()
        avg_train = epoch_train_loss / len(dataloader_train)

        # ── Evaluate ───────────────────────────────────────────────────────
        avg_test = model.evaluate(dataloader_test, criterion)

        train_loss_history.append(avg_train)
        test_loss_history.append(avg_test)

        scheduler.step(avg_test)
        current_lr = optimizer.param_groups[0]['lr']

        # ── Best checkpoint ────────────────────────────────────────────────
        if avg_test < best_test_loss:
            best_test_loss   = avg_test
            patience_counter = 0
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     {k: v.cpu()
                                         for k, v in model.state_dict().items()},
                'optimizer_state_dict': optimizer.state_dict(),
                'best_test_loss':       best_test_loss,
            }, best_modelname)
        else:
            patience_counter += 1

        print(
            f"  Epoch [{epoch:4d}/{finetune_epochs}] | "
            f"Train: {avg_train:.10f} | Test: {avg_test:.10f} | "
            f"LR: {current_lr:.2e} | "
            f"Patience: {patience_counter}/{early_stop_patience}"
        )

        # ── Live plot ──────────────────────────────────────────────────────
        if epoch % 10 == 0:
            epochs_range = list(range(len(train_loss_history)))
            line_train.set_data(epochs_range, train_loss_history)
            line_test.set_data(epochs_range,  test_loss_history)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)

        # ── Early stopping ─────────────────────────────────────────────────
        if patience_counter >= early_stop_patience:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {early_stop_patience} epochs).")
            break

    print(f"\n  Best test loss after fine-tuning: {best_test_loss:.10f}")

else:
    # No fine-tuning — treat the ridge solution as the best model
    optimizer = torch.optim.SGD(model.readout.parameters(), lr=0.0)
    torch.save({
        'epoch':                0,
        'model_state_dict':     {k: v.cpu() for k, v in model.state_dict().items()},
        'optimizer_state_dict': optimizer.state_dict(),
        'best_test_loss':       test_loss_ridge,
    }, best_modelname)
    print("Fine-tuning disabled  (finetune_epochs=0).")

# ─────────────────────────────────────────────
# Save last checkpoint + loss history
# ─────────────────────────────────────────────
save_checkpoint(model, optimizer,
                len(train_loss_history),
                train_loss_history, test_loss_history,
                modelname)
print(f"\nSaved last checkpoint : {modelname}")
print(f"Saved best checkpoint : {best_modelname}")

savemat(training_error, {
    'train_err': train_loss_history,
    'test_err':  test_loss_history,
    'ridge_train_loss': train_loss_ridge,
    'ridge_test_loss':  test_loss_ridge,
})

# ─────────────────────────────────────────────
# Final plot — info box
# ─────────────────────────────────────────────
if train_loss_history:
    epochs_range = list(range(len(train_loss_history)))
    line_train.set_data(epochs_range, train_loss_history)
    line_test.set_data(epochs_range,  test_loss_history)
    ax.relim()
    ax.autoscale_view()

hparam_text = (
    f"reservoir_dim       = {reservoir_dim}\n"
    f"num_layers          = {num_layers}\n"
    f"activation          = {activation}\n"
    f"ridge_alpha         = {ridge_alpha:.1e}\n"
    f"finetune_epochs     = {finetune_epochs}\n"
    f"finetune_lr         = {finetune_lr:.1e}\n"
    f"grad_clip_norm      = {grad_clip_norm}\n"
    f"early_stop_patience = {early_stop_patience}\n"
    f"─────────────────────────────\n"
    f"ridge train L1      = {train_loss_ridge:.6f}\n"
    f"ridge test  L1      = {test_loss_ridge:.6f}"
)
ax.text(
    0.98, 0.97,
    hparam_text,
    transform           = ax.transAxes,
    fontsize            = 8,
    verticalalignment   = 'top',
    horizontalalignment = 'right',
    bbox                = dict(boxstyle='round,pad=0.4',
                               facecolor='white',
                               alpha=0.8,
                               edgecolor='gray'),
    fontfamily          = 'monospace',
)
fig.canvas.draw()
fig.canvas.flush_events()

# ─────────────────────────────────────────────
# ONNX export — from best weights, on CPU
# ─────────────────────────────────────────────
cpu_device = torch.device("cpu")
best_ckpt  = torch.load(best_modelname, map_location=cpu_device)
model_cpu  = model.to(cpu_device)
model_cpu.load_state_dict(best_ckpt['model_state_dict'])
model_cpu.eval()

dummy_input = next(iter(dataloader_train))[0][:1].to(cpu_device)
torch.onnx.export(
    model_cpu,
    dummy_input,
    onnx_name,
    export_params       = True,
    opset_version       = 17,
    do_constant_folding = True,
    input_names         = ['input'],
    output_names        = ['output'],
    dynamic_axes        = {
        'input':  {0: 'batch_size'},
        'output': {0: 'batch_size'},
    }
)
print(f"Saved ONNX model : {onnx_name}  (exported from best checkpoint)")

# ─────────────────────────────────────────────
# Finalise
# ─────────────────────────────────────────────
plt.ioff()
plt.show()
print("Done.")
