import torch
import torch.onnx
import matplotlib.pyplot as plt
from train_common_EL import get_device, setup_problem, train_and_eval

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
inputfile      = 'input_tensor.csv'
outputfile     = 'output_tensor.csv'
dataname       = 'JX_NN'
num_epochs     = 1000
ln_rate        = 3e-4
b_size         = 256          # larger batches → better CPU BLAS utilisation

modelname      = dataname + '_model.pth'
best_modelname = dataname + '_best_model.pth'
training_error = dataname + '_error.mat'
onnx_name      = dataname + '_model.onnx'

early_stop_patience = 500

# ─────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────
device = get_device()
dataloader_train, dataloader_test, model, criterion = \
    setup_problem(inputfile, outputfile, b_size, device)

def update_plot(epoch_idx, train_hist, test_hist):
    if epoch_idx % 10 == 0:
        epochs_range = list(range(len(train_hist)))
        line_train.set_data(epochs_range, train_hist)
        line_test.set_data(epochs_range,  test_hist)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

# ── Weight initialisation (Kaiming for ReLU networks) ───────────────────────
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        torch.nn.init.zeros_(m.bias)

model.apply(init_weights)

# ── Optimiser + LR scheduler ─────────────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=ln_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode      = 'min',
    factor    = 0.5,
    patience  = 100,
    min_lr    = 1e-7,
    threshold = 1e-6,
)

# ─────────────────────────────────────────────
# Real-time plot setup
# ─────────────────────────────────────────────
plt.ion()
fig, ax = plt.subplots(figsize=(10, 6))
train_loss_history = []
test_loss_history  = []
line_train, = ax.plot([], [], 'b-', label='Train Loss', linewidth=1.5)
line_test,  = ax.plot([], [], 'r-', label='Test Loss',  linewidth=1.5)
ax.set_yscale('log')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss (Log Scale)')
ax.set_title(f'Real-time Training Monitor: {dataname}')
ax.legend()
ax.grid(True, which="both", ls="-", alpha=0.3)

# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
print(f"Starting training on {device} for up to {num_epochs} epochs...")

model, optimizer, stopped_epoch, train_loss_history, test_loss_history = \
    train_and_eval(
        model                = model,
        optimizer            = optimizer,
        dataloader_train     = dataloader_train,
        dataloader_test      = dataloader_test,
        criterion            = criterion,
        num_epochs           = num_epochs,
        modelname            = modelname,
        best_modelname       = best_modelname,
        training_error       = training_error,
        scheduler            = scheduler,
        early_stop_patience  = early_stop_patience,
        grad_clip_norm       = 1.0,
        plot_callback        = update_plot,
    )

# ─────────────────────────────────────────────
# Final plot update
# ─────────────────────────────────────────────
epochs_range = list(range(len(train_loss_history)))
line_train.set_data(epochs_range, train_loss_history)
line_test.set_data(epochs_range, test_loss_history)
ax.relim()
ax.autoscale_view()
fig.canvas.draw()
fig.canvas.flush_events()

# ─────────────────────────────────────────────
# ONNX export — always from CPU + best weights
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
# Finalise plot
# ─────────────────────────────────────────────
plt.ioff()
plt.show()
print("Done.")
