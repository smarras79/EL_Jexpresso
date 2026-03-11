import torch
import torch.onnx
import matplotlib.pyplot as plt
from scipy.io import savemat
from train_common_EL import get_device, setup_problem

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
inputfile      = 'input_tensor.csv'
outputfile     = 'output_tensor.csv'
dataname       = 'JX_NN'
num_epochs     = 1000
ln_rate = 3e-4     # slightly lower to avoid spikes and find a better minimum
b_size  = 64       # smaller batches → noisier gradients help escape the plateau

modelname      = dataname + '_model.pth'
best_modelname = dataname + '_best_model.pth'
training_error = dataname + '_error.mat'
onnx_name      = dataname + '_model.onnx'

# ─────────────────────────────────────────────
# Early-stopping settings
# ─────────────────────────────────────────────
early_stop_patience = 500   # stop if val loss doesn't improve for this many epochs

# ─────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────
device = get_device()
dataloader_train, dataloader_test, model, criterion = \
    setup_problem(inputfile, outputfile, b_size, device)

# ── Weight initialisation (Kaiming for ReLU networks) ──────────────────────
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        torch.nn.init.zeros_(m.bias)

model.apply(init_weights)

# ── Optimiser + LR scheduler ────────────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=ln_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode      = 'min',
    factor    = 0.1,        # multiply LR by 0.1 on plateau
    patience  = 200,        # epochs to wait before reducing
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
# Training loop
# ─────────────────────────────────────────────
print(f"Starting training on {device} for up to {num_epochs} epochs "
      f"(early-stop patience = {early_stop_patience})...")

best_test_loss    = float('inf')
patience_counter  = 0
stopped_at_epoch  = num_epochs          # will be updated if early-stopped

for epoch in range(num_epochs):

    # ── Train ──────────────────────────────────────────────────────────────
    model.train()
    running_train_loss = 0.0
    for inputs, targets in dataloader_train:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_train_loss += loss.item()
    avg_train_loss = running_train_loss / len(dataloader_train)
    train_loss_history.append(avg_train_loss)

    # ── Validate ───────────────────────────────────────────────────────────
    model.eval()
    running_test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader_test:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            running_test_loss += criterion(outputs, targets).item()
    avg_test_loss = running_test_loss / len(dataloader_test)
    test_loss_history.append(avg_test_loss)

    # ── LR scheduler step ──────────────────────────────────────────────────
    scheduler.step(avg_test_loss)

    # ── Save best model ────────────────────────────────────────────────────
    if avg_test_loss < best_test_loss:
        best_test_loss   = avg_test_loss
        patience_counter = 0
        torch.save({
            'epoch':                epoch + 1,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_test_loss':       best_test_loss,
        }, best_modelname)
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1} "
                  f"(no improvement for {early_stop_patience} epochs).")
            stopped_at_epoch = epoch + 1
            break

    # ── Live plot update (every 10 epochs) ────────────────────────────────
    if epoch % 10 == 0 or epoch == num_epochs - 1:
        epochs_range = list(range(len(train_loss_history)))
        line_train.set_data(epochs_range, train_loss_history)
        line_test.set_data(epochs_range, test_loss_history)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

    # ── Console output (every 10 epochs) ──────────────────────────────────
    if epoch % 10 == 0 or epoch == num_epochs - 1:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {avg_train_loss:.10f} | "
              f"Test Loss: {avg_test_loss:.10f} | "
              f"LR: {current_lr:.2e} | "
              f"Patience: {patience_counter}/{early_stop_patience}")

# ─────────────────────────────────────────────
# Save 1: Full checkpoint of LAST epoch (.pth)
# ─────────────────────────────────────────────
checkpoint = {
    'model_state_dict':     model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch':                stopped_at_epoch,
    'train_loss_history':   train_loss_history,
    'test_loss_history':    test_loss_history,
}
torch.save(checkpoint, modelname)
print(f"\nSaved last checkpoint : {modelname}")
print(f"Saved best checkpoint : {best_modelname}  (test loss = {best_test_loss:.10f})")

# ─────────────────────────────────────────────
# Save 2: Loss histories (.mat)
# ─────────────────────────────────────────────
savemat(training_error, {
    'train_err': train_loss_history,
    'test_err':  test_loss_history,
})
print(f"Saved loss history    : {training_error}")

# ─────────────────────────────────────────────
# Save 3: ONNX export — load the BEST weights first
# ─────────────────────────────────────────────
best_ckpt = torch.load(best_modelname, map_location=device)
model.load_state_dict(best_ckpt['model_state_dict'])
model.eval()

dummy_input = next(iter(dataloader_train))[0][:1].to(device)
torch.onnx.export(
    model,
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
print(f"Saved ONNX model      : {onnx_name}  (exported from best checkpoint)")

# ─────────────────────────────────────────────
# Finalise plot
# ─────────────────────────────────────────────
plt.ioff()
plt.show()
print("Done.")
