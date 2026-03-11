import torch
import torch.onnx
import matplotlib.pyplot as plt
from scipy.io import savemat
from train_common_EL import get_device, setup_problem

# --- Configuration ---
inputfile      = 'input_tensor.csv'
outputfile     = 'output_tensor.csv'
dataname       = 'JX_NN'
num_epochs     = 300
b_size         = 50
ln_rate        = 1e-5
modelname      = dataname + '_model.pth'
training_error = dataname + '_error.mat'
onnx_name      = dataname + '_model.onnx'

# --- Setup ---
device = get_device()
dataloader_train, dataloader_test, model, criterion = \
    setup_problem(inputfile, outputfile, b_size, device)
optimizer = torch.optim.Adam(model.parameters(), lr=ln_rate)

# --- Real-Time Plotting ---
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

# --- Training Loop ---
print(f"Starting training on {device} for {num_epochs} epochs...")
for epoch in range(num_epochs):
    # ---- Train ----
    model.train()
    running_train_loss = 0.0
    for inputs, targets in dataloader_train:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
    avg_train_loss = running_train_loss / len(dataloader_train)
    train_loss_history.append(avg_train_loss)

    # ---- Validate ----
    model.eval()
    running_test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader_test:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            running_test_loss += criterion(outputs, targets).item()
    avg_test_loss = running_test_loss / len(dataloader_test)
    test_loss_history.append(avg_test_loss)

    # ---- Live Plot Update (every 10 epochs) ----
    if epoch % 10 == 0 or epoch == num_epochs - 1:
        epochs_range = list(range(len(train_loss_history)))
        line_train.set_data(epochs_range, train_loss_history)
        line_test.set_data(epochs_range, test_loss_history)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

    # ---- Console Output (every 10 epochs) ----
    if epoch % 10 == 0 or epoch == num_epochs - 1:
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {avg_train_loss:.10f} | "
              f"Test Loss:  {avg_test_loss:.10f}")

# --- Save 1: Full checkpoint (.pth) ---
checkpoint = {
    'model_state_dict':     model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch':                num_epochs,
    'train_loss_history':   train_loss_history,
    'test_loss_history':    test_loss_history,
}
torch.save(checkpoint, modelname)
print(f"Saved checkpoint : {modelname}")

# --- Save 2: Loss histories (.mat) ---
savemat(training_error, {
    'train_err': train_loss_history,
    'test_err':  test_loss_history,
})
print(f"Saved loss history: {training_error}")

# --- Save 3: ONNX export (.onnx  +  .onnx.data) ---
model.eval()
dummy_input = next(iter(dataloader_train))[0][:1].to(device)
torch.onnx.export(
    model,
    dummy_input,
    onnx_name,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input':  {0: 'batch_size'},
        'output': {0: 'batch_size'},
    }
)
print(f"Saved ONNX model  : {onnx_name}  (+ {onnx_name}.data if weights are large)")

# --- Finalize Plot ---
plt.ioff()
plt.show()
print("Done.")
