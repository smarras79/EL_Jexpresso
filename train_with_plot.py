import torch
import matplotlib.pyplot as plt
from train_common_EL import get_device, setup_problem

# --- Configuration ---
inputfile = 'input_tensor.csv'
outputfile = 'output_tensor.csv'
dataname = 'JX_NN'

num_epochs = 6000  
b_size     = 50    
ln_rate    = 2e-5  

modelname      = dataname + '_model.pth'
training_error = dataname + '_error.mat'

device = get_device()

# Setup problem
dataloader_train, dataloader_test, model, criterion = \
    setup_problem(inputfile, outputfile, b_size, device)

optimizer = torch.optim.Adam(model.parameters(), lr=ln_rate)

# --- Initialize Real-Time Plotting ---
plt.ion()  # Interactive mode ON
fig, ax = plt.subplots(figsize=(10, 6))

train_loss_history = []
test_loss_history = []

# Initialize lines with a single point to avoid the "empty" -0.04 range issue
line_train, = ax.plot([], [], 'b-', label='Train Loss', linewidth=1.5)
line_test, = ax.plot([], [], 'r-', label='Test Loss', linewidth=1.5)

ax.set_yscale('log') # Essential for regression loss
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss (Log Scale)')
ax.set_title(f'Real-time Training Monitor: {dataname}')
ax.legend()
ax.grid(True, which="both", ls="-", alpha=0.3)

# --- Training Loop ---
print(f"Starting training on {device}...")

for epoch in range(num_epochs):
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

    # Validation Phase
    model.eval()
    running_test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader_test:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            v_loss = criterion(outputs, targets)
            running_test_loss += v_loss.item()
            
    avg_test_loss = running_test_loss / len(dataloader_test)
    test_loss_history.append(avg_test_loss)

    # --- Robust Plot Update ---
    # Update every 10 epochs to keep the UI responsive
    if epoch % 10 == 0 or epoch == num_epochs - 1:
        epochs_range = list(range(len(train_loss_history)))
        
        # Update line data
        line_train.set_data(epochs_range, train_loss_history)
        line_test.set_data(epochs_range, test_loss_history)
        
        # Force the axes to expand to fit the new data
        ax.relim()
        ax.autoscale_view()
        
        # Force redraw and handle GUI events
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001) 
        
    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{num_epochs}] | Train Loss: {avg_train_loss:.6e} | Test Loss: {avg_test_loss:.6e}")

# --- Finalize ---
torch.save(model.state_dict(), modelname)
print("Training Complete.")

plt.ioff() # Turn off interactive mode so the window stays open
plt.show()
