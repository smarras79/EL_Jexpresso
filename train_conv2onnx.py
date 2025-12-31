import torch
from IO_EL import csv2pyt_fc
from NN_EL import FCNN
from SLmodel_EL import load_checkpoint

# Adjust this as needed
inputfile = 'input_tensor.csv'
outputfile = 'output_tensor.csv'
dataname = 'JX_NN'

modelname = dataname + '_model.pth'
onnxname = dataname + '_model.onnx'

device = 'cpu'
print(f"Device on: {device}")

dataloader_train, dataloader_test, N_samp, N_in, N_out = \
    csv2pyt_fc(inputfile, outputfile, device, b_size=100, test_split=0.2)
model = FCNN(input_size=N_in, output_size=N_out).to(device)

# Set the loss function and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Adam optimizer

model, optimizer, start_epoch, train_loss_history, test_loss_history = load_checkpoint(model, optimizer, modelname)
model.eval()

# You'll need an example input to trace the operations
example_input = torch.randn(1, N_in, device=device)

# Export the model
torch.onnx.export(model,               # model being run
                  example_input,       # model input (or a tuple for multiple inputs)
                  onnxname,        # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=12,    # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})
