import torch
from train_common_EL import get_device, setup_problem, train_and_eval

# Adjust this as needed
inputfile = 'input_tensor.csv'
outputfile = 'output_tensor.csv'
dataname = 'JX_NN'

num_epochs = 2000   # was 1000
b_size     = 50      # smaller batches often help for regression
ln_rate    = 2e-5    # reduce after initial training


modelname      = dataname + '_model.pth'
training_error = dataname + '_error.mat'

device = get_device()

dataloader_train, dataloader_test, model, criterion = \
    setup_problem(inputfile, outputfile, b_size, device)

optimizer = torch.optim.Adam(model.parameters(), lr=ln_rate)

model, optimizer, last_epoch, train_loss_history, test_loss_history = train_and_eval(
    model=model,
    optimizer=optimizer,
    dataloader_train=dataloader_train,
    dataloader_test=dataloader_test,
    criterion=criterion,
    num_epochs=num_epochs,
    start_epoch=0,
    train_loss_history=[],
    test_loss_history=[],
    modelname=modelname,
    training_error=training_error,
)
