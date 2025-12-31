# train_common_EL.py
import torch
from NN_EL import FCNN
from IO_EL import csv2pyt_fc
from SLmodel_EL import save_checkpoint
from scipy.io import savemat

def get_device():
    """Select device and print it."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Training on: {device}")
    return device


def setup_problem(inputfile, outputfile, b_size, device):
    """
    Create dataloaders, model, lam_tv, loss function.
    Returns:
        dataloader_train, dataloader_test, model, lam_tv, criterion
    """
    dataloader_train, dataloader_test, N_samp, N_in, N_out = \
        csv2pyt_fc(inputfile, outputfile, device, b_size, test_split=0.2)

    model = FCNN(input_size=N_in, output_size=N_out).to(device)
    criterion = torch.nn.L1Loss()

    return dataloader_train, dataloader_test, model, criterion


def train_and_eval(
    model,
    optimizer,
    dataloader_train,
    dataloader_test,
    criterion,
    num_epochs,
    start_epoch=0,
    train_loss_history=None,
    test_loss_history=None,
    modelname=None,
    training_error=None,
):
    """
    Core training + testing loop.
    Optionally continues histories and saves checkpoint / .mat file.
    """
    if train_loss_history is None:
        train_loss_history = []
    if test_loss_history is None:
        test_loss_history = []

    total_epochs = start_epoch + num_epochs
    epoch_idx = start_epoch

    for _ in range(num_epochs):
        epoch_idx += 1
        epoch_train_loss = 0.0
        epoch_test_loss = 0.0

        # ---- Train ----
        model.train()
        for inputs, targets in dataloader_train:
            outputs = model(inputs)            

            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(dataloader_train)
        train_loss_history.append(avg_train_loss)

        # ---- Test ----
        model.eval()
        with torch.no_grad():
            for inputs, targets in dataloader_test:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                epoch_test_loss += loss.item()

        avg_test_loss = epoch_test_loss / len(dataloader_test)
        test_loss_history.append(avg_test_loss)

        print(
            f"Epoch [{epoch_idx}/{total_epochs}], "
            f"Train Loss: {avg_train_loss:.10f}, Test Loss: {avg_test_loss:.10f}"
        )

    # Save checkpoint and training errors if requested
    if modelname is not None:
        save_checkpoint(model, optimizer, epoch_idx,
                        train_loss_history, test_loss_history, modelname)

    if training_error is not None:
        savemat(training_error, {
            'train_err': train_loss_history,
            'test_err': test_loss_history
        })

    return model, optimizer, epoch_idx, train_loss_history, test_loss_history
