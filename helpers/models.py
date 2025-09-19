
from pathlib import Path
import pickle

import torch
import pandas
import matplotlib.pyplot as plt

from .data import Data_Loader


def select_device():

    """
    Select a device to compute with.

    Returns the name of the selected device.
    "cuda" if cuda is available, otherwise "cpu".
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)
    return device


def print_gpu_memory_info():
    
    """
    Print GPU memory summary.
    """
    
    print(
        "Peak GPU memory usage: ",
        f"{torch.cuda.max_memory_allocated()/1024**3:.2f} GB" 
    )


def print_epoch_loss(epoch, loss_train, loss_val):
    
    print(
        f"\nEpoch {epoch} complete:\n"
        f"    Train loss: {loss_train}\n"
        f"    Val loss: {loss_val}\n"
    )


def print_prev_lr(scheduler):
    
    print(f"Learning rate: {scheduler.get_last_lr()}")


def train_batch(x, y, model, loss_fn, optimizer):
    
    model.train()
    yhat = model(x)    
    train_loss = loss_fn(yhat, y)
    train_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return train_loss
    

def evaluate_batch(x, y, model, loss_fn):
    
    model.eval()
    with torch.no_grad():
        yhat = model(x)
        eval_loss = loss_fn(yhat, y)
        return eval_loss
    

def train_epoch(dataloader, model, loss_fn, optimizer, device):
    
    total_batch_loss = 0
    for x, y in dataloader:
        if device != "cpu":
            x = x.to(device)
            y = y.to(device)
        batch_loss = train_batch(x, y, model, loss_fn, optimizer)
        total_batch_loss += batch_loss

    num_batches = len(dataloader)
    avg_batch_loss = total_batch_loss / num_batches
    return avg_batch_loss


def evaluate_epoch(dataloader, model, loss_fn, device, scheduler=None):
    
    total_batch_loss = 0
    for x, y in dataloader:
        if device != "cpu":
            x = x.to(device)
            y = y.to(device)
        batch_loss = evaluate_batch(x, y, model, loss_fn)
        total_batch_loss += batch_loss
    
    num_batches = len(dataloader)
    avg_batch_loss = total_batch_loss / num_batches
    if scheduler:
        scheduler.step(avg_batch_loss)
    return avg_batch_loss


def train(
    model,
    name, 
    loss_fn,
    dataset_train, 
    dataset_val, 
    device, 
    lr, 
    lr_reduce_factor, 
    lr_reduce_patience, 
    batch_size_train, 
    batch_size_val,
    epochs,
    epochs_checkpoint,
):

    def setup_model_dir(name, parent_dir):

        path = Path(parent_dir).joinpath(name)

        if path.is_dir():

            raise ValueError(f"Model directory exists (delete to continue): {path}")
        
        path.mkdir()

        return path

    def save_model(model, filename, save_dir):

        torch.save(model.state_dict(), Path(save_dir).joinpath(filename)) 

    def save_loss_table(table, save_dir):

        pandas.DataFrame(table).to_parquet(Path(save_dir).joinpath("loss_table.parquet"))

    def plot_loss(table, save_dir): 

        plt.plot(table["epoch"], table["loss_train"], label="train")
        plt.plot(table["epoch"], table["loss_val"], label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(Path(save_dir).joinpath("loss.png"), bbox_inches="tight")
        plt.close()

    save_dir = setup_model_dir(name, "models")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, 
        factor=lr_reduce_factor, 
        patience=lr_reduce_patience, 
        threshold=0,
        eps=0
    )

    dataloader_train = Data_Loader(
        dataset_train, 
        batch_size_train, 
        shuffle=True
    )

    dataloader_val = Data_Loader(
        dataset_val, 
        batch_size_val, 
        shuffle=True
    )

    model = model.to(device)

    loss_table = {
        "epoch" : [], 
        "loss_train": [], 
        "loss_val": []
    } 

    for ep in range(epochs):
        
        loss_train = train_epoch(
            dataloader_train, 
            model, 
            loss_fn, 
            optimizer, 
            device
        ).item()
        
        loss_val = evaluate_epoch(
            dataloader_val, 
            model, 
            loss_fn, 
            device, 
            scheduler=lr_scheduler
        ).item()

        loss_table["epoch"].append(ep)
        loss_table["loss_train"].append(loss_train)
        loss_table["loss_val"].append(loss_val)
        
        print_epoch_loss(ep, loss_train, loss_val)
        print_prev_lr(lr_scheduler)
        print_gpu_memory_info()

        if (ep % epochs_checkpoint) == 0:

            save_model(model, f"epoch_{ep}.pt", save_dir)
            save_loss_table(loss_table, save_dir)
            plot_loss(loss_table, save_dir)
            
    save_model(model, "final.pt", save_dir)
    save_loss_table(loss_table, save_dir)
    plot_loss(loss_table, save_dir)
    
    print("Training completed.")


def open_model_state_dict(model_name, epoch="final"):

    filename = "final.pt" if epoch == "final" else f"epoch_{epoch}.pt"

    path = Path("models").joinpath(model_name).joinpath(filename)

    state_dict = torch.load(path, weights_only=True)

    return state_dict


def predict_values_set_model(model, sets_features, device):
        
    with torch.no_grad():

        preds = torch.tensor(
            [
                model(set_.to(device).unsqueeze(dim=0))
                for set_ in sets_features
            ]
        )

        return preds
    

def predict_log_probs_event_model(model, sets_features, device):

    def calc_log_probs(model, set_, device):

        event_logits = model(set_.to(device))
        event_log_probs = torch.nn.functional.log_softmax(event_logits, dim=1)
        set_logits = torch.sum(event_log_probs, dim=0)
        set_log_probs = torch.nn.functional.log_softmax(set_logits, dim=0)
        return set_log_probs
    
    with torch.no_grad():
    
        return torch.cat(
            [
                calc_log_probs(model, set_, device).unsqueeze(dim=0)
                for set_ in sets_features
            ]
        )
    

def predict_values_event_model(log_probs, bin_map, device):

    def calc_expectation(log_probs, bin_map, device):

        bin_shift = 5
        log_bin_map = torch.log(bin_map.to(device) + bin_shift)
        expectation = torch.exp(torch.logsumexp(log_bin_map + log_probs.to(device), dim=0)) - bin_shift
        return expectation
    
    with torch.no_grad():

        return torch.tensor(
            [
                calc_expectation(log_p, bin_map, device)
                for log_p in log_probs
            ]
        )


def run_linearity_test(preds, labels):

    def sort_by_label(preds, labels):

        sorted_labels, sorted_indices = torch.sort(labels)
        sorted_preds = preds[sorted_indices]
        return sorted_preds, sorted_labels
    
    def count_num_sets_per_label(labels):

        unique_labels = torch.unique(labels)
        counts_per_label = torch.tensor(
            [
                len(labels[labels==unique_label])
                for unique_label in unique_labels
            ]
        )
        assert torch.all(counts_per_label == counts_per_label[0])
        return counts_per_label[0]

    assert len(preds) == len(labels)

    with torch.no_grad():

        sorted_preds, sorted_labels = sort_by_label(preds, labels)
        num_sets_per_label = count_num_sets_per_label(labels)
        sorted_preds = sorted_preds.reshape(-1, num_sets_per_label)
        unique_labels = torch.unique(sorted_labels, sorted=False)
        avgs = sorted_preds.mean(dim=1)
        stds = sorted_preds.std(dim=1)

    unique_labels = unique_labels.cpu().detach().numpy()
    avgs = avgs.cpu().detach().numpy()
    stds = stds.cpu().detach().numpy()

    return unique_labels, avgs, stds
        

def run_sensitivity_test(preds, labels):
    
    def get_label(labels):

        unique_labels = torch.unique(labels)
        if len(unique_labels) > 1:
            raise ValueError("Sensitivity test runs on dataset with one label.")
        label = unique_labels.item()
        return label

    with torch.no_grad():

        label = get_label(labels)
        avg = preds.mean()
        std = preds.std()
        bias = avg - label

    preds = preds.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    avg = avg.cpu().detach().numpy()
    std = std.cpu().detach().numpy()
    bias = bias.cpu().detach().numpy()

    return preds, label, avg, std, bias


def run_error_test(preds, labels):

    with torch.no_grad():

        mse = torch.nn.functional.mse_loss(preds, labels)
        mae = torch.nn.functional.l1_loss(preds, labels)

    mse = mse.cpu().detach().numpy()
    mae = mae.cpu().detach().numpy()

    return mse, mae


def make_test_result_filename(result_name, model_name, num_signal):

    return f"{model_name}_{result_name}_{num_signal}.pkl"


def save_test_result(result, result_name, num_signal, model_name):

    filename = make_test_result_filename(result_name, model_name, num_signal)

    path = Path("results").joinpath(filename)

    with open(path, 'wb') as file:

        pickle.dump(result, file)


def open_test_result(result_name, num_signal, model_name):

    filename = make_test_result_filename(result_name, model_name, num_signal)

    path = Path("results").joinpath(filename)

    with open(path, 'rb') as file:

        result = pickle.load(file)

    return result 


class Results_Table:

    def __init__(self):

        self.table = pandas.DataFrame(
            index=self.make_index(), 
            columns=("MSE", "MAE", "Std at NP", "Mean at NP", "Bias at NP")
        )

    def add_item(self, value, column, level, model_name, signal_per_set):

        if type(value) is torch.Tensor:
        
            value = value.item()
        
        self.table.loc[(level, model_name, signal_per_set), column] = value

    def make_index(self):

        names = (
            "level",
            "model",
            "signal_per_set",
        )

        values = (
            ("gen", "det", "det_bkg"), 
            ("cnn", "deep_sets", "ebe"),
            (8_000, 16_000, 32_000),
        )

        return pandas.MultiIndex.from_product(values, names=names)


class Deep_Sets_Model(torch.nn.Module):

    def __init__(self):

        super().__init__()
        
        self.event_layers = torch.nn.Sequential(
            torch.nn.Linear(4, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32)
        )

        self.set_layers = torch.nn.Sequential(
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x):

        x = self.event_layers(x)
        x = torch.mean(x, dim=1)
        x = self.set_layers(x)
        x = torch.squeeze(x)
        return x


class CNN_Model(torch.nn.Module):

    def __init__(self):

        super().__init__()
        
        self.convolution_layers = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels=1, 
                out_channels=64, 
                kernel_size=7, 
                stride=2, 
                padding=1, 
                bias=False
            ),
            torch.nn.BatchNorm3d(num_features=64),
            torch.nn.MaxPool3d(
                kernel_size=3, 
                stride=2, 
                padding=1
            ),
            torch.nn.ReLU(),
            self.Residual_Block_A(64, 64),
            self.Residual_Block_A(64, 64),
            self.Residual_Block_A(64, 64),
            self.Residual_Block_B(64, 128),
            self.Residual_Block_A(128, 128),
            self.Residual_Block_A(128, 128),
            self.Residual_Block_A(128, 128),
            self.Residual_Block_B(128, 256),
            self.Residual_Block_A(256, 256),
            self.Residual_Block_A(256, 256),
            self.Residual_Block_A(256, 256),
            self.Residual_Block_A(256, 256),
            self.Residual_Block_A(256, 256),
            self.Residual_Block_B(256, 512),
            self.Residual_Block_A(512, 512),
            self.Residual_Block_A(512, 512),
        )

        self.dense_layers = torch.nn.Sequential(
            torch.nn.Linear(512, 1000),
            torch.nn.Dropout(p=0.5), 
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 1),
        )
        
    def forward(self, x):

        x = self.convolution_layers(x)
        x = torch.mean(x, dim=(2,3,4))
        x = self.dense_layers(x)
        x = torch.squeeze(x)
        return x

    class Residual_Block_A(torch.nn.Module):
        
        def __init__(self, in_channels, out_channels):
            
            super().__init__()

            self.convolution_block = torch.nn.Sequential(
                torch.nn.Conv3d(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=3, 
                    stride=1, 
                    padding="same"
                ),
                torch.nn.BatchNorm3d(num_features=out_channels),
                torch.nn.ReLU(),
                torch.nn.Conv3d(
                    in_channels=out_channels, 
                    out_channels=out_channels, 
                    kernel_size=3, 
                    stride=1, 
                    padding="same"
                ),
                torch.nn.BatchNorm3d(num_features=out_channels)
            )
        
        def forward(self, x):

            x = self.convolution_block(x) + x
            x = torch.nn.functional.relu(x)
            return x

    class Residual_Block_B(torch.nn.Module):

        def __init__(self, in_channels, out_channels):
            
            super().__init__()

            self.convolution_block_1 = torch.nn.Sequential(
                torch.nn.Conv3d(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=3, 
                    stride=2, 
                    padding=1
                ),
                torch.nn.BatchNorm3d(num_features=out_channels),
                torch.nn.ReLU(),
                torch.nn.Conv3d(
                    in_channels=out_channels, 
                    out_channels=out_channels, 
                    kernel_size=3, 
                    stride=1, 
                    padding="same"
                ),
                torch.nn.BatchNorm3d(num_features=out_channels),
            )
            
            self.convolution_block_2 = torch.nn.Sequential(
                torch.nn.Conv3d(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=3, 
                    stride=2, 
                    padding=1
                ),
                torch.nn.BatchNorm3d(num_features=out_channels),
            )
            
        def forward(self, x):

            x = self.convolution_block_1(x) + self.convolution_block_2(x)
            x = torch.nn.functional.relu(x)            
            return x


class Event_by_Event_Model(torch.nn.Module):

    def __init__(self):

        super().__init__()
        
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(4, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 44),
        )

    def forward(self, x):
        
        return self.layers(x) # logits
