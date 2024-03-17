import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch import nn
# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class DefenseTransformationDataset(Dataset):
    def __init__(self, labels, representations, transform=None): # , transform=None, target_transform=None):
        self.labels = labels
        self.vectors = representations
        self.transform = transform
        # self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        vector = self.vectors[idx]
        if self.transform:
            vector = self.transform(vector)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return vector, label
    
class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.dense = nn.Sequential(
            nn.Linear(2003, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = self.dense(x)
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
    
class Net(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, n_outputs)

    def forward(self, x):
        x = self.fc1(x)
        return x


class MapperNet(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_hidden):
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_outputs)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
def train_one_epoch(epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(train_dataloader):
            # Every data instance is an input + label pair
            inputs, labels = data

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(train_dataloader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss

def test_step(model: nn.Module, test_loader: DataLoader) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    confusion_matrix = np.zeros((10, 10))
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            for l, p in zip(labels, predicted):
                confusion_matrix[l, p] += 1

    return confusion_matrix


    
if __name__ == "__main__":
    data = np.load(
        "defensetransformation/data/2003_combine_with_two_Evaluate.npz"
        
    )

    #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    #full_dataset = DefenseTransformationDataset(data["labels"], data["representations"], transform = transforms.ToTensor())

    full_dataset = DefenseTransformationDataset(data["labels"], data["representations"])
    full_dataset_len = len(full_dataset)
    train_size = int(0.8 * full_dataset_len)
    test_size = len(full_dataset) - train_size
    train_data, test_data = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    model = MLP()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('defensetransformation/runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 15
    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer)

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(test_dataloader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'defensetransformation/models/model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1


        confusion_matrix = test_step(
            model=model, test_loader=test_dataloader)

        print(
            f"Test accuracy {np.trace(confusion_matrix) / np.sum(confusion_matrix):.4f}\n"
        )

# test accuracy 0.95
        
# saved_model = MLP()
# saved_model.load_state_dict(torch.load(PATH))