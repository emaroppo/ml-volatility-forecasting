import torch
from torch.utils.data import Dataset
import os


class VolatilityDataset(Dataset):
    def __init__(self, data_path, seq_len=22, n_features=3):
        self.data_path = data_path
        self.seq_len = seq_len
        if os.path.exists(self.data_path):
            data = torch.load(self.data_path)
            self.X = data["inputs"]
            self.y = data["targets"]
        else:
            self.X = torch.empty(0, seq_len, n_features)
            self.y = torch.empty(0)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X = self.X[idx].unsqueeze(0)  # Add batch dimension
        y = self.y[idx].unsqueeze(0)  # Add batch dimension

        if X.shape == (1, self.seq_len):
            X = X.transpose(0, 1)
        return X, y

    def append_data(self, new_X, new_y):
        # Append new data to the existing dataset

        # check if new_X is empty (0,)
        if new_X.shape[0] == 0:
            return

        # convert to tensor if not already
        if not torch.is_tensor(new_X):
            new_X = torch.tensor(new_X, dtype=torch.float32)
        if not torch.is_tensor(new_y):
            new_y = torch.tensor(new_y, dtype=torch.float32)

        if len(self.X) > 0:
            self.X = torch.cat((self.X, new_X), dim=0)
            self.y = torch.cat((self.y, new_y), dim=0)
        else:
            self.X = new_X
            self.y = new_y

    def save(self):
        torch.save({"inputs": self.X, "targets": self.y}, self.data_path)
        print(f"Training data saved to {self.data_path}")
