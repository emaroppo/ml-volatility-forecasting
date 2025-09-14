import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers, fc1_size, fc2_size, output_size
    ):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, fc1_size)  # First FC layer
        self.fc2 = nn.Linear(fc1_size, fc2_size)  # Second FC layer
        self.relu = nn.ReLU()  # Activation function
        self.fc3 = nn.Linear(fc2_size, output_size)  # Second FC layer

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Take the last time step output
        out = out[:, -1, :]  # Shape: (batch_size, hidden_size)
        out = self.relu(out)
        # Fully connected layers
        out = self.fc1(out)  # First FC layer
        out = self.relu(out)  # Activation
        out = self.fc2(out)  # Second FC layer
        out = self.relu(out)  # Activation
        out = self.fc3(out)  # Output layer

        return out
