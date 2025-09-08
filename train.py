import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from models.lstm import LSTMModel
from data.processing.VolatilityDataset import VolatilityDataset


def calculate_mape(y_true, y_pred, epsilon=1e-6, clip_threshold=100):
    """
    Calculate the Mean Absolute Percentage Error (MAPE).
    Args:
    - y_true: Tensor of true values
    - y_pred: Tensor of predicted values
    - epsilon: Small value added to avoid division by zero
    - clip_threshold: Threshold to clip very high MAPE values (in percentage)
    Returns:
    - MAPE: Mean Absolute Percentage Error (as a float)
    """
    # Calculate absolute percentage error
    percentage_error = torch.abs((y_true - y_pred) / (y_true + epsilon)) * 100
    percentage_error = torch.clamp(percentage_error, max=clip_threshold)

    return torch.mean(percentage_error)


def train_lstm_model(
    train_dataset_path,
    val_dataset_path,
    input_size,
    hidden_size,
    num_layers,
    fc1_size,
    output_size,
    batch_size=1024,
    num_epochs=10,
    learning_rate=0.001,
    validation_split=0.1,
    test_split=0.1,
):
    # Load the dataset from the saved .pt file
    train_dataset = VolatilityDataset(
        data_path=train_dataset_path,
        seq_len=22,
        n_features=input_size,
    )
    val_dataset = VolatilityDataset(
        data_path=val_dataset_path,
        seq_len=22,
        n_features=input_size,
    )
    # print sample
    print(len(val_dataset))

    # Split the dataset into training, validation, and test sets
    test_size = int(len(val_dataset) / 2)
    val_dataset, test_dataset = random_split(val_dataset, [test_size, test_size])

    # Create DataLoader for training, validation, and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        fc1_size=fc1_size,
        output_size=output_size,
    )

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss function and optimizer
    criterion = (
        nn.MSELoss()
    )  # Mean Squared Error loss (assuming you're forecasting a continuous value)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Lists to store losses for training and validation
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_train_mape = []
    epoch_val_mape = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_train_loss = 0.0
        running_train_mape = 0.0

        # Training loop
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            # Move data to the same device as the model (GPU or CPU)
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(batch_X)

            # Compute the loss
            loss = criterion(output, batch_y)
            mape = calculate_mape(batch_y, output)  # Calculate MAPE

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Track running loss and MAPE for reporting
            running_train_loss += loss.item()
            running_train_mape += mape.item()

        # Calculate average training loss and MAPE for this epoch
        train_loss = running_train_loss / len(train_loader)
        train_mape = running_train_mape / len(train_loader)
        epoch_train_losses.append(train_loss)
        epoch_train_mape.append(train_mape)

        # Validation loop
        model.eval()
        running_val_loss = 0.0
        running_val_mape = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                # Forward pass
                output = model(batch_X)

                # Compute the loss
                loss = criterion(output, batch_y)
                mape = calculate_mape(batch_y, output)  # Calculate MAPE

                # Track validation loss and MAPE
                running_val_loss += loss.item()
                running_val_mape += mape.item()

        # Calculate average validation loss and MAPE for this epoch
        val_loss = running_val_loss / len(val_loader)
        val_mape = running_val_mape / len(val_loader)
        epoch_val_losses.append(val_loss)
        epoch_val_mape.append(val_mape)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.8f}, Train MAPE: {train_mape:.2f}%, Val Loss: {val_loss:.8f}, Val MAPE: {val_mape:.2f}%"
        )

    # Save the trained model
    torch.save(model.state_dict(), "lstm_volatility_model.pth")
    print("Model trained and saved!")

    # Plot the loss curves
    plt.plot(range(1, num_epochs + 1), epoch_train_losses, label="Training Loss")
    plt.plot(range(1, num_epochs + 1), epoch_val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_loss.png")
    plt.show()

    # Plot the MAPE curves
    plt.plot(range(1, num_epochs + 1), epoch_train_mape, label="Training MAPE")
    plt.plot(range(1, num_epochs + 1), epoch_val_mape, label="Validation MAPE")
    plt.xlabel("Epoch")
    plt.ylabel("MAPE (%)")
    plt.title("Training and Validation MAPE over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_mape.png")
    plt.show()

    # Test the model on the test set
    model.eval()
    running_test_loss = 0.0
    running_test_mape = 0.0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Forward pass
            output = model(batch_X)

            # Compute the loss
            loss = criterion(output, batch_y)
            mape = calculate_mape(batch_y, output)  # Calculate MAPE

            # Track test loss and MAPE
            running_test_loss += loss.item()
            running_test_mape += mape.item()

    # Calculate average test loss and MAPE
    test_loss = running_test_loss / len(test_loader)
    test_mape = running_test_mape / len(test_loader)
    print(f"Test Loss: {test_loss:.8f}, Test MAPE: {test_mape:.2f}%")


input_size = 1
hidden_size = 64
num_layers = 3
fc1_size = 64
output_size = 1
train_dataset_path = "data/processed/univariate_train.pt"
val_dataset_path = "data/processed/univariate_val.pt"

# Train the model
train_lstm_model(
    train_dataset_path=train_dataset_path,
    val_dataset_path=val_dataset_path,
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    fc1_size=fc1_size,
    output_size=output_size,
    batch_size=1024,
    num_epochs=50,
    learning_rate=0.001,
)
