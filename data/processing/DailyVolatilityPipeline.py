from pydantic import BaseModel
from typing import Literal
import numpy as np
import pandas as pd


class DailyVolatilityPipeline(BaseModel):
    name: str = "Daily Volatility Pipeline"
    description: str = "Pipeline to compute daily volatility"

    def compute_realised_volatility(
        self, log_returns: pd.Series, method: Literal["squared", "abs"] = "squared"
    ) -> pd.Series:
        if method == "squared":
            return log_returns**2
        elif method == "abs":
            return log_returns.abs()
        else:
            raise ValueError("Invalid method specified.")

    def process_ticker(self, log_returns):
        raise NotImplementedError("Subclasses should implement this method.")

    def create_sequences(self, data, seq_length=22):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data.iloc[i : i + seq_length].values.reshape(-1, 1))
            y.append(data.iloc[i + seq_length])
        return np.array(X), np.array(y)

    def split_data(self, data, seq_length=22, split_date="2024-01-01", n_features=1):
        # data before 2024-01-01 for training, after for validation
        split_date = pd.Timestamp(split_date)
        val_data = data[data.index >= split_date]
        train_data = data[data.index < split_date]

        X_train, y_train = self.create_sequences(train_data, seq_length)
        X_val, y_val = self.create_sequences(val_data, seq_length)

        train_data = {
            "inputs": np.array(X_train),  # (N, 22, 1)
            "targets": np.array(y_train).reshape(
                -1, n_features
            ),  # (N_Samples, N_Features)
        }
        validation_data = {
            "inputs": np.array(X_val),  # (M, 22, 1)
            "targets": np.array(y_val).reshape(
                -1, n_features
            ),  # (N_Samples, N_Features)
        }

        return train_data, validation_data

    def run(self, log_returns, training_dataset, validation_dataset):
        processed_data = self.process_ticker(log_returns)

        training_data, validation_data = self.split_data(processed_data)

        training_dataset.append_data(
            new_X=training_data["inputs"], new_y=training_data["targets"]
        )
        validation_dataset.append_data(
            new_X=validation_data["inputs"], new_y=validation_data["targets"]
        )

        return training_dataset, validation_dataset
