from data.processing.DailyVolatilityPipeline import DailyVolatilityPipeline
import pandas as pd
import numpy as np


class UnivariateDailyVolatilityPipeline(DailyVolatilityPipeline):
    name: str = "Univariate Daily Volatility Pipeline"
    description: str = "Pipeline to compute univariate daily volatility"

    def process_ticker(self, log_returns):
        print(log_returns.head())
        data = pd.Series(log_returns).dropna()  # revisit
        print(data.head())

        processed_data = self.compute_realised_volatility(data)
        print(processed_data.head())

        # data before 2024-01-01 for training, after for validation
        split_date = pd.Timestamp("2024-01-01")
        val_data = processed_data[processed_data.index >= split_date]
        train_data = processed_data[processed_data.index < split_date]

        X_train, y_train = [], []
        for i in range(len(train_data) - 22):
            X_train.append(train_data.iloc[i : i + 22].values.reshape(-1, 1))
            y_train.append(train_data.iloc[i + 22])

        X_val, y_val = [], []
        for i in range(len(val_data) - 22):
            X_val.append(val_data.iloc[i : i + 22].values.reshape(-1, 1))
            y_val.append(val_data.iloc[i + 22])

        train_data = {
            "inputs": np.array(X_train),  # (N, 22, 1)
            "targets": np.array(y_train).reshape(-1, 1),  # (N, 1)
        }
        validation_data = {
            "inputs": np.array(X_val),  # (M, 22, 1)
            "targets": np.array(y_val).reshape(-1, 1),  # (M, 1)
        }

        return train_data, validation_data
