from data.processing.DailyVolatilityPipeline import DailyVolatilityPipeline
import pandas as pd
import numpy as np


class UnivariateDailyVolatilityPipeline(DailyVolatilityPipeline):
    name: str = "Univariate Daily Volatility Pipeline"
    description: str = "Pipeline to compute univariate daily volatility"

    def process_ticker(self, log_returns):
        data = pd.Series(log_returns).dropna()  # revisit

        processed_data = self.compute_realised_volatility(data)

        # data before 2024-01-01 for training, after for validation
        split_date = pd.Timestamp("2024-01-01")
        val_data = processed_data[processed_data.index >= split_date]
        train_data = processed_data[processed_data.index < split_date]

        X_train, y_train = [], []
        for i in range(len(train_data) - 22):
            X_train.append(processed_data.iloc[i : i + 22].values)
            y_train.append(processed_data.iloc[i + 22])

        if not val_data.empty:

            X_val, y_val = [], []
            for i in range(len(val_data) - 22):
                X_val.append(processed_data.iloc[i : i + 22].values)
                y_val.append(processed_data.iloc[i + 22])

        train_data = {
            "inputs": np.array(X_train),
            "targets": np.array(y_train).reshape(-1, 1),
        }
        validation_data = {
            "inputs": np.array(X_val) if not val_data.empty else np.empty((0, 22)),
            "targets": (
                np.array(y_val).reshape(-1, 1)
                if not val_data.empty
                else np.empty((0, 1))
            ),
        }

        print(train_data["inputs"].shape, train_data["targets"].shape)
        print(validation_data["inputs"].shape, validation_data["targets"].shape)

        return train_data, validation_data
