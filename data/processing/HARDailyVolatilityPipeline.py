from data.processing.DailyVolatilityPipeline import DailyVolatilityPipeline
from typing import Union, List
import pandas as pd
import numpy as np
from data.processing.VolatilityDataset import VolatilityDataset


class HARDailyVolatilityPipeline(DailyVolatilityPipeline):
    name: str = "HAR Daily Volatility Pipeline"
    description: str = "Pipeline to compute daily volatility using HAR model"

    def compute_weekly_monthly_volatility(
        self,
        log_returns: pd.Series,
    ) -> Union[pd.Series, pd.DataFrame]:

        daily_vol = self.compute_realised_volatility(log_returns)
        weekly_vol = daily_vol.rolling(window=5).mean()
        monthly_vol = daily_vol.rolling(window=22).mean()
        return pd.DataFrame(
            {
                "realised_daily_volatility": daily_vol,
                "realised_weekly_volatility": weekly_vol,
                "realised_monthly_volatility": monthly_vol,
            }
        )

    def process_ticker(self, log_returns):
        data = pd.Series(log_returns).dropna()
        processed_data = self.compute_weekly_monthly_volatility(data)
        processed_data = processed_data.dropna()
        X, y = [], []
        for i in range(len(processed_data) - 22):
            X.append(processed_data.iloc[i : i + 22].values)
            y.append(processed_data.iloc[i + 22]["realised_daily_volatility"])

        # split into training and validation sets
        split_idx = int(0.8 * len(X))
        processed_data = {
            "inputs": np.array(X[:split_idx]),
            "targets": np.array(y[:split_idx]),
        }

        validation_data = {
            "inputs": np.array(X[split_idx:]),
            "targets": np.array(y[split_idx:]),
        }

        return processed_data, validation_data
