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
        return processed_data
