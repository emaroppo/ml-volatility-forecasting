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
        return processed_data
