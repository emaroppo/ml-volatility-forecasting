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

    def run(self, log_returns, training_dataset, validation_dataset):
        processed_data, validation_data = self.process_ticker(log_returns)

        training_dataset.append_data(
            new_X=processed_data["inputs"], new_y=processed_data["targets"]
        )
        validation_dataset.append_data(
            new_X=validation_data["inputs"], new_y=validation_data["targets"]
        )

        return training_dataset, validation_dataset
