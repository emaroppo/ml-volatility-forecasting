from data.processing.HARDailyVolatilityPipeline import HARDailyVolatilityPipeline
from data.processing.VolatilityDataset import VolatilityDataset
from data.db.DBManager import DBManager
from models.har import HARModel
import numpy as np
import pandas as pd

db_manager = DBManager(db_path="data/db/tickers.db")
data = db_manager.get_daily_stock_data("MSFT")
data["log_return"] = (data["ADJ_CLOSE"] / data["ADJ_CLOSE"].shift(1)).apply(
    lambda x: np.log(x)
)
log_returns = data["log_return"]
log_returns.index = data["DATE"]
log_returns = log_returns.dropna()

pipeline = HARDailyVolatilityPipeline()
processed_data = pipeline.process_ticker(log_returns)
processed_data = processed_data.dropna()

split_date = pd.Timestamp("2024-01-01")
training_data = processed_data[processed_data.index < split_date]

training_targets = training_data["realised_daily_volatility"].shift(-1).dropna()
training_data = training_data.iloc[:-1]  # align with targets

har_model = HARModel(use_log=False)
har_model.fit(training_data.values, training_targets.values)

validation_data = processed_data[processed_data.index >= split_date]
validation_targets = validation_data["realised_daily_volatility"].shift(-1).dropna()
validation_data = validation_data.iloc[:-1]  # align with targets

print(har_model.model.coef_)

print(har_model.evaluate(validation_data.values, validation_targets.values))


print(har_model.coefficients())
print("Model fitted.")
