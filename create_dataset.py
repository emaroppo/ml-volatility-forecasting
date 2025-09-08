from data.db.DBManager import DBManager
from data.processing.HARDailyVolatilityPipeline import HARDailyVolatilityPipeline
from data.processing.UnivariateDailyVolatilityPipeline import (
    UnivariateDailyVolatilityPipeline,
)
from data.processing.VolatilityDataset import VolatilityDataset
from tqdm import tqdm
import numpy as np


def create_univariate_dataset(tickers=None, db_path="data/db/tickers.db"):
    db_manager = DBManager(db_path=db_path)
    # get all tickers if none provided
    if not tickers:
        tickers = db_manager.get_all_tickers()
    univariate_pipeline = UnivariateDailyVolatilityPipeline()
    train_dataset = VolatilityDataset(
        data_path="data/processed/univariate_train.pt",
        seq_len=22,
        n_features=1,
    )
    val_dataset = VolatilityDataset(
        data_path="data/processed/univariate_val.pt",
        seq_len=22,
        n_features=1,
    )

    for i, ticker in tqdm(enumerate(tickers)):
        print(f"Processing ticker: {ticker}")
        stock_data = db_manager.get_daily_stock_data(ticker)
        stock_data = stock_data.sort_values(by="DATE")
        # compute log returns
        stock_data["log_return"] = (
            stock_data["ADJ_CLOSE"] / stock_data["ADJ_CLOSE"].shift(1)
        ).apply(lambda x: np.log(x))
        stock_data = stock_data.dropna(subset=["log_return"])
        log_returns = stock_data["log_return"]

        # set date as log_returns index
        log_returns.index = stock_data["DATE"]

        univariate_pipeline.run(
            log_returns, training_dataset=train_dataset, validation_dataset=val_dataset
        )
        if (i + 1) % 50 == 0:
            print("Saving intermediate datasets...")
            train_dataset.save()
            val_dataset.save()
    train_dataset.save()
    val_dataset.save()
    print("Univariate dataset creation complete.")


def create_har_dataset(tickers=None, db_path="data/db/tickers.db"):
    db_manager = DBManager(db_path=db_path)
    # get all tickers if none provided
    if not tickers:
        tickers = db_manager.get_all_tickers()
    har_pipeline = HARDailyVolatilityPipeline()
    train_dataset = VolatilityDataset(
        data_path="data/processed/har_train.pt",
        seq_len=22,
        n_features=3,
    )
    val_dataset = VolatilityDataset(
        data_path="data/processed/har_val.pt",
        seq_len=22,
        n_features=3,
    )

    for i, ticker in tqdm(enumerate(tickers)):
        print(f"Processing ticker: {ticker}")
        stock_data = db_manager.get_daily_stock_data(ticker)
        stock_data = stock_data.sort_values(by="DATE")
        # compute log returns
        stock_data["log_return"] = (
            stock_data["ADJ_CLOSE"] / stock_data["ADJ_CLOSE"].shift(1)
        ).apply(lambda x: np.log(x))
        stock_data = stock_data.dropna(subset=["log_return"])
        log_returns = stock_data["log_return"]

        har_pipeline.run(
            log_returns=log_returns,
            training_dataset=train_dataset,
            validation_dataset=val_dataset,
        )
        if (i + 1) % 50 == 0:
            print("Saving intermediate datasets...")
            train_dataset.save()
            val_dataset.save()
    train_dataset.save()
    val_dataset.save()
    print("HAR dataset creation complete.")


if __name__ == "__main__":

    create_univariate_dataset()
