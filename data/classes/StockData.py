from pydantic import BaseModel, validator, Field
from typing import Optional, List, Dict, Any, ClassVar, Tuple
import math
import pandas as pd
import numpy as np
from datetime import datetime
from data.db.DBManager import db_manager


class StockData(BaseModel):
    interval: str
    symbol: str
    start_timestamp: int
    open: Optional[float] = None
    adj_open: Optional[float] = None
    adj_close: Optional[float] = None
    adj_high: Optional[float] = None
    adj_low: Optional[float] = None
    adj_volume: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[int] = None
    dividend: Optional[float] = None
    split: Optional[float] = None

    @validator("*", pre=True)
    def replace_nan_with_none(cls, value):
        if isinstance(value, float) and math.isnan(value):
            return None
        return value

    def to_dict(self) -> Dict[str, Any]:
        """Convert the stock data to a dictionary suitable for DataFrame creation"""
        return {
            "DATE": datetime.fromtimestamp(self.start_timestamp),
            "TICKER": self.symbol,
            "CLOSE": self.close,
            "SPLIT": self.split if self.split is not None else 1.0,
            "DIVIDEND": self.dividend if self.dividend is not None else 0.0,
        }

    @classmethod
    def list_to_dataframe(cls, stock_data_list: List["StockData"]) -> pd.DataFrame:
        """Convert a list of StockData objects to a pandas DataFrame"""
        if not stock_data_list:
            return pd.DataFrame()

        records = [data.to_dict() for data in stock_data_list]
        df = pd.DataFrame(records)

        # Ensure proper types
        if not df.empty:
            df["DATE"] = pd.to_datetime(df["DATE"])
            df["SPLIT"] = df["SPLIT"].fillna(1.0)
            df["DIVIDEND"] = df["DIVIDEND"].fillna(0.0)

        return df.sort_values("DATE")


class DailyStockData(StockData):
    """
    Represents daily stock data.
    """

    interval: str = "daily"

    @classmethod
    def get_stock_data(cls, ticker: str) -> pd.DataFrame:
        return db_manager.get_daily_stock_data(ticker)


class MinuteStockData(StockData):
    """
    Represents minute stock data.
    """

    interval: str = "minute"


class HourlyStockData(StockData):
    """
    Represents hourly stock data.
    """

    interval: str = "hourly"
