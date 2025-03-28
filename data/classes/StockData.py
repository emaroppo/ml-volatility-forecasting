from pydantic import BaseModel


class StockData(BaseModel):
    interval: str
    symbol: str
    start_timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: int
    dividend: float
    split: float


class DailyStockData(StockData):
    """
    Represents daily stock data.
    """

    interval: str = "daily"


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
