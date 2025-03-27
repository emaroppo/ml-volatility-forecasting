from pydantic import BaseModel


class StockData(BaseModel):
    interval: str
    start_date: str
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

    date: str
    adjusted_close: float
    ticker: str


class MinuteStockData(StockData):
    """
    Represents minute stock data.
    """

    date: str
    adjusted_close: float
    ticker: str


class HourlyStockData(StockData):
    """
    Represents hourly stock data.
    """

    date: str
    adjusted_close: float
    ticker: str
