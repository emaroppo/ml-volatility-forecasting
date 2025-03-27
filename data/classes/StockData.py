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
