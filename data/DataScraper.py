from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import yfinance as yf
import json
from tqdm import tqdm
from data.classes.Company import Company
from data.classes.Industry import Industry
from data.classes.StockData import DailyStockData, MinuteStockData, HourlyStockData
from data.db.DBManager import DBManager


class DataScraper(BaseModel):

    def fetch_top_companies(self):
        """
        Fetches all top companies for each industry of each sector.
        """
        db_manager = DBManager()
        industries = Industry.get_all_industries()

        for industry in tqdm(industries):
            top_companies = yf.Industry(industry.name).top_companies
            if top_companies is None:
                print(f"Error with {industry.name}")
                continue
            else:
                print(f"{industry.name} - {len(top_companies)}")
                print(f"Top companies in {industry.name}:")
                print(top_companies)

                for company in top_companies.iterrows():

                    print(company)
                    company_obj = Company(
                        name=company[1]["name"],
                        symbol=company[0],
                        industry_id=industry.industry_id,
                    )
                    db_manager.insert_company(company_obj)

    def fetch_minute_data(
        self,
        symbol: List[str],
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> Optional[List[dict]]:
        """
        Fetches minute-level data for given symbols and inserts into database.
        """
        try:
            tickers = yf.Tickers(" ".join(symbol))
            if start_date and end_date:
                data = tickers.download(start=start_date, end=end_date, interval="1m")
            else:
                data = tickers.download("5d", interval="1m")

            data.columns = data.columns.swaplevel(0, 1)
            data = data.sort_index(axis=1)

            minute_data = []
            for ticker in symbol:
                company_data = data[ticker]
                company_data.loc[:, ["Open", "Close"]] = company_data[
                    ["Open", "Close"]
                ].ffill()
                company_data = company_data.dropna(subset=["Close"])
                for observation in company_data.iterrows():
                    minute_data.append(
                        MinuteStockData(
                            interval="minute",
                            symbol=ticker,
                            start_timestamp=int(observation[0].timestamp()),
                            open=observation[1]["Open"],
                            high=observation[1]["High"],
                            low=observation[1]["Low"],
                            close=observation[1]["Close"],
                            volume=observation[1]["Volume"],
                            dividend=observation[1]["Dividends"],
                            split=observation[1]["Stock Splits"],
                        )
                    )

            db_manager = DBManager()
            db_manager.insert_price_data(minute_data)
            return data

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

    def fetch_daily_data(
        self,
        symbol: List[str],
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> Optional[List[dict]]:
        tickers = yf.Tickers(" ".join(symbol))
        if start_date and end_date:
            data = tickers.download(
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=False,
                group_by="ticker",
            )
        else:
            data = tickers.download(
                "50y", interval="1d", auto_adjust=False, group_by="ticker"
            )

        data = data.sort_index(axis=1)

        daily_data = []
        for ticker in symbol:
            company_data = data[ticker]

            # Fill missing data in raw prices
            company_data.loc[:, ["Open", "High", "Low", "Close", "Adj Close"]] = (
                company_data[["Open", "High", "Low", "Close", "Adj Close"]].ffill()
            )

            # Drop rows with missing critical values
            company_data = company_data.dropna(subset=["Close", "Adj Close"])

            # Calculate adjustment factor
            adj_factor = company_data["Adj Close"] / company_data["Close"]

            # Apply factor to all prices and volume
            company_data["Adj Open"] = company_data["Open"] * adj_factor
            company_data["Adj High"] = company_data["High"] * adj_factor
            company_data["Adj Low"] = company_data["Low"] * adj_factor
            company_data["Adj Volume"] = company_data["Volume"] / adj_factor

            for timestamp, row in company_data.iterrows():
                daily_data.append(
                    DailyStockData(
                        interval="daily",
                        symbol=ticker,
                        start_timestamp=int(timestamp.timestamp()),
                        open=row["Open"],
                        high=row["High"],
                        low=row["Low"],
                        close=row["Close"],
                        volume=row["Volume"],
                        dividend=row.get("Dividends", 0.0),
                        split=row.get("Stock Splits", 0.0),
                        adj_open=row["Adj Open"],
                        adj_high=row["Adj High"],
                        adj_low=row["Adj Low"],
                        adj_close=row["Adj Close"],
                        adj_volume=row["Adj Volume"],
                    )
                )

        db_manager = DBManager()
        db_manager.insert_price_data(daily_data)
        return data

    def fetch_hourly_data(
        self, symbol: List[str], start_date: datetime = None, end_date: datetime = None
    ) -> Optional[List[dict]]:
        """
        Fetches weekly data for a given symbol between start_date and end_date.
        """
        tickers = yf.Tickers(" ".join(symbol))
        print(start_date, end_date)
        print(tickers)
        if start_date and end_date:
            data = tickers.download(start=start_date, end=end_date, interval="1h")
        else:
            data = tickers.download("2y", interval="1h")

        data.columns = data.columns.swaplevel(0, 1)
        data = data.sort_index(axis=1)

        # for each company, create HourlyStockData object
        hourly_data = list()
        print(data.head())
        for ticker in symbol:
            company_data = data[ticker]
            # ffill missing values for Open and Close
            company_data.loc[:, ["Open", "Close"]] = company_data[
                ["Open", "Close"]
            ].ffill()
            # drop rows where Close is NaN
            company_data = company_data.dropna(subset=["Close"])
            # create HourlyStockData object
            for observation in company_data.iterrows():
                hourly_data.append(
                    HourlyStockData(
                        interval="hourly",
                        symbol=ticker,
                        start_timestamp=int(observation[0].timestamp()),
                        open=observation[1]["Open"],
                        high=observation[1]["High"],
                        low=observation[1]["Low"],
                        close=observation[1]["Close"],
                        volume=observation[1]["Volume"],
                        dividend=observation[1]["Dividends"],
                        split=observation[1]["Stock Splits"],
                    )
                )
        # insert data into database
        db_manager = DBManager()
        db_manager.insert_price_data(hourly_data)
        return data
