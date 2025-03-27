from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import yfinance as yf
import json
from tqdm import tqdm
from data.classes.Company import Company
from data.classes.Industry import Industry
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
        Fetches minute data for a given symbol between start_date and end_date.
        """
        try:
            tickers = yf.Tickers(" ".join(symbol))
            if start_date and end_date:
                data = tickers.download(start=start_date, end=end_date, interval="1m")
            else:
                data = tickers.download("5d", interval="1m")
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
        """
        Fetches daily data for a given symbol between start_date and end_date.
        """
        try:
            tickers = yf.Tickers(" ".join(symbol))
            if start_date and end_date:
                data = tickers.download(start=start_date, end=end_date, interval="1d")
            else:
                data = tickers.download("50y", interval="1d")

            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

    def fetch_hourly_data(
        self, symbol: List[str], start_date: datetime = None, end_date: datetime = None
    ) -> Optional[List[dict]]:
        """
        Fetches weekly data for a given symbol between start_date and end_date.
        """
        try:
            tickers = yf.Tickers(" ".join(symbol))
            print(start_date, end_date)
            print(tickers)
            if start_date and end_date:
                data = tickers.download(start=start_date, end=end_date, interval="1h")
            else:
                data = tickers.download("2y", interval="1h")
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
