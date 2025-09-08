import time
import random
from datetime import datetime
from data.DataScraper import DataScraper
from data.classes.Company import Company

# Initialize scraper
scraper = DataScraper()

# retrieve tickers from database
tickers = [company.symbol for company in Company.get_all_companies()]

# Config
BATCH_SIZE = 50
PAUSE_RANGE = (5, 15)  # seconds
MAX_RETRIES = 5


def chunkify(lst, size):
    """Split list into chunks of size `size`"""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def safe_fetch(fetch_func, batch, *args, **kwargs):
    for attempt in range(MAX_RETRIES):
        return fetch_func(batch, *args, **kwargs)

    print(f"[Failed] Giving up on batch: {batch}")
    return None


def scrape_data(minute=True, hourly=True, daily=True):
    print("Starting scrape...")

    if minute:

        # 1. Minute data (last 5 days)
        print("\n[1m Interval] Fetching recent minute-level data")
        for batch in chunkify(tickers, BATCH_SIZE):
            data = safe_fetch(scraper.fetch_minute_data, batch)
            if data is not None:
                print(f"✔️ Minute data fetched for batch: {batch}")
            time.sleep(random.uniform(*PAUSE_RANGE))

    # 2. Hourly data (up to 2 years)
    if hourly:
        print("\n[1h Interval] Fetching hourly data")
        for batch in chunkify(tickers, BATCH_SIZE):
            data = safe_fetch(scraper.fetch_hourly_data, batch)
            if data is not None:
                print(f"✔️ Hourly data fetched for batch: {batch}")
            time.sleep(random.uniform(*PAUSE_RANGE))

    # 3. Daily data (up to 50 years)
    if daily:
        print("\n[1d Interval] Fetching long-term daily data")
        for batch in chunkify(tickers, BATCH_SIZE):
            data = safe_fetch(scraper.fetch_daily_data, batch)
            if data is not None:
                print(f"✔️ Daily data fetched for batch: {batch}")
            time.sleep(random.uniform(*PAUSE_RANGE))

    print("\n✅ All data fetched!")


if __name__ == "__main__":
    scrape_data(minute=False, hourly=False, daily=True)
