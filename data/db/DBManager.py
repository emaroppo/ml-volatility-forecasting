# add setup for db table
# update industries and sectors tables
import sqlite3
import json
from pydantic import BaseModel
import pandas as pd

from typing import List, Union

DB_INIT = """CREATE TABLE IF NOT EXISTS sectors (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    NAME TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS industries (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    NAME TEXT NOT NULL,
    SECTOR_ID INTEGER NOT NULL,
    FOREIGN KEY (SECTOR_ID) REFERENCES sectors(ID)
);

CREATE TABLE IF NOT EXISTS companies (
    TICKER TEXT PRIMARY KEY,
    NAME TEXT,
    INDUSTRY_ID INTEGER NOT NULL,
    FOREIGN KEY (INDUSTRY_ID) REFERENCES industries(ID)
);

CREATE TABLE IF NOT EXISTS daily (
    DATE TEXT NOT NULL,
    TICKER TEXT NOT NULL,
    OPEN REAL NOT NULL,
    HIGH REAL NOT NULL,
    LOW REAL NOT NULL,
    CLOSE REAL NOT NULL,
    VOLUME INTEGER NOT NULL,
    DIVIDEND REAL NOT NULL,
    SPLIT REAL NOT NULL,
    PRIMARY KEY (TICKER, DATE),
    FOREIGN KEY (TICKER) REFERENCES companies(TICKER)
);

CREATE TABLE IF NOT EXISTS hourly (
    DATE TEXT NOT NULL,
    TICKER TEXT NOT NULL,
    OPEN REAL NOT NULL,
    HIGH REAL NOT NULL,
    LOW REAL NOT NULL,
    CLOSE REAL NOT NULL,
    VOLUME INTEGER NOT NULL,
    DIVIDEND REAL NOT NULL,
    SPLIT REAL NOT NULL,
    PRIMARY KEY (TICKER, DATE),
    FOREIGN KEY (TICKER) REFERENCES companies(TICKER)
);

CREATE TABLE IF NOT EXISTS minute (
    DATE TEXT NOT NULL,
    TICKER TEXT NOT NULL,
    OPEN REAL NOT NULL,
    HIGH REAL NOT NULL,
    LOW REAL NOT NULL,
    CLOSE REAL NOT NULL,
    VOLUME INTEGER NOT NULL,
    DIVIDEND REAL NOT NULL,
    SPLIT REAL NOT NULL,
    PRIMARY KEY (TICKER, DATE),
    FOREIGN KEY (TICKER) REFERENCES companies(TICKER)
);

    """


class DBManager(BaseModel):
    db_path: str = "data/db/tickers.db"

    def create_db(self, init_string=DB_INIT):
        """
        Creates a SQLite database and initializes it with the given schema.
        """
        # Connect to SQLite database (or create it if it doesn't exist)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Execute the DB_INIT script to create tables
        cursor.executescript(init_string)

        # Commit changes and close the connection
        conn.commit()
        conn.close()

    def populate_sectors_industries(
        self, sectors_industries_path: str = "data/db/sectors_industries.json"
    ):
        """
        Populates the sectors and industries tables with data from a JSON file.
        """
        # Connect to SQLite database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Load sectors and industries from JSON file
        with open(sectors_industries_path, "r") as f:
            sectors_industries = json.load(f)
            for sector, industries in sectors_industries.items():
                # Insert sector into the database
                cursor.execute("INSERT INTO sectors (NAME) VALUES (?)", (sector,))
                sector_id = cursor.lastrowid

                for industry in industries:
                    # Insert industry into the database
                    cursor.execute(
                        "INSERT INTO industries (NAME, SECTOR_ID) VALUES (?, ?)",
                        (industry, sector_id),
                    )
        conn.commit()
        conn.close()
        print("Sectors and industries populated successfully.")
        return True

    def retrieve_table(self, table_name: str):
        """
        Retrieves all data from the specified table.
        """
        # Connect to SQLite database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Execute SQL query to retrieve all data from the specified table
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()

        # Close the connection
        conn.close()

        return rows

    def get_all_tickers(self) -> List[str]:
        """
        Retrieves all tickers from the companies table.
        """
        # Connect to SQLite database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Execute SQL query to retrieve all tickers
        cursor.execute("SELECT TICKER FROM companies")
        rows = cursor.fetchall()

        # Close the connection
        conn.close()

        return [row[0] for row in rows]

    def retrieve_industries(self, sector_id: int = None):
        """
        Retrieves all industries from the database.
        """
        # Connect to SQLite database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if sector_id:
            # Execute SQL query to retrieve all data from the specified table
            cursor.execute(f"SELECT * FROM industries WHERE SECTOR_ID={sector_id}")
        else:
            # Execute SQL query to retrieve all data from the specified table
            cursor.execute(f"SELECT * FROM industries")

        rows = cursor.fetchall()

        # Close the connection
        conn.close()

        return rows

    def retrieve_companies(self):
        """
        Retrieves all companies from the database.
        """
        # Connect to SQLite database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Execute SQL query to retrieve all data from the specified table
        cursor.execute(f"SELECT * FROM companies")
        rows = cursor.fetchall()

        # Close the connection
        conn.close()

        return rows

    def get_daily_stock_data(
        self, ticker: str, start: Union[int, None] = None, end: Union[int, None] = None
    ) -> pd.DataFrame:
        """
        Retrieves daily stock data for a specific ticker.
        """
        query = """
            SELECT *
            FROM daily
            WHERE TICKER = ?
            ORDER BY DATE
        """

        if start and end:
            query = """
                SELECT *
                FROM daily
                WHERE TICKER = ?
                AND DATE BETWEEN ? AND ?
                ORDER BY DATE
            """
            params = (ticker, start, end)
        elif start:
            query = """
                SELECT *
                FROM daily
                WHERE TICKER = ?
                AND DATE >= ?
                ORDER BY DATE
            """
            params = (ticker, start)
        elif end:
            query = """
                SELECT *
                FROM daily
                WHERE TICKER = ?
                AND DATE <= ?
                ORDER BY DATE
            """
            params = (ticker, end)
        else:
            params = (ticker,)

        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql(query, conn, params=params, parse_dates=["DATE"])

        conn.close()
        return df

    def insert_company(self, company):
        """
        Inserts a company into the database.
        """
        # Connect to SQLite database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Insert company into the database
        cursor.execute(
            "INSERT INTO companies (TICKER, NAME, INDUSTRY_ID) VALUES (?, ?, ?)",
            (company.symbol, company.name, company.industry_id),
        )

        # Commit changes and close the connection
        conn.commit()
        conn.close()

    def insert_price_data(self, data):
        """
        Inserts price data into the database.
        """
        table = f"{data[0].interval}"
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for observation in data:
            cursor.execute(
                f"""
                INSERT INTO {table} (
                    DATE, TICKER, OPEN, HIGH, LOW, CLOSE, VOLUME,
                    DIVIDEND, SPLIT,
                    ADJ_OPEN, ADJ_CLOSE, ADJ_HIGH, ADJ_LOW
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    observation.start_timestamp,
                    observation.symbol,
                    observation.open,
                    observation.high,
                    observation.low,
                    observation.close,
                    observation.volume,
                    observation.dividend,
                    observation.split,
                    observation.adj_open,
                    observation.adj_close,
                    observation.adj_high,
                    observation.adj_low,
                ),
            )

        conn.commit()
        conn.close()


db_manager = DBManager()
