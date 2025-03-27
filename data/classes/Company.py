from pydantic import BaseModel
from data.db.DBManager import db_manager
from typing import Union


class Company(BaseModel):
    """
    Represents a company with its details.
    """

    name: Union[str | None]
    symbol: str
    industry_id: int

    @classmethod
    def get_all_companies(cls):
        """
        Fetches all companies from the database.
        """
