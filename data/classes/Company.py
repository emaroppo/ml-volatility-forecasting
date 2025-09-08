from pydantic import BaseModel, Field
from data.db.DBManager import db_manager
from typing import Optional, List, ClassVar


class Company(BaseModel):
    """
    Represents a company with its details.
    """

    name: Optional[str] = None
    symbol: str
    industry_id: int

    # Class-level DBManager reference for easier testing/mocking
    _db_manager: ClassVar = db_manager

    @classmethod
    def get_all_companies(cls) -> List["Company"]:
        """
        Fetches all companies from the database.
        Returns a list of Company objects.
        """
        companies = cls._db_manager.retrieve_companies()
        if not companies:
            return []

        return [
            cls(name=company[1], symbol=company[0], industry_id=company[2])
            for company in companies
        ]

    @classmethod
    def get_by_symbol(cls, symbol: str) -> Optional["Company"]:
        """
        Fetches a company by its ticker symbol.
        """
        company_data = cls._db_manager.retrieve_company_by_symbol(symbol)
        if not company_data:
            return None

        return cls(
            name=company_data[1], symbol=company_data[0], industry_id=company_data[2]
        )
