from pydantic import BaseModel
from data.db.DBManager import DBManager


class Industry(BaseModel):
    industry_id: int
    name: str
    sector_id: int

    @classmethod
    def get_all_industries(cls) -> list["Industry"]:
        """
        Fetches all industries from the database.
        """
        db_manager = DBManager()
        industries = db_manager.retrieve_industries()
        return [
            cls(industry_id=industry[0], name=industry[1], sector_id=industry[2])
            for industry in industries
        ]
