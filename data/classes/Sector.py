from pydantic import BaseModel
from data.db.DBManager import DBManager


class Sector(BaseModel):
    sector_id: int
    name: str

    @classmethod
    def get_all_sectors(cls) -> list["Sector"]:
        """
        Fetches all sectors from the database.
        """
        db_manager = DBManager()
        sectors = db_manager.retrieve_table("sectors")
        return [cls(sector_id=sector[0], name=sector[1]) for sector in sectors]
