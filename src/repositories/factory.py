import os
from repositories.csv_repo import CSVInventoryRepository, CSVDemandRepository

DATA_BACKEND = os.getenv("DATA_BACKEND", "csv")


def get_inventory_repository():
    if DATA_BACKEND == "csv":
        return CSVInventoryRepository()
    raise NotImplementedError("Postgres repository not implemented yet")


def get_demand_repository():
    if DATA_BACKEND == "csv":
        return CSVDemandRepository()
    raise NotImplementedError("Postgres repository not implemented yet")
