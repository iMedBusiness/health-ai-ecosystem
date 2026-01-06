import pandas as pd
from datetime import date
from pathlib import Path
from typing import List

from data_contracts.validate import validate_df
from data_contracts.models import (
    LotContract,
    InventoryBalanceContract,
    ConsumptionDailyContract,
    ForecastDailyContract,
)


DATA_DIR = Path("data/curated")


class CSVInventoryRepository:

    def __init__(self):
        self.lots = pd.read_csv(DATA_DIR / "lots.csv", parse_dates=["expiry_date", "received_date"])
        self.balance = pd.read_csv(DATA_DIR / "inventory_balance.csv")

        validate_df(self.lots, "lots")
        validate_df(self.balance, "inventory_balance")

    def get_lots(self, facility_id: str, item_id: str) -> List[LotContract]:
        df = self.lots[self.lots["item_id"] == item_id]
        return [LotContract(**row) for row in df.to_dict(orient="records")]

    def get_inventory_balance(self, facility_id: str, item_id: str) -> List[InventoryBalanceContract]:
        df = self.balance[self.balance["facility_id"] == facility_id]
        return [InventoryBalanceContract(**row) for row in df.to_dict(orient="records")]


class CSVDemandRepository:

    def __init__(self):
        self.consumption = pd.read_csv(DATA_DIR / "consumption.csv", parse_dates=["service_date"])
        self.forecast = pd.read_csv(DATA_DIR / "forecast_daily.csv", parse_dates=["forecast_period"])

        validate_df(self.consumption, "consumption")
        validate_df(self.forecast, "forecast_daily")

    def get_consumption(
        self,
        facility_id: str,
        item_id: str,
        start_date: date,
        end_date: date,
    ) -> List[ConsumptionDailyContract]:

        df = self.consumption[
            (self.consumption["facility_id"] == facility_id)
            & (self.consumption["item_id"] == item_id)
            & (self.consumption["service_date"] >= start_date)
            & (self.consumption["service_date"] <= end_date)
        ]

        return [ConsumptionDailyContract(**row) for row in df.to_dict(orient="records")]

    def get_forecast(
        self,
        facility_id: str,
        item_id: str,
        start_date: date,
        end_date: date,
        run_id: str | None = None,
    ) -> List[ForecastDailyContract]:

        df = self.forecast[
            (self.forecast["facility_id"] == facility_id)
            & (self.forecast["item_id"] == item_id)
            & (self.forecast["forecast_period"] >= start_date)
            & (self.forecast["forecast_period"] <= end_date)
        ]

        if run_id:
            df = df[df["run_id"] == run_id]

        return [ForecastDailyContract(**row) for row in df.to_dict(orient="records")]
