from abc import ABC, abstractmethod
from datetime import date
from typing import List

from data_contracts.models import (
    ItemContract,
    FacilityContract,
    SupplierContract,
    LotContract,
    InventoryBalanceContract,
    ConsumptionDailyContract,
    ForecastDailyContract,
)


class InventoryRepository(ABC):

    @abstractmethod
    def get_lots(
        self, facility_id: str, item_id: str
    ) -> List[LotContract]:
        pass

    @abstractmethod
    def get_inventory_balance(
        self, facility_id: str, item_id: str
    ) -> List[InventoryBalanceContract]:
        pass


class DemandRepository(ABC):

    @abstractmethod
    def get_consumption(
        self,
        facility_id: str,
        item_id: str,
        start_date: date,
        end_date: date,
    ) -> List[ConsumptionDailyContract]:
        pass

    @abstractmethod
    def get_forecast(
        self,
        facility_id: str,
        item_id: str,
        start_date: date,
        end_date: date,
        run_id: str | None = None,
    ) -> List[ForecastDailyContract]:
        pass


class MasterDataRepository(ABC):

    @abstractmethod
    def get_item(self, item_id: str) -> ItemContract:
        pass

    @abstractmethod
    def get_facility(self, facility_id: str) -> FacilityContract:
        pass

    @abstractmethod
    def get_supplier(self, supplier_id: str) -> SupplierContract:
        pass
