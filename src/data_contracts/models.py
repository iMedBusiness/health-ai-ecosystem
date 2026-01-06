from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date
from enum import Enum


# =========================
# ENUMS (shared, canonical)
# =========================

class ItemCategory(str, Enum):
    drug = "drug"
    consumable = "consumable"
    equipment = "equipment"


class FacilityType(str, Enum):
    warehouse = "warehouse"
    clinic = "clinic"
    hub = "hub"
    port = "port"
    airport = "airport"


class InventoryStatus(str, Enum):
    usable = "usable"
    expired = "expired"
    quarantine = "quarantine"


# =========================
# MASTER DATA CONTRACTS
# =========================

class ItemContract(BaseModel):
    item_id: str
    item_code: str
    item_name: str
    category: ItemCategory
    dosage_form: Optional[str]
    strength: Optional[str]
    unit: str
    shelf_life_months: Optional[int]
    cold_chain_required: bool
    hazardous: bool
    essential_medicine_flag: bool


class LocationContract(BaseModel):
    location_id: str
    name: str
    type: FacilityType
    country: str
    region: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    security_risk_level: Optional[float]
    road_access: Optional[bool]


class FacilityContract(BaseModel):
    facility_id: str
    facility_name: str
    facility_type: FacilityType
    location_id: str
    cold_storage_capacity: Optional[float]
    backup_power: bool
    catchment_population: Optional[int]


class SupplierContract(BaseModel):
    supplier_id: str
    name: str
    location_id: str
    avg_lead_time_days: int
    lead_time_std: Optional[float]
    cold_chain_capable: bool
    risk_score: Optional[float]
    certifications: Optional[List[str]]


# =========================
# LOT & INVENTORY
# =========================

class LotContract(BaseModel):
    lot_id: str
    item_id: str
    batch_no: str
    mfg_date: Optional[date]
    expiry_date: date
    supplier_id: Optional[str]
    donor_id: Optional[str]
    unit_cost: Optional[float]
    received_date: date


class InventoryBalanceContract(BaseModel):
    lot_id: str
    facility_id: str
    quantity_on_hand: float
    status: InventoryStatus


# =========================
# DEMAND & FORECAST
# =========================

class ConsumptionDailyContract(BaseModel):
    facility_id: str
    item_id: str
    service_date: date
    qty_used: float
    patient_count: Optional[int]
    program: Optional[str]


class ForecastDailyContract(BaseModel):
    facility_id: str
    item_id: str
    forecast_period: date
    forecast_qty: float
    confidence_score: Optional[float]
    run_id: Optional[str]  # scenario / model run
