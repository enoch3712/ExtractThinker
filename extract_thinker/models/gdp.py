from typing import List, Optional
from pydantic import Field
from .contract import Contract

class CountryGDP(Contract):
    country: str
    location: str
    gdp_real: float

class ProvinceData(Contract):
    name: str
    gdp_million: Optional[float] = Field(None, description="GDP (€ million)")
    share_in_eu27_gdp: Optional[float] = Field(None, description="Share in EU27/national GDP (%)")
    gdp_per_capita: Optional[int] = Field(None, description="GDP per capita (€)")

class RegionData(Contract):
    region: str
    gdp_million: Optional[float] = Field(None, description="GDP (€ million)")
    share_in_eu27_gdp: Optional[float] = Field(None, description="Share in EU27/national GDP (%)")
    gdp_per_capita: Optional[int] = Field(None, description="GDP per capita (€)")
    provinces: List[ProvinceData] = Field(default_factory=list)

class CountryData(Contract):
    country: str
    total_gdp_million: Optional[float] = Field(None, description="Total GDP (€ million)")
    regions: List[RegionData] = Field(default_factory=list, description="Make sure to ignore Extra-regio*/Extra-region")

class EUData(Contract):
    eu_total_gdp_million_27: float = Field(None, description="EU27 Total GDP (€ million)")
    eu_total_gdp_million_28: float = Field(None, description="EU28 Total GDP (€ million)")
    countries: List[CountryData] 