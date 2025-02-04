from typing import List, Optional
from pydantic import Field
from extract_thinker.models.contract import Contract

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
    country: str = Field(
        ...,
        description="Country name as it appears in the PDF. IMPORTANT: Extract this value from every page and aggregate unique entries, not just the first occurrence."
    )
    total_gdp_million: Optional[float] = Field(
        None,
        description="Total GDP (€ million) for the country, using the value from any page in the document."
    )
    regions: List[RegionData] = Field(
        default_factory=list,
        description="List of regions for the country. Aggregate all regions from every page and ignore any formatting variations like 'Extra-regio*/Extra-region'."
    )

class EUData(Contract):
    thinking: str = Field(None, description="Think step by step. You have 2 pages dont forget to add them.")
    eu_total_gdp_million_27: float = Field(None, description="EU27 Total GDP (€ million)")
    eu_total_gdp_million_28: float = Field(None, description="EU28 Total GDP (€ million)")
    countries: List[CountryData] = Field(None, description="List of countries. Make sure you add all countries of every page, not just the first one.")