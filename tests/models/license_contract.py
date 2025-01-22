from extract_thinker.models.contract import Contract

class PaginateContract(Contract):
    us_state: str
    driver_license_number: str
    expiration_date: str
    registered_for_personal_use: bool 