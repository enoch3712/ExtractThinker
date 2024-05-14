from extract_thinker.models.contract import Contract


class DriverLicense(Contract):
    name: str
    age: int
    license_number: str
