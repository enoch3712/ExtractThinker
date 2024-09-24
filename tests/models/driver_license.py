from extract_thinker.models.contract import Contract

class IdentificationContract(Contract):
    name: str
    age: int
    id_number: str

class DriverLicense(Contract):
    name: str
    age: int
    license_number: str