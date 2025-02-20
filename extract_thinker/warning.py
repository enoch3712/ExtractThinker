import warnings

def filter_pydantic_v2_warnings():
    warnings.filterwarnings(
        "ignore",
        message="Valid config keys have changed in V2:*"
    )

filter_pydantic_v2_warnings()