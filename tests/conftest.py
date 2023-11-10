import pandas as pd
import pytest
from faker import Faker

FAKE_ROWS = 100


@pytest.fixture(scope="session")
def synthetic_data() -> pd.DataFrame:
    """Generate synthetic data"""
    fake = Faker()
    Faker.seed(42)
    df = {
        "datetime": [fake.date() for _ in range(FAKE_ROWS)],
        "county": [fake.pyint(min_value=0, max_value=15) for _ in range(FAKE_ROWS)],
        "is_business": [fake.pyint(min_value=0, max_value=1) for _ in range(FAKE_ROWS)],
        "product_type": [
            fake.pyint(min_value=0, max_value=4) for _ in range(FAKE_ROWS)
        ],
        "is_consumption": [
            fake.pyint(min_value=0, max_value=1) for _ in range(FAKE_ROWS)
        ],
        "target": [fake.pyfloat(min_value=0, max_value=100) for _ in range(FAKE_ROWS)],
    }
    df = pd.DataFrame(data=df)
    df["datetime"] = pd.to_datetime(df["datetime"])
    
    return pd.DataFrame(data=df)
