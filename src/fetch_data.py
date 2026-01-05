import pandas as pd
from ucimlrepo import fetch_ucirepo
import os

os.makedirs("data/raw", exist_ok=True)

cdc = fetch_ucirepo(id=891)

df = pd.concat(
    [cdc.data.features, cdc.data.targets],
    axis=1
)

df.to_csv("data/raw/cdc_diabetes.csv", index=False)

print("✅ Dataset saved to data/raw/cdc_diabetes.csv")
