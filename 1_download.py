from pathlib import Path

import pandas as pd
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

dataset = load_dataset("mteb/banking77")
df_train = dataset["train"].to_pandas()
df_train.to_csv(RAW_DIR / "banking77_train.csv", index=False)

df_test = dataset["test"].to_pandas()
df_test.to_csv(RAW_DIR / "banking77_test.csv", index=False)

print(f"Saved {len(df_train):,} training rows to {RAW_DIR / 'banking77_train.csv'}")
print(f"Saved {len(df_test):,} test rows to {RAW_DIR / 'banking77_test.csv'}")
