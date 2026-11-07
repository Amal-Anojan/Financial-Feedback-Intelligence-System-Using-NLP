from pathlib import Path

import pandas as pd
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

dataset = load_dataset("mteb/banking77")
df_train = dataset["train"].to_pandas()
label_feature = dataset["train"].features.get("label")
for split in ("train", "test"):
    if "label_text" not in dataset[split].column_names and not hasattr(label_feature, "int2str"):
        raise ValueError("Banking77 source lacks label names. Do not guess the integer-to-label mapping.")
if "label_text" not in df_train:
    df_train["label_text"] = df_train["label"].map(label_feature.int2str)
df_train.to_csv(RAW_DIR / "banking77_train.csv", index=False)

df_test = dataset["test"].to_pandas()
if "label_text" not in df_test:
    df_test["label_text"] = df_test["label"].map(label_feature.int2str)
df_test.to_csv(RAW_DIR / "banking77_test.csv", index=False)

print(f"Saved {len(df_train):,} training rows to {RAW_DIR / 'banking77_train.csv'}")
print(f"Saved {len(df_test):,} test rows to {RAW_DIR / 'banking77_test.csv'}")
