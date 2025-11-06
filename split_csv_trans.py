import os
import random
import pandas as pd

# Constants
# DATA_DIR = "/home/rochak/Desktop/QCT/QCT_Segmentation/data/data_uci_converted"
# PREPROCESSED_REL_PATH = "../../../data/data_uci_converted/preprocessed_training_data"

# Constants
DATA_DIR = "/home/rochak/Desktop/QCT/QCT_Segmentation/data/data_tulane_converted"
PREPROCESSED_REL_PATH = "../../../data/data_tulane_converted/preprocessed_training_data"

# Extract sample IDs (e.g., BH6002)
sample_ids = sorted([f.replace("_x.npy", "") for f in os.listdir(DATA_DIR) if f.endswith("_x.npy")])
random.shuffle(sample_ids)

# Split 80/20
split_idx = int(0.8 * len(sample_ids))
train_ids = sample_ids[:split_idx]
val_ids = sample_ids[split_idx:]

# Build DataFrames
def make_df(ids):
    return pd.DataFrame({
        "data_path": [os.path.join(PREPROCESSED_REL_PATH, sid) for sid in ids],
        "case_name": ids
    })

# Save CSVs
make_df(train_ids).to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
make_df(val_ids).to_csv(os.path.join(DATA_DIR, "validation.csv"), index=False)
