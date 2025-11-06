import os
import random
import numpy as np
import pandas as pd

def create_train_val_split(
    folder_dir: str,
    append_dir: str = "",
    save_dir: str = "./",
    train_split_perc: float = 0.85,
    val_split_perc: float = 0.15,
):
    assert os.path.exists(folder_dir), f"{folder_dir} does not exist"
    assert 0.0 < train_split_perc < 1.0
    assert 0.0 < val_split_perc < 1.0

    # For reproducibility
    random.seed(0)
    np.random.seed(0)

    # List all case folders like BH6002, BH6004, etc.
    all_cases = [
        d for d in os.listdir(folder_dir)
        if os.path.isdir(os.path.join(folder_dir, d))
    ]
    total_cases = len(all_cases)

    # Generate full relative paths
    all_data_paths = [os.path.join(append_dir, case) for case in all_cases]

    # Shuffle and split
    indices = np.arange(total_cases)
    np.random.shuffle(indices)

    train_idx, val_idx = np.split(
        indices,
        [int(train_split_perc * total_cases)]
    )

    # Get splits
    train_paths = np.array(all_data_paths)[train_idx]
    train_names = np.array(all_cases)[train_idx]
    val_paths = np.array(all_data_paths)[val_idx]
    val_names = np.array(all_cases)[val_idx]

    # Create DataFrames
    train_df = pd.DataFrame({
        "data_path": train_paths,
        "case_name": train_names
    })
    val_df = pd.DataFrame({
        "data_path": val_paths,
        "case_name": val_names
    })

    # Save CSVs
    os.makedirs(save_dir, exist_ok=True)
    train_csv = os.path.join(save_dir, "train.csv")
    val_csv = os.path.join(save_dir, "validation.csv")

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    print(f"[✓] train.csv saved to {train_csv} ({len(train_df)} samples)")
    print(f"[✓] validation.csv saved to {val_csv} ({len(val_df)} samples)")

    # Function to preview shapes of .npy files in a given case folder


if __name__ == "__main__":
    create_train_val_split(
        folder_dir="/workspace/QCT_Segmentation/SegFormer3D/data/tulane_qct/preprocessed_training_data/",
        append_dir="/workspace/QCT_Segmentation/SegFormer3D/data/tulane_qct/preprocessed_training_data/",
        save_dir="/workspace/QCT_Segmentation/SegFormer3D/data/tulane_qct/",
        train_split_perc=0.80,
        val_split_perc=0.20,
    )
