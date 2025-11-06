import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# === Configurations ===
PREDICTIONS_CSV = "/workspace/QCT_Segmentation/pfdaformer/runs/run_0/all_cases.csv"
GROUND_TRUTH_CSV = "/workspace/QCT_Segmentation/pfdaformer/data/tulane_qct/validation.csv"
PREDICTIONS_DIR = "/workspace/QCT_Segmentation/pfdaformer/runs/run_0/"
VISUALIZATION_DIR = "/workspace/QCT_Segmentation/pfdaformer/visualize/"
ALPHA = 0.4  # overlay transparency

# Create base output directory
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Load CSVs
pred_df = pd.read_csv(PREDICTIONS_CSV)
gt_df = pd.read_csv(GROUND_TRUTH_CSV)

# Match case_ids
matched_cases = pred_df[pred_df["case_name"].isin(gt_df["case_name"])]

# Loop over all matched cases
for _, row in tqdm(matched_cases.iterrows(), total=len(matched_cases), desc="Visualizing all sagittal slices"):
    case_id = row["case_name"]

    pred_path = os.path.join(PREDICTIONS_DIR, case_id, f"{case_id}_label.pt")
    gt_row = gt_df[gt_df["case_name"] == case_id].iloc[0]
    gt_path = os.path.join(gt_row["data_path"], f"{case_id}_label.pt")

    if not os.path.exists(pred_path) or not os.path.exists(gt_path):
        print(f"[warning] Missing data for case: {case_id}")
        continue

    # Load tensors and convert to numpy
    pred = torch.load(pred_path).squeeze().cpu().numpy()  # (D, H, W)
    gt = torch.load(gt_path).squeeze().cpu().numpy()      # (D, H, W)

    if pred.shape != gt.shape:
        print(f"[warning] Shape mismatch in {case_id}")
        continue

    case_dir = os.path.join(VISUALIZATION_DIR, case_id)
    os.makedirs(case_dir, exist_ok=True)

    num_slices = pred.shape[2]  # sagittal (width) direction
    for i in range(num_slices):
        gt_slice = gt[:, :, i]
        pred_slice = pred[:, :, i]

        # Overlay: red = pred, green = GT
        overlay = np.stack([
            pred_slice * 255,       # R
            gt_slice * 255,         # G
            np.zeros_like(gt_slice) # B
        ], axis=-1).astype(np.uint8)

        # Background grayscale from GT
        background = np.stack([gt_slice * 255] * 3, axis=-1).astype(np.uint8)

        # Blend
        blended = ((1 - ALPHA) * background + ALPHA * overlay).astype(np.uint8)

        # Plot and save
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(gt_slice, cmap='gray')
        axes[0].set_title(f"GT - Sagittal {i}")
        axes[0].axis("off")

        axes[1].imshow(pred_slice, cmap='gray')
        axes[1].set_title(f"Pred - Sagittal {i}")
        axes[1].axis("off")

        axes[2].imshow(blended)
        axes[2].set_title(f"Overlay - Sagittal {i}")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(case_dir, f"slice_{i:03d}.png"))
        plt.close()

print(f"\n All sagittal slices saved under: {VISUALIZATION_DIR}")

