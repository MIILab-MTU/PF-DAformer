# 192*192*192 #not

import os
import numpy as np
import torch

# root_dir = "data_uci_converted"
# output_dir = "/home/rochak/Desktop/QCT/QCT_Segmentation/SegFormer3D/data/uci_qct/preprocessed_training_data"

root_dir = "data_tulane_converted"
output_dir = "/workspace/QCT_Segmentation/SegFormer3D/data/tulane_qct/preprocessed_training_data/"

target_shape = (192, 192, 192)

def pad_or_crop(volume, target_shape):
    """
    Pads or crops the 3D volume to the target shape (C, D, H, W).
    """
    current_shape = volume.shape
    pad_width = []
    for i in range(3):
        diff = target_shape[i] - current_shape[i]
        if diff > 0:
            pad_before = diff // 2
            pad_after = diff - pad_before
            pad_width.append((pad_before, pad_after))
        else:
            crop_start = (-diff) // 2
            crop_end = crop_start + target_shape[i]
            volume = volume.take(indices=range(crop_start, crop_end), axis=i)
            pad_width.append((0, 0))
    volume = np.pad(volume, pad_width, mode='constant')
    return volume

def convert_npy_to_brats_format():
    all_files = os.listdir(root_dir)
    case_ids = sorted(set(f.split("_")[0] for f in all_files if f.endswith(".npy")))

    for case_id in case_ids:
        volume_path = os.path.join(root_dir, f"{case_id}_x.npy")
        label_path = os.path.join(root_dir, f"{case_id}_y.npy")

        volume = np.load(volume_path)
        label = np.load(label_path)

        volume = pad_or_crop(volume, target_shape)
        label = pad_or_crop(label, target_shape)

        volume = volume[np.newaxis, ...]  # (1, D, H, W)
        label = label[np.newaxis, ...]

        case_out_dir = os.path.join(output_dir, case_id)
        os.makedirs(case_out_dir, exist_ok=True)

        torch.save(torch.from_numpy(volume), os.path.join(case_out_dir, f"{case_id}_modalities.pt"))
        torch.save(torch.from_numpy(label), os.path.join(case_out_dir, f"{case_id}_label.pt"))

        print(f"[✓] Saved: {case_id}")

if __name__ == "__main__":
    convert_npy_to_brats_format()