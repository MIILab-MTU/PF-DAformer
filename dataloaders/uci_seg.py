

import os
import torch
import pandas as pd
from torch.utils.data import Dataset


class QCTDataset(Dataset):

    def __init__(
        self, root_dir: str, is_train: bool = True, transform=None, fold_id: int = None
    ):
        """
        root_dir: path to (Data) folder
        is_train: whether or nor it is train or validation
        transform: composition of the pytorch transforms
        fold_id: fold index in kfold dataheld out
        """
        super().__init__()
        if fold_id is not None:
            csv_name = (
                f"train_fold_{fold_id}.csv"
                if is_train
                else f"validation_fold_{fold_id}.csv"
            )
            csv_fp = os.path.join(root_dir, csv_name)
            assert os.path.exists(csv_fp)
        else:
            csv_name = "train.csv" if is_train else "validation.csv"
            csv_fp = os.path.join(root_dir, csv_name)
            assert os.path.exists(csv_fp)
        self.csv = pd.read_csv(csv_fp)
        self.transform = transform

    def __len__(self):
        return self.csv.__len__()

    def __getitem__(self, idx):
        data_path = self.csv["data_path"][idx]
        case_name = self.csv["case_name"][idx]
        volume_fp = os.path.join(data_path, f"{case_name}_modalities.pt")
        label_fp = os.path.join(data_path, f"{case_name}_label.pt")
        

        volume = torch.load(volume_fp)
        label = torch.load(label_fp)

        # print(f"Input{volume.shape}")
        # print(f"OP{label.shape}")

        if not isinstance(volume, torch.Tensor):
            volume = torch.from_numpy(volume)
        if not isinstance(label, torch.Tensor):
            label = torch.from_numpy(label)


        data = {
            "image": volume.float(),               # shape: [1, D, H, W]
            "label": label.squeeze(0).long()       # shape: [D, H, W]
        }
        # print(f"After Input Dice{data["image"].shape}")
        # print(f"After Out Dice{data["label"].shape}")

        if self.transform:
            data = self.transform(data)

        return data

