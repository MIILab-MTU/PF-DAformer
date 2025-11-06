import sys
import torch

sys.path.append("../")

from typing import Dict
from monai.data import DataLoader
from augmentations.augmentations import build_augmentations


######################################################################
def build_dataset(dataset_type: str, dataset_args: Dict):
    if dataset_type == "qct_seg":
        from .uci_seg import UCIDataset
        dataset = UCIDataset(
            root_dir=dataset_args["root"],
            is_train=dataset_args["train"],
            # transform=build_augmentations(False),
            # transform=False
            # fold_id=dataset_args["fold_id"],
        )
        return dataset
    
    else:
        raise ValueError(
            "only qct_seg currently supported!"
        )



def pad_collate_fn(batch):
    # Flatten if needed
    if isinstance(batch[0], (list, tuple)) and isinstance(batch[0][0], dict):
        batch = [b[0] for b in batch]

    images = []
    labels = []

    for i in range(len(batch)):
        images.append(batch[i]["image"])
        labels.append(batch[i]["label"])

    max_depth = max(img.shape[-1] for img in images)  # Find max depth (D)

    padded_images = []
    padded_labels = []
    for i in range(len(images)):
        img = images[i]
        lbl = labels[i]
        pad_d = max_depth - img.shape[-1]
        padded_images.append(torch.nn.functional.pad(img, (0, pad_d)))
        padded_labels.append(torch.nn.functional.pad(lbl, (0, pad_d)))

    return {
        "image": torch.stack(padded_images),
        "label": torch.stack(padded_labels),
    }

######################################################################
def build_dataloader(
    dataset, dataloader_args: Dict, config: Dict = None, train: bool = True
) -> DataLoader:
    """builds the dataloader for given dataset

    Args:
        dataset (_type_): _description_
        dataloader_args (Dict): _description_
        config (Dict, optional): _description_. Defaults to None.
        train (bool, optional): _description_. Defaults to True.

    Returns:
        DataLoader: _description_
    """
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=dataloader_args["batch_size"],
        shuffle=dataloader_args["shuffle"],
        num_workers=dataloader_args["num_workers"],
        drop_last=dataloader_args["drop_last"],
        pin_memory=True,
        collate_fn=pad_collate_fn
    )
    return dataloader
