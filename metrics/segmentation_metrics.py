import torch
import numpy as np
from monai.metrics import DiceMetric, compute_hausdorff_distance, compute_average_surface_distance
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, Activations, AsDiscrete
from monai.data import decollate_batch
from medpy.metric.binary import precision, recall, hd, hd95, asd
import wandb

################################################################################
class SlidingWindowInference:
    def __init__(self, roi: tuple, sw_batch_size: int):
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.post_transform = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        self.sw_batch_size = sw_batch_size
        self.roi = roi

    def __call__(self, val_inputs: torch.Tensor, val_labels: torch.Tensor, model: torch.nn.Module):
        self.dice_metric.reset()
        logits = sliding_window_inference(
            inputs=val_inputs,
            roi_size=self.roi,
            sw_batch_size=self.sw_batch_size,
            predictor=model,
            overlap=0.5,
        )

        val_labels_list = decollate_batch(val_labels)
        val_outputs_list = decollate_batch(logits)
        val_output_convert = [self.post_transform(pred) for pred in val_outputs_list]

        self.dice_metric(y_pred=val_output_convert, y=val_labels_list)
        dice_score = self.dice_metric.aggregate().item() * 100

        all_precision, all_recall, all_hd, all_hd95, all_asd = [], [], [], [], []

        for pred, label in zip(val_output_convert, val_labels_list):
            p = pred.squeeze().cpu().numpy().astype(bool)
            l = label.squeeze().cpu().numpy().astype(bool)

            try:
                all_precision.append(precision(p, l))
                all_recall.append(recall(p, l))
                all_hd.append(hd(p, l))
                all_hd95.append(hd95(p, l))
                all_asd.append(asd(p, l))
            except Exception:
                all_precision.append(0.0)
                all_recall.append(0.0)
                all_hd.append(0.0)
                all_hd95.append(0.0)
                all_asd.append(0.0)

        return {
            "dice": dice_score,
            "precision": np.mean(all_precision) * 100,
            "recall": np.mean(all_recall) * 100,
            "hd": np.mean(all_hd),
            "hd95": np.mean(all_hd95),
            "asd": np.mean(all_asd),
        }