# # # import os
# # # import torch
# # # import argparse
# # # from glob import glob
# # # from tqdm import tqdm
# # # import matplotlib.pyplot as plt
# # # import pandas as pd
# # # import numpy as np

# # # from architectures.segformer3d import SegFormer3D
# # # from losses.losses import build_loss_fn
# # # from metrics.segmentation_metrics import build_metric_fn


# # # # ----------------- Dice Score -----------------
# # # def dice_score(pred_logits, target):
# # #     probs = torch.sigmoid(pred_logits)
# # #     preds = (probs > 0.5).float()
    
# # #     target = target.float()

# # #     intersection = (preds * target).sum()
# # #     union = preds.sum() + target.sum()

# # #     dice = (2. * intersection + 1e-5) / (union + 1e-5)
# # #     return dice


# # # # ----------------- Evaluation Function -----------------
# # # def evaluate_model(config):
# # #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # #     # Load model
# # #     model = SegFormer3D()
# # #     model.load_state_dict(torch.load(config["checkpoint_path"], map_location=device))
# # #     model.to(device)
# # #     model.eval()

# # #     # Load loss function
# # #     criterion = build_loss_fn(
# # #         loss_type=config["loss_fn"]["loss_type"],
# # #         loss_args=config["loss_fn"].get("loss_args", {})
# # #     )

# # #     # Load CSV with cases
# # #     df = pd.read_csv(config["csv_path"])

# # #     total_dice = 0.0
# # #     losses = []

# # #     for i, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
# # #         case_dir = row["data_path"]
# # #         case_id = row["case_name"]

# # #         input_path = os.path.join(case_dir, f"{case_id}_modalities.pt")
# # #         label_path = os.path.join(case_dir, f"{case_id}_label.pt")
# """  """
# # #         x = torch.load(input_path).float().to(device)  # (1, D, H, W)
# # #         y = torch.load(label_path).long().to(device)   # (1, D, H, W)
# # #         x = x.unsqueeze(0)  # (1, 1, D, H, W)

# # #         with torch.no_grad():
# # #             logits = model(x)  # (1, 1, D, H, W)

# # #             # Apply sigmoid and threshold for binary prediction
# # #             probs = torch.sigmoid(logits)
# # #             pred = (probs > 0.5).long().squeeze(1)  # (1, D, H, W)

# # #             y_input = y.unsqueeze(1)  # Match shape (1, 1, D, H, W) for loss
# # #             loss = criterion(logits, y_input)
# # #             losses.append(loss.item())

# # #             dice = dice_score(pred, y)
# # #             total_dice += dice.item()

# # #             print(f"{case_id} - Dice: {dice.item():.4f} | Loss: {loss.item():.4f}")

# # #             # Visualization
# # #             os.makedirs("predictions", exist_ok=True)

# # #             # Extract 91st slice
# # #             slice_idx = 100
# # #             input_img = x.squeeze().cpu().numpy()       # (D, H, W)
# # #             gt_mask = y.squeeze().cpu().numpy()         # (D, H, W)
# # #             pred_mask = pred.squeeze().cpu().numpy()    # (D, H, W)

# # #             vol_slice = input_img[:, :, slice_idx]
# # #             gt_slice = gt_mask[:, :, slice_idx]
# # #             pred_slice = pred_mask[:, :, slice_idx]

# # #             fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# # #             axs[0].imshow(vol_slice, cmap="gray")
# # #             axs[0].set_title(f"Input Slice {slice_idx}")
# # #             axs[0].axis("off")

# # #             axs[1].imshow(gt_slice, cmap="gray")
# # #             axs[1].set_title("Ground Truth")
# # #             axs[1].axis("off")

# # #             axs[2].imshow(pred_slice, cmap="gray")
# # #             axs[2].set_title("Predicted")
# # #             axs[2].axis("off")

# # #             plt.tight_layout()
# # #             out_path = f"predictions/{case_id}_slice_{slice_idx}.png"
# # #             plt.savefig(out_path, dpi=150)
# # #             plt.close()

# # #             print(f"📸 Saved visualization to {out_path}")

# # #     # Summary
# # #     avg_dice = total_dice / len(df)
# # #     avg_loss = sum(losses) / len(losses)

# # #     print("\n✅ Evaluation Complete:")
# # #     print(f"   Average Dice Score: {avg_dice:.4f}")
# # #     print(f"   Average Loss:       {avg_loss:.4f}")


# # # # ----------------- Main Entry -----------------
# # # if __name__ == "__main__":
# # #     parser = argparse.ArgumentParser(description="Evaluate SegFormer3D Model")
# # #     parser.add_argument(
# # #         "--config",
# # #         type=str,
# # #         default="/home/rochak/Desktop/QCT/QCT_Segmentation/SegFormer3D/experiments/brats_2017/template_experiment/config.yaml",
# # #         help="Path to config YAML"
# # #     )
# # #     args = parser.parse_args()

# # #     import yaml
# # #     with open(args.config, "r") as f:
# # #         full_config = yaml.safe_load(f)
# # #         config = full_config["evaluate_parameters"]

# # #     evaluate_model(config)




















# import os
# import torch
# import argparse
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import yaml

# from architectures.segformer3d import SegFormer3D
# from losses.losses import build_loss_fn
# from metrics.segmentation_metrics import SlidingWindowInference


# def dice_score(pred, target, eps=1e-5):
#     intersection = (pred == target).float().sum()
#     return (2. * intersection + eps) / (pred.numel() + target.numel() + eps)


# class Evaluator:
#     def __init__(self, config):
#         self.config = config
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model = SegFormer3D().to(self.device)
#         print(config["checkpoint_path"])
#         self.model.load_state_dict(torch.load(config["checkpoint_path"], map_location=self.device))
#         self.model.eval()

#         self.criterion = build_loss_fn(
#             loss_type=config["loss_fn"]["loss_type"],
#             loss_args=config["loss_fn"].get("loss_args", {})
#         )
#         print(config)

#         self.sliding_window_inference = SlidingWindowInference(
#             config["sliding_window_inference"]["roi"],
#             config["sliding_window_inference"]["sw_batch_size"],
#         )

#         self.use_ema = config.get("use_ema", False)
#         self.ema_model = None  # If you use EMA, load or assign it here

#         self.calculate_metrics = True

#     def _calc_dice_metric(self, data, labels, use_ema: bool) -> float:
#         if use_ema:
#             avg_dice_score = self.sliding_window_inference(data, labels, self.ema_model)
#         else:
#             avg_dice_score = self.sliding_window_inference(data, labels, self.model)
#         return avg_dice_score

#     def evaluate(self):
#         df = pd.read_csv(self.config["csv_path"])
#         total_dice = 0.0
#         losses = []

#         for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
#             case_dir = row["data_path"]
#             case_id = row["case_name"]

#             input_path = os.path.join(case_dir, f"{case_id}_modalities.pt")
#             label_path = os.path.join(case_dir, f"{case_id}_label.pt")

#             x = torch.load(input_path).float().to(self.device)  # (1, D, H, W)
#             y = torch.load(label_path).long().to(self.device)   # (1, D, H, W)
#             x = x.unsqueeze(0)  # (1, 1, D, H, W)

#             with torch.no_grad():
#                 logits = self.model(x)  # (1, 1, D, H, W)

#                 probs = torch.sigmoid(logits)
#                 pred = (probs > 0.5).long().squeeze(1)

#                 y_input = y.unsqueeze(1)
#                 loss = self.criterion(logits, y_input)
#                 losses.append(loss.item())

#                 if self.calculate_metrics:
#                     mean_dice = self._calc_dice_metric(x, y.unsqueeze(1), self.use_ema)
#                     total_dice += mean_dice

#                 print(f"{case_id} - Dice: {mean_dice:.4f} | Loss: {loss.item():.4f}")

#                 # Visualization
#                 os.makedirs("predictions", exist_ok=True)
#                 slice_idx = 100
#                 vol_slice = x.squeeze().cpu().numpy()[:, :, slice_idx]
#                 gt_slice = y.squeeze().cpu().numpy()[:, :, slice_idx]
#                 pred_slice = pred.squeeze().cpu().numpy()[:, :, slice_idx]

#                 fig, axs = plt.subplots(1, 3, figsize=(12, 4))
#                 axs[0].imshow(vol_slice, cmap="gray")
#                 axs[0].set_title(f"Input Slice {slice_idx}")
#                 axs[0].axis("off")

#                 axs[1].imshow(gt_slice, cmap="gray")
#                 axs[1].set_title("Ground Truth")
#                 axs[1].axis("off")

#                 axs[2].imshow(pred_slice, cmap="gray")
#                 axs[2].set_title("Predicted")
#                 axs[2].axis("off")

#                 out_path = f"predictions/{case_id}_slice_{slice_idx}.png"
#                 plt.tight_layout()
#                 plt.savefig(out_path, dpi=150)
#                 plt.close()
#                 print(f"📸 Saved visualization to {out_path}")

#         avg_dice = total_dice / len(df)
#         avg_loss = sum(losses) / len(losses)

#         print("\n✅ Evaluation Complete:")
#         print(f"   Average Dice Score: {avg_dice:.4f}")
#         print(f"   Average Loss:       {avg_loss:.4f}")


# # ----------------- Main Entry -----------------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Evaluate SegFormer3D Model")
#     parser.add_argument(
#         "--config",
#         type=str,
#         default="/home/rochak/Desktop/QCT/QCT_Segmentation/SegFormer3D/experiments/brats_2017/template_experiment/config.yaml",
#         help="Path to config YAML"
#     )
#     args = parser.parse_args()

#     with open(args.config, "r") as f:
#         full_config = yaml.safe_load(f)
#         config = full_config["evaluate_parameters"]

#     evaluator = Evaluator(config)
#     evaluator.evaluate()


# import os
# import torch
# import argparse
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import yaml

# from architectures.segformer3d import SegFormer3D
# from losses.losses import build_loss_fn
# from metrics.segmentation_metrics import SlidingWindowInference


# class Evaluator:
#     def __init__(self, config):
#         self.config = config
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model = SegFormer3D().to(self.device)
#         print(config["checkpoint_path"])
#         self.model.load_state_dict(torch.load(config["checkpoint_path"], map_location=self.device))
#         self.model.eval()

#         self.criterion = build_loss_fn(
#             loss_type=config["loss_fn"]["loss_type"],
#             loss_args=config["loss_fn"].get("loss_args", {})
#         )

#         self.sliding_window_inference = SlidingWindowInference(
#             config["sliding_window_inference"]["roi"],
#             config["sliding_window_inference"]["sw_batch_size"],
#         )

#         self.use_ema = config.get("use_ema", False)
#         self.ema_model = None  # If you use EMA, load or assign it here

#         self.calculate_metrics = True

#     def _calc_metrics(self, data, labels, use_ema: bool) -> dict:
#         model = self.ema_model if use_ema else self.model
#         return self.sliding_window_inference(data, labels, model)

#     def evaluate(self):
#         df = pd.read_csv(self.config["csv_path"])
#         total_dice = 0.0
#         total_precision = 0.0
#         total_recall = 0.0
#         total_hd = 0.0
#         total_hd95 = 0.0
#         total_asd = 0.0
#         losses = []
#         per_case_metrics = []

#         for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
#             case_dir = row["data_path"]
#             case_id = row["case_name"]

#             input_path = os.path.join(case_dir, f"{case_id}_modalities.pt")
#             label_path = os.path.join(case_dir, f"{case_id}_label.pt")

#             x = torch.load(input_path).float().to(self.device)  # (1, D, H, W)
#             y = torch.load(label_path).long().to(self.device)   # (1, D, H, W)
#             x = x.unsqueeze(0)  # (1, 1, D, H, W)

#             with torch.no_grad():
#                 logits = self.model(x)
#                 probs = torch.sigmoid(logits)
#                 pred = (probs > 0.5).long().squeeze(1)
#                 y_input = y.unsqueeze(1)

#                 loss = self.criterion(logits, y_input)
#                 losses.append(loss.item())

#                 metrics = self._calc_metrics(x, y_input, use_ema=False)
#                 total_dice += metrics["dice"]
#                 total_precision += metrics["precision"]
#                 total_recall += metrics["recall"]
#                 total_hd += metrics["hd"]
#                 total_hd95 += metrics["hd95"]
#                 total_asd += metrics["asd"]

#                 per_case_metrics.append({
#                     "case_id": case_id,
#                     "dice": metrics["dice"],
#                     "precision": metrics["precision"],
#                     "recall": metrics["recall"],
#                     "hd": metrics["hd"],
#                     "hd95": metrics["hd95"],
#                     "asd": metrics["asd"],
#                     "loss": loss.item()
#                 })

#         num_cases = len(df)
#         avg_metrics = {
#             "avg_dice": total_dice / num_cases,
#             "avg_precision": total_precision / num_cases,
#             "avg_recall": total_recall / num_cases,
#             "avg_hd": total_hd / num_cases,
#             "avg_hd95": total_hd95 / num_cases,
#             "avg_asd": total_asd / num_cases,
#             "avg_loss": sum(losses) / num_cases
#         }

#         # Save per-case metrics as CSV
#         result_df = pd.DataFrame(per_case_metrics)
#         save_dir = self.config.get("save_dir", ".")
#         os.makedirs(save_dir, exist_ok=True)
#         output_path = os.path.join(save_dir, "case_based_report.csv")
#         result_df.to_csv(output_path, index=False)

#         # Print summary
#         print("\n✅ Evaluation Complete:")
#         for k, v in avg_metrics.items():
#             print(f"   {k.replace('_', ' ').capitalize()}: {v:.4f}")


# # ----------------- Main Entry -----------------
# if __name__ == "__main__":
#     if __name__ == "__main__":
#         parser = argparse.ArgumentParser(description="Evaluate SegFormer3D Model")
#         parser.add_argument(
#             "--config",
#             type=str,
#             default="/home/rochak/Desktop/QCT/QCT_Segmentation/SegFormer3D/experiments/brats_2017/template_experiment/config.yaml",
#             help="Path to config YAML"
#         )
#         args = parser.parse_args()

#     with open(args.config, "r") as f:
#         full_config = yaml.safe_load(f)
#         config = full_config["evaluate_parameters"]

#     evaluator = Evaluator(config)
#     evaluator.evaluate()





# import os
# import torch
# import argparse
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import yaml

# from architectures.segformer3d import SegFormer3D
# from losses.losses import build_loss_fn
# from metrics.segmentation_metrics import SlidingWindowInference


# class Evaluator:
#     def __init__(self, config):
#         self.config = config
#         self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#         # Initialize model
#         self.model = SegFormer3D(
#         in_channels=full_config["model_parameters"]["in_channels"],
#         sr_ratios=full_config["model_parameters"]["sr_ratios"],
#         embed_dims=full_config["model_parameters"]["embed_dims"],
#         patch_kernel_size=full_config["model_parameters"]["patch_kernel_size"],
#         patch_stride=full_config["model_parameters"]["patch_stride"],
#         patch_padding=full_config["model_parameters"]["patch_padding"],
#         mlp_ratios=full_config["model_parameters"]["mlp_ratios"],
#         num_heads=full_config["model_parameters"]["num_heads"],
#         depths=full_config["model_parameters"]["depths"],
#         decoder_head_embedding_dim=full_config["model_parameters"][
#             "decoder_head_embedding_dim"
#         ],
#         num_classes=full_config["model_parameters"]["num_classes"],
#         decoder_dropout=full_config["model_parameters"]["decoder_dropout"],
#         ).to(self.device)
#         #self.model = SegFormer3D().to(self.device)

#         # Load checkpoint
#         print(f"[info] -- Loading checkpoint from: {config['checkpoint_path']}")
#         checkpoint = torch.load(config["checkpoint_path"], map_location=self.device)
#         # Print a few keys from the checkpoint
#         print("\n[Debug] 🔑 Checkpoint keys sample:")
#         for k in list(checkpoint["model"].keys()):
#             print("  -", k)

#         # Print a few keys from the current model
#         print("\n[Debug] 🔑 Model keys sample:")
#         for k in list(self.model.state_dict().keys()):
#             print("  -", k)
#         self.model.load_state_dict(checkpoint["model"])

#         self.model.eval()

#         # Optional: print best metrics from checkpoint
#         self.best_val_dice = checkpoint.get("best_val_dice", None)
#         if self.best_val_dice is not None:
#             print(f"[info] -- Loaded model with best validation Dice: {self.best_val_dice:.4f}")

#         # Loss function
#         self.criterion = build_loss_fn(
#             loss_type=config["loss_fn"]["loss_type"],
#             loss_args=config["loss_fn"].get("loss_args", {})
#         )

#         # Sliding window inference setup
#         self.sliding_window_inference = SlidingWindowInference(
#             config["sliding_window_inference"]["roi"],
#             config["sliding_window_inference"]["sw_batch_size"],
#         )

#         self.use_ema = config.get("use_ema", False)
#         self.ema_model = None  # Optional: load EMA weights if needed

#         self.calculate_metrics = True

#     def _calc_metrics(self, data, labels, use_ema: bool) -> dict:
#         model = self.ema_model if use_ema else self.model
#         return self.sliding_window_inference(data, labels, model)

#     def evaluate(self):
#         df = pd.read_csv(self.config["csv_path"])
#         total_dice = 0.0
#         total_precision = 0.0
#         total_recall = 0.0
#         total_hd = 0.0
#         total_hd95 = 0.0
#         total_asd = 0.0
#         losses = []
#         per_case_metrics = []

#         for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
#             case_dir = row["data_path"]
#             case_id = row["case_name"]

#             input_path = os.path.join(case_dir, f"{case_id}_modalities.pt")
#             label_path = os.path.join(case_dir, f"{case_id}_label.pt")

#             x = torch.load(input_path).float().to(self.device)  # (1, D, H, W)
#             y = torch.load(label_path).long().to(self.device)   # (1, D, H, W)
#             x = x.unsqueeze(0)  # (1, 1, D, H, W)

#             with torch.no_grad():
#                 logits = self.model(x)
#                 probs = torch.sigmoid(logits)
#                 pred = (probs > 0.5).long().squeeze(1)
#                 y_input = y.unsqueeze(1)

#                 loss = self.criterion(logits, y_input)
#                 losses.append(loss.item())

#                 metrics = self._calc_metrics(x, y_input, use_ema=False)
#                 total_dice += metrics["dice"]
#                 total_precision += metrics["precision"]
#                 total_recall += metrics["recall"]
#                 total_hd += metrics["hd"]
#                 total_hd95 += metrics["hd95"]
#                 total_asd += metrics["asd"]

#                 per_case_metrics.append({
#                     "case_id": case_id,
#                     "dice": metrics["dice"],
#                     "precision": metrics["precision"],
#                     "recall": metrics["recall"],
#                     "hd": metrics["hd"],
#                     "hd95": metrics["hd95"],
#                     "asd": metrics["asd"],
#                     "loss": loss.item()
#                 })

#         num_cases = len(df)
#         avg_metrics = {
#             "avg_dice": total_dice / num_cases,
#             "avg_precision": total_precision / num_cases,
#             "avg_recall": total_recall / num_cases,
#             "avg_hd": total_hd / num_cases,
#             "avg_hd95": total_hd95 / num_cases,
#             "avg_asd": total_asd / num_cases,
#             "avg_loss": sum(losses) / num_cases
#         }

#         # Save per-case metrics as CSV
#         #result_df = pd.DataFrame(per_case_metrics)
#         #save_dir = self.config.get("save_dir", ".")
#         #os.makedirs(save_dir, exist_ok=True)
#         #output_path = os.path.join(save_dir, "case_based_report.csv")
#         #result_df.to_csv(output_path, index=False)

#         # Print summary
#         print("\n✅ Evaluation Complete:")
#         for k, v in avg_metrics.items():
#             print(f"   {k.replace('_', ' ').capitalize()}: {v:.4f}")


# # ----------------- Main Entry -----------------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Evaluate SegFormer3D Model")
#     parser.add_argument(
#         "--config",
#         type=str,
#         default="/home/rochak/Desktop/QCT/QCT_Segmentation/SegFormer3D/experiments/brats_2017/template_experiment/config.yaml",
#         help="Path to config YAML"
#     )
#     args = parser.parse_args()

#     with open(args.config, "r") as f:
#         full_config = yaml.safe_load(f)
#         config = full_config["evaluate_parameters"]

#     evaluator = Evaluator(config)
#     evaluator.evaluate()




import os
import torch
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import yaml

from architectures.segformer3d import SegFormer3D
from architectures.transUNET3D import TransUNet3D
from losses.losses import build_loss_fn
from metrics.segmentation_metrics import SlidingWindowInference


class Evaluator:
    def __init__(self, config, full_config):
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Initialize model
        # self.model = SegFormer3D(
        #     in_channels=full_config["model_parameters"]["in_channels"],
        #     sr_ratios=full_config["model_parameters"]["sr_ratios"],
        #     embed_dims=full_config["model_parameters"]["embed_dims"],
        #     patch_kernel_size=full_config["model_parameters"]["patch_kernel_size"],
        #     patch_stride=full_config["model_parameters"]["patch_stride"],
        #     patch_padding=full_config["model_parameters"]["patch_padding"],
        #     mlp_ratios=full_config["model_parameters"]["mlp_ratios"],
        #     num_heads=full_config["model_parameters"]["num_heads"],
        #     depths=full_config["model_parameters"]["depths"],
        #     decoder_head_embedding_dim=full_config["model_parameters"]["decoder_head_embedding_dim"],
        #     num_classes=full_config["model_parameters"]["num_classes"],
        #     decoder_dropout=full_config["model_parameters"]["decoder_dropout"],
        # ).to(self.device)

        self.model = TransUNet3D(
            in_channels=1,
            n_classes=1,
            base_channels=16,
            img_size=192,
            patch_size=8,
            hidden_size=512,
            num_layers=6,
            num_heads=8,
            mlp_dim=2048,
            use_domain_adapt = True
        ).to(self.device)

        print(f"[info] -- Loading checkpoint from: {config['checkpoint_path']}")
        checkpoint = torch.load(config["checkpoint_path"], map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

        if checkpoint.get("best_val_dice") is not None:
            print(f"[info] -- Loaded model with best validation Dice: {checkpoint['best_val_dice']:.4f}")

        self.criterion = build_loss_fn(
            loss_type=config["loss_fn"]["loss_type"],
            loss_args=config["loss_fn"].get("loss_args", {})
        )

        self.sliding_window_inference = SlidingWindowInference(
            config["sliding_window_inference"]["roi"],
            config["sliding_window_inference"]["sw_batch_size"],
        )

        self.use_ema = config.get("use_ema", False)
        self.ema_model = None
        self.save_predictions = config.get("save_predictions", False)
        self.sample_n = config.get("sample_n", None)
        self.run_folder = None

        if self.save_predictions:
            base_dir = config.get("save_dir", "./runs")
            run_id = 0
            while os.path.exists(os.path.join(base_dir, f"run_{run_id}")):
                run_id += 1
            self.run_folder = os.path.join(base_dir, f"run_{run_id}")
            os.makedirs(self.run_folder, exist_ok=True)
            print(f"[info] -- Saving predictions to: {self.run_folder}")

    def _calc_metrics(self, data, labels, use_ema: bool) -> dict:
        model = self.ema_model if use_ema else self.model
        return self.sliding_window_inference(data, labels, model)

    def evaluate(self):
        df = pd.read_csv(self.config["csv_path"])

        if self.sample_n is not None:
            df = df.sample(n=min(self.sample_n, len(df)), random_state=42).reset_index(drop=True)

        total_dice = total_precision = total_recall = 0.0
        total_hd = total_hd95 = total_asd = 0.0
        losses = []
        per_case_metrics = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            case_dir = row["data_path"]
            case_id = row["case_name"]

            input_path = os.path.join(case_dir, f"{case_id}_modalities.pt")
            label_path = os.path.join(case_dir, f"{case_id}_label.pt")

            x = torch.load(input_path).float().to(self.device)
            y = torch.load(label_path).long().to(self.device)
            x = x.unsqueeze(0)  # (1, 1, D, H, W)

            with torch.no_grad():
                logits = self.model(x)
                probs = torch.sigmoid(logits)
                pred = (probs > 0.5).long().squeeze(1)
                y_input = y.unsqueeze(1)

                loss = self.criterion(logits, y_input)
                losses.append(loss.item())

                metrics = self._calc_metrics(x, y_input, use_ema=False)
                total_dice += metrics["dice"]
                total_precision += metrics["precision"]
                total_recall += metrics["recall"]
                total_hd += metrics["hd"]
                total_hd95 += metrics["hd95"]
                total_asd += metrics["asd"]

                per_case_metrics.append({
                    "case_id": case_id,
                    "dice": metrics["dice"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "hd": metrics["hd"],
                    "hd95": metrics["hd95"],
                    "asd": metrics["asd"],
                    "loss": loss.item()
                })

                # Save prediction
                if self.save_predictions:
                    case_folder = os.path.join(self.run_folder, case_id)
                    os.makedirs(case_folder, exist_ok=True)
                    torch.save(pred.cpu(), os.path.join(case_folder, f"{case_id}_label.pt"))

        num_cases = len(df)
        avg_metrics = {
            "avg_dice": total_dice / num_cases,
            "avg_precision": total_precision / num_cases,
            "avg_recall": total_recall / num_cases,
            "avg_hd": total_hd / num_cases,
            "avg_hd95": total_hd95 / num_cases,
            "avg_asd": total_asd / num_cases,
            # "avg_loss": sum(losses) / num_cases
        }

        # Save per-case metrics as CSV
        result_df = pd.DataFrame(per_case_metrics)
        save_dir = self.config.get("save_dir", ".")
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, "case_based_report_da_tulane_0.6_0.4_full_DA.csv")
        result_df.to_csv(output_path, index=False)

        print("\nEvaluation Complete:")
        for k, v in avg_metrics.items():
            print(f"   {k.replace('_', ' ').capitalize()}: {v:.4f}")

# ----------------- Main Entry -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SegFormer3D Model")
    parser.add_argument(
        "--config",
        type=str,
        default="/workspace/QCT_Segmentation/SegFormer3D/experiments/brats_2017/template_experiment/config.yaml",
        help="Path to config YAML"
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        default=True,
        help="Save predicted outputs as .pt"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./runs",
        help="Base directory to store prediction outputs"
    )
    parser.add_argument(
        "--sample_n",
        type=int,
        default=300,
        help="Number of cases to randomly sample from CSV"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        full_config = yaml.safe_load(f)
        config = full_config["evaluate_parameters"]

    config["save_predictions"] = args.save_predictions
    config["save_dir"] = args.save_dir
    config["sample_n"] = args.sample_n

    evaluator = Evaluator(config, full_config)
    evaluator.evaluate()
