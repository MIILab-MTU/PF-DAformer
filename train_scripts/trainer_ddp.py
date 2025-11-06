#####################################################################DA#########################################################
import os
import torch
print(f"torch{torch.version}")
import evaluate
from tqdm import tqdm
from typing import Dict
from copy import deepcopy
from termcolor import colored
from torch.utils.data import DataLoader
import monai
from metrics.segmentation_metrics import SlidingWindowInference
import kornia
import wandb
from losses.losses import BinaryCrossEntropyWithLogits,CrossEntropyLoss

###############################################################################################
class FeatureAccumulator:
    #Before MMD Calcuation we accumulated more data as our batch size was low.
    def __init__(self, max_samples=16):
        self.source_feats = []
        self.target_feats = []
        self.max_samples = max_samples

    def update(self, source_feat, target_feat):
        # Detach and move to CPU to save memory
        self.source_feats.append(source_feat.detach().cpu())
        self.target_feats.append(target_feat.detach().cpu())

        if len(self.source_feats) > self.max_samples:
            self.source_feats.pop(0)
        if len(self.target_feats) > self.max_samples:
            self.target_feats.pop(0)

    def get_features(self):
        return torch.cat(self.source_feats), torch.cat(self.target_feats)

    def is_full(self):
        return len(self.source_feats) == self.max_samples

    def reset(self):
        self.source_feats.clear()
        self.target_feats.clear()

################################################################################################

class Segmentation_Trainer:
    def __init__(
        self,
        config: Dict,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        train_dataloader_source: DataLoader,
        train_dataloader_target: DataLoader,
        val_dataloader: DataLoader,
        val_dataloader_target: DataLoader = None,
        warmup_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        training_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        mmd_loss_fn: torch.nn.Module = None,
        lmmd_loss_fn: torch.nn.Module = None,
        coral_loss_fn: callable = None,
        accelerator=None,
    ) -> None:
        self.config = config
        self._configure_trainer()

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataloader_source = train_dataloader_source
        self.train_dataloader_target = train_dataloader_target
        self.val_dataloader = val_dataloader
        self.val_dataloader_target = val_dataloader_target
        self.mmd_loss_fn = mmd_loss_fn
        self.lmmd_loss_fn = lmmd_loss_fn
        self.coral_loss_fn = coral_loss_fn
        self.accelerator = accelerator
        self.wandb_tracker = accelerator.get_tracker("wandb")

        self.sliding_window_inference = SlidingWindowInference(
            config["sliding_window_inference"]["roi"],
            config["sliding_window_inference"]["sw_batch_size"]
        )

        self.warmup_scheduler = warmup_scheduler
        self.training_scheduler = training_scheduler
        self.scheduler = None

        self.current_epoch = 0
        self.epoch_train_loss = 0.0
        self.best_train_loss = float('inf')
        self.epoch_val_loss = 0.0
        self.best_val_loss = float('inf')
        self.epoch_val_dice = 0.0
        self.best_val_dice = 0.0

        self.val_ema_model = None
        self.ema_model = self._create_ema_model() if self.ema_enabled else None
        self.epoch_val_ema_dice = 0.0
        self.best_val_ema_dice = 0.0
        self.domain_loss_fn = CrossEntropyLoss()

        self.start_epoch = 0

    def _configure_trainer(self) -> None:
        """
        Configures useful config variables
        """
        self.num_epochs = self.config["training_parameters"]["num_epochs"]
        self.print_every = self.config["training_parameters"]["print_every"]
        self.ema_enabled = self.config["ema"]["enabled"]
        self.val_ema_every = self.config["ema"]["val_ema_every"]
        self.warmup_enabled = self.config["warmup_scheduler"]["enabled"]
        self.warmup_epochs = self.config["warmup_scheduler"]["warmup_epochs"]
        self.cutoff_epoch = self.config["training_parameters"]["cutoff_epoch"]
        self.calculate_metrics = self.config["training_parameters"]["calculate_metrics"]
        self.checkpoint_save_dir = self.config["training_parameters"][
            "checkpoint_save_dir"
        ]


    def _create_ema_model(self):
        return torch.optim.swa_utils.AveragedModel(
            self.model,
            device=self.accelerator.device,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(
                self.config["ema"]["ema_decay"]
            )
        )

    def _calc_dice_metric(self, data, labels, use_ema: bool) -> float:
        if use_ema:
            avg_dice_score = self.sliding_window_inference(data, labels, self.ema_model)
        else:
            avg_dice_score = self.sliding_window_inference(data, labels, self.model)
        return avg_dice_score

    def _train_step(self) -> float:
        self.model.train()
        epoch_avg_loss = 0.0
        epoch_domain_loss = 0.0
        epoch_mmd_loss = 0.0
        feature_accumulator = FeatureAccumulator(max_samples=16)

        for index, (source_batch, target_batch) in enumerate(zip(self.train_dataloader_source, self.train_dataloader_target)):
            with self.accelerator.accumulate(self.model):
                source_data, source_labels = source_batch["image"], source_batch["label"]
                target_data = target_batch["image"]

                B_src = source_data.size(0)
                B_tgt = target_data.size(0)

                self.optimizer.zero_grad()

                # 🔹 Forward pass first
                source_preds, source_feats, src_domain_pred = self.model(source_data, return_features=True,use_domain_adapt=True)
                target_preds, target_feats, tgt_domain_pred = self.model(target_data, return_features=True,use_domain_adapt=True)
                
                feature_accumulator.update(source_feats, target_feats)

                if feature_accumulator.is_full():
                    accumulated_source, accumulated_target = feature_accumulator.get_features()
                    loss_mmd = self.mmd_loss_fn(accumulated_source, accumulated_target)
                    feature_accumulator.reset()
                else:
                    loss_mmd = torch.tensor(0.0, device=source_feats.device, requires_grad=True)

                domain_labels_src = torch.zeros(B_src, dtype=torch.long, device=source_data.device)
                domain_labels_tgt = torch.ones(B_tgt, dtype=torch.long, device=target_data.device)
                domain_preds = torch.cat([src_domain_pred, tgt_domain_pred], dim=0)
                domain_labels = torch.cat([domain_labels_src, domain_labels_tgt], dim=0)

                # Losses
                loss_cls = self.criterion(source_preds, source_labels.unsqueeze(1))
                loss_domain = self.domain_loss_fn(domain_preds, domain_labels)

                #  Replace NaN with 0.0
                loss_cls = torch.nan_to_num(loss_cls, nan=0.0)
                loss_mmd = torch.nan_to_num(loss_mmd, nan=0.0)
                loss_domain = torch.nan_to_num(loss_domain, nan=0.0)

                lambda_mmd = self.config["loss_fn"].get("lambda_mmd", 0.1)
                lambda_domain = self.config["loss_fn"].get("lambda_domain", 0.1)

                # Abalation Study -> Only MMD 
                # loss = loss_cls + lambda_mmd * loss_mmd

                # Abalation Study -> GRL 
                # loss = loss_cls + lambda_domain * loss_domain

                loss = loss_cls + lambda_domain * loss_domain + lambda_domain * loss_domain

                self.accelerator.backward(loss)
                self.optimizer.step()

                #Not Enabled
                if self.ema_enabled and self.accelerator.is_main_process:
                    self.ema_model.update_parameters(self.model)

                epoch_avg_loss += loss.item()
                epoch_mmd_loss += loss_mmd.item()
                epoch_domain_loss += loss_domain.item()

                if self.print_every and index % self.print_every == 0:
                    self.accelerator.print(
                        f"epoch: {str(self.current_epoch).zfill(4)} -- "
                        f"train loss: {(epoch_avg_loss / (index + 1)):.5f} -- "
                        f"mmd: {(epoch_mmd_loss/ (index + 1)):.5f} --"
                        f"domain_loss: {(epoch_domain_loss/(index + 1)):.5f} --"
                        f"lr: {self.scheduler.get_last_lr()[0]:.8f}"
                    )
        epoch_mmd_loss /= (index + 1)
        epoch_domain_loss /= (index + 1)
        self.mmd_loss_train = epoch_mmd_loss
        self.domain_loss_train = epoch_domain_loss

        return epoch_avg_loss / (index + 1)
    
    def _val_step(self, use_ema: bool = False) -> float:
        self.model.eval()

        epoch_val_total_loss = 0.0
        epoch_val_domain_loss = 0.0
        total_dice = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_hd = 0.0
        total_hd95 = 0.0
        total_asd = 0.0

        with torch.no_grad():
            for idx, (src_batch, tgt_batch) in enumerate(zip(self.val_dataloader, self.val_dataloader_target)):
                src_data, src_labels = src_batch["image"], src_batch["label"]
                tgt_data = tgt_batch["image"]

                B_src, B_tgt = src_data.size(0), tgt_data.size(0)

                # Forward: source
                src_pred, _, src_domain_pred = self.model(src_data, return_features=True, use_domain_adapt=True)
                
                # Forward: target (only domain)
                _, _, tgt_domain_pred = self.model(tgt_data, return_features=True, use_domain_adapt=True)

                # Compute segmentation/classification loss
                loss_cls = self.criterion(src_pred, src_labels.unsqueeze(1))

                # Compute domain loss
                domain_labels = torch.cat([
                    torch.zeros(B_src, dtype=torch.long, device=src_data.device),
                    torch.ones(B_tgt, dtype=torch.long, device=tgt_data.device)
                ], dim=0)

                domain_preds = torch.cat([src_domain_pred, tgt_domain_pred], dim=0)
                loss_domain = self.domain_loss_fn(domain_preds, domain_labels)

                lambda_domain = self.config["loss_fn"].get("lambda_domain", 0.1)
                
                #Abalation Study 
                # total_loss = loss_cls  #no DA loss 
                
                total_loss = loss_cls + lambda_domain * loss_domain
                epoch_val_total_loss += total_loss.item()
                epoch_val_domain_loss += loss_domain.item()
                
                # === Metrics Calculation ===
                if self.calculate_metrics:
                    self.metrics = self._calc_metrics(src_data, src_labels.unsqueeze(1), use_ema=use_ema)
                    total_dice += self.metrics["dice"]
                    total_precision += self.metrics["precision"]
                    total_recall += self.metrics["recall"]
                    total_hd += self.metrics["hd"]
                    total_hd95 += self.metrics["hd95"]
                    total_asd += self.metrics["asd"]

                    # === Visualization  ===
                    # case_id = src_batch["case_id"][0]  # assumes batch size = 1
                    # x_np = src_data[0].cpu().squeeze().numpy()
                    # y_np = src_labels[0].cpu().squeeze().numpy()
                    # pred_np = torch.sigmoid(src_pred[0]).cpu().squeeze().numpy()
                    # pred_bin = (pred_np > 0.5).astype(np.float32)
                    # vis_img = tile_directional_slices(x_np, pred_bin, view_axis=2)
                    # viz_dir = os.path.join(os.getcwd(), "visualization")
                    # os.makedirs(viz_dir, exist_ok=True)
                    # plt.imsave(os.path.join(viz_dir, f"{case_id}.png"), vis_img)

        n_batches = idx + 1
        # === Final Averaging ===
        avg_loss = epoch_val_total_loss / n_batches
        avg_domain_loss = epoch_val_domain_loss / n_batches

        if use_ema:
            self.epoch_val_ema_dice = total_dice / n_batches
        else:
            self.epoch_val_dice = total_dice / n_batches
            self.epoch_val_precision = total_precision / n_batches
            self.epoch_val_recall = total_recall / n_batches
            self.epoch_val_hd = total_hd / n_batches
            self.epoch_val_hd95 = total_hd95 / n_batches
            self.epoch_val_asd = total_asd / n_batches

        self.avg_domain_loss = avg_domain_loss

        return avg_loss
    
    def _calc_metrics(self, data, labels, use_ema: bool) -> float:
        """_summary_

        Args:
            predicted (_type_): _description_
            labels (_type_): _description_

        Returns:
            float: _description_
        """
        if use_ema:
            avg_dice_score = self.sliding_window_inference(
                data,
                labels,
                self.ema_model,
            )
        else:
            avg_dice_score = self.sliding_window_inference(
                data,
                labels,
                self.model,
            )
        return avg_dice_score


    def _run_train_val(self) -> None:
        if self.accelerator.is_main_process:
            self.wandb_tracker.run.watch(
                self.model, self.criterion, log="all", log_freq=10, log_graph=True
            )

        for epoch in tqdm(range(self.start_epoch,self.num_epochs)):
            self.current_epoch = epoch
            self._update_scheduler()

            self.epoch_train_loss = self._train_step()
            self.epoch_val_loss = self._val_step()
            self._val_ema_model()
            self._update_metrics()
            self._log_metrics()
            self._save_and_print()
            self.scheduler.step()

    def _log_metrics(self) -> None:
        log_data = {
            "epoch": self.current_epoch,
            "train_loss": self.epoch_train_loss,
            "mmd_loss": self.mmd_loss_train,
            "val/main_loss": self.epoch_val_loss,
            "val/mean_dice": self.epoch_val_dice,
            "val/domain_loss": self.avg_domain_loss,
            "val/precision": self.epoch_val_precision,
            "val/recall": self.epoch_val_recall,
            "val/hd": self.epoch_val_hd,
            "val/hd95": self.epoch_val_hd95,
            "val/asd": self.epoch_val_asd
        }
        self.accelerator.log(log_data)

    def _update_scheduler(self) -> None:
        """_summary_"""
        if self.warmup_enabled:
            if self.current_epoch == 0:
                self.accelerator.print(
                    colored(f"\n[info] -- warming up learning rate \n", color="red")
                )
                self.scheduler = self.warmup_scheduler
            elif self.current_epoch == self.warmup_epochs:
                self.accelerator.print(
                    colored(
                        f"\n[info] -- switching to learning rate decay schedule \n",
                        color="red",
                    )
                )
                self.scheduler = self.training_scheduler
        elif self.current_epoch == 0:
            self.accelerator.print(
                colored(
                    f"\n[info] -- setting learning rate decay schedule \n",
                    color="red",
                )
            )
            self.scheduler = self.training_scheduler

    def _update_metrics(self) -> None:
        """_summary_"""
        # update training loss
        if self.epoch_train_loss <= self.best_train_loss:
            self.best_train_loss = self.epoch_train_loss

        # update validation loss
        if self.epoch_val_loss <= self.best_val_loss:
            self.best_val_loss = self.epoch_val_loss

        if self.calculate_metrics:
            if self.epoch_val_dice >= self.best_val_dice:
                self.best_val_dice = self.epoch_val_dice

    def _val_ema_model(self):
        if self.ema_enabled and (self.current_epoch % self.val_ema_every == 0):
            self.val_ema_model = self.ema_model
            _ = self._val_step()

    def _save_and_print(self) -> None:
        """_summary_"""
        # print only on the first gpu
        if self.epoch_val_dice >= self.best_val_dice:
            # change path name based on cutoff epoch
            if self.current_epoch <= self.cutoff_epoch:
                save_path = os.path.join(self.checkpoint_save_dir, "checkpoint.pth")
            else:
                save_path = os.path.join(
                    self.checkpoint_save_dir,
                    "best_dice_model_post_cutoff",
                    "checkpoint.pth"
                )
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True) 
            # save checkpoint and log
            self._save_checkpoint(save_path)

            self.accelerator.print(
                f"epoch -- {colored(str(self.current_epoch).zfill(4), color='green')} || "
                f"train loss -- {colored(f'{self.epoch_train_loss:.5f}', color='green')} || "
                f"val loss -- {colored(f'{self.epoch_val_loss:.5f}', color='green')} || "
                f"val domain loss -- {colored(f'{self.avg_domain_loss:.5f}', color='green')} || "
                f"lr -- {colored(f'{self.scheduler.get_last_lr()[0]:.8f}', color='green')} || "
                f"val mean_dice -- {colored(f'{self.best_val_dice:.5f}', color='green')} -- saved || "
                f"precision -- {colored(f'{self.epoch_val_precision:.5f}', color='green')} || "
                f"recall -- {colored(f'{self.epoch_val_recall:.5f}', color='green')} || "
                f"hd -- {colored(f'{self.epoch_val_hd:.5f}', color='green')} || "
                f"hd95 -- {colored(f'{self.epoch_val_hd95:.5f}', color='green')} || "
                f"asd -- {colored(f'{self.epoch_val_asd:.5f}', color='green')}"
            )
        else:
            self.accelerator.print(
                f"epoch -- {str(self.current_epoch).zfill(4)} || "
                f"train loss -- {self.epoch_train_loss:.5f} || "
                f"val loss -- {self.epoch_val_loss:.5f} || "
                f"val domain loss -- {self.epoch_val_loss:.5f} || "
                f"lr -- {self.scheduler.get_last_lr()[0]:.8f} || "
                f"val mean_dice -- {self.epoch_val_dice:.5f} || "
                f"precision -- {self.epoch_val_precision:.5f} || "
                f"recall -- {self.epoch_val_recall:.5f} || "
                f"hd -- {self.epoch_val_hd:.5f} || "
                f"hd95 -- {self.epoch_val_hd95:.5f} || "
                f"asd -- {self.epoch_val_asd:.5f}"
            )

    def _save_checkpoint(self, filename: str) -> None:
        """
        Saves the model checkpoint with full training state, optimizer, scheduler, and best metrics.
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": self.current_epoch,
            "best_train_loss": self.best_train_loss,
            "best_val_dice": self.best_val_dice,
            "best_val_precision": self.epoch_val_precision,
            "best_val_recall": self.epoch_val_recall,
            "best_val_hd": self.epoch_val_hd,
            "best_val_hd95": self.epoch_val_hd95,
            "best_val_asd": self.epoch_val_asd,
        }

        torch.save(checkpoint, filename)
        self.accelerator.print(f"[info] -- Saved checkpoint to {filename}")
    
    def _load_checkpoint(self):
        """
        Loads checkpoint for full training resume or model-only fine-tuning, based on config flags.
        """
        load_cfg = self.config["training_parameters"]["load_checkpoint"]

        if not (load_cfg["load_full_checkpoint"] or load_cfg["load_model_only"]):
            self.accelerator.print("[info] -- Skipping checkpoint loading.")
            return

        if load_cfg["load_full_checkpoint"] and load_cfg["load_model_only"]:
            raise ValueError("Only one of `load_full_checkpoint` or `load_model_only` can be True.")

        ckpt_path = os.path.join(load_cfg["load_checkpoint_path"],"checkpoint.pth")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.accelerator.device,weights_only=False)

        # Load model weights
        self.model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint["state_dict"])
        self.accelerator.print(f"[info] -- Loaded model weights from {ckpt_path}")

        if load_cfg["load_full_checkpoint"]:
            if "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.accelerator.print("[info] -- Optimizer state loaded.")
            else:
                self.accelerator.print("[warn] -- Optimizer state not found in checkpoint.")

            # if "scheduler" in checkpoint and self.scheduler:
            #     self.scheduler.load_state_dict(checkpoint["scheduler"])
            #     self.scheduler.last_epoch = self.current_epoch 
            #     self.accelerator.print("[info] -- Scheduler state loaded.")
            # else:
            #     self.accelerator.print("[warn] -- Scheduler state not found or not initialized.")
            
            #This was done to load correct schedular when the training resumes. We used deffernt schedular as the epoch progress 
            self.current_epoch = checkpoint.get("epoch", 0)  # <-- set before scheduler
            # Select the correct scheduler based on saved epoch
            if self.current_epoch < self.warmup_epochs:
                self.scheduler = self.warmup_scheduler
            else:
                self.scheduler = self.training_scheduler

            # Now load scheduler state safely
            if "scheduler" in checkpoint and self.scheduler:
                try:
                    self.scheduler.load_state_dict(checkpoint["scheduler"])
                    self.scheduler.last_epoch = self.current_epoch
                    self.accelerator.print("[info] -- Scheduler state loaded.")
                except KeyError as e:
                    self.accelerator.print(f"[warn] -- Scheduler state could not be loaded: {e}")

            self.start_epoch = checkpoint.get("epoch", 0)
            self.best_train_loss = checkpoint.get("best_train_loss", 100.0)
            self.best_val_dice = checkpoint.get("best_val_dice", 0.0)
            self.best_val_precision = checkpoint.get("best_val_precision", 0.0)
            self.best_val_recall = checkpoint.get("best_val_recall", 0.0)
            self.best_val_hd = checkpoint.get("best_val_hd", 0.0)
            self.best_val_hd95 = checkpoint.get("best_val_hd95", 0.0)
            self.best_val_asd = checkpoint.get("best_val_asd", 0.0)

            self.accelerator.print(f"[info] -- Resumed from epoch {self.start_epoch}")
            self.accelerator.print(f"[info] -- Best train loss: {self.best_train_loss:.5f}")
            self.accelerator.print(f"[info] -- Best validation metrics:")
            self.accelerator.print(f"        Dice     : {self.best_val_dice:.5f}")
            self.accelerator.print(f"        Precision: {self.best_val_precision:.5f}")
            self.accelerator.print(f"        Recall   : {self.best_val_recall:.5f}")
            self.accelerator.print(f"        HD       : {self.best_val_hd:.5f}")
            self.accelerator.print(f"        HD95     : {self.best_val_hd95:.5f}")
            self.accelerator.print(f"        ASD      : {self.best_val_asd:.5f}")


        elif load_cfg["load_model_only"]:
            self.accelerator.print("[info] -- Loaded model for finetuning (no optimizer/scheduler state).")


    def train(self) -> None:
        self._run_train_val()
        self.accelerator.end_traninig()

    def evaluate(self) -> None:
        raise NotImplementedError("Testing Evaltion is Done By Seperate Script called evaulate_wandb_model.py")