##################################NO DA################################################
import os
import torch
import evaluate
import matplotlib.pyplot as plt
import monai
import numpy as np
from tqdm import tqdm
from typing import Dict
from copy import deepcopy
from termcolor import colored
from torch.utils.data import DataLoader
from metrics.segmentation_metrics import SlidingWindowInference
############################################################################################################



class Segmentation_Trainer:
    def __init__(
        self,
        config: Dict,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        warmup_scheduler: torch.optim.lr_scheduler.LRScheduler,
        training_scheduler: torch.optim.lr_scheduler.LRScheduler,
        accelerator=None,
    ) -> None:
        """classification trainer class init function

        Args:
            config (Dict): _description_
            model (torch.nn.Module): _description_
            optimizer (torch.optim.Optimizer): _description_
            criterion (torch.nn.Module): _description_
            train_dataloader (DataLoader): _description_
            val_dataloader (DataLoader): _description_
            warmup_scheduler (torch.optim.lr_scheduler.LRScheduler): _description_
            training_scheduler (torch.optim.lr_scheduler.LRScheduler): _description_
            accelerator (_type_, optional): _description_. Defaults to None.
        """
        # config
        self.config = config
        self._configure_trainer()

        # model components
        self.model = model
        
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # accelerate object
        self.accelerator = accelerator

        # get wandb object
        self.wandb_tracker = accelerator.get_tracker("wandb")

        # metrics
        self.current_epoch = 0  # current epoch
        self.epoch_train_loss = 0.0  # epoch train loss
        self.best_train_loss = 100.0  # best train loss
        self.epoch_val_loss = 0.0  # epoch validation loss
        self.best_val_loss = 100.0  # best validation loss
        self.epoch_val_dice = 0.0  # epoch validation accuracy
        self.best_val_dice = 0.0  # best validation accuracy

        # external metric functions we can add
        self.sliding_window_inference = SlidingWindowInference(
            config["sliding_window_inference"]["roi"],
            config["sliding_window_inference"]["sw_batch_size"],
        )

        # training scheduler
        self.warmup_scheduler = warmup_scheduler
        self.training_scheduler = training_scheduler
        self.scheduler = None

        # temp ema model copy
        self.val_ema_model = None
        self.ema_model = self._create_ema_model() if self.ema_enabled else None
        self.epoch_val_ema_dice = 0.0
        self.best_val_ema_dice = 0.0

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
        self.checkpoint_save_dir = self.config["training_parameters"]["checkpoint_save_dir"]

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

        ckpt_path = os.path.join(load_cfg["load_checkpoint_path"], "checkpoint.pth")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.accelerator.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint["state_dict"])
        self.accelerator.print(f"[info] -- Loaded model weights from {ckpt_path}")

        if load_cfg["load_full_checkpoint"]:
            if "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.accelerator.print("[info] -- Optimizer state loaded.")
            else:
                self.accelerator.print("[warn] -- Optimizer state not found in checkpoint.")

            # Determine correct scheduler before loading state
            self.current_epoch = checkpoint.get("epoch", 0)
            if self.current_epoch < self.warmup_epochs:
                self.scheduler = self.warmup_scheduler
            else:
                self.scheduler = self.training_scheduler

            # Safe scheduler state loading
            if "scheduler" in checkpoint and self.scheduler:
                try:
                    self.scheduler.load_state_dict(checkpoint["scheduler"])
                    self.scheduler.last_epoch = self.current_epoch
                    self.accelerator.print("[info] -- Scheduler state loaded.")
                except KeyError as e:
                    self.accelerator.print(f"[warn] -- Scheduler state could not be loaded: {e}")
            else:
                self.accelerator.print("[warn] -- Scheduler state not found or not initialized.")

            # Restore training stats
            self.best_train_loss = checkpoint.get("best_train_loss", 100.0)
            self.best_val_dice = checkpoint.get("best_val_dice", 0.0)
            self.best_val_precision = checkpoint.get("best_val_precision", 0.0)
            self.best_val_recall = checkpoint.get("best_val_recall", 0.0)
            self.best_val_hd = checkpoint.get("best_val_hd", 0.0)
            self.best_val_hd95 = checkpoint.get("best_val_hd95", 0.0)
            self.best_val_asd = checkpoint.get("best_val_asd", 0.0)

            self.accelerator.print(f"[info] -- Resumed from epoch {self.current_epoch}")
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

    def _create_ema_model(self) -> torch.nn.Module:
        self.accelerator.print(f"[info] -- creating ema model")
        ema_model = torch.optim.swa_utils.AveragedModel(
            self.model,
            device=self.accelerator.device,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(
                self.config["ema"]["ema_decay"]
            ),
        )
        return ema_model

    def _train_step(self) -> float:
        # Initialize the training loss for the current epoch
        epoch_avg_loss = 0.0

        # set model to train
        self.model.train()

        # set epoch to shift data order each epoch
        # self.train_dataloader.sampler.set_epoch(self.current_epoch)
        for index, raw_data in enumerate(self.train_dataloader):
            with self.accelerator.accumulate(self.model):
                # get data ex: (data, target)
                data, labels = (
                    raw_data["image"],
                    raw_data["label"],
                )
                # print("data ", data.shape, "label ", labels.shape)
                # zero out existing gradients
                
                self.optimizer.zero_grad()
                # forward pass
                predicted = self.model.forward(data)
                # print("predicted_shape")
                
                # calculate loss
                loss = self.criterion(predicted, labels.unsqueeze(1))

                # backward pass
                self.accelerator.backward(loss)

                # update gradients
                self.optimizer.step()

                # model update with ema if available
                if self.ema_enabled and (self.accelerator.is_main_process):
                    self.ema_model.update_parameters(self.model)

                # update loss
                epoch_avg_loss += loss.item()

                if self.print_every:
                    if index % self.print_every == 0:
                        self.accelerator.print(
                            f"epoch: {str(self.current_epoch).zfill(4)} -- "
                            f"train loss: {(epoch_avg_loss / (index + 1)):.5f} -- "
                            f"lr: {self.scheduler.get_last_lr()[0]}"
                        )

        epoch_avg_loss = epoch_avg_loss / (index + 1)

        return epoch_avg_loss

    def _val_step(self, use_ema: bool = False) -> float:
        """_summary_

        Args:
            use_ema (bool, optional): if use_ema runs validation with ema_model. Defaults to False.

        Returns:
            float: _description_
        """
        # Initialize the training loss for the current Epoch
        epoch_avg_loss = 0.0
        total_dice = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_hd = 0.0
        total_hd95 = 0.0
        total_asd = 0.0

        # set model to train mode
        self.model.eval()
        if use_ema:
            self.val_ema_model.eval()

        # set epoch to shift data order each epoch
        with torch.no_grad():
            for index, (raw_data) in enumerate(self.val_dataloader):
                # get data ex: (data, target)
                data, labels = (
                    raw_data["image"],
                    raw_data["label"]
                    # raw_data["case_id"],
                )
                # forward pass
                if use_ema:
                    predicted = self.ema_model.forward(data)
                else:
                    predicted = self.model.forward(data)
                # calculate loss
                loss = self.criterion(predicted, labels.unsqueeze(1))

                # calculate metrics
                if self.calculate_metrics:
                    self.metrics = self._calc_metrics(data, labels.unsqueeze(1), use_ema)
                    total_dice += self.metrics["dice"]
                    total_precision += self.metrics["precision"]
                    total_recall += self.metrics["recall"]
                    total_hd += self.metrics["hd"]
                    total_hd95 += self.metrics["hd95"]
                    total_asd += self.metrics["asd"]

                epoch_avg_loss += loss.item()

                # === Visualization for each case ===
                # x_np = data[0].detach().cpu().squeeze().numpy()
                # y_np = labels[0].detach().cpu().squeeze().numpy()
                # pred_np = predicted[0].detach().cpu().squeeze().numpy()
                # pred_bin = (pred_np > 0.5).astype(np.float32)

                # vis_img = tile_directional_slices(x_np, pred_bin, view_axis=2)

                # # Create the visualization directory under root
                # root_dir = os.path.dirname(os.path.abspath(__file__))
                # viz_dir = os.path.join(root_dir, "visualization")
                # os.makedirs(viz_dir, exist_ok=True)
                # out_path = os.path.join(viz_dir, f"{case_id}.png")

                # plt.imsave(out_path, vis_img)

        if use_ema:
            self.epoch_val_ema_dice = total_dice / float(index + 1)
        else:
            self.epoch_val_dice = total_dice / float(index + 1)
            self.epoch_val_precision= total_precision / float(index + 1)
            self.epoch_val_recall= total_recall / float(index + 1)
            self.epoch_val_hd = total_hd / float(index + 1)
            self.epoch_val_hd95 = total_hd95 / float(index + 1)
            self.epoch_val_asd = total_asd / float(index + 1)

        epoch_avg_loss = epoch_avg_loss / float(index + 1)

        return epoch_avg_loss


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
        """_summary_"""
        # Tell wandb to watch the model and optimizer values
        if self.accelerator.is_main_process:
            self.wandb_tracker.run.watch(
                self.model, self.criterion, log="all", log_freq=10, log_graph=True
            )

        # Run Training and Validation
        for epoch in tqdm(range(self.current_epoch, self.num_epochs)):
            print(f"epoch:{self.current_epoch}")
            # update epoch
            self.current_epoch = epoch
            self._update_scheduler()

            # run a single training step
            train_loss = self._train_step()
            self.epoch_train_loss = train_loss

            # run a single validation step
            val_loss = self._val_step(use_ema=False)
            self.epoch_val_loss = val_loss

            # if enabled run ema every x steps
            self._val_ema_model()

            # update metrics
            self._update_metrics()

            # log metrics
            self._log_metrics()

            # save and print
            self._save_and_print()

            # update schduler
            self.scheduler.step()

            # ==== Log resume info ====
            if self.current_epoch > 0:
                current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else None
                self.accelerator.print(
                    f"[info] -- Resuming training from epoch {self.current_epoch} with LR {current_lr:.8f}"
                )


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

    def _log_metrics(self) -> None:
        """_summary_"""
        # data to be logged
        log_data = {
            "epoch": self.current_epoch,
            "train_loss": self.epoch_train_loss,
            "val_loss": self.epoch_val_loss,
            "mean_dice": self.epoch_val_dice,
            "val/precision": self.epoch_val_precision,
            "val/recall": self.epoch_val_recall,
            "val/hd": self.epoch_val_hd,
            "val/hd95": self.epoch_val_hd95,
            "val/asd": self.epoch_val_asd
        }
        # log the data
        self.accelerator.log(log_data)

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

    #Not Enambled
    def _val_ema_model(self):
        if self.ema_enabled and (self.current_epoch % self.val_ema_every == 0):
            self.val_ema_model = self._update_ema_bn(duplicate_model=False)
            _ = self._val_step(use_ema=True)
            self.accelerator.print(
                f"[info] -- gpu id: {self.accelerator.device} -- "
                f"ema val dice: {colored(f'{self.epoch_val_ema_dice:.5f}', color='red')}"
            )

        if self.epoch_val_ema_dice > self.best_val_ema_dice:
            torch.save(self.val_ema_model.module, "best_ema_model_ckpt.pth")
            self.best_val_ema_dice = self.epoch_val_ema_dice

    #Not Enabled
    def _update_ema_bn(self, duplicate_model: bool = True):
        """
        Updates the BatchNorm stats of the EMA model using only image tensors.
        """
        self.accelerator.print(colored("[info] -- updating ema batch norm stats", color="red"))

        def _image_only_dataloader():
            for batch in self.train_dataloader:
                if isinstance(batch, dict) and "image" in batch:
                    yield batch["image"].to(self.accelerator.device)
                else:
                    raise ValueError("Expected a dict with an 'image' key.")

        model = deepcopy(self.ema_model).to(self.accelerator.device) if duplicate_model else self.ema_model

        torch.optim.swa_utils.update_bn(
            _image_only_dataloader(),  # 💡 DON'T pass train_dataloader directly!
            model,
            device=self.accelerator.device,
        )

        return self.ema_model  


    def train(self) -> None:
        """
        Runs a full training and validation of the dataset.
        """
        self._run_train_val()
        self.accelerator.end_traninig()

    def evaluate(self) -> None:
        raise NotImplementedError("Testing Evaltion is Done By Seperate Script called evaulate_wandb_model.py")