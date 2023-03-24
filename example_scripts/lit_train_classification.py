"""
This is an example script for vision classification using the BigEarthNet dataset.
It is basically a 1-to-1 application of the process described in the documentation under
Supervised Vision Classification.
"""
# import packages
import os
from os.path import isfile
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import typer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import optim
from torchmetrics.classification import MultilabelAveragePrecision
from torchmetrics.classification import MultilabelF1Score
from wandb.sdk import finish as wandb_finish
from wandb.sdk import login as wandb_login

from configvlm import ConfigVLM
from configvlm.ConfigVLM import VLMConfiguration
from configvlm.ConfigVLM import VLMType
from configvlm.extra.BEN_DataModule_LMDB_Encoder import BENDataModule
from configvlm.extra.BEN_lmdb_utils import resolve_ben_data_dir


__author__ = "Leonard Hackel - BIFOLD/RSiM TU Berlin"
os.environ["WANDB_START_METHOD"] = "thread"
wandb_api_key = os.environ["WANDB_API_KEY"]


class LitVisionEncoder(pl.LightningModule):
    """
    Wrapper around a pytorch module, allowing this module to be used in automatic
    training with pytorch lightning.
    Among other things, the wrapper allows us to do automatic training and removes the
    need to manage data on different devices (e.g. GPU and CPU).
    """

    def __init__(
        self,
        config: ConfigVLM.VLMConfiguration,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.lr = lr
        self.config = config
        self.model = ConfigVLM.ConfigVLM(config)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.model(x)
        loss = F.binary_cross_entropy_with_logits(x_hat, y)
        self.log("train/loss", loss)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        return optimizer

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.model(x)
        loss = F.binary_cross_entropy_with_logits(x_hat, y)
        return {"loss": loss, "outputs": x_hat, "labels": y}

    def validation_epoch_end(self, outputs):
        metrics = self.get_metrics(outputs)

        self.log("val/loss", metrics["avg_loss"])
        self.log("val/f1", metrics["avg_f1_score"])
        self.log("val/mAP (Micro)", metrics["map_score"]["micro"])
        self.log("val/mAP (Macro)", metrics["map_score"]["macro"])

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.model(x)
        loss = F.binary_cross_entropy_with_logits(x_hat, y)
        return {"loss": loss, "outputs": x_hat, "labels": y}

    def test_epoch_end(self, outputs):
        metrics = self.get_metrics(outputs)

        self.log("test/loss", metrics["avg_loss"])
        self.log("test/f1", metrics["avg_f1_score"])
        self.log("test/mAP (Micro)", metrics["map_score"]["micro"])
        self.log("test/mAP (Macro)", metrics["map_score"]["macro"])

    def forward(self, batch):
        # because we are a wrapper, we call the inner function manually
        return self.model(batch)

    def get_metrics(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        logits = torch.cat([x["outputs"].cpu() for x in outputs], 0)
        labels = torch.cat(
            [x["labels"].cpu() for x in outputs], 0
        )  # Tensor of size (#samples x classes)

        f1_score = MultilabelF1Score(num_labels=self.config.classes, average=None).to(
            logits.device
        )(logits, labels)

        # calculate AP
        ap_micro = MultilabelAveragePrecision(
            num_labels=self.config.classes, average="micro"
        ).to(logits.device)(logits, labels.int())

        ap_macro = MultilabelAveragePrecision(
            num_labels=self.config.classes, average="macro"
        ).to(logits.device)(logits, labels.int())

        ap_score = {"micro": float(ap_micro), "macro": float(ap_macro)}

        avg_f1_score = float(
            torch.sum(f1_score) / self.config.classes
        )  # macro average f1 score

        return {
            "avg_loss": avg_loss,
            "avg_f1_score": avg_f1_score,
            "map_score": ap_score,
        }


def main(
    vision_model: str = "resnet18",
    data_dir: Optional[str] = None,
    number_of_channels: int = 12,
    image_size: int = 120,
    batch_size: int = 32,
    num_workers: int = 4,
    max_img_index: int = 7 * 128,
    epochs: int = 10,
    lr: float = 5e-4,
    drop_rate: float = 0.2,
    seed: int = 42,
    disable_logging: bool = False,
    logging_model_name: str = "Example Script",
    offline: bool = False,
    val_epoch_interval: Optional[int] = None,
    early_stopping_patience: int = 5,
    vision_checkpoint: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None,
):
    assert (
        resume_from_checkpoint is None or vision_checkpoint is None
    ), "Provided both checkpoints, please use only one"

    # for ampere GPUs set precision -> can also be 'high', see details at
    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision("medium")

    # seed for pytorch, numpy, python.random, Dataloader workers, spawned subprocesses
    pl.seed_everything(seed, workers=True)

    # Key is available by wandb, project name can be chosen at will
    wandb_login(key=wandb_api_key)
    # disable logging gets priority over online/offline
    if disable_logging:
        wandb_mode = "disabled"
    else:
        wandb_mode = "offline" if offline else "online"

    model_config = VLMConfiguration(
        timm_model_name=vision_model,
        classes=19,
        image_size=image_size,
        channels=number_of_channels,
        drop_rate=drop_rate,
        network_type=VLMType.VISION_CLASSIFICATION,
    )

    wandb_logger = WandbLogger(
        project="BEN_classification",
        log_model=not offline,
        tags=[logging_model_name],
        # keyword arg directly to wandb.init()
        mode=wandb_mode,
    )

    monitor = "val/f1"
    monitor_str = "F1_score"
    # checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor="val/f1",
        dirpath="./checkpoints",
        filename=f"{wandb_logger.experiment.name}-{logging_model_name}-seed="
        + str(seed)
        + "-epoch={epoch:03d}-"
        + f"{monitor_str}"
        + "={"
        + f"{monitor}"
        + ":.3f}",
        auto_insert_metric_name=False,
        save_top_k=1,
        mode="max",
        save_last=True,
    )
    early_stopping_callback = EarlyStopping(
        monitor=monitor,
        min_delta=0.00,
        patience=early_stopping_patience,
        verbose=False,
        mode="max",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        log_every_n_steps=1,
        logger=wandb_logger,
        callbacks=[early_stopping_callback, lr_monitor, checkpoint_callback],
        check_val_every_n_epoch=val_epoch_interval,
    )

    model = LitVisionEncoder(config=model_config, lr=lr)
    dm = BENDataModule(
        data_dir=resolve_ben_data_dir(data_dir),  # path to dataset
        img_size=(number_of_channels, image_size, image_size),
        num_workers_dataloader=num_workers,
        max_img_idx=max_img_index,
        batch_size=batch_size,
    )

    if vision_checkpoint is not None:
        # try to load checkpoint
        if not isfile(vision_checkpoint):
            print("Pretrained vision model not available, cannot load checkpoint")
        else:
            # get model and pretrained state dicts
            if torch.cuda.is_available():
                pretrained_dict = torch.load(vision_checkpoint)
            else:
                pretrained_dict = torch.load(
                    vision_checkpoint, map_location=torch.device("cpu")
                )
            model_dict = model.state_dict()

            # filter out unnecessary keys
            # this allows to load lightning or pytorch model loading
            if "pytorch-lightning_version" in pretrained_dict.keys():
                # checkpoint is a Pytorch-Lightning Checkpoint
                pretrained_dict = {
                    k: v
                    for k, v in pretrained_dict["state_dict"].items()
                    if k in model_dict
                }
            else:
                pretrained_dict = {
                    k: v for k, v in pretrained_dict.items() if k in model_dict
                }

            # filter keys that have a size mismatch
            mismatch_keys = [
                x
                for x in pretrained_dict.keys()
                if pretrained_dict[x].shape != model_dict[x].shape
            ]
            for key in mismatch_keys:
                del pretrained_dict[key]
                print(f"Key '{key}' size mismatch, removing from loading")

            # overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)

            # load the new state dict
            model.load_state_dict(model_dict)
            print("Vision Model checkpoint loaded")

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path="best")

    wandb_finish()
    print("=== Training finished ===")


if __name__ == "__main__":
    typer.run(main)
