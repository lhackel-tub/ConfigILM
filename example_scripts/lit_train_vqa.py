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
from sklearn.metrics import accuracy_score
from torch import optim
from torchmetrics.classification import MultilabelF1Score
from tqdm import tqdm
from wandb.sdk import finish as wandb_finish
from wandb.sdk import login as wandb_login

from configilm import ConfigILM
from configilm.ConfigILM import ILMConfiguration
from configilm.ConfigILM import ILMType
from configilm.extra.BEN_lmdb_utils import resolve_ben_data_dir
from configilm.extra.RSVQAxBEN_DataModule_LMDB_Encoder import RSVQAxBENDataModule


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
        config: ConfigILM.ILMConfiguration,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.lr = lr
        self.config = config
        self.model = ConfigILM.ConfigILM(config)
        self.val_output_list = []
        self.test_output_list = []

    def _disassemble_batch(self, batch):
        images, questions, labels = batch
        # transposing tensor, needed for Huggingface-Dataloader combination
        questions = torch.tensor(
            [x.tolist() for x in questions], device=self.device
        ).T.int()
        return (images, questions), labels

    def training_step(self, batch, batch_idx):
        x, y = self._disassemble_batch(batch)
        x_hat = self.model(x)
        loss = F.binary_cross_entropy_with_logits(x_hat, y)
        self.log("train/loss", loss)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        return optimizer

    def validation_step(self, batch, batch_idx):
        x, y = self._disassemble_batch(batch)
        x_hat = self.model(x)
        loss = F.binary_cross_entropy_with_logits(x_hat, y)
        self.val_output_list += [{"loss": loss, "outputs": x_hat, "labels": y}]

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.val_output_list = []

    def on_validation_epoch_end(self):
        metrics = self.get_metrics(self.val_output_list)

        self.log("val/loss", metrics["avg_loss"])
        self.log("val/f1", metrics["avg_f1_score"])
        self.log("val/Accuracy (LULC)", metrics["accuracy"]["LULC"])
        self.log("val/Accuracy (Yes-No)", metrics["accuracy"]["Yes/No"])
        self.log("val/Accuracy (Overall)", metrics["accuracy"]["Overall"])
        self.log("val/Accuracy (Average)", metrics["accuracy"]["Average"])

    def test_step(self, batch, batch_idx):
        x, y = self._disassemble_batch(batch)
        x_hat = self.model(x)
        loss = F.binary_cross_entropy_with_logits(x_hat, y)
        self.test_output_list += [{"loss": loss, "outputs": x_hat, "labels": y}]

    def on_test_epoch_end(self):
        metrics = self.get_metrics(self.test_output_list)

        self.log("test/loss", metrics["avg_loss"])
        self.log("test/f1", metrics["avg_f1_score"])
        self.log("test/Accuracy (LULC)", metrics["accuracy"]["LULC"])
        self.log("test/Accuracy (Yes-No)", metrics["accuracy"]["Yes/No"])
        self.log("test/Accuracy (Overall)", metrics["accuracy"]["Overall"])
        self.log("test/Accuracy (Average)", metrics["accuracy"]["Average"])

    def forward(self, batch):
        # because we are a wrapper, we call the inner function manually
        return self.model(batch)

    def get_metrics(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        logits = torch.cat([x["outputs"].cpu() for x in outputs], 0)
        labels = torch.cat(
            [x["labels"].cpu() for x in outputs], 0
        )  # Tensor of size (#samples x classes)

        selected_answers = self.trainer.datamodule.selected_answers

        argmax_out = torch.argmax(logits, dim=1)
        argmax_lbl = torch.argmax(labels, dim=1)

        # get answers and predictions per type
        yn_preds = []
        yn_gts = []
        lulc_preds = []
        lulc_gts = []

        for i, ans in enumerate(tqdm(argmax_lbl, desc="Counting answers")):
            # Yes/No question
            if selected_answers[ans] in ["yes", "no"]:

                # stored for global Yes/No
                yn_preds.append(argmax_out[i])
                yn_gts.append(ans)

            # LC question
            else:
                # stored for global LC
                lulc_preds.append(argmax_out[i])
                lulc_gts.append(ans)

        acc_yn = accuracy_score(yn_gts, yn_preds)
        acc_lulc = accuracy_score(lulc_gts, lulc_preds)

        accuracy_dict = {
            "Yes/No": acc_yn,
            "LULC": acc_lulc,
            "Overall": accuracy_score(
                argmax_lbl, argmax_out
            ),  # micro average on classes
            "Average": (acc_yn + acc_lulc) / 2,  # macro average on types
        }

        f1_score = MultilabelF1Score(num_labels=self.config.classes, average=None).to(
            logits.device
        )(logits, labels)

        avg_f1_score = float(
            torch.sum(f1_score) / self.config.classes
        )  # macro average f1 score

        return {
            "avg_loss": avg_loss,
            "avg_f1_score": avg_f1_score,
            "accuracy": accuracy_dict,
        }


def main(
    vision_model: str = "resnet18",
    text_model: str = "prajjwal1/bert-tiny",
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
    val_epoch_interval: Optional[int] = 5,
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

    model_config = ILMConfiguration(
        timm_model_name=vision_model,
        hf_model_name=text_model,
        classes=1000,
        image_size=image_size,
        channels=number_of_channels,
        drop_rate=drop_rate,
        network_type=ILMType.VQA_CLASSIFICATION,
    )

    wandb_logger = WandbLogger(
        project="RSVQAxBEN_classification",
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
    dm = RSVQAxBENDataModule(
        data_dir=resolve_ben_data_dir(data_dir, allow_mock=True),  # path to dataset
        img_size=(number_of_channels, image_size, image_size),
        max_img_idx=max_img_index,
        num_workers_dataloader=num_workers,
        batch_size=batch_size,
        tokenizer=model.model.get_tokenizer(),
        seq_length=64,
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
