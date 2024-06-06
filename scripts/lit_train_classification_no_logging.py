"""
This is an example script for vision classification using the BigEarthNet dataset.
It is basically a 1-to-1 application of the process described in the documentation under
Supervised Image Classification.
"""
# import packages
from typing import List

try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import typer
from torch import optim
from torchmetrics.classification import AveragePrecision

from configilm import ConfigILM
from configilm.ConfigILM import ILMConfiguration
from configilm.ConfigILM import ILMType
from configilm.extra import BENv1_utils
from configilm.extra.DataModules.BENv1_DataModule import BENv1DataModule

__author__ = "Leonard Hackel - BIFOLD/RSiM TU Berlin"


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
        self.val_output_list: List[dict] = []
        self.test_output_list: List[dict] = []

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
        self.val_output_list += [{"loss": loss, "outputs": x_hat, "labels": y}]

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.val_output_list = []

    def on_validation_epoch_end(self):
        metrics = self.get_metrics(self.val_output_list)

        self.log("val/loss", metrics["avg_loss"])
        self.log("val/mAP (Macro)", metrics["map_macro"])

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.model(x)
        loss = F.binary_cross_entropy_with_logits(x_hat, y)
        self.test_output_list += [{"loss": loss, "outputs": x_hat, "labels": y}]

    def on_test_epoch_end(self):
        metrics = self.get_metrics(self.test_output_list)

        self.log("test/loss", metrics["avg_loss"])
        self.log("test/mAP (Macro)", metrics["map_macro"])

    def forward(self, batch):
        # because we are a wrapper, we call the inner function manually
        return self.model(batch)

    def get_metrics(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        logits = torch.cat([x["outputs"].cpu() for x in outputs], 0)
        labels = torch.cat([x["labels"].cpu() for x in outputs], 0)  # Tensor of size (#samples x classes)

        # calculate AP
        ap_macro = AveragePrecision(num_labels=self.config.classes, average="macro", task="multilabel").to(
            logits.device
        )(logits, labels.int())

        return {
            "avg_loss": avg_loss,
            "map_macro": ap_macro,
        }


def main(
    vision_model: str = "resnet18",
    number_of_channels: int = 12,
    image_size: int = 120,
    batch_size: int = 32,
    num_workers: int = 4,
    max_img_index: int = 7 * 128,
    epochs: int = 10,
    lr: float = 5e-4,
):
    # some static parameters
    seed = 42
    # for ampere GPUs set precision -> can also be 'high', see details at
    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision("medium")

    # seed for pytorch, numpy, python.random, Dataloader workers, spawned subprocesses
    pl.seed_everything(seed, workers=True)

    model_config = ILMConfiguration(
        timm_model_name=vision_model,
        classes=19,
        image_size=image_size,
        channels=number_of_channels,
        network_type=ILMType.IMAGE_CLASSIFICATION,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        log_every_n_steps=1,
        logger=False,
    )

    model = LitVisionEncoder(config=model_config, lr=lr)
    dm = BENv1DataModule(
        # just using a mock data dir here - you should replace this with the path to your data
        # (consider the dict structure needed)
        data_dirs=BENv1_utils.resolve_data_dir(None, allow_mock=True),
        img_size=(number_of_channels, image_size, image_size),
        num_workers_dataloader=num_workers,
        batch_size=batch_size,
        max_len=max_img_index,
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path="best")

    print("=== Training finished ===")


if __name__ == "__main__":
    typer.run(main)
