"""
This is an example script for supervised image classification using the BigEarthNet v2.0 dataset.
"""
# import packages
from typing import List

try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl
import torch
import torch.nn.functional as F

try:
    import typer
except ImportError:
    raise OSError("Please install typer to run this script.")
from torch import optim

from configilm import ConfigILM
from configilm.ConfigILM import ILMConfiguration
from configilm.ConfigILM import ILMType
from configilm.extra.BENv2_utils import STANDARD_BANDS, NEW_LABELS, resolve_data_dir
from configilm.extra.DataModules.BENv2_DataModule import BENv2DataModule
from configilm.metrics import get_classification_metric_collection
from configilm.extra.CustomTorchClasses import LinearWarmupCosineAnnealingLR
from configilm.util import Messages

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
        config: ILMConfiguration,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.lr = lr
        self.config = config
        assert config.network_type == ILMType.IMAGE_CLASSIFICATION
        assert config.classes == 19
        self.model = ConfigILM.ConfigILM(config)
        self.val_output_list: List[dict] = []
        self.test_output_list: List[dict] = []
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.val_metrics_micro = get_classification_metric_collection(
            "multilabel", "micro", num_labels=config.classes, prefix="val/"
        )
        self.val_metrics_macro = get_classification_metric_collection(
            "multilabel", "macro", num_labels=config.classes, prefix="val/"
        )
        self.val_metrics_samples = get_classification_metric_collection(
            "multilabel", "sample", num_labels=config.classes, prefix="val/"
        )
        self.val_metrics_class = get_classification_metric_collection(
            "multilabel", None, num_labels=config.classes, prefix="val/"
        )
        self.test_metrics_micro = get_classification_metric_collection(
            "multilabel", "micro", num_labels=config.classes, prefix="test/"
        )
        self.test_metrics_macro = get_classification_metric_collection(
            "multilabel", "macro", num_labels=config.classes, prefix="test/"
        )
        self.test_metrics_samples = get_classification_metric_collection(
            "multilabel", "sample", num_labels=config.classes, prefix="test/"
        )
        self.test_metrics_class = get_classification_metric_collection(
            "multilabel", None, num_labels=config.classes, prefix="test/"
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.model(x)
        loss = self.loss(x_hat, y)
        self.log("train/loss", loss)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)

        # these are steps if interval is set to step
        max_intervals = int(
            self.trainer.max_epochs * len(self.trainer.datamodule.train_ds) / self.trainer.datamodule.batch_size
        )
        warmup = 10000 if max_intervals > 10000 else 100 if max_intervals > 100 else 0

        print(f"Optimizing for {max_intervals} steps with warmup for {warmup} steps")

        lr_scheduler = {
            "scheduler": LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=warmup,
                max_epochs=max_intervals,
                warmup_start_lr=self.lr / 10,
                eta_min=self.lr / 10,
            ),
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.model(x)
        loss = self.loss(x_hat, y)
        self.val_output_list += [{"loss": loss, "outputs": x_hat, "labels": y}]

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.val_output_list = []

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x["loss"] for x in self.val_output_list]).mean()
        self.log("val/loss", avg_loss)

        preds = torch.cat([x["outputs"] for x in self.val_output_list])
        labels = torch.cat([x["labels"] for x in self.val_output_list]).long()
        metrics_macro = self.val_metrics_macro(preds, labels)
        self.log_dict(metrics_macro)
        metrics_micro = self.val_metrics_micro(preds, labels)
        self.log_dict(metrics_micro)
        metrics_samples = self.val_metrics_samples(preds.unsqueeze(-1), labels.unsqueeze(-1))
        metrics_samples = {k: v.mean() for k, v in metrics_samples.items()}
        self.log_dict(metrics_samples)

        # get class names from datamodule
        class_names = NEW_LABELS
        metrics_class = self.val_metrics_class(preds, labels)
        classwise_acc = {
            f"val/ClasswiseAccuracy/{class_names[i]}": metrics_class["val/MultilabelAccuracy_class"][i]
            for i in range(len(class_names))
        }
        self.log_dict(classwise_acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.model(x)
        loss = F.binary_cross_entropy_with_logits(x_hat, y)
        self.test_output_list += [{"loss": loss, "outputs": x_hat, "labels": y}]

    def on_test_epoch_end(self):
        avg_loss = torch.stack([x["loss"] for x in self.test_output_list]).mean()
        self.log("test/loss", avg_loss)

        preds = torch.cat([x["outputs"] for x in self.test_output_list])
        labels = torch.cat([x["labels"] for x in self.test_output_list]).long()
        metrics_macro = self.test_metrics_macro(preds, labels)
        self.log_dict(metrics_macro)
        metrics_micro = self.test_metrics_micro(preds, labels)
        self.log_dict(metrics_micro)
        metrics_samples = self.test_metrics_samples(preds.unsqueeze(-1), labels.unsqueeze(-1))
        metrics_samples = {k: v.mean() for k, v in metrics_samples.items()}
        self.log_dict(metrics_samples)

        class_names = NEW_LABELS
        metrics_class = self.test_metrics_class(preds, labels)
        classwise_acc = {
            f"test/ClasswiseAccuracy/{class_names[i]}": metrics_class["test/MultilabelAccuracy_class"][i]
            for i in range(len(class_names))
        }
        self.log_dict(classwise_acc)

    def forward(self, batch):
        # because we are a wrapper, we call the inner function manually
        return self.model(batch)


def main(
    architecture: str = typer.Option("resnet18", help="Model name"),
    seed: int = typer.Option(42, help="Random seed"),
    lr: float = typer.Option(0.001, help="Learning rate"),
    epochs: int = typer.Option(100, help="Number of epochs"),
    bs: int = typer.Option(32, help="Batch size"),
    workers: int = typer.Option(8, help="Number of workers"),
    bandconfig: str = typer.Option("all", help="Band configuration, one of all, s2, s1, all_full, s2_full, s1_full"),
    use_wandb: bool = typer.Option(False, help="Use wandb for logging"),
):
    # set seed
    pl.seed_everything(seed)
    torch.set_float32_matmul_precision("medium")

    if bandconfig == "all":
        bands = STANDARD_BANDS[12]  # 10m + 20m Sentinel-2 + 10m Sentinel-1
    elif bandconfig == "s2":
        bands = STANDARD_BANDS[10]  # 10m + 20m Sentinel-2
    elif bandconfig == "s1":
        bands = STANDARD_BANDS[2]  # Sentinel-1
    elif bandconfig == "all_full":
        bands = STANDARD_BANDS["ALL"]
    elif bandconfig == "s2_full":
        bands = STANDARD_BANDS["S2"]
    elif bandconfig == "s1_full":
        bands = STANDARD_BANDS["S1"]
    else:
        raise ValueError(
            f"Unknown band configuration {bandconfig}, select one of all, s2, s1 or all_full, s2_full, s1_full. The "
            f"full versions include all bands whereas the non-full versions only include the 10m & 20m bands."
        )
    channels = len(bands)
    data_dirs = resolve_data_dir(None, allow_mock=True)

    dm = BENv2DataModule(
        data_dirs=data_dirs,
        batch_size=bs,
        num_workers_dataloader=workers,
        img_size=(channels, 120, 120),
    )

    # fixed model parameters
    num_classes = 19
    img_size = 120
    dropout = 0.25
    config = ILMConfiguration(
        network_type=ILMType.IMAGE_CLASSIFICATION,
        classes=num_classes,
        image_size=img_size,
        drop_rate=dropout,
        timm_model_name=architecture,
        channels=channels,
    )

    model = LitVisionEncoder(config, lr=lr)

    try:
        if use_wandb:
            logger = pl.loggers.WandbLogger(project="BENv2", log_model=True)
        else:
            logger = pl.loggers.WandbLogger(project="BENv2", log_model=False, mode="disabled")
    except ModuleNotFoundError:
        Messages.warn("Wandb is not installed. Please install wandb to use this feature.\n Continuing without logging.")
        Messages.info("Logging with DummyLogger.")
        logger = pl.loggers.logger.DummyLogger()
    logger.log_hyperparams(
        {
            "architecture": architecture,
            "seed": seed,
            "lr": lr,
            "epochs": epochs,
            "batch_size": bs,
            "workers": workers,
            "channels": channels,
        }
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/MultilabelAveragePrecision_macro",
        dirpath="./checkpoints",
        filename=f"{architecture}-{seed}-{channels}-val_mAP_macro-" + "{val/MultilabelAveragePrecision_macro:.2f}",
        save_top_k=1,
        mode="max",
        auto_insert_metric_name=False,
        enable_version_counter=False,  # remove version counter from filename (v1, v2, ...)
    )
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="val/MultilabelAveragePrecision_macro",
        patience=5,
        verbose=True,
        mode="max",
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=logger,
        accelerator="auto",
        callbacks=[checkpoint_callback, lr_monitor, early_stopping_callback],
    )

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm, ckpt_path="best")

    print("=== Training finished ===")


if __name__ == "__main__":
    typer.run(main)
