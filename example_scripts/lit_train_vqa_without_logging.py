"""
This is an example script for vision classification using the BigEarthNet dataset.
It is basically a 1-to-1 application of the process described in the documentation under
Supervised Vision Classification.
"""
# import packages
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import typer
from sklearn.metrics import accuracy_score
from torch import optim
from torchmetrics.classification import MultilabelF1Score
from tqdm import tqdm

from configilm import ConfigVLM
from configilm.ConfigVLM import get_hf_model as get_huggingface_model
from configilm.ConfigVLM import VLMConfiguration
from configilm.ConfigVLM import VLMType
from configilm.extra.BEN_lmdb_utils import resolve_ben_data_dir
from configilm.extra.RSVQAxBEN_DataModule_LMDB_Encoder import RSVQAxBENDataModule


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
        config: ConfigVLM.ILMConfiguration,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.lr = lr
        self.config = config
        self.model = ConfigVLM.ConfigVLM(config)

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
        return {"loss": loss, "outputs": x_hat, "labels": y}

    def validation_epoch_end(self, outputs):
        metrics = self.get_metrics(outputs)

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
        return {"loss": loss, "outputs": x_hat, "labels": y}

    def test_epoch_end(self, outputs):
        metrics = self.get_metrics(outputs)

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
    val_epoch_interval: Optional[int] = 5,
):

    # for ampere GPUs set precision -> can also be 'high', see details at
    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision("medium")

    # seed for pytorch, numpy, python.random, Dataloader workers, spawned subprocesses
    pl.seed_everything(seed, workers=True)

    # get data
    hf_tokenizer, _ = get_huggingface_model(
        model_name=text_model, load_pretrained_if_available=False
    )

    model_config = VLMConfiguration(
        timm_model_name=vision_model,
        hf_model_name=text_model,
        classes=1000,
        image_size=image_size,
        channels=number_of_channels,
        drop_rate=drop_rate,
        network_type=VLMType.VQA_CLASSIFICATION,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        log_every_n_steps=1,
        check_val_every_n_epoch=val_epoch_interval,
        logger=False,
    )

    model = LitVisionEncoder(config=model_config, lr=lr)
    dm = RSVQAxBENDataModule(
        data_dir=resolve_ben_data_dir(data_dir, allow_mock=True),
        img_size=(number_of_channels, image_size, image_size),
        max_img_idx=max_img_index,
        num_workers_dataloader=num_workers,
        batch_size=batch_size,
        tokenizer=hf_tokenizer,
        seq_length=64,
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path="best")
    print("=== Training finished ===")


if __name__ == "__main__":
    typer.run(main)
