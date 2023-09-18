# import packages
from os.path import isfile
from typing import List
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import typer
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import parameter_count
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import accuracy_score
from torch import optim
from torchmetrics.classification import MultilabelF1Score
from tqdm import tqdm

from configilm import ConfigILM
from configilm.ConfigILM import _get_hf_model as get_huggingface_model
from configilm.ConfigILM import ILMConfiguration
from configilm.ConfigILM import ILMType
from configilm.extra.BEN_lmdb_utils import resolve_data_dir
from configilm.extra.CustomTorchClasses import LinearWarmupCosineAnnealingLR
from configilm.extra.DataModules.RSVQAxBEN_DataModule import RSVQAxBENDataModule

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

    def get_stats(self):
        # create example image
        dummy_input = [
            torch.rand(
                [
                    1,
                    self.config.channels,
                    self.config.image_size,
                    self.config.image_size,
                ],
                device=self.device,
            ),
            torch.ones([1, 32], device=self.device, dtype=torch.int),
        ]
        params = parameter_count(self)
        flops = FlopCountAnalysis(self, dummy_input)
        return {"flops": flops.total(), "params": params[""]}

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

        # these are steps if interval is set to step
        max_intervals = int(
            self.trainer.max_epochs
            * len(self.trainer.datamodule.train_ds)
            / self.trainer.datamodule.batch_size
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

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.val_output_list = []

    def validation_step(self, batch, batch_idx):
        x, y = self._disassemble_batch(batch)
        x_hat = self.model(x)
        loss = F.binary_cross_entropy_with_logits(x_hat, y)
        self.val_output_list += [{"loss": loss, "outputs": x_hat, "labels": y}]

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


def overwrite_vision_weights(model, vision_checkpoint):
    if vision_checkpoint is None:
        return model
    if not isfile(vision_checkpoint):
        print("Pretrained vision model not available, cannot load checkpoint")
        return model
    # load weights
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
            k: v for k, v in pretrained_dict["state_dict"].items() if k in model_dict
        }
    else:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

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
    return model


def main(
    vision_model: str = "mobilevit_s",
    text_model: str = "prajjwal1/bert-tiny",
    lr: float = 5e-4,
    epochs: int = 10,
    batch_size: int = 512,
    seed: int = 42,
    data_dir: Optional[str] = None,
    test_run: bool = False,
    num_workers_dataloader: int = 4,
    vision_checkpoint: Optional[str] = None,
    matmul_precision: str = "medium",
):
    if test_run:
        max_img_index = 10 * batch_size
        epochs = 10
    else:
        max_img_index = -1
    torch.set_float32_matmul_precision(matmul_precision)

    pl.seed_everything(seed, workers=True)

    img_size = 120
    channels = 10

    model_config = ILMConfiguration(
        timm_model_name=vision_model,
        hf_model_name=text_model,
        classes=1000,
        image_size=img_size,
        channels=channels,
        network_type=ILMType.VQA_CLASSIFICATION,
    )

    tags = ["Training", vision_model, text_model]
    if test_run:
        tags += ["Test Run"]
    if vision_checkpoint is not None:
        tags += ["Vision Pretraining"]

    monitor = "val/Accuracy (Average)"
    monitor_str = "AA"
    # checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        dirpath="./checkpoints",
        filename="lit4rsvqa-seed="
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
    lr_monitor = LearningRateMonitor(logging_interval="step")

    logger = TensorBoardLogger(save_dir="./logs/")
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        log_every_n_steps=5,
        logger=logger,
        check_val_every_n_epoch=2,
        callbacks=[checkpoint_callback, lr_monitor],
    )

    model = LitVisionEncoder(config=model_config, lr=lr)
    model = overwrite_vision_weights(model, vision_checkpoint)

    print(
        f"Model Stats: Params: {model.get_stats()['params']:15,d}\n"
        f"              Flops: {model.get_stats()['flops']:15,d}"
    )

    hf_tokenizer, _ = get_huggingface_model(
        model_name=text_model, load_pretrained_if_available=False
    )
    dm = RSVQAxBENDataModule(
        data_dir=resolve_data_dir(data_dir=data_dir, allow_mock=True),
        img_size=(channels, img_size, img_size),
        num_workers_dataloader=num_workers_dataloader,
        batch_size=batch_size,
        max_img_idx=max_img_index,
        tokenizer=hf_tokenizer,
    )

    logger.log_hyperparams(
        {
            "Vision Model": vision_model,
            "Text Model": text_model,
            "Learning Rate": lr,
            "Epochs": epochs,
            "Batch Size": batch_size,
            "Seed": seed,
            "# Workers": num_workers_dataloader,
            "Vision Checkpoint": vision_checkpoint,
            "GPU": torch.cuda.get_device_name(),
            "MatMul Precision": matmul_precision,
        }
    )

    trainer.fit(model=model, datamodule=dm)
    trainer.test(model=model, datamodule=dm, ckpt_path="best")

    print("=== Training finished ===")


if __name__ == "__main__":
    typer.run(main)
