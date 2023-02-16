from datetime import datetime

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import parameter_count
from sklearn.metrics import accuracy_score
from torch import optim
from torchmetrics.classification import MultilabelAveragePrecision
from torchmetrics.classification import MultilabelF1Score
from tqdm import tqdm

from configvlm import ConfigVLM
from configvlm.util import Messages

# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


class LitVQAEncoder(pl.LightningModule):
    def __init__(
        self,
        config: ConfigVLM.VLMConfiguration,
        lr: float = 1e-3,
        max_epochs: int = 5,
    ):
        super().__init__()
        self.lr = lr
        self.max_epochs = max_epochs
        self.config = config

        self.model = ConfigVLM.ConfigVLM(config)

        self.start_time = datetime.now()
        Messages.hint(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M')}")
        self.twait = self.start_time

    def get_stats(self):
        # create example image
        if self.config.network_type == ConfigVLM.VLMType.VISION_CLASSIFICATION:
            dummy_input = torch.rand(
                [
                    1,
                    self.config.channels,
                    self.config.image_size,
                    self.config.image_size,
                ],
                device=self.device,
            )
        elif self.config.network_type == ConfigVLM.VLMType.VQA_CLASSIFICATION:
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
                torch.ones(
                    [1, self.config.max_sequence_length],
                    device=self.device,
                    dtype=torch.int,
                ),
            ]
        else:
            raise ValueError("Configuration type unknown")

        params = parameter_count(self)
        flops = FlopCountAnalysis(self, dummy_input)
        return {"flops": flops.total(), "params": params[""]}

    def _approximate_end_time(self):
        now = datetime.now()
        time_passed = now - self.start_time
        if self.current_epoch != 0:
            time_per_epoch = time_passed / (self.current_epoch + 1)
            remaining = time_per_epoch * (self.max_epochs - self.current_epoch - 1)
            final_time = now + remaining
            remaining_hrs = remaining.days * 24 + remaining.seconds // 3600
            remaining_mins = remaining.seconds // 60 - (remaining.seconds // 3600) * 60
            Messages.hint(
                f"Done at approx. {final_time.strftime('%Y-%m-%d %H:%M')}"
                f" ({remaining_hrs:02d}:{remaining_mins:02d}"
                f" remaining)"
            )

    def _disassemble_batch(self, batch):
        if self.config.network_type == ConfigVLM.VLMType.VISION_CLASSIFICATION:
            return batch
        elif self.config.network_type == ConfigVLM.VLMType.VQA_CLASSIFICATION:
            images, questions, labels = batch
            # For some reason questions come in here transposed as a list of Tensors
            # where the first elements of the question are in the first tensor (first
            # element of the list), all the second elements are in the second tensor
            # which is the second element of the list and so on.
            # So we first make it a list of lists and then a big tensor and then
            # transpose this tensor.
            # Now each tensor contains one question
            questions = torch.tensor(
                [x.tolist() for x in questions], device=self.device
            ).T.int()
            return (images, questions), labels
        else:
            raise ValueError(f"Configuration type '{self.config.network_type}' unknown")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = self._disassemble_batch(batch)
        x_hat = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(x_hat, y)
        self.log("train/loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = self._disassemble_batch(batch)
        x_hat = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(x_hat, y)
        return {"loss": loss, "outputs": x_hat, "labels": y}

    def validation_epoch_end(self, outputs):
        metrics = self.get_metrics(outputs)
        self.log("val/loss", metrics["avg_loss"])
        self.log("val/f1", metrics["avg_f1_score"])

        if self.config.network_type == ConfigVLM.VLMType.VISION_CLASSIFICATION:
            self.log("val/mAP (Micro)", metrics["map_score"]["micro"])
            self.log("val/mAP (Macro)", metrics["map_score"]["macro"])
        elif self.config.network_type == ConfigVLM.VLMType.VQA_CLASSIFICATION:
            self.log("val/Accuracy (LULC)", metrics["accuracy"]["LULC"])
            self.log("val/Accuracy (Yes-No)", metrics["accuracy"]["Yes/No"])
            self.log("val/Accuracy (Overall)", metrics["accuracy"]["Overall"])
            self.log("val/Accuracy (Average)", metrics["accuracy"]["Average"])
        else:
            raise ValueError(f"Configuration type '{self.config.network_type}' unknown")

        self._approximate_end_time()

    def test_step(self, batch, batch_idx):
        x, y = self._disassemble_batch(batch)
        x_hat = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(x_hat, y)
        return {"loss": loss, "outputs": x_hat, "labels": y}

    def test_epoch_end(self, outputs):
        metrics = self.get_metrics(outputs)
        self.log("test/loss", metrics["avg_loss"])
        self.log("test/f1", metrics["avg_f1_score"])

        if self.config.network_type == ConfigVLM.VLMType.VISION_CLASSIFICATION:
            self.log("test/mAP (Micro)", metrics["map_score"]["micro"])
            self.log("test/mAP (Macro)", metrics["map_score"]["macro"])
        elif self.config.network_type == ConfigVLM.VLMType.VQA_CLASSIFICATION:
            self.log("test/Accuracy (LULC)", metrics["accuracy"]["LULC"])
            self.log("test/Accuracy (Yes-No)", metrics["accuracy"]["Yes/No"])
            self.log("test/Accuracy (Overall)", metrics["accuracy"]["Overall"])
            self.log("test/Accuracy (Average)", metrics["accuracy"]["Average"])
        else:
            raise ValueError("Configuration type unknown")

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        # optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.01)
        interval = "step"  # Use this for batch_step
        # interval = "epoch"

        # these are steps if interval is set to step
        max_intervals = (
            self.trainer.max_epochs
            if interval == "epoch"
            else int(
                self.trainer.max_epochs
                * self.trainer.limit_train_batches
                * len(self.trainer.datamodule.train_ds)
                / self.trainer.datamodule.batch_size
            )
        )
        max_intervals = int(max_intervals)

        if interval == "step":
            # warmup epochs if updated each step
            # -> epoch means step in this case
            warmup = (
                10000 if max_intervals > 10000 else 100 if max_intervals > 100 else 0
            )
        else:
            # warmup epochs if updated each epoch
            warmup = 5 if max_intervals > 5 else 1 if max_intervals > 1 else 0

        Messages.hint(
            f"Optimizing for {max_intervals} {interval}s with warmup for "
            f"{warmup} {interval}s"
        )

        # lr_scheduler = {
        #    "scheduler": LinearWarmupCosineAnnealingLR(
        #        optimizer,
        #        warmup_epochs=warmup,
        #        max_epochs=max_intervals,
        #        warmup_start_lr=self.lr / 10,
        #    ),
        #    "name": "learning_rate",
        #    "interval": interval,
        #    "frequency": 1,
        # }
        return [optimizer]  # , [lr_scheduler]

    def get_metrics(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        logits = torch.cat([x["outputs"].cpu() for x in outputs], 0)
        labels = torch.cat(
            [x["labels"].cpu() for x in outputs], 0
        )  # Tensor of size (#samples x classes)

        # calculate accuracy by argmax only
        # this is the way it is done in the VBFormer paper
        # only valid for VQA task, not used in vision-only
        if self.config.network_type == ConfigVLM.VLMType.VQA_CLASSIFICATION:
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
        else:
            accuracy_dict = None

        f1_score = MultilabelF1Score(num_labels=self.config.classes, average=None).to(
            logits.device
        )(logits, labels)

        # calculate AP only for vision-only
        if self.config.network_type == ConfigVLM.VLMType.VISION_CLASSIFICATION:
            ap_micro = MultilabelAveragePrecision(
                num_labels=self.config.classes, average="micro"
            ).to(logits.device)(logits, labels.int())

            ap_macro = MultilabelAveragePrecision(
                num_labels=self.config.classes, average="macro"
            ).to(logits.device)(logits, labels.int())

            ap_score = {"micro": float(ap_micro), "macro": float(ap_macro)}
        else:
            ap_score = None

        avg_f1_score = float(
            torch.sum(f1_score) / self.config.classes
        )  # macro average f1 score

        return {
            "avg_loss": avg_loss,
            "avg_f1_score": avg_f1_score,
            "map_score": ap_score,
            "accuracy": accuracy_dict,
        }
