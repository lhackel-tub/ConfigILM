import os
from dataclasses import dataclass
from os.path import isfile
from typing import Optional
from typing import Union

import pytorch_lightning as pl
import torch
import typer
import wandb
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from timm_huggingface_vqa import LitVQAEncoder

from configvlm.ConfigVLM import get_hf_model as get_huggingface_model
from configvlm.ConfigVLM import VLMConfiguration
from configvlm.util import Messages

__author__ = "Leonard Hackel - BIFOLD/RSiM TU Berlin"
os.environ["WANDB_START_METHOD"] = "thread"
wandb_api_key = os.environ["WANDB_API_KEY"]


@dataclass
class DatasetConfig:
    num_classes: int
    data_module: pl.LightningDataModule
    project_logging_name: str
    text_model_used: bool

    def __init__(
        self,
        data_module: pl.LightningDataModule,
        num_classes: int,
        project_logging_name: str,
        text_model_used: bool,
    ):
        self.data_module = data_module
        self.num_classes = num_classes
        self.project_logging_name = project_logging_name
        self.text_model_used = text_model_used


def get_dataset(
    dataset_name: str,
    image_shape: Union[tuple, list],
    num_workers: int,
    batch_size: int,
    max_img_index: int,
    data_dir: Optional[str],
    tokenizer=None,
):
    dataset_name = dataset_name.lower()
    if dataset_name in ["ben", "bigearthnet"]:
        from configvlm.extra.BEN_DataModule_LMDB_Encoder import BENDataModule
        from configvlm.extra.BEN_lmdb_utils import resolve_ben_data_dir

        dm = BENDataModule(
            data_dir=resolve_ben_data_dir(data_dir),
            img_size=image_shape,
            max_img_idx=max_img_index,
            num_workers_dataloader=num_workers,
            batch_size=batch_size,
        )
        return DatasetConfig(
            data_module=dm,
            num_classes=19,
            project_logging_name="BigEarthNet19",
            text_model_used=False,
        )
    elif dataset_name in ["rsvqaxben", "ben-vqa", "benvqa"]:
        from configvlm.extra.RSVQAxBEN_DataModule_LMDB_Encoder import (
            RSVQAxBENDataModule,
        )
        from configvlm.extra.BEN_lmdb_utils import resolve_ben_data_dir

        dm = RSVQAxBENDataModule(
            data_dir=resolve_ben_data_dir(data_dir),
            img_size=image_shape,
            max_img_idx=max_img_index,
            num_workers_dataloader=num_workers,
            batch_size=batch_size,
            tokenizer=tokenizer,
            seq_length=64,
        )
        return DatasetConfig(
            data_module=dm,
            num_classes=1000,
            project_logging_name="RSVQAxBEN",
            text_model_used=True,
        )
    elif dataset_name in ["cocoqa", "coco-qa"]:
        from configvlm.extra.COCOQA_DataModule import (
            COCOQADataModule,
            resolve_cocoqa_data_dir,
        )

        dm = COCOQADataModule(
            data_dir=resolve_cocoqa_data_dir(data_dir),
            img_size=image_shape,
            max_img_idx=max_img_index,
            num_workers_dataloader=num_workers,
            batch_size=batch_size,
            tokenizer=tokenizer,
            seq_length=64,
        )
        return DatasetConfig(
            data_module=dm,
            num_classes=430,
            project_logging_name="COCO-QA",
            text_model_used=True,
        )
    else:
        raise NotImplementedError(f"Dataset '{dataset_name}' not known.")


def main(
    data_dir: Optional[str] = None,
    seed: int = 42,
    number_of_channels: int = 12,
    image_size: int = 120,
    batch_size: int = 32,
    num_workers: int = 4,
    max_img_index: int = 7 * 128,
    epochs: int = 10,
    text_model: str = "prajjwal1/bert-tiny",
    vision_model: str = "mobilevit_xxs",
    vision_checkpoint: Optional[str] = None,
    logging_model_name: str = "Unnamed Model",
    val_epoch_interval: Optional[int] = None,
    resume_from_checkpoint: Optional[str] = None,
    early_stopping_patience: int = 5,
    lr: float = 5e-4,
    drop_rate: float = 0.2,
    offline: bool = False,
    disable_logging: bool = False,
    dataset: str = "BEN",
):
    assert (
        resume_from_checkpoint is None or vision_checkpoint is None
    ), "Provided both checkpoints, please use only one"

    # seed for pytorch, numpy, python.random, Dataloader workers, spawned subprocesses
    pl.seed_everything(seed, workers=True)

    # Key is available by wandb, project name can be chosen at will
    wandb.login(key=wandb_api_key)
    # disable logging gets priority over online/offline
    if disable_logging:
        wandb_mode = "disabled"
    else:
        wandb_mode = "offline" if offline else "online"

    # get data
    hf_tokenizer, _ = get_huggingface_model(
        model_name=text_model, load_pretrained_if_available=False
    )
    dataset_config = get_dataset(
        dataset_name=dataset,
        image_shape=(number_of_channels, image_size, image_size),
        num_workers=num_workers,
        batch_size=batch_size,
        max_img_index=max_img_index,
        data_dir=data_dir,
        tokenizer=hf_tokenizer,
    )

    model_config = VLMConfiguration(
        timm_model_name=vision_model,
        hf_model_name=text_model,
        classes=dataset_config.num_classes,
        image_size=image_size,
        channels=number_of_channels,
        drop_rate=drop_rate
        # TODO add how to set it to VQA
    )
    model = LitVQAEncoder(config=model_config, max_epochs=epochs, lr=lr)

    model_stats = model.get_stats()
    Messages.hint(
        f"Model stats:\n"
        f"    Flops:  {model_stats['flops']:18,d}\n"
        f"    Params: {model_stats['params']:18,d} "
    )

    wandb_logger = WandbLogger(
        project=dataset_config.project_logging_name,
        log_model=not offline,
        tags=[logging_model_name],
        # keyword arg directly to wandb.init()
        mode=wandb_mode,
    )

    # logger config
    wandb_logger.watch(model, log="all")

    monitor = "val/mAP (Micro)" if not dataset_config.text_model_used else "val/f1"
    monitor_str = "mAP_micro" if not dataset_config.text_model_used else "F1_Score"
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

    val_epoch_interval = (
        val_epoch_interval
        if val_epoch_interval is not None
        else 5
        if not dataset_config.text_model_used
        else 1
    )
    Messages.hint(f"Validation every {val_epoch_interval} epochs")

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices="auto",
        check_val_every_n_epoch=val_epoch_interval,
        logger=wandb_logger,
        log_every_n_steps=5,
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        resume_from_checkpoint=resume_from_checkpoint,
    )

    if vision_checkpoint is not None:
        # try to load checkpoint
        if not isfile(vision_checkpoint):
            Messages.warn(
                "Pretrained vision model not available, cannot load checkpoint"
            )
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
                Messages.warn(f"Key '{key}' size mismatch, removing from loading")

            # overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)

            # load the new state dict
            model.load_state_dict(model_dict)
            Messages.success("Vision Model checkpoint loaded")

    wandb_logger.log_hyperparams(
        {
            "Vision Model": vision_model,
            "Text Model": text_model if dataset_config.text_model_used else "unused",
            "Seed": seed,
            "Epochs": epochs,
            "Channels": number_of_channels,
            "Image Size": image_size,
            "Max. Image Index": max_img_index,
            "Batch Size": batch_size,
            "# Workers": num_workers,
            "Vision Checkpoint": vision_checkpoint,
            "GPU": torch.cuda.get_device_name(),
            "Validation Interval": val_epoch_interval,
            "From Checkpoint": resume_from_checkpoint,
            "Early Stopping Patience": early_stopping_patience,
            "Learning Rate": lr,
            "Drop Rate": drop_rate,
        }
    )

    trainer.fit(model, datamodule=dataset_config.data_module)

    trainer.test(model, datamodule=dataset_config.data_module, ckpt_path="best")

    Messages.success("Finished Training")


if __name__ == "__main__":
    typer.run(main)
