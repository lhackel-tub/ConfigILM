"""
This is an example script for vision classification using the BigEarthNet dataset.
It is basically a 1-to-1 application of the process described in the documentation under
Supervised Vision Classification.
"""
# import packages
from typing import Callable
from typing import Optional

import torch
import torch.nn.functional as F
import typer
from torch.utils.data import DataLoader
from torchmetrics.classification import AveragePrecision

from configilm import ConfigILM
from configilm.ConfigILM import ILMConfiguration
from configilm.ConfigILM import ILMType
from configilm.extra import BEN_lmdb_utils
from configilm.extra._defaults import default_train_transform
from configilm.extra._defaults import default_transform
from configilm.extra.DataSets.BEN_DataSet import BENDataSet

__author__ = "Leonard Hackel - BIFOLD/RSiM TU Berlin"


def get_metrics(outputs, classes: int):
    avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
    logits = torch.cat([x["outputs"].cpu() for x in outputs], 0)
    labels = torch.cat([x["labels"].cpu() for x in outputs], 0)  # Tensor of size (#samples x classes)

    # calculate AP
    ap_macro = AveragePrecision(num_labels=classes, average="macro", task="multilabel").to(logits.device)(
        logits, labels.int()
    )

    return {
        "avg_loss": avg_loss,
        "map_macro": ap_macro,
    }


def eval_model(
    model: torch.nn.Module,
    dl: torch.utils.data.DataLoader,
    loss_fn: Callable,
    device: torch.device,
    classes: int,
):
    model.eval()
    with torch.no_grad():
        output_list = []
        for i, batch in enumerate(dl):
            img, label = batch
            img = img.to(device)
            label = label.to(device)

            output = model(img)
            loss = loss_fn(output, label)
            output_list += [{"loss": loss, "outputs": output, "labels": label}]
    # print metrics
    metrics = get_metrics(output_list, classes)
    return metrics


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
):
    # some static parameters
    seed = 42

    torch.manual_seed(seed)
    # for ampere GPUs set precision -> can also be 'high', see details at
    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision("medium")

    # seed for pytorch, numpy, python.random, Dataloader workers, spawned subprocesses

    model_config = ILMConfiguration(
        timm_model_name=vision_model,
        classes=19,
        image_size=image_size,
        channels=number_of_channels,
        network_type=ILMType.IMAGE_CLASSIFICATION,
    )

    model = ConfigILM.ConfigILM(model_config)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = F.binary_cross_entropy_with_logits

    img_size = (number_of_channels, image_size, image_size)

    ben_mean, ben_std = BEN_lmdb_utils.band_combi_to_mean_std(img_size[0])
    train_transform = default_train_transform(img_size=(img_size[1], img_size[2]), mean=ben_mean, std=ben_std)
    transform = default_transform(img_size=(img_size[1], img_size[2]), mean=ben_mean, std=ben_std)

    train_ds = BENDataSet(
        BEN_lmdb_utils.resolve_data_dir(data_dir, allow_mock=True),
        split="train",
        transform=train_transform,
        max_img_idx=max_img_index,
        img_size=img_size,
    )
    val_ds = BENDataSet(
        BEN_lmdb_utils.resolve_data_dir(data_dir, allow_mock=True),
        split="val",
        transform=transform,
        max_img_idx=max_img_index,
        img_size=img_size,
    )

    test_ds = BENDataSet(
        BEN_lmdb_utils.resolve_data_dir(data_dir, allow_mock=True),
        split="test",
        transform=transform,
        max_img_idx=max_img_index,
        img_size=img_size,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(epochs):
        running_loss = 0.0
        # train loop
        model.train()
        for i, batch in enumerate(train_dl):
            img, label = batch
            img = img.to(device)
            label = label.to(device)

            optim.zero_grad()
            output = model(img)
            loss = loss_fn(output, label)
            loss.backward()
            optim.step()
            if i % 10 == 9:  # print every 10 mini-batches
                print(f"[{epoch + 1:3d}, {i + 1:5d}] loss: {running_loss / 10:.4f}")
                running_loss = 0.0

        # validation
        # print metrics
        metrics = eval_model(model, val_dl, loss_fn, device, model_config.classes)
        print(f"Epoch {epoch + 1:3d} validation loss: {metrics['avg_loss']:.4f}")
        print(f"Epoch {epoch + 1:3d} validation mAP (Macro): {metrics['map_macro']:.4f}")

    # test
    # print metrics
    metrics = eval_model(model, test_dl, loss_fn, device, model_config.classes)
    print(f"Test loss: {metrics['avg_loss']:.4f}")
    print(f"Test mAP (Macro): {metrics['map_macro']:.4f}")

    print("\n=== Training finished ===")


if __name__ == "__main__":
    typer.run(main)
