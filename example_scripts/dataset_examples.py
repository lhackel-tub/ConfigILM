from typing import List
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import torch
import typer
from tqdm import tqdm

from configilm.extra import BEN_DataModule_LMDB_Encoder
from configilm.extra import COCOQA_DataModule
from configilm.extra import RSVQAxBEN_DataModule_LMDB_Encoder
from configilm.extra.BEN_lmdb_utils import resolve_ben_data_dir


def speedtest(
    datamodule: Tuple[
        str,
        Union[
            COCOQA_DataModule.COCOQADataModule,
            BEN_DataModule_LMDB_Encoder.BENDataModule,
            RSVQAxBEN_DataModule_LMDB_Encoder.RSVQAxBENDataModule,
        ],
    ],
):
    ds_name, dm = datamodule
    dl = dm.train_dataloader()
    print("\n==== Got dataloader ====")
    print(
        f"Testing with:\n"
        f"  Batchsize: {dm.batch_size:6d}\n"
        f"  # workers: {dm.num_workers_dataloader:6d}\n"
        f"       imgs: {dm.max_img_idx:6d}"
    )
    batch_voting: Union[List, None] = None
    if hasattr(dm, "seq_length"):
        # VQA dataset, show voting for seq_length
        batch_voting = [0] * dm.seq_length

    for i in range(1):
        for batch in tqdm(iter(dl), desc=f"Data Loading speed test epoch {i}"):
            if batch_voting is None:
                continue
            q = batch[1]
            pad = [torch.sum(x) for x in q]
            batch_voting[pad.index(0) - 1] += 1
    if batch_voting is not None:
        while batch_voting[-1] == 0:
            del batch_voting[-1]
        print(
            f"Done, voting length result = {batch_voting}"
            f"\n      len = {len(batch_voting)}"
        )
    else:
        print("Done")


def display_img(
    dataset: Tuple[
        str,
        Union[
            COCOQA_DataModule.COCOQADataSet,
            BEN_DataModule_LMDB_Encoder.BENDataSet,
            RSVQAxBEN_DataModule_LMDB_Encoder.RSVQAxBENDataSet,
        ],
    ],
    img_id: int,
):
    ds_name, ds = dataset
    # get img
    sample = ds[img_id]
    if len(sample) == 3:
        img, q, a = sample
    elif len(sample) == 2:
        img, lbl = sample
    else:
        img = None  # just for "Var not assigned"-warnings
        RuntimeError(
            f"Don't know how to interpret datasets returning "
            f"{len(sample)} elements."
        )
    assert img.shape[0] in [1, 3], (
        f"Don't know how to display {img[0]}-dimensional images. "
        f"Please use 1 or 3 channels for display."
    )
    # select channels and transpose (= move channels to last dim as expected by mpl)
    img = img.permute(2, 1, 0)
    # move to range 0-1
    img -= img.min()
    img /= img.max()

    # display
    # no axis
    plt.axis("off")

    # if interpolate we want to keep the "blockiness" of the pixels
    plt.imshow(img, interpolation="nearest")
    # first save then show, otherwise data is deleted by mpl
    # also remove most of the white boarder and increase base resolution to 200
    # (~780x780)
    try:
        plt.savefig(
            f"{ds_name}_{img_id}_ID[{ds.get_patchname_from_index(img_id)}].png",
            bbox_inches="tight",
            dpi=200,
        )
    except AttributeError:
        plt.savefig(f"{ds_name}_{img_id}.png", bbox_inches="tight", dpi=200)

    plt.show()


def main(
    dataset: str,
    function: str,
    data_dir: Union[str, None] = None,
    channels: int = 3,
    img_size: int = 120,
    # only relevant for display
    img_id: int = 5,
    # only relevant for speedtest
    max_img_index: int = 1024 * 100,
    workers: int = 8,
    bs: int = 64,
    seq_length: int = 64,
):
    dataset = dataset.lower()
    if dataset.lower() in ["ben", "bigearthnet"]:
        data_dir = resolve_ben_data_dir(data_dir, allow_mock=True)

        dm = BEN_DataModule_LMDB_Encoder.BENDataModule(
            data_dir=data_dir,
            img_size=(channels, img_size, img_size),
            max_img_idx=max_img_index,
            num_workers_dataloader=workers,
            batch_size=bs,
        )
    elif dataset.lower() in ["rsvqaxben"]:
        data_dir = resolve_ben_data_dir(data_dir)

        dm = RSVQAxBEN_DataModule_LMDB_Encoder.RSVQAxBENDataModule(
            data_dir=data_dir,
            img_size=(channels, img_size, img_size),
            max_img_idx=max_img_index,
            num_workers_dataloader=workers,
            batch_size=bs,
            seq_length=seq_length,
        )
    elif dataset.lower() in ["cocoqa", "coco-qa"]:
        data_dir = COCOQA_DataModule.resolve_cocoqa_data_dir(data_dir)

        dm = COCOQA_DataModule.COCOQADataModule(
            data_dir=data_dir,
            img_size=(3, img_size, img_size),
            max_img_idx=max_img_index,
            num_workers_dataloader=workers,
            batch_size=bs,
            seq_length=seq_length,
        )
    else:
        raise ValueError(f"Dataset '{dataset}' not known")
    dm.setup("fit")
    if dm.train_ds is not None:
        if function.lower() in ["display", "display_img"]:
            display_img((dataset, dm.train_ds), img_id=img_id)
        elif function.lower() in ["speedtest"]:
            speedtest((dataset, dm))
        else:
            raise ValueError(f"Function '{function}' not known")
    else:
        raise ValueError("Datamodule does not contain a train_ds")


if __name__ == "__main__":
    typer.run(main)
