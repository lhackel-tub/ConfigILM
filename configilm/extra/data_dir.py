from os.path import isdir
from pathlib import Path
from typing import Mapping
from typing import Optional
from typing import Union

from configilm.util import Messages

dataset_paths = {
    "benv1": [
        "/data/leonard/BEN_VQA/",  # MARS
        "/faststorage/leonard/",  # ERDE
        "/mnt/storagecube/leonard/",  # last resort: storagecube (MARS)
        "/media/storagecube/leonard/",  # (ERDE)
    ],
    "cocoqa": [],
    "hrvqa": [
        "/mnt/storagecube/data/datasets/HRVQA-1.0 release",  # MARS Storagecube
        "/media/storagecube/data/datasets/HRVQA-1.0 release",  # ERDE Storagecube
    ],
    "rsvqa-hr": [
        "/mnt/storagecube/data/datasets/RSVQA/RSVQA-HR",  # MARS Storagecube
        "/media/storagecube/data/datasets/RSVQA/RSVQA-HR",  # ERDE Storagecube
    ],
    "rsvqa-lr": [
        "/mnt/storagecube/data/datasets/RSVQA/RSVQA-LR",  # MARS Storagecube
        "/media/storagecube/data/datasets/RSVQA/RSVQA-LR",  # ERDE Storagecube
    ],
}

mock_data_dir = Path(__file__).parent / "mock_data"
mock_data_path = {
    "benv1": {
        "images_lmdb": mock_data_dir / "BENv1" / "BigEarthNetEncoded.lmdb",
        "train.csv": mock_data_dir / "BENv1" / "train.csv",
        "val.csv": mock_data_dir / "BENv1" / "val.csv",
        "test.csv": mock_data_dir / "BENv1" / "test.csv",
    },
    "cocoqa": {
        "images": mock_data_dir / "COCO-QA" / "images",
        "train_data": mock_data_dir / "COCO-QA" / "cocoqa-2015-05-17" / "train",
        "test_data": mock_data_dir / "COCO-QA" / "cocoqa-2015-05-17" / "test",
    },
    "hrvqa": {
        "images": mock_data_dir / "HR-VQA" / "images",
        "train_data": mock_data_dir / "HR-VQA" / "jsons",
        "val_data": mock_data_dir / "HR-VQA" / "jsons",
        "test_data": mock_data_dir / "HR-VQA" / "jsons",
    },
    "rsvqa-hr": {
        "images": mock_data_dir / "RSVQA-HR" / "Images" / "Data",
        "train_data": mock_data_dir / "RSVQA-HR",
        "val_data": mock_data_dir / "RSVQA-HR",
        "test_data": mock_data_dir / "RSVQA-HR",
        "test_data_phili": mock_data_dir / "RSVQA-HR",
    },
    "rsvqa-lr": {
        "images": mock_data_dir / "RSVQA-LR" / "Images_LR",
        "train_data": mock_data_dir / "RSVQA-LR",
        "val_data": mock_data_dir / "RSVQA-LR",
        "test_data": mock_data_dir / "RSVQA-LR",
    },
    "rsvqaxben": {
        "images_lmdb": mock_data_dir / "BENv1" / "BigEarthNetEncoded.lmdb",
        "train_data": mock_data_dir / "VQA_RSVQAxBEN",
        "val_data": mock_data_dir / "VQA_RSVQAxBEN",
        "test_data": mock_data_dir / "VQA_RSVQAxBEN",
    },
}


def resolve_data_dir_for_ds(
    dataset_name: str,
    data_dir: Optional[Mapping[str, Union[str, Path]]],
    allow_mock: bool = False,
    force_mock: bool = False,
) -> Mapping[str, Union[str, Path]]:
    """
    Resolves the data directory for the given dataset name.

    :param dataset_name: Name of the dataset to resolve the data directory for.
    :param data_dir: Optional path to the data directory. If None, the default data
        directory will be used.
    :param allow_mock: allows mock data path to be returned
    :param force_mock: only mock data path will be returned. Useful for debugging with
        small data
    :return: a valid dir to the dataset if data_dir was none, otherwise data_dir
    """
    dataset_name = dataset_name.lower()
    if data_dir in [None, "none", "None"]:
        Messages.warn("No data directory provided, trying to resolve")
        path_dicts = dataset_paths.get(dataset_name, {})
        for pd in path_dicts:
            # check that all paths are valid
            valid = True
            for p in pd:
                if not isdir(Path(p).resolve()):
                    valid = False
                    break
            if valid:
                data_dir = {k: Path(v).resolve() for k, v in pd.items()}
                Messages.warn(f"Changing path to {data_dir}")
                break

    # using mock data if allowed and no other found or forced
    if data_dir in [None, "none", "None"] and allow_mock:
        Messages.warn("Mock data being used, no alternative available.")
        data_dir = mock_data_path[dataset_name]
    if force_mock:
        Messages.warn("Forcing Mock data")
        data_dir = mock_data_path[dataset_name]

    if data_dir in [None, "none", "None"]:
        raise AssertionError("Could not resolve data directory")
    else:
        return data_dir
