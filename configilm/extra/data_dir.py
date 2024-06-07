from os.path import isdir
from pathlib import Path
from typing import Mapping
from typing import Optional

from configilm.util import Messages

mars_storagecube = Path("/mnt/storagecube")
erde_storagecube = Path("/media/storagecube")
mars_storagecube_datasets = mars_storagecube / "data" / "datasets"
erde_storagecube_datasets = erde_storagecube / "data" / "datasets"
mars_data_dir = Path("/data/leonard")
erde_data_dir = Path("/faststorage/leonard")
pluto_local = Path("/home/leonard/data/")

dataset_paths = {
    "benv1": [
        # MARS
        {
            "images_lmdb": mars_data_dir / "BEN_VQA" / "BigEarthNetEncoded.lmdb",
            "train_data": mars_data_dir / "BEN_VQA" / "train.csv",
            "val_data": mars_data_dir / "BEN_VQA" / "val.csv",
            "test_data": mars_data_dir / "BEN_VQA" / "test.csv",
        },
        # ERDE
        {
            "images_lmdb": erde_data_dir / "BigEarthNetEncoded.lmdb",
            "train_data": erde_data_dir / "train.csv",
            "val_data": erde_data_dir / "val.csv",
            "test_data": erde_data_dir / "test.csv",
        },
        # MARS Storagecube
        {
            "images_lmdb": mars_storagecube / "leonard" / "BigEarthNetEncoded.lmdb",
            "train_data": mars_storagecube / "leonard" / "train.csv",
            "val_data": mars_storagecube / "leonard" / "val.csv",
            "test_data": mars_storagecube / "leonard" / "test.csv",
        },
        # ERDE Storagecube
        {
            "images_lmdb": erde_storagecube / "leonard" / "BigEarthNetEncoded.lmdb",
            "train_data": erde_storagecube / "leonard" / "train.csv",
            "val_data": erde_storagecube / "leonard" / "val.csv",
            "test_data": erde_storagecube / "leonard" / "test.csv",
        },
    ],
    "benv2": [
        # MARS
        {"images_lmdb": "INVALID_PATH"},
        # ERDE
        {
            "images_lmdb": Path("/faststorage") / "BigEarthNet-V2" / "BigEarthNet-V2-LMDB",
            "split_csv": Path("/faststorage") / "BigEarthNet-V2" / "patch_id_split_mapping.csv",
            "s1_mapping_csv": Path("/faststorage") / "BigEarthNet-V2" / "patch_id_s1_mapping.csv",
            "labels_csv": Path("/faststorage") / "BigEarthNet-V2" / "patch_id_label_mapping.csv",
        },
        # PLUTO
        {
            "images_lmdb": pluto_local / "BigEarthNet-V2" / "BigEarthNet-V2-LMDB",
            "split_csv": pluto_local / "BigEarthNet-V2" / "patch_id_split_mapping.csv",
            "s1_mapping_csv": pluto_local / "BigEarthNet-V2" / "patch_id_s1_mapping.csv",
            "labels_csv": pluto_local / "BigEarthNet-V2" / "patch_id_label_mapping.csv",
        },
    ],
    "cocoqa": [],
    "hrvqa": [
        # MARS Storagecube
        {
            "images": mars_storagecube_datasets / "HRVQA-1.0 release" / "images",
            "train_data": mars_storagecube_datasets / "HRVQA-1.0 release" / "jsons",
            "val_data": mars_storagecube_datasets / "HRVQA-1.0 release" / "jsons",
            "test_data": mars_storagecube_datasets / "HRVQA-1.0 release" / "jsons",
        },
        # ERDE Storagecube
        {
            "images": erde_storagecube_datasets / "HRVQA-1.0 release" / "images",
            "train_data": erde_storagecube_datasets / "HRVQA-1.0 release" / "jsons",
            "val_data": erde_storagecube_datasets / "HRVQA-1.0 release" / "jsons",
            "test_data": erde_storagecube_datasets / "HRVQA-1.0 release" / "jsons",
        },
    ],
    "rsvqa-hr": [
        # MARS Storagecube
        {
            "images": mars_storagecube_datasets / "RSVQA" / "RSVQA-HR" / "Images" / "Data",
            "train_data": mars_storagecube_datasets / "RSVQA" / "RSVQA-HR",
            "val_data": mars_storagecube_datasets / "RSVQA" / "RSVQA-HR",
            "test_data": mars_storagecube_datasets / "RSVQA" / "RSVQA-HR",
            "test_phili_data": mars_storagecube_datasets / "RSVQA" / "RSVQA-HR",
        },
        # ERDE Storagecube
        {
            "images": erde_storagecube_datasets / "RSVQA" / "RSVQA-HR" / "Images" / "Data",
            "train_data": erde_storagecube_datasets / "RSVQA" / "RSVQA-HR",
            "val_data": erde_storagecube_datasets / "RSVQA" / "RSVQA-HR",
            "test_data": erde_storagecube_datasets / "RSVQA" / "RSVQA-HR",
            "test_phili_data": erde_storagecube_datasets / "RSVQA" / "RSVQA-HR",
        },
    ],
    "rsvqa-lr": [
        # MARS Storagecube
        {
            "images": mars_storagecube_datasets / "RSVQA" / "RSVQA-LR" / "Images_LR",
            "train_data": mars_storagecube_datasets / "RSVQA" / "RSVQA-LR",
            "val_data": mars_storagecube_datasets / "RSVQA" / "RSVQA-LR",
            "test_data": mars_storagecube_datasets / "RSVQA" / "RSVQA-LR",
        },
        # ERDE Storagecube
        {
            "images": erde_storagecube_datasets / "RSVQA" / "RSVQA-LR" / "Images_LR",
            "train_data": erde_storagecube_datasets / "RSVQA" / "RSVQA-LR",
            "val_data": erde_storagecube_datasets / "RSVQA" / "RSVQA-LR",
            "test_data": erde_storagecube_datasets / "RSVQA" / "RSVQA-LR",
        },
    ],
    "rsvqaxben": [
        # MARS Storagecube
        {
            "images_lmdb": mars_storagecube / "leonard" / "BigEarthNetEncoded.lmdb",
            "train_data": mars_storagecube_datasets / "RSVQAxBEN",
            "val_data": mars_storagecube_datasets / "RSVQAxBEN",
            "test_data": mars_storagecube_datasets / "RSVQAxBEN",
        },
        # ERDE Storagecube
        {
            "images_lmdb": erde_storagecube / "leonard" / "BigEarthNetEncoded.lmdb",
            "train_data": erde_storagecube_datasets / "RSVQAxBEN",
            "val_data": erde_storagecube_datasets / "RSVQAxBEN",
            "test_data": erde_storagecube_datasets / "RSVQAxBEN",
        },
    ],
    "throughput_test": [
        {
            # just use this file as a placeholder, as the result is not used anyway but needed for compatibility with
            # other datasets
            "current_path": Path(__file__).parent,
        }
    ],
}

mock_data_dir = Path(__file__).parent / "mock_data"
mock_data_path = {
    "benv1": {
        "images_lmdb": mock_data_dir / "BENv1" / "BigEarthNetEncoded.lmdb",
        "train_data": mock_data_dir / "BENv1" / "train.csv",
        "val_data": mock_data_dir / "BENv1" / "val.csv",
        "test_data": mock_data_dir / "BENv1" / "test.csv",
    },
    "benv2": {
        "images_lmdb": mock_data_dir / "BENv2" / "BigEarthNet-V2-LMDB",
        "split_csv": mock_data_dir / "BENv2" / "patch_id_split_mapping.csv",
        "s1_mapping_csv": mock_data_dir / "BENv2" / "patch_id_s1_mapping.csv",
        "labels_csv": mock_data_dir / "BENv2" / "patch_id_label_mapping.csv",
    },
    "cocoqa": {
        "images": mock_data_dir / "COCO-QA" / "images",
        "train_data": mock_data_dir / "COCO-QA" / "cocoqa-2015-05-17" / "train",
        "test_data": mock_data_dir / "COCO-QA" / "cocoqa-2015-05-17" / "test",
    },
    "hrvqa": {
        "images": mock_data_dir / "HRVQA" / "images",
        "train_data": mock_data_dir / "HRVQA" / "jsons",
        "val_data": mock_data_dir / "HRVQA" / "jsons",
        "test_data": mock_data_dir / "HRVQA" / "jsons",
    },
    "rsvqa-hr": {
        "images": mock_data_dir / "RSVQA-HR" / "Images" / "Data",
        "train_data": mock_data_dir / "RSVQA-HR",
        "val_data": mock_data_dir / "RSVQA-HR",
        "test_data": mock_data_dir / "RSVQA-HR",
        "test_phili_data": mock_data_dir / "RSVQA-HR",
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
    "throughput_test": {
        # empty dict, as the result is not used anyway but needed for compatibility with other datasets
    },
}


def resolve_data_dir_for_ds(
    dataset_name: str,
    data_dir_mapping: Optional[Mapping[str, Path]],
    allow_mock: bool = False,
    force_mock: bool = False,
) -> Mapping[str, Path]:
    """
    Resolves the data directory for the given dataset name.

    :param dataset_name: Name of the dataset to resolve the data directory for.
    :param data_dir_mapping: Optional path to the data directory. If None, the default data
        directory will be used.
    :param allow_mock: allows mock data path to be returned
    :param force_mock: only mock data path will be returned. Useful for debugging with
        small data
    :return: a valid dir to the dataset if data_dir was none, otherwise data_dir
    """
    dataset_name = dataset_name.lower()
    if data_dir_mapping is None:
        Messages.info("No data directory provided, trying to resolve")
        path_dicts = dataset_paths.get(dataset_name, {})
        assert type(path_dicts) == list, f"Invalid path_dicts for {dataset_name}"
        for pd in path_dicts:
            # check that all paths are valid
            valid = True
            for p in pd:
                if not isdir(Path(p).resolve()):
                    valid = False
                    break
            if valid:
                data_dir_mapping = {k: Path(v).resolve() for k, v in pd.items()}
                Messages.info(f"Changing path to {data_dir_mapping}")
                break

    # using mock data if allowed and no other found or forced
    if data_dir_mapping is None and allow_mock:
        Messages.warn("Mock data being used, no alternative available.")
        mock_dir = mock_data_path[dataset_name]
        assert isinstance(mock_dir, Mapping), f"Invalid mock_dir for {dataset_name}"
        data_dir_mapping = mock_dir
    if force_mock:
        Messages.warn("Forcing Mock data")
        mock_dir = mock_data_path[dataset_name]
        assert isinstance(mock_dir, Mapping), f"Invalid mock_dir for {dataset_name}"
        data_dir_mapping = mock_dir

    if data_dir_mapping is None:
        raise AssertionError("Could not resolve data directory")
    else:
        # convert all values to a Path
        return {k: Path(v) for k, v in data_dir_mapping.items()}
