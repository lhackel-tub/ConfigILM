import csv
from pathlib import Path

import lmdb
import pandas as pd


def gen_csv_subset(
    from_file: str,
    to_file: str,
    patch_ids: set,
):
    # read the csv file
    with open(from_file) as f:
        reader = csv.reader(f)
        data = list(reader)
    header = data[0]
    data = data[1:]
    patch_id_col = header.index("patch_id")
    # create the new csv file's folder
    Path(to_file).parent.mkdir(parents=True, exist_ok=True)
    # write the header
    with open(to_file, mode="w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        # write the data (only the patch ids in patch_ids)
        for row in data:
            if row[patch_id_col] in patch_ids:
                writer.writerow(row)

    # read the new csv file
    with open(to_file) as f:
        reader = csv.reader(f)
        data = list(reader)
    print(f"New csv file ({Path(to_file).name}) has {len(data)} lines")


def gen_lmdb_subset(
    from_file: str,
    to_file: str,
    metadata: pd.DataFrame,
):
    # get the patch ids
    patch_ids = set(metadata["patch_id"])
    s1_patch_ids = set(metadata["s1_name"])

    # open the lmdb file
    env_read = lmdb.open(from_file, readonly=True, lock=False, meminit=False, readahead=True)
    # create the new lmdb file
    env_write = lmdb.open(to_file, map_size=1000 * 1000 * 1000)
    with env_write.begin(write=True) as txn:
        # write the patch data
        for patch_id in patch_ids:
            with env_read.begin(write=False) as txn_read:
                binary_patch_data = txn_read.get(patch_id.encode())
                txn.put(patch_id.encode(), binary_patch_data)
        # write the s1 patch data
        for s1_patch_id in s1_patch_ids:
            with env_read.begin(write=False) as txn_read:
                binary_patch_data = txn_read.get(s1_patch_id.encode())
                txn.put(s1_patch_id.encode(), binary_patch_data)
    env_read.close()
    env_write.close()

    # open the new lmdb file
    env = lmdb.open(to_file, readonly=True, lock=False, meminit=False, readahead=True)
    with env.begin(write=False) as txn:
        num_elems = txn.stat()["entries"]
    print(f"New lmdb file ({Path(to_file).name}) has {num_elems} elements")
    print(
        f"    It should be {len(patch_ids) + len(s1_patch_ids)} elements ({len(patch_ids)} S2 patches and "
        f"{len(s1_patch_ids)} S1 patches)"
    )
    env.close()


def gen_metadata_subset(
    from_file: str,
    to_file: str,
    patches_per_split: int,
):
    # read the metadata file
    metadata = pd.read_parquet(from_file)
    # create the new metadata file's folder
    Path(to_file).parent.mkdir(parents=True, exist_ok=True)
    # get rows with patches_per_split patches for each of train, validation, test
    train_metadata = metadata[metadata["split"] == "train"][:patches_per_split]
    val_metadata = metadata[metadata["split"] == "validation"][:patches_per_split]
    test_metadata = metadata[metadata["split"] == "test"][:patches_per_split]
    # write the new metadata file
    metadata_subset = pd.concat([train_metadata, val_metadata, test_metadata])
    metadata_subset.to_parquet(to_file)


def gen_extra_metadata_file(
    from_file: str,
    to_file: str,
    patches_per_version: int,
):
    # read the metadata file
    metadata = pd.read_parquet(from_file)
    # create the new metadata file's folder
    Path(to_file).parent.mkdir(parents=True, exist_ok=True)
    # get rows with patches_per_version patches for each of contains_seasonal_snow, contains_cloud_or_shadow for each
    # split (train, validation, test)
    train_snow_metadata = metadata[metadata["split"] == "train"][metadata["contains_seasonal_snow"]][
        :patches_per_version
    ]
    val_snow_metadata = metadata[metadata["split"] == "validation"][metadata["contains_seasonal_snow"]][
        :patches_per_version
    ]
    test_snow_metadata = metadata[metadata["split"] == "test"][metadata["contains_seasonal_snow"]][:patches_per_version]
    train_cloud_metadata = metadata[metadata["split"] == "train"][metadata["contains_cloud_or_shadow"]][
        :patches_per_version
    ]
    val_cloud_metadata = metadata[metadata["split"] == "validation"][metadata["contains_cloud_or_shadow"]][
        :patches_per_version
    ]
    test_cloud_metadata = metadata[metadata["split"] == "test"][metadata["contains_cloud_or_shadow"]][
        :patches_per_version
    ]
    # write the new metadata file
    metadata_subset = pd.concat(
        [
            train_snow_metadata,
            val_snow_metadata,
            test_snow_metadata,
            train_cloud_metadata,
            val_cloud_metadata,
            test_cloud_metadata,
        ]
    )
    metadata_subset.to_parquet(to_file)


if __name__ == "__main__":
    lmdb_file = "/home/leonard/data/BigEarthNet-V2/BigEarthNet-V2-LMDB"
    new_lmdb_file = "./BENv2/BigEarthNet-V2-LMDB"

    metadata_file = "/home/leonard/data/BigEarthNet-V2/metadata.parquet"
    new_metadata_file = "./BENv2/metadata.parquet"

    metadata_extra_file = "/home/leonard/data/BigEarthNet-V2/metadata_for_patches_with_snow_cloud_or_shadow.parquet"
    new_metadata_extra_file = "./BENv2/metadata_for_patches_with_snow_cloud_or_shadow.parquet"

    gen_metadata_subset(from_file=metadata_file, to_file=new_metadata_file, patches_per_split=6)
    gen_extra_metadata_file(from_file=metadata_extra_file, to_file=new_metadata_extra_file, patches_per_version=1)

    # read new metadata
    metadata = pd.read_parquet(new_metadata_file)
    metadata_extra = pd.read_parquet(new_metadata_extra_file)
    metadata = pd.concat([metadata, metadata_extra])

    gen_lmdb_subset(from_file=lmdb_file, to_file=new_lmdb_file, metadata=metadata)
