import csv
from pathlib import Path

import lmdb


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
    patch_ids: list,
    s1s2_mapping_file: str,
):
    # read the s1s2 mapping csv file
    with open(s1s2_mapping_file) as f:
        reader = csv.reader(f)
        data = list(reader)
    header = data[0]
    data = data[1:]
    patch_id_col = header.index("patch_id")
    s1_col = header.index("s1_name")
    s1_patch_ids = [x[s1_col] for x in data if x[patch_id_col] in patch_ids]
    assert len(s1_patch_ids) == len(
        patch_ids
    ), f"Number of S1 patches ({len(s1_patch_ids)}) does not match number of S2 patches ({len(patch_ids)})"
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
    env.close()


def gen_split_subset(
    from_file: str,
    to_file: str,
    num_elems_per_split: int,
):
    # read the csv file
    with open(from_file) as f:
        reader = csv.reader(f)
        data = list(reader)
    header = data[0]
    data = data[1:]
    splits = {x[1] for x in data}
    print(f"Found splits: {splits} in original csv file")
    # create the new csv file's folder
    Path(to_file).parent.mkdir(parents=True, exist_ok=True)
    # write the header
    with open(to_file, mode="w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        # write the data (first num_elems_per_split elements of each split)
        for split in splits:
            split_data = [x for x in data if x[1] == split]
            writer.writerows(split_data[:num_elems_per_split])
    # read the new csv file
    with open(to_file) as f:
        reader = csv.reader(f)
        data = list(reader)
    print(f"New csv file ({Path(to_file).name}) has {len(data)} lines")


if __name__ == "__main__":
    lmdb_file = "/home/leonard/data/BigEarthNet-V2/BigEarthNet-V2-LMDB"
    new_lmdb_file = "./BENv2/BigEarthNet-V2-LMDB"

    split_file = "/home/leonard/data/BigEarthNet-V2/patch_id_split_mapping.csv"
    new_split_file = "./BENv2/patch_id_split_mapping.csv"

    s1_file = "/home/leonard/data/BigEarthNet-V2/patch_id_s1_mapping.csv"
    new_s1_file = "./BENv2/patch_id_s1_mapping.csv"

    label_file = "/home/leonard/data/BigEarthNet-V2/patch_id_label_mapping.csv"
    new_label_file = "./BENv2/patch_id_label_mapping.csv"

    country_file = "/home/leonard/data/BigEarthNet-V2/patch_id_country_mapping.csv"
    new_country_file = "./BENv2/patch_id_country_mapping.csv"

    gen_split_subset(from_file=split_file, to_file=new_split_file, num_elems_per_split=9)

    # read new generated csv file, extract patch ids
    with open(new_split_file) as f:
        reader = csv.reader(f)
        data = list(reader)
    header = data[0]
    data = data[1:]
    patch_id_col = header.index("patch_id")
    patch_ids = {x[patch_id_col] for x in data}

    gen_csv_subset(from_file=s1_file, to_file=new_s1_file, patch_ids=patch_ids)

    gen_csv_subset(from_file=label_file, to_file=new_label_file, patch_ids=patch_ids)

    gen_csv_subset(from_file=country_file, to_file=new_country_file, patch_ids=patch_ids)

    gen_lmdb_subset(
        from_file=lmdb_file, to_file=new_lmdb_file, patch_ids=list(patch_ids), s1s2_mapping_file=new_s1_file
    )
