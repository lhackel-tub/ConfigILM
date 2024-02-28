import csv
import json
import pathlib
import random
from os.path import join
from typing import Sequence

import lmdb


def copy_part_of_json(base_path: str, f_name: str, elems: int = 10, random_seed: int = 0):
    random.seed(random_seed)
    with open(join(base_path, f_name)) as read_file:
        data = json.load(read_file)

    new_keys = list(data.keys())
    random.shuffle(new_keys)
    data_new = {}
    for key in new_keys:
        data_new[key] = data[key]
        if len(data_new) >= elems:
            break

    path = pathlib.Path(".").joinpath(f_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data_new, f)


def lmdb_subset_from_jsons(
    base_path: str,
    f_name: str,
    json_base_path: str,
    jsons: Sequence[str],
    target_csv_names: Sequence[str],
):
    assert len(jsons) == len(target_csv_names), "target names have to be as many as jsons"
    # collect names of all S2 patches
    subset_elems = []
    for i, json_f_name in enumerate(jsons):
        with open(join(json_base_path, json_f_name)) as read_file:
            data = json.load(read_file)
            s2_names = [data[x]["S2_name"] for x in data]
            subset_elems += s2_names
        with open(join(json_base_path, target_csv_names[i]), mode="w") as f:
            writer = csv.writer(f)
            writer.writerows([[x] for x in s2_names])

    data = {}
    # read data
    lmdb_dir = join(base_path, f_name)
    env = lmdb.open(lmdb_dir, readonly=True, lock=False, meminit=False, readahead=True)
    for key in subset_elems:
        bin_key = str(key).encode()
        with env.begin(write=False) as txn:
            binary_patch_data = txn.get(bin_key)
            data[key] = binary_patch_data
    env.close()

    # write data
    env = lmdb.open(f_name, map_size=1000 * 1000 * 1000)
    txn = env.begin(write=True)
    i = 0
    for key in data:
        bin_key = str(key).encode()
        txn.put(key=bin_key, value=data[key])
        i += 1

        if i % 100 == 0:
            txn.commit()
            txn = env.begin(write=True)
    if i % 100 != 0:
        txn.commit()
    env.close()


if __name__ == "__main__":
    base_path = "/media/lhackel/My Passport/lhackel/Datasets/BEN"
    rsvqaxben_jsons = [
        "VQA_RSVQAxBEN/RSVQAxBEN_QA_train.json",
        "VQA_RSVQAxBEN/RSVQAxBEN_QA_val.json",
        "VQA_RSVQAxBEN/RSVQAxBEN_QA_test.json",
    ]
    for f in rsvqaxben_jsons:
        copy_part_of_json(base_path=base_path, f_name=f)

    lmdb_subset_from_jsons(
        base_path=base_path,
        f_name="BigEarthNetEncoded.lmdb",
        json_base_path="./",
        jsons=rsvqaxben_jsons,
        target_csv_names=["train.csv", "val.csv", "test.csv"],
    )
