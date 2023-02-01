import json
from os.path import join
from typing import Collection

import lmdb


def copy_part_of_json(base_path: str, f_name: str, elems: int = 25):
    with open(join(base_path, f_name)) as read_file:
        data = json.load(read_file)

    data_new = {}
    for key in data:
        data_new[key] = data[key]
        if len(data_new) >= elems:
            break

    with open(f_name, "w") as f:
        json.dump(data_new, f)


def lmdb_subset_from_jsons(
    base_path: str, f_name: str, json_base_path: str, jsons: Collection[str]
):
    # collect names of all S2 patches
    subset_elems = []
    for json_f_name in jsons:
        with open(join(json_base_path, json_f_name)) as read_file:
            data = json.load(read_file)
            for k in data:
                subset_elems += [data[k]["S2_name"]]

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
    base_path_json = "/home/lhackel/Documents/datasets/BEN/VQA_RSVQAxBEN"
    copy_part_of_json(base_path=base_path_json, f_name="RSVQAxBEN_QA_train.json")
    copy_part_of_json(base_path=base_path_json, f_name="RSVQAxBEN_QA_val.json")
    copy_part_of_json(base_path=base_path_json, f_name="RSVQAxBEN_QA_test.json")

    lmdb_subset_from_jsons(
        base_path="/home/lhackel/Documents/datasets/BEN/",
        f_name="BigEarthNetEncoded.lmdb",
        json_base_path="",
        jsons=[
            "RSVQAxBEN_QA_train.json",
            "RSVQAxBEN_QA_val.json",
            "RSVQAxBEN_QA_test.json",
        ],
    )
