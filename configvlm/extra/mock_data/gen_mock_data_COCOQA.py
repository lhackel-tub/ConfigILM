import json
import os
import pathlib
import random
import shutil
from os.path import join
from typing import Sequence


def copy_part_of_json(
    base_path: str, f_name: str, elems: int = 25, random_seed: int = 0
):
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


def img_subset_from_jsons(
    base_path: str,
    json_base_path: str,
    jsons: Sequence[str],
):
    # collect names of all images
    subset_elems = []
    for i, json_f_name in enumerate(jsons):
        with open(join(json_base_path, json_f_name)) as read_file:
            data = json.load(read_file)
            subset_elems += [data[x]["img_id"] for x in data]

    image_name_mapping = {
        int(x[-14:-4]): x for x in os.listdir(join(base_path, "images"))
    }

    path = pathlib.Path(".").joinpath("images")
    path.mkdir(parents=True, exist_ok=True)
    for elem in subset_elems:
        img_name = image_name_mapping[int(elem)]
        shutil.copy(
            pathlib.Path(base_path).joinpath("images").joinpath(img_name),
            path.joinpath(img_name).resolve(),
        )


if __name__ == "__main__":
    base_path = "/home/lhackel/Documents/datasets/COCO-QA/"
    cocoqa_jsons = [
        "COCO-QA_QA_train.json",
        "COCO-QA_QA_test.json",
    ]
    for f in cocoqa_jsons:
        copy_part_of_json(base_path=base_path, f_name=f)

    img_subset_from_jsons(base_path=base_path, json_base_path="./", jsons=cocoqa_jsons)
