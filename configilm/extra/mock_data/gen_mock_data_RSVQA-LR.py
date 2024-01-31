import json
import os
import pathlib
import random
import shutil
from os.path import join
from typing import Sequence


def copy_part_of_json(base_path: str, f_name: str, elems: int = 100, random_seed: int = 0):
    random.seed(random_seed)
    with open(join(base_path, f_name)) as read_file:
        data = json.load(read_file)["images"]
    assert len(data) >= elems, f"Cannot select {elems} from {len(data)} elements."
    random.shuffle(data)
    data_new = data[:elems]
    data_new = sorted(data_new, key=lambda x: x["id"])

    assert len({x["id"] for x in data_new}) == elems, (
        f"Seed {random_seed} selects elements multiple times - please select different"
        f" seed or change number of elements"
    )

    print(f"Selected {len(data_new)} elements of which " f"{len([x for x in data_new if x['active']])} are active")

    path = pathlib.Path(".") / "RSVQA-LR" / f_name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"images": data_new}, f)

    # copy answers to same questions
    qids_l = [x["questions_ids"] for x in data_new if x["active"]]
    qids = {x for sublist in qids_l for x in sublist}
    f_name_q = f_name.split("_images")[0] + "_questions.json"
    if os.path.isfile(join(base_path, f_name_q)):
        with open(join(base_path, f_name_q)) as read_file:
            data = json.load(read_file)["questions"]
        data_new = [x for x in data if x["id"] in qids]
        path = pathlib.Path(".") / "RSVQA-LR" / f_name_q
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"questions": data_new}, f)
    else:
        print(f"No question file found for {f_name}")

    qids_l = [x["answers_ids"] for x in data_new if x["active"]]
    qids = {x for sublist in qids_l for x in sublist}
    f_name_a = f_name_q.split("_questions")[0] + "_answers.json"
    if os.path.isfile(join(base_path, f_name_a)):
        with open(join(base_path, f_name_a)) as read_file:
            data = json.load(read_file)["answers"]
        data_new = [x for x in data if x["id"] in qids]
        path = pathlib.Path(".") / "RSVQA-LR" / f_name_a
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"answers": data_new}, f)
    else:
        print(f"No question file found for {f_name}")


def img_subset_from_jsons(
    base_path: str,
    json_base_path: str,
    jsons: Sequence[str],
):
    # collect names of all images
    subset_elems = []
    for i, json_f_name in enumerate(jsons):
        with open(join(json_base_path, json_f_name)) as read_file:
            data = json.load(read_file)["images"]
            subset_elems += [x["id"] for x in data if x["active"]]

    print(f"There are {len(subset_elems)} images ({len(set(subset_elems))} unique) " f"selected.")

    path = pathlib.Path(".") / "RSVQA-LR" / "Images_LR"
    path.mkdir(parents=True, exist_ok=True)
    for elem in subset_elems:
        img_name = f"{elem}.tif"
        shutil.copy(
            pathlib.Path(base_path).joinpath("Images_LR").joinpath(img_name),
            path.joinpath(img_name).resolve(),
        )


if __name__ == "__main__":
    base_path = "/media/lhackel/My Passport/lhackel/Datasets/RSVQA-LR"
    jsons = [
        "LR_split_train_images.json",
        "LR_split_val_images.json",
        "LR_split_test_images.json",
    ]
    for f in jsons:
        elems = 5 if "train" in f else 25
        copy_part_of_json(base_path=base_path, f_name=f, elems=elems)

    img_subset_from_jsons(base_path=base_path, json_base_path="./RSVQA-LR", jsons=jsons)
