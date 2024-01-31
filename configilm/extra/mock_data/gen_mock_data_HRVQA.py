import json
import os
import pathlib
import random
import shutil
from os.path import join
from typing import Sequence


def copy_part_of_json(base_path: str, f_name: str, elems: int = 10, random_seed: int = 0):
    random.seed(random_seed)
    with open(join(base_path, "jsons", f_name)) as read_file:
        data = json.load(read_file)["questions"]
    assert len(data) >= elems, f"Cannot select {elems} from {len(data)} elements."
    random.shuffle(data)
    data_new = data[:elems]
    data_new = sorted(data_new, key=lambda x: x["question_id"])

    assert len({x["question_id"] for x in data_new}) == elems, (
        f"Seed {random_seed} selects elements multiple times - please select different"
        f" seed or change number of elements"
    )
    print(f"Selected {len(data_new)} elements for {f_name}.")

    path = pathlib.Path(".") / "HRVQA" / "jsons" / f_name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"questions": data_new}, f)

    # copy answers to same questions
    qids = {x["question_id"] for x in data_new}
    f_name_a = f_name.split("_")[0] + "_answer.json"
    if os.path.isfile(join(base_path, "jsons", f_name_a)):
        with open(join(base_path, "jsons", f_name_a)) as read_file:
            data = json.load(read_file)["annotations"]
        data_new = [x for x in data if x["question_id"] in qids]
        path = pathlib.Path(".") / "HRVQA" / "jsons" / f_name_a
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"annotations": data_new}, f)
    else:
        print(f"No answer file found for {f_name}")


def img_subset_from_jsons(
    base_path: str,
    json_base_path: str,
    jsons: Sequence[str],
):
    # collect names of all images
    subset_elems = []
    for i, json_f_name in enumerate(jsons):
        with open(join(json_base_path, json_f_name)) as read_file:
            data = json.load(read_file)["questions"]
            subset_elems += [x["image_id"] for x in data]

    path = pathlib.Path(".") / "HRVQA" / "images"
    path.mkdir(parents=True, exist_ok=True)
    for elem in subset_elems:
        img_name = f"{elem}.png"
        shutil.copy(
            pathlib.Path(base_path).joinpath("images").joinpath(img_name),
            path.joinpath(img_name).resolve(),
        )
    print(f"Copied {len(subset_elems)} images.")


if __name__ == "__main__":
    base_path = "/media/lhackel/My Passport/lhackel/Datasets/HRVQA-1.0 release"
    jsons = [
        "train_question.json",
        "val_question.json",
        "test_question.json",
    ]
    for f in jsons:
        copy_part_of_json(base_path=base_path, f_name=f, elems=5)

    img_subset_from_jsons(base_path=base_path, json_base_path="./HRVQA/jsons/", jsons=jsons)
