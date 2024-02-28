import json
import os
import pathlib
import random
import shutil
from os.path import join
from typing import Sequence


def copy_part_of_json(base_path: str, f_name: str, elems: int = 25, random_seed: int = 0):
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

    image_name_mapping = {int(x[-14:-4]): x for x in os.listdir(join(base_path, "images"))}

    path = pathlib.Path(".").joinpath("images")
    path.mkdir(parents=True, exist_ok=True)
    for elem in subset_elems:
        img_name = image_name_mapping[int(elem)]
        shutil.copy(
            pathlib.Path(base_path).joinpath("images").joinpath(img_name),
            path.joinpath(img_name).resolve(),
        )


if __name__ == "__main__":
    base_path = "/home/leonard/data/COCO-QA/"

    new_path = pathlib.Path(".").joinpath("COCO-QA").resolve()
    samples = 10

    train_files = [
        "cocoqa-2015-05-17/train/answers.txt",
        "cocoqa-2015-05-17/train/img_ids.txt",
        "cocoqa-2015-05-17/train/questions.txt",
        "cocoqa-2015-05-17/train/types.txt",
    ]

    test_files = [
        "cocoqa-2015-05-17/test/answers.txt",
        "cocoqa-2015-05-17/test/img_ids.txt",
        "cocoqa-2015-05-17/test/questions.txt",
        "cocoqa-2015-05-17/test/types.txt",
    ]

    # copy first n elements of each txt file to new file in mock_data
    # do for train and test
    for files in [train_files, test_files]:
        for f_name in files:
            full_path = join(base_path, f_name)
            path = new_path.joinpath(f_name)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path) as f:
                with open(path, "w") as f_new:
                    for i in range(samples):
                        f_new.write(f.readline())

    # copy first n images to mock_data
    # collect names of all images
    # 1. read img_ids.txt
    # 2. map the ids to the image names
    # 3. copy the images to mock_data
    for files in [train_files, test_files]:
        img_id_file = [x for x in files if "img_ids.txt" in x][0]
        with open(join(new_path, img_id_file)) as f:
            img_ids = [x.strip() for x in f.readlines()]
        if "train" in img_id_file:
            img_ids = [f"train2014/COCO_train2014_{x:0>12}.jpg" for x in img_ids]
        elif "test" in img_id_file:
            img_ids = [f"val2014/COCO_val2014_{x:0>12}.jpg" for x in img_ids]

        for img_id in img_ids:
            source = pathlib.Path(base_path).joinpath("images").joinpath(img_id)
            target = new_path.joinpath("images").joinpath(img_id).resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(
                source,
                target,
            )

    print("done")
