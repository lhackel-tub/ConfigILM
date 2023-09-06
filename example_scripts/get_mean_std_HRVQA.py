import pathlib

from PIL import Image
from torchvision import transforms
from tqdm import tqdm

if __name__ == "__main__":
    # r, g, b, grey
    means = [0, 0, 0, 0]
    stds = [0, 0, 0, 0]
    positive = 0
    pre_transforms = transforms.Compose([transforms.ToTensor()])
    for i in tqdm(range(1, 53513), desc="Calculation Mean"):
        root_dir = pathlib.Path(
            "/media/lhackel/My Passport/lhackel/Datasets/HRVQA-1.0 release"
        )

        img_path = root_dir / "images" / f"{i}.png"
        v = Image.open(img_path.resolve()).convert("RGB")
        v = pre_transforms(v)

        r, g, b = v.mean([1, 2])
        grey = v.mean()
        means[0] += r
        means[1] += g
        means[2] += b
        means[3] += grey

        r, g, b = v.std([1, 2])
        grey = v.std()
        stds[0] += r
        stds[1] += g
        stds[2] += b
        stds[3] += grey

        positive += 1
    print("GOT   ", positive)
    print("SHOULD", 53512)

    print([m / positive for m in means])
    print([s / positive for s in stds])
