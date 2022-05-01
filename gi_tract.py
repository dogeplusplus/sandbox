import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

CLASS_MAPPING = {
    "background": 0,
    "large_bowel": 1,
    "small_bowel": 2,
    "stomach": 3,
}

def parse_segmentation(
    rle_segmentation: str,
    label: str,
    image_size: np.ndarray,
) -> np.ndarray:
    mask = np.zeros(image_size)

    coordinates = [int(x) for x in rle_segmentation.split()]
    coordinates = list(zip(coordinates[::2], coordinates[1::2]))

    for y, x in coordinates:
        mask[y, x] = CLASS_MAPPING[label]

    return mask


def read_labels(df_path: Path) -> pd.DataFrame:
    df = pd.read_csv(df_path)
    import pdb; pdb.set_trace()
    # df["mask"] = df["segmentation"].apply(parse_segmentation)


def generate_masks(df: pd.DataFrame, data_dir: Path, label_dir: Path):

    label_dir.mkdir(parents=True, exist_ok=True)

    for case in data_dir.iterdir():
        for day in case.iterdir():
            for scan in (day / "scans").iterdir():
                parts = scan.stem.split("_")
                _, slice_num, height, width, pixel_height, pixel_width = parts


def main():
    data_dir = Path("train")
    cases = list(data_dir.iterdir())

    idx = 10
    case = cases[idx]

    case_days = list(case.iterdir())
    case_idx = 0
    scans = list((case_days[case_idx] / "scans").iterdir())

    image_path = scans[100]
    image = cv2.imread(str(image_path), cv2.IMREAD_ANYDEPTH)

    from collections import Counter
    # image_shapes = Counter([cv2.imread(str(p), cv2.IMREAD_ANYDEPTH).shape for p in scans])

    # all_images = list(data_dir.rglob("*/**/*.png"))
    # image_shapes = Counter([cv2.imread(str(p), cv2.IMREAD_ANYDEPTH).shape for p in all_images])

    labels_path = Path("train.csv")
    read_labels(labels_path)


if __name__ == "__main__":
    main()

