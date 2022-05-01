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
    flat_mask = np.zeros(np.product(image_size), dtype=np.uint16)
    runs = [int(x) for x in rle_segmentation.split()]

    for idx in range(0, len(runs), 2):
        start = runs[idx]
        length = runs[idx+1]
        flat_mask[start:start+length] = CLASS_MAPPING[label]

    mask = np.reshape(flat_mask, image_size)
    return mask


def generate_mask(segments: pd.DataFrame, image_size: np.ndarray) -> np.ndarray:
    mask = np.zeros(image_size, dtype=np.uint16)

    for _, seg in segments.iterrows():
        if pd.isna(seg["segmentation"]):
            continue
        segment_mask = parse_segmentation(seg["segmentation"], seg["class"], image_size)
        # Avoid cases where the RLE mask overlap
        mask = np.maximum(mask, segment_mask)

    return mask


def generate_masks(df: pd.DataFrame, data_dir: Path, label_dir: Path):
    label_dir.mkdir(parents=True, exist_ok=True)

    for case_dir in data_dir.iterdir():
        for case_day in case_dir.iterdir():
            case = case_day.name
            for scan in (case_day / "scans").iterdir():
                parts = scan.stem.split("_")
                _, slice_num, height, width, pixel_height, pixel_width = parts
                height = int(height)
                width = int(width)
                pixel_height = float(pixel_height)
                pixel_width = float(pixel_width)
                slice_id = f"{case}_slice_{slice_num}"
                rows = df.loc[df["id"] == slice_id]
                mask = generate_mask(rows, (width, height))
                destination = Path(label_dir, f"{slice_id}.png")
                cv2.imwrite(str(destination), mask)


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
    df = pd.read_csv(labels_path)
    label_dir = Path("labels")
    generate_masks(df, data_dir, label_dir)




if __name__ == "__main__":
    main()

