import cv2
import torch
import shutil
import typing as t
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from einops import rearrange
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split


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
    flat_mask = np.zeros(np.product(image_size), dtype=np.int8)
    runs = [int(x) for x in rle_segmentation.split()]

    for idx in range(0, len(runs), 2):
        start = runs[idx]
        length = runs[idx+1]
        flat_mask[start:start+length] = CLASS_MAPPING[label]

    mask = np.reshape(flat_mask, image_size)
    return mask


def generate_mask(segments: pd.DataFrame, image_size: np.ndarray) -> np.ndarray:
    mask = np.zeros(image_size, dtype=np.int8)

    for _, seg in segments.iterrows():
        if pd.isna(seg["segmentation"]):
            continue
        segment_mask = parse_segmentation(seg["segmentation"], seg["class"], image_size)
        # Avoid cases where the RLE mask overlap
        mask = np.maximum(mask, segment_mask)

    return mask


def preprocess_dataset(df: pd.DataFrame, input_dir: Path, dataset_dir: Path):
    image_dir = dataset_dir / "images"
    label_dir = dataset_dir / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for case_dir in input_dir.iterdir():
        case_image_dir = image_dir / case_dir.name
        case_label_dir = label_dir / case_dir.name

        case_image_dir.mkdir(exist_ok=True, parents=True)
        case_label_dir.mkdir(exist_ok=True, parents=True)

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
                image_path = case_image_dir / f"{slice_id}.png"
                label_path = case_label_dir / f"{slice_id}.png"
                shutil.copy(scan, image_path)
                cv2.imwrite(str(label_path), mask)


class GITract(Dataset):
    def __init__(self, images: t.List[Path], labels: t.List[Path]):
        assert len(images) == len(labels), f"Images and Labels unequal length, Images: {len(images)}, Labels: {len(labels)}"
        self.images = images
        self.labels = labels

    def __len__(self):
        return min(len(self.images), len(self.labels))

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        label_path = self.labels[idx]

        img = cv2.imread(str(img_path), cv2.IMREAD_ANYDEPTH)
        label = cv2.imread(str(label_path), cv2.IMREAD_ANYDEPTH)

        img = np.asarray(img, dtype=np.float32)
        label = np.asarray(label, dtype=np.uint8)

        return torch.from_numpy(img), torch.from_numpy(label)


def collate_fn(batch, image_size=320):
    resize_layer = transforms.Resize((image_size, image_size))
    resize = lambda x: resize_layer(rearrange(x, "h w -> 1 h w"))

    images, labels = zip(*batch)
    images = [resize(x) for x in images]
    labels = [resize(y) for y in labels]

    return torch.cat(images), torch.cat(labels)


@dataclass
class DataPaths:
    images: t.List[Path]
    labels: t.List[Path]


def split_train_test_cases(input_dir: Path, val_ratio: float) -> t.Tuple[DataPaths, DataPaths]:
    image_dir = input_dir / "images"
    label_dir = input_dir / "labels"

    cases = np.array([x.name for x in image_dir.iterdir()])
    val_len = int(val_ratio * len(cases))
    train_len = len(cases) - val_len

    train_idx, val_idx = random_split(np.arange(len(cases)), [train_len, val_len])
    train_cases = cases[train_idx]
    val_cases = cases[val_idx]

    train_images = list(chain.from_iterable((image_dir / case).rglob("**/*.png") for case in train_cases))
    val_images = list(chain.from_iterable((image_dir / case).rglob("**/*.png") for case in val_cases))

    train_labels = list(chain.from_iterable((label_dir / case).rglob("**/*.png") for case in train_cases))
    val_labels = list(chain.from_iterable((label_dir / case).rglob("**/*.png") for case in val_cases))

    return DataPaths(train_images, train_labels), DataPaths(val_images, val_labels)


def main():
    input_dir = Path("train")
    labels_path = Path("train.csv")

    df = pd.read_csv(labels_path)

    dataset_dir = Path("tract_gi_dataset")

    val_ratio = 0.2
    batch_size = 16
    train_set, val_set = split_train_test_cases(dataset_dir, val_ratio)

    train_ds = GITract(train_set.images, train_set.labels)
    val_ds = GITract(val_set.images, val_set.labels)

    train_ds = DataLoader(train_ds, batch_size, shuffle=True, collate_fn=collate_fn)
    val_ds = DataLoader(val_ds, batch_size, shuffle=True, collate_fn=collate_fn)

    for x, y in train_ds:
        pass


if __name__ == "__main__":
    main()

