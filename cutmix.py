import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class CutMix(object):
    def __init__(self, proportion):
        self.proportion = proportion

    def __call__(self, samples):
        images, labels = samples
        batch, height, width = samples.shape[1:3]

        pairs = np.array_split(np.random.shuffle(np.arange(0, batch)), batch // 2)

        for (a, b) in pairs:
            image_a = images[a]
            image_b = images[b]
            lab_a = labels[a]
            lab_b = labels[b]


            x = np.random.uniform(0, width)
            y = np.random.uniform(0, height)
            w = self.proportion * width
            h = self.proportion * height

            image_a[y:y+h, x:x+w], image_b[y:y+h, x:x+w] = image_b[y:y+h, x:x+w], image_a[y:y+h, x:x+w]

            labels[a] = (1 - self.proportion) * lab_a + self.proportion * lab_b
            labels[b] = self.proportion * lab_a + (1 - self.proportion) * lab_b

        return images, labels


if __name__ == "__main__":
    batch_size = 8
    ds = datasets.FashionMNIST(root="data", download=True)
    loader = DataLoader(ds, batch_size, transform=transforms.Compose([CutMix(0.2)]))

    for x, y in loader:
        import pdb; pdb.set_trace()

