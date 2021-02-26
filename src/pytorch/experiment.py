#!/usr/bin/env python3

from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# This line comes from the PyTorch Cheatsheet (https://pytorch.org/tutorials/beginner/ptcheat.html)
from torch.utils.data import Dataset, DataLoader  # dataset representation and loading
import torchvision.transforms as T

torch.set_printoptions(edgeitems=2)
torch.manual_seed(42)  # The Answer to Life, the Universe, and Everything

import os
from pathlib import Path

from voc_xml import read_bounding_box_and_labels

if "COLAB_GPU" in os.environ:
    print("Running on Colab!")
    COLAB = True
    from google.colab import drive

    drive.mount("/content/gdrive")
    dataset_root = Path("/content/gdrive/MyDrive/dishwasher_training_data")
else:
    print("Nope, not on Colab")
    COLAB = False
    dataset_root = Path("/home/aardvark/dev/dishwasher_training_data/roboflow")


class MyDataset(Dataset):
    def __init__(self, root, transforms):
        """Initialization.

        root: Path object.
        """
        Dataset.__init__(self)
        self.root = root
        self.transforms = transforms
        self.imgs = [f for f in self.root.glob("*jpg")]
        self.labels = [f for f in self.root.glob("*xml")]
        try:
            assert len(self.imgs) == len(self.labels)
        except AssertionError:
            print("ERROR: number of images and labels differ!")
            raise AssertionError

    def __len__(self):
        return len(self.imgs)

    def __get_item__(self, idx):
        """Need to return a dict containing:
        - boxes
        - labels
        - image_id
        - area
        - iscrowd
        - (optionally) masks
        - (optionally) keypoints
        """
        img_path = self.imgs[idx]
        labels_path = self.labels[idx]

        img_filename, boxes = read_bounding_box_and_labels(labels_path)
        num_items = len(boxes)
        boxes_t = torch.ones([num_items, 4], dtype=torch.float)
        labels_t = torch.ones([num_items], dtype=torch.int64)
        image_id = img_filename

        for i in range(len(boxes)):
            box = boxes[i][0:4]  # up-to-but-not-including!
            boxes_t[i] = torch.FloatTensor(boxes[i][0:4])
            # FIXME: Going to need to think about transforming label
            # string to list.  For now, will just return 42.
            label = boxes[-1]
            label = 42
            labels_t[i] = label

        image_id = torch.tensor([idx])
        area = (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_items), dtype=torch.int64)

        return {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }


# When creating a dataset, a set of transformers is required.  This is
# a function to create that list of transformers.  Following the tutorial.


def get_transform(train=False):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    """Main entry point"""
    training_ds = MyDataset(dataset_root / "train", get_transform(train=True))
    print(training_ds.__get_item__(0))


if __name__ == "__main__":
    main()
