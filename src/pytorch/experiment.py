#!/usr/bin/env python3

from matplotlib import pyplot as plt
import numpy as np

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# This line comes from the PyTorch Cheatsheet (https://pytorch.org/tutorials/beginner/ptcheat.html)
from torch.utils.data import Dataset, DataLoader  # dataset representation and loading

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import torchvision_utils.transforms as T
from torchvision_utils.utils import collate_fn
from torchvision_utils import engine

torch.set_printoptions(edgeitems=2)
torch.manual_seed(42)  # The Answer to Life, the Universe, and Everything

import os
from pathlib import Path

from voc_xml import read_bounding_box_and_labels, get_unique_labels

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
        self.label_files = [f for f in self.root.glob("*xml")]
        # FIXME: This is a hash; come up with a better name for this.
        # It makes for confusing use in build_unique_labels.
        self.unique_labels = self.build_unique_labels()
        try:
            assert len(self.imgs) == len(self.label_files)
        except AssertionError:
            print("ERROR: number of images and labels differ!")
            raise AssertionError

    def build_unique_labels(self):
        """Build unique labels from all the label files
        Returns a dictionary:
        { "this": 1, "that": 2, "the other": 3 ... }
        """
        labels = set()
        for f in self.label_files:
            new_set = get_unique_labels(f)
            labels |= new_set
        i = 1  # Start at 1 because 0 is always background class
        d = {"background": 0}

        # FIXME: For reasons I don't understand, I need to convert
        # this to a list here.  If I don't, I end up iterating over
        # each *letter* that comes out of labels.pop().
        for label in list(labels):
            d[label] = i
            i += 1

        return d

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
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
        labels_path = self.label_files[idx]

        img_filename, boxes = read_bounding_box_and_labels(labels_path)
        img = Image.open(img_path).convert("RGB")
        num_items = len(boxes)
        boxes_t = torch.ones([num_items, 4], dtype=torch.float)
        labels_t = torch.ones([num_items], dtype=torch.int64)
        image_id = img_filename

        for i in range(len(boxes)):
            box = boxes[i][0:4]  # up-to-but-not-including!
            boxes_t[i] = torch.FloatTensor(boxes[i][0:4])
            label = boxes[i][-1]
            labels_t[i] = self.unique_labels[label]

        image_id = torch.tensor([idx])
        area = (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_items), dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


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

    num_labels = len(training_ds.unique_labels)
    print(training_ds.unique_labels)
    print(training_ds.__getitem__(0))
    # fasterrcnn_resnet50_fpn was default for Detecto, so stick with that for now
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # About to replace the box_predictor with one set up for the
    # number of classes we have.
    # FIXME: I'm using "labels" and "classes" interchangeably, which is a bit confusing.
    print("Before:")
    print(model.roi_heads)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_labels)
    print("After:")
    print(model.roi_heads)

    training_dl = DataLoader(
        training_ds,
        batch_size=2,
        shuffle=True,
        num_workers=1,
        collate_fn=utils.collate_fn,
    )
    print(next(iter(training_dl)))
    images, targets = next(iter(training_dl))
    images = list(image for image in images)
    targets = [{k: v for k, v in t.items()} for t in targets]
    output = model(images, targets)  # Return losses & detections
    print(output)


if __name__ == "__main__":
    main()
