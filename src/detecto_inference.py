#!/usr/bin/env python3

from glob import glob
import os
import sys


from detecto.core import Model
from detecto.utils import read_image
from detecto.visualize import show_labeled_image, plot_prediction_grid
import torch
from torchvision import transforms

SRC_ROOT = os.path.dirname(__file__)
LABEL_FILE = f'{SRC_ROOT}/../src/labels.txt'
MODEL_1 = f'{SRC_ROOT}/../models/detecto_model-0.0.1.pth'
MODEL_2 = f'{SRC_ROOT}/../models/detecto_model-0.0.2-paperspace.pth'
MODEL_3 = f'{SRC_ROOT}/../models/detecto_model-0.0.3-paperspace.pth'
VAL_IMG_ROOT = f'{SRC_ROOT}/../../dishwasher_training_data/sorted/validation'

def load_labels(label_file):
    labels = []
    with open(label_file, 'r') as f:
        for line in f:
            labels.append(line[:-1])
    return labels

def load_model_1(labels):
    """Load model 1 (which has fewer labels)
    """
    return Model.load(MODEL_1, labels[0:5])

def load_model_2(labels):
    """Load model 2 (which uses all the labels)
    """
    return Model.load(MODEL_2, labels)

def load_model_3(labels):
    """Load model 3 (which uses all the labels)
    """
    return Model.load(MODEL_3, labels)

def load_val_images():
    """Load validation images
    """
    val_imgs = []
    for img in glob(f'{VAL_IMG_ROOT}/**/*[jJ][pP][gG]'):
        val_imgs.append(read_image(img))
    return val_imgs

def main(args=[]):
    """
    Main entry point.
    """
    labels = load_labels(LABEL_FILE)
    if len(args) > 0:
        print("Model 3")
        model = load_model_3(labels)
    else:
        print("Model 1")
        model = load_model_1(labels)

    val_imgs = load_val_images()
    int_model = model.get_internal_model()

    try:
        print(model.predict_top(val_imgs[0]))
        plot_prediction_grid(model, val_imgs, dim=(4, 2), figsize=(64, 64), score_filter=0.7)
    except RuntimeError:
        # Note: originally I was running into this error:
        #
        # RuntimeError: cannot reshape tensor of 0 elements into shape
        # [0, -1] because the unspecified dimension size -1 can be any
        # value and is ambiguous
        #
        # What I think is going on here: it looks like maybe the model
        # is not making any predictions (ie, not finding anything to
        # label); thus, there are no boxes to resize.  This seems to
        # be an exploding gradient (where very high numbers get
        # converted to NaN), and you need to try lowering the learning
        # rate or gradient clipping.  Ref:
        # - https://github.com/pytorch/vision/issues/1568#issuecomment-660782838
        #
        # In fact, one person reports they have it with FasterRCNN
        # (https://github.com/pytorch/vision/issues/1568#issuecomment-739259675),
        # which is Detecto's default
        # (https://github.com/alankbi/detecto/blob/master/detecto/core.py#L255)
        #
        # Changing the learning rate from default 0.005 to 0.0005 did
        # the trick -- predictions now work!

        int_model.eval()
        with torch.no_grad():
            int_model([transforms.ToTensor(val_imgs[0])])

if __name__ == '__main__':
    main(sys.argv[1:])
