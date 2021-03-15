import json
import os
from pathlib import Path

import torch
from torchvision import models
import transform

# A note about where I'm leaving this:

# Prediction using the modelzoo option works just fine (if a little
# inaccurate).  That part I've got down.

# The next part is taking the model I'd built for dishwasher stuff and
# loading _that_.  Which in turn means thinking about a few things:

# - Do I want to use detecto, or try to get this working in PyTorch?
#   - Answer: Use detecto for now.  Leave PyTorch for a future improvement.

# - How to get detecto's output to work with Flask?  It would be nice
#   to display some bounding box images for the user.

# - And at some point, thinking about how to turn all this into a
#   score/evaluation ("yeah, you totally loaded that dishwasher
#   wrong.")

SRC_ROOT = os.path.dirname(__file__)
DEFAULT_LABEL_FILE = f'{SRC_ROOT}/../../src/labels.txt'
MODEL_1 = f'{SRC_ROOT}/../../models/detecto_model-0.0.1.pth'
MODEL_2 = f'{SRC_ROOT}/../../models/detecto_model-0.0.2-paperspace.pth'
MODEL_3 = f'{SRC_ROOT}/../../models/detecto_model-0.0.3-paperspace.pth'
MODEL_4 = f'{SRC_ROOT}/../../models/notebook_6-detecto_roboflow.pth'

# FIXME: This would be better off as option in UI, different endpoint, or a
# config variable.
USE_MODELZOO = False

def get_labels(model_file):
    """Get labels for the model we're loading

    Return: list of labels
    """
    label_file = DEFAULT_LABEL_FILE
    model_specific_label_file =  Path(model_file).with_suffix('.labels')
    if model_specific_label_file.exists():
        label_file = model_specific_label_file

    labels = []
    with open(label_file, 'rb') as f:
        for line in f:
            labels.append(line[:-1])

    return labels

def get_pytorch_prediction(image_bytes):
    """Generate a prediction from PyTorch model
    """
    tensor = transform.transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    # FIXME: Now that I'm returning the output of predict_top, the
    # return from get_pytorch_prediction is broken.k
    return class_index[predicted_idx]

# FIXME: At some point I need to think about bounding boxes.
def get_detecto_prediction(image_bytes):
    """Generate a prediction from Detecto model

    param image_bytes: bytes representing an image

    returns: tuple of labels, boxes, confidence
    """
    image_tensor = transform.transform_image(image_bytes=image_bytes)

    # Detecto expects tensor in form (C x W x H), rather than the
    # 4-dimensional array that transform_image returns.
    image_tensor = torch.squeeze(image_tensor)

    prediction = model.predict_top(image_tensor)
    labels, boxes, confidence = model.predict_top(image_tensor)

    # FIXME: At some point, I need to think about whether I need a
    # custom object -- if this output is specific to Detecto, I'll
    # probably want somthing.
    return (labels, boxes, confidence)

def get_prediction(image_bytes):
    """Dumb wrapper while I make up my mind

    param image_bytes: bytes representing an image

    returns: tuple of labels, boxes, confidence
    """
    if USE_MODELZOO is True:
        return get_pytorch_prediction(image_bytes)
    else:
        return get_detecto_prediction(image_bytes)

if USE_MODELZOO is True:
    model = models.densenet121(pretrained=True)
    imagenet_class_dir = os.path.dirname(os.path.abspath(__file__))
    # Downloaded from
    # https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
    imagenet_class_file = f"{imagenet_class_dir}/static/imagenet_class_index.json"
    class_index = json.load(open(imagenet_class_file))
    # Make sure we're in `eval` mode
    model.eval()

else:
    # FIXME: copy-pasta from detecto_inference.py
    model_file = MODEL_4
    labels = get_labels(model_file)
    from detecto.core import Model
    model = Model.load(model_file, labels)

if __name__ == '__main__':
    with open(transform.test_img, "rb") as f:
        image_bytes = f.read()
        print(get_prediction(image_bytes=image_bytes))
