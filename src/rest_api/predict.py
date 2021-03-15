import json
import os

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
LABEL_FILE = f'{SRC_ROOT}/../../src/labels.txt'
MODEL_1 = f'{SRC_ROOT}/../../models/detecto_model-0.0.1.pth'
MODEL_2 = f'{SRC_ROOT}/../../models/detecto_model-0.0.2-paperspace.pth'
MODEL_3 = f'{SRC_ROOT}/../../models/detecto_model-0.0.3-paperspace.pth'
MODEL_4 = f'{SRC_ROOT}/../../models/notebook_6-detecto_roboflow.pth'

# FIXME: This would be better off as option in UI, different endpoint, or a
# config variable.
USE_MODELZOO = False

if USE_MODELZOO is True:
    model = models.densenet121(pretrained=True)
    imagenet_class_dir = os.path.dirname(os.path.abspath(__file__))
    imagenet_class_file = f"{imagenet_class_dir}/static/imagenet_class_index.json"
    class_index = json.load(open(imagenet_class_file))
    # Make sure we're in `eval` mode
    model.eval()

else:
    # FIXME: copy-pasta from detecto_inference.py
    labels = []
    with open(LABEL_FILE, 'rb') as f:
        for line in f:
            labels.append(line[:-1])

    from detecto.core import Model
    model = Model.load(MODEL_3, labels)


# Downloaded from https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json

def get_pytorch_prediction(image_bytes):
    """Generate a prediction from PyTorch model
    """
    tensor = transform.transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return class_index[predicted_idx]

# FIXME: At some point I need to think about bounding boxes.
def get_detecto_prediction(image_bytes):
    """Generate a prediction from Detecto model
    """
    tensor = transform.transform_image(image_bytes=image_bytes)
    # Detecto expects tensor in form (C x W x H), rather than the
    # 4-dimensional array that transform_image returns.
    tensor = torch.squeeze(tensor)
    # Zeroth element of what predict_top returns is the list of string
    # labels of detected objects; I am assuming that the zeroth
    # element of that is the top prediction.
    prediction = model.predict_top(tensor)[0][0]
    # Note bogus class number here.
    return ["1234", str(prediction)]

def get_prediction(image_bytes):
    """Dumb wrapper while I make up my mind
    """
    if USE_MODELZOO is True:
        return get_pytorch_prediction(image_bytes)
    else:
        return get_detecto_prediction(image_bytes)

if __name__ == '__main__':
    with open(transform.test_img, "rb") as f:
        image_bytes = f.read()
        print(get_prediction(image_bytes=image_bytes))
