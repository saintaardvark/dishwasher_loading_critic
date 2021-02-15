import json
import os

from torchvision import models
import transform

model = models.densenet121(pretrained=True)
# Make sure we're in `eval` mode
model.eval()

# Downloaded from https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
imagenet_class_dir = os.path.dirname(os.path.abspath(__file__))
imagenet_class_file = f"{imagenet_class_dir}/static/imagenet_class_index.json"
imagenet_class_index = json.load(open(imagenet_class_file))

def get_prediction(image_bytes):
    """Generate a prediction from our model
    """
    tensor = transform.transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]

if __name__ == '__main__':
    with open(transform.test_img, "rb") as f:
        image_bytes = f.read()
        print(get_prediction(image_bytes=image_bytes))
