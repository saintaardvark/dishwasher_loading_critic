import io

import torchvision.transforms as transforms
from PIL import Image

test_img = '/home/aardvark/dev//dishwasher_training_data/raw/Dishwasher Training/20-09-07 09-01-49 9698.jpg'

def transform_image(image_bytes):
    """Transform image appropriately for inference
    """
    my_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


if __name__ == '__main__':
    with open(test_img, "rb") as f:
        image_bytes = f.read()
        tensor = transform_image(image_bytes=image_bytes)
        print(tensor)
