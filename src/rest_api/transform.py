import io

import torchvision.transforms as transforms
from PIL import Image

test_img = '/home/aardvark/dev/src/dishwasher_training_data/raw/Dishwasher Training/20-09-07 09-01-49 9698.jpg'

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

def thumbnailify_image(image_bytes, size=(120, 120)):
    """Convenience method for shrinking image for thumbnails
    """
    img = Image.open(io.BytesIO(image_bytes))
    img.thumbnail(size)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

if __name__ == '__main__':
    with open(test_img, "rb") as f:
        image_bytes = f.read()
        tensor = transform_image(image_bytes=image_bytes)
        print(tensor)
