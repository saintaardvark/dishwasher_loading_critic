import io

import torchvision.transforms as transforms
from numpy import asarray, uint8
from PIL import Image
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

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
    image = img_from_bytes(image_bytes)
    return my_transforms(image).unsqueeze(0)

def thumbnailify_image(image_bytes, size=(120, 120)):
    """Convenience method for shrinking image for thumbnails
    """
    img = img_from_bytes(image_bytes)
    img.thumbnail(size)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

def draw_bounding_boxes(image_bytes, boxes, labels, limit=0):
    """Draw bounding boxes as needed

    If limit > 0: only use the top limit number of boxes

    Returns: numpy ndarray.  Use imgdata_from_ndarray() to convert to bytes

    TODO: Figure out what format would make sense to return in this function
    """
    # https://stackoverflow.com/questions/384759/how-to-convert-a-pil-image-into-a-numpy-array
    image = asarray(img_from_bytes(image_bytes))

    all_bounding_boxes = []
    if limit == 0:
        limit = len(boxes)
    for i in range(limit):
        box = boxes[i]
        # FIXME: Document what I *think* is dependence on detecto's
        # predict output here
        bbox = BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3])
        bbox.label = labels[i]
        all_bounding_boxes.append(bbox)

    img_with_bbs = BoundingBoxesOnImage(all_bounding_boxes, shape=image.shape).draw_on_image(image, size=2)
    return img_with_bbs

def img_from_bytes(image_bytes):
    """Convenience method

    Returns: PIL Image converted from image_bytes
    """
    return Image.open(io.BytesIO(image_bytes))

def img_from_ndarray(nd_array):
    """Convenience method

    Returns PIL Image 
    """
    return Image.fromarray(nd_array, 'RGB')

def imgdata_from_ndarray(nd_array):
    """Convenience method

    Returns series of bytes for JPEG imgage
    """
    img = Image.fromarray(nd_array, 'RGB')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()


if __name__ == '__main__':
    with open(test_img, "rb") as f:
        image_bytes = f.read()
        tensor = transform_image(image_bytes=image_bytes)
        print(tensor)
