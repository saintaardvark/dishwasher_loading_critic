# https://stackoverflow.com/questions/53317592/reading-pascal-voc-annotations-in-python
import xml.etree.ElementTree as ET


# TODO: Misleading name, odd choice of things to return
# TODO: Find a proper library for VOC stuff
def read_bounding_box_and_labels(xml_file: str):
    """Get the bounding boxes and labels contained within a VOC XML file.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter("object"):

        filename = root.find("filename").text

        xmin, ymin, xmax, ymax = None, None, None, None

        xmin = int(boxes.find("bndbox/xmin").text)
        ymin = int(boxes.find("bndbox/ymin").text)
        xmax = int(boxes.find("bndbox/xmax").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        name = str(boxes.find("name").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax, name]
        list_with_all_boxes.append(list_with_single_boxes)

    return filename, list_with_all_boxes
