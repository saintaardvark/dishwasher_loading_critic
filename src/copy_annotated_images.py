#!/usr/bin/env python3

# A Small but Useful tool that will copy images annotated in VOC format by makesense.ai.
#
# It's assumed that
# - the XML_DIR is a directory containing # VOC files, but no images
# - the ROOT_IMAGES_DIR contains the images somewhere (may be down a few directories)
# - the XML files in XML_DIR match image names, with the exception of the extension (XML vs whatever)
# - all we want is the first file in ROOT_IMAGES_DIR that has the same basename; not getting fancier
#   here because the extension may change (jpg vs jpeg vs JPG vs...)

import os
import re
import shutil
from pathlib import Path


ROOT_IMAGES_DIR = '/home/aardvark/dev/dishwasher_training_data/raw'
XML_DIR = '/home/aardvark/dev/dishwasher_training_data/annotated/labels_my-project-name_2020-12-26-02-09-53'


def find_file_matching(target, root):
    """Find file matching target (anchored to beginning of filename) in root.
    """
    for dirpath, _, files in os.walk(root):
        for f in files:
            if re.match(target, f):
                # Just the first match
                return f"{dirpath}/{f}"

def main():
    xml_candidates = Path(XML_DIR).glob('**/*xml')
    xml_files = [f for f in xml_candidates if f.is_file()]
    for f in xml_files:
        basename = os.path.basename(f)
        img_base = os.path.splitext(basename)[0]
        matching_img = find_file_matching(img_base, ROOT_IMAGES_DIR)
        if matching_img:
            print("Found match! {}".format(matching_img))
            shutil.copy(matching_img, XML_DIR)

if __name__ == '__main__':
    main()
