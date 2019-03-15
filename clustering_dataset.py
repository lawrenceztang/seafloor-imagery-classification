
import os
import time
import numpy as np

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil
from PIL import Image
from copy import deepcopy
import numpy
from random import randint

from config import Config
import utils
import model as modellib


ROOT_DIR = os.getcwd()

class Stack:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[len(self.items) - 1]

    def size(self):
        return len(self.items)


def flood_fill(filled, imageArr, origx, origy):
    color = imageArr[origx][origy]
    s = Stack()
    s.push([origx, origy])
    mask = [[False for i in range(len(imageArr[0]))] for j in range(len(imageArr))]
    while(not s.isEmpty()):
        x, y = s.pop()
        if(x >= len(imageArr) or x < 0 or y >= len(imageArr[0]) or y < 0 or filled[x][y] or not get_supercategory(imageArr[x][y]) == get_supercategory(color)):
            continue
        filled[x][y] = True
        mask[x][y] = True
        s.push([x, y + 1])
        s.push([x + 1, y])
        s.push([x - 1, y])
        s.push([x, y - 1])
    return np.array(mask)

def get_supercategory(id):
    if id == 5:
        return 2
    elif id == 12:
        return 3
    elif id == 1:
        return 4
    else:
        return 1

class seafloor_dataset(utils.Dataset):


    def load_seafloor(self, dir, configDir, ann_path):
        self.add_class("seafloor", 1, "other")
        self.add_class("seafloor", 2, "weird_big_lump")
        self.add_class("seafloor", 3, "potato")
        self.add_class("seafloor", 4, "brain")
        import json
        import pickle
        config = json.load(open(configDir))

        i = 0
        annotations = []

        makeAnnotations = True
        import os.path

        if os.path.isfile(ann_path):
            makeAnnotations = False
            annotations = pickle.load(open(ann_path, "rb"))

        for file in os.listdir(dir):
            if "mask" in file and not "watershed" in file and not "color" in file:
                width, height = 320, 180
                if makeAnnotations:
                    annotations.append(self.loadAnns(dir + "/" + file))
                self.add_image(
                    "seafloor", image_id=dir + "/" + file[:-9] + ".png",
                    path=dir + "/" + file[:-9] + ".png",
                    width=width,
                    height=height,
                    annotations=annotations[i])
                i += 1
                print(i) if i % 10 == 0 else ""
        pickle.dump(annotations, open(ann_path, "wb"), pickle.HIGHEST_PROTOCOL)


    def loadAnns(self, dir):

        image = Image.open(dir)
        imageArr = numpy.array(image)
        imageArr = imageArr[:, :, 0]

        # flood fill to find individual objects and change their color
        filled = [[False for i in range(len(imageArr[0]))] for j in range(len(imageArr))]
        masks = []
        ids = []

        for x in range(len(imageArr)):
            for y in range(len(imageArr[0])):
                if (not filled[x][y] and imageArr[x][y] != 0):
                    masks.append(flood_fill(filled, imageArr, x, y))
                    ids.append(get_supercategory(imageArr[x][y]))

        ids = numpy.array(ids)

        # tests if mask reading works
        # if (len(masks) > 1):
        #     temp = [[100 if j else 0 for j in i] for i in masks[1]]
        #     import matplotlib.pyplot as plt
        #     plt.imshow(temp, cmap="gray")
        #     plt.show()

        masks = numpy.array(masks)
        masks = np.stack(masks, axis=2)
        return masks, ids

    def load_mask(self, image_id):
        return self.image_info[image_id]["annotations"]


class seafloor_config(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "seafloor"

    # We use one GPU with 8GB memory, which can fit one image.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 4

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # COCO has 80 classes
    LEARNING_RATE = 0.001


