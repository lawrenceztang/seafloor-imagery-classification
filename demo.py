import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import coco
import utils
import model as modellib
import visualize
from seafloor_dataset import seafloor_config
from seafloor_dataset import seafloor_dataset

import torch


# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
MODEL_PATH = "/home/lawrence/PycharmProjects/pytorch-mask-rcnn/logs/seafloor20190423T2237/mask_rcnn_seafloor_0060.pth"

# Directory of images to run detection on
IMAGE_DIR = "/home/lawrence/PycharmProjects/pytorch-mask-rcnn/seafloor/test"


class InferenceConfig(seafloor_config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = .6

config = InferenceConfig()
config.display()

# Create model object.
model = modellib.MaskRCNN(model_dir=MODEL_DIR, config=config)
if config.GPU_COUNT:
    model = model.cuda()

# Load weights trained on MS-COCO
model.load_weights(MODEL_PATH)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ["BG", "Brain Coral", "Fire Coral", "Tube Coral", "Sea Rod", "Yellow Green Big Lump", "Other Coral", "Sand", "Rock", "Algae Rock", "Fish", "Potato Coral"]
class_names = ["BG", "Coral", "Sand", "Rock", "Fish"]

coco_ann_dir = "/home/lawrence/PycharmProjects/pytorch-mask-rcnn/seafloor/annotations/test.json"
DATASET_PATH = "/home/lawrence/PycharmProjects/pytorch-mask-rcnn/seafloor"
dataset_val = seafloor_dataset()
coco = dataset_val.load_coco(dataset_dir=DATASET_PATH, subset="test", return_coco=True)

visualize.initializeColors(len(class_names))

plt.figure(figsize=(15, 9))
plt.ion()

ax = plt.figure(1).subplots(ncols=2)
im = ax[0].imshow(skimage.io.imread(os.path.join(IMAGE_DIR, os.listdir(IMAGE_DIR)[0])))

for i, file in enumerate(sorted(os.listdir(IMAGE_DIR))):
    if "mask" not in file:
        image = skimage.io.imread(os.path.join(IMAGE_DIR, file))
        results = model.detect([image], mode="inference")
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    class_names, r['scores'], axes=ax, im=im)
        visualize.display_coco(coco, DATASET_PATH + "/test",  file, ax[1])
        plt.pause(2)

