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

import torch


# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
# Download this file and place in the root of your
# project (See README file for details)
MODEL_PATH = "/home/lawrence/PycharmProjects/pytorch-mask-rcnn/logs/seafloor20190312T2318/mask_rcnn_seafloor_0060.pth"

# Directory of images to run detection on
IMAGE_DIR = "/home/lawrence/PycharmProjects/pytorch-mask-rcnn/seafloor/Examples"
IMAGE_DIR = "/home/lawrence/PycharmProjects/pytorch-mask-rcnn/seafloor/demo"


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
class_names = ["BG", "Coral", "Terrain", "Fish"]

plt.figure(figsize=(15, 9))
plt.ion()
import time

for i, file in enumerate(sorted(os.listdir(IMAGE_DIR))):
    if "mask" not in file:
        image = skimage.io.imread(os.path.join(IMAGE_DIR, file))
        results = model.detect([image])
        r = results[0]
        ax = plt.figure(1).subplots(1)
        temp = time.time()
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    class_names, r['scores'], ax=ax)


#TODO:
#1. evaluate performance on training data
#2. evaluate performance on untrained images - same species, different species
#3. zero shot learning - feature clustering to predict similarity. Could define new species. 10 training images per class

