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
CLUSTER_TRAIN_DIR = ""

# Path to trained weights file
# Download this file and place in the root of your
# project (See README file for details)
MODEL_PATH = "/home/lawrence/PycharmProjects/pytorch-mask-rcnn/logs/seafloor20190311T1813/mask_rcnn_seafloor_0010.pth"



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
class_names = ["BG", "Other", "Big_Lump", "Potato", "Brain"]

dir = "/home/lawrence/PycharmProjects/pytorch-mask-rcnn/seafloor/clustering_images"
configDir = "/home/lawrence/PycharmProjects/pytorch-mask-rcnn/seafloor/config.json"
ann_path = "/home/lawrence/PycharmProjects/pytorch-mask-rcnn/seafloor/annotation_zero_shot.pkl"
IMAGE_DIR = "/home/lawrence/PycharmProjects/pytorch-mask-rcnn/seafloor/clustering_images"

from clustering_dataset import seafloor_dataset
dataset_train = seafloor_dataset()
dataset_train.load_seafloor(dir, configDir, ann_path)
dataset_train.prepare()
model.train_clustering(dataset_train)

plt.figure(figsize=(15, 9))
plt.ion()
import time

for i, file in enumerate(sorted(os.listdir(IMAGE_DIR))):
    if "mask" not in file:
        image = skimage.io.imread(os.path.join(IMAGE_DIR, file))
        results = model.detect([image], clusterCategory=True)
        r = results[0]
        ax = plt.figure(1).subplots(1)
        temp = time.time()
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    class_names, r['scores'], ax=ax)
