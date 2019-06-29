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
import seafloor_dataset

import torch


# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
CLUSTER_TRAIN_DIR = ""

# Path to trained weights file
# Download this file and place in the root of your
# project (See README file for details)
MODEL_PATH = "/home/lawrence/PycharmProjects/pytorch-mask-rcnn/logs/seafloor20190423T2237/mask_rcnn_seafloor_0010.pth"



class InferenceConfig(seafloor_config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = .7

config = InferenceConfig()
config.display()

# Create model object.
model = modellib.MaskRCNN(model_dir=MODEL_DIR, config=config)
if config.GPU_COUNT:
    model = model.cuda()

# Load weights trained on MS-COCO
model.load_weights(MODEL_PATH)

class_names = ["BG", "Coral", "Sand", "Rock", "Fish"]
class_names = ["BG", "Brain Coral", "Fire Coral", "Tube Coral", "Sea Rod", "Yellow Green Big Lump", "Other Coral", "Sand", "Sand2", "Rock", "Algae Rock", "Fish", "Potato Coral"]

# class_names = [str(i) for i in range(100)]

dir = "/home/lawrence/PycharmProjects/pytorch-mask-rcnn/seafloor/subset-20"
configDir = "/home/lawrence/PycharmProjects/pytorch-mask-rcnn/seafloor/config.json"
ann_path = "/home/lawrence/PycharmProjects/pytorch-mask-rcnn/seafloor/annotation_zero_shot.pkl"
IMAGE_DIR = "/home/lawrence/PycharmProjects/pytorch-mask-rcnn/seafloor/demo"

from clustering_dataset import seafloor_dataset
dataset_train = seafloor_dataset()
dataset_train.load_seafloor(dir, configDir, ann_path)
dataset_train.prepare()
model.train_clustering(dataset_train)

plt.figure(figsize=(15, 9))

# ax = plt.figure(1).subplots(1)
# im = ax.imshow(skimage.io.imread(os.path.join(IMAGE_DIR, os.listdir(IMAGE_DIR)[0])))

visualize.initializeColors(len(class_names))

for i, file in enumerate(sorted(os.listdir(IMAGE_DIR))):
    if "mask" not in file:
        image = skimage.io.imread(os.path.join(IMAGE_DIR, file))
        results = model.detect([image], mode="zeroshot")
        r = results[0]
        # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
        #                             class_names, r['scores'], axes=[ax], im=im)
        # plt.pause(2)

model.tsne(class_names = class_names)

#improve speed of silouette optimization

#maskutils.iou

#display annotated image in clustering_dataset

#training and evaluation of mask r-cnn in seafloor_dataset.py

#demo mask rcnn in demo.py

#demo zero shot and pca in demo_zero_shot.py