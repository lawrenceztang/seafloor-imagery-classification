
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
    if 1 <= id <= 6 or id == 12:
        return numpy.int64(1)
    elif 6 < id <= 10:
        return numpy.int64(2)
    elif id == 11:
        return numpy.int64(3)

class seafloor_dataset(utils.Dataset):


    def load_seafloor(self, dir, configDir, ann_path):
        self.add_class("seafloor", 1, "coral")
        self.add_class("seafloor", 2, "terrain")
        self.add_class("seafloor", 3, "fish")
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

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")



# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.pth")

def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image])[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"], r["masks"])
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--model', required=False,
                        metavar="/path/to/weights.pth",
                        help="Path to weights .pth file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')

    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = seafloor_config()
    else:
        class InferenceConfig(seafloor_config):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(config=config,
                                  model_dir=args.logs)
    if config.GPU_COUNT:
        model = model.cuda()

    # Select weights file to load
    if args.model:
        if args.model.lower() == "coco":
            model_path = COCO_MODEL_PATH
            model.load_weights_pretrained(model_path)
        elif args.model.lower() == "last":
            # Find last trained weights
            model_path = model.find_last()[1]
            model.load_weights(model_path)
        else:
            model_path = args.model
            model.load_weights(model_path)
    else:
        model_path = ""
        model.load_weights(model_path)

    # model.load_weights("/home/lawrence/PycharmProjects/pytorch-mask-rcnn/logs/seafloor20190311T1813/mask_rcnn_seafloor_0009.pth")

    # Load weights
    print("Loading weights ", model_path)


    dir = "/home/lawrence/PycharmProjects/pytorch-mask-rcnn/seafloor/subset-20"
    configDir = "/home/lawrence/PycharmProjects/pytorch-mask-rcnn/seafloor/config.json"
    ann_path = "/home/lawrence/PycharmProjects/pytorch-mask-rcnn/seafloor/annotation.pkl"

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = seafloor_dataset()
        dataset_train.load_seafloor(dir, configDir, ann_path)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = dataset_train

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train_model(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=20,
                    layers='all')

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train_model(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='4+')

        # Training - Stage 1
        print("Training network heads")
        model.train_model(dataset_train, dataset_val,
                          learning_rate=config.LEARNING_RATE,
                          epochs=60,
                          layers='heads')

    # elif args.command == "evaluate":
    #     # Validation dataset
    #     dataset_val = CocoDataset()
    #     coco = dataset_val.load_coco(args.dataset, "minival", year=args.year, return_coco=True, auto_download=args.download)
    #     dataset_val.prepare()
    #     print("Running COCO evaluation on {} images.".format(args.limit))
    #     evaluate_(model, dataset_val, coco, "bbox", limit=int(args.limit))
    #     evaluate_coco(model, dataset_val, coco, "segm", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
