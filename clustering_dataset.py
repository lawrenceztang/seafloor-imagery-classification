
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
    # if id == 5:
    #     return 2
    # elif id == 12:
    #     return 3
    # else:
    #     return 1
    return id

class seafloor_dataset(utils.Dataset):


    def load_seafloor(self, dir, configDir, ann_path):

        class_names = ["BG", "Brain Coral", "Fire Coral", "Tube Coral", "Sea Rod", "Yellow Green Big Lump",
                       "Other Coral", "Sand", "Rock", "Algae Rock", "Fish", "Potato Coral"]

        for i in range(len(class_names)):
            self.add_class("seafloor", i, class_names[i])

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

    def load_coco(self, dataset_dir, subset, annotation_dir, class_ids=None, return_coco=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """


        coco = COCO(annotation_dir)
        image_dir = "{}/{}".format(dataset_dir, subset)

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco



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
    NUM_CLASSES = 1 + 4  # COCO has 80 classes
    LEARNING_RATE = 0.001

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results

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
        r = model.detect([image], mode="zeroshot")[0]
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

    MODEL_PATH = "/home/lawrence/PycharmProjects/pytorch-mask-rcnn/logs/seafloor20190423T2237/mask_rcnn_seafloor_0060.pth"
    DATASET_PATH = "/home/lawrence/PycharmProjects/pytorch-mask-rcnn/seafloor"
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    configDir = "/home/lawrence/PycharmProjects/pytorch-mask-rcnn/seafloor/config.json"
    ann_path = "/home/lawrence/PycharmProjects/pytorch-mask-rcnn/seafloor/annotation.pkl"
    dir = "/home/lawrence/PycharmProjects/pytorch-mask-rcnn/seafloor/subset-20"
    coco_ann_dir = "/home/lawrence/PycharmProjects/pytorch-mask-rcnn/seafloor/annotations/test.json"

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

    model.load_weights(MODEL_PATH)

    import matplotlib.pyplot as plt
    import skimage.io as io

    dataset_val = seafloor_dataset()
    coco = dataset_val.load_coco(dataset_dir=DATASET_PATH, annotation_dir=coco_ann_dir, subset="test", return_coco=True)
    dataset_val.prepare()

    #display annotated image
    # imgIds = coco.getImgIds()
    # img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
    # I = io.imread("/home/lawrence/PycharmProjects/pytorch-mask-rcnn/seafloor/subset-20/" + img['file_name'])
    # plt.axis('off')
    # plt.imshow(I)
    # plt.show()
    # plt.imshow(I)
    # plt.axis('off')
    # catIds = coco.getCatIds()
    # annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    # anns = coco.loadAnns(annIds)
    # coco.showAnns(anns)

    dataset_train_clustering = seafloor_dataset()
    dataset_train_clustering.load_seafloor(dir, configDir, ann_path)
    dataset_train_clustering.prepare()
    model.train_clustering(dataset_train_clustering)
    print("Running COCO evaluation")
    evaluate_coco(model, dataset_val, coco, "bbox", limit=100)
    evaluate_coco(model, dataset_val, coco, "segm", limit=100)
