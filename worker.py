#!/usr/bin/env python

import os
import sys
import torch
import redis
from collections import OrderedDict
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2
import time
import logging

from torch.nn.functional import upsample

import networks.deeplab_resnet as resnet
from mypath import Path
from dataloaders import helpers as helpers

logging.basicConfig(
    handlers=[logging.StreamHandler()],
    format="%(asctime)s %(levelname)s:%(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def init_model():
    """Initializes the model"""
    modelName = 'dextr_pascal-sbd'
    pad = 50
    thres = 0.6
    gpu_id = 0
    device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")

    #  Create the network and load the weights
    net = resnet.resnet101(1, nInputChannels=4, classifier='psp')
    logger.info("Initializing weights from: {}".format(os.path.join(Path.models_dir(), modelName + '.pth')))
    state_dict_checkpoint = torch.load(os.path.join(Path.models_dir(), modelName + '.pth'),
                                       map_location=lambda storage, loc: storage)
    logger.info("FINISHED TORCH LOAD")
    # Remove the prefix .module from the model when it is trained using DataParallel
    if 'module.' in list(state_dict_checkpoint.keys())[0]:
        new_state_dict = OrderedDict()
        for k, v in state_dict_checkpoint.items():
            name = k[7:]  # remove `module.` from multi-gpu training
            new_state_dict[name] = v
    else:
        new_state_dict = state_dict_checkpoint
    net.load_state_dict(new_state_dict)
    net.eval()
    net.to(device)
    return net

def init_redis():
    """Initializes redis client"""
    # connect to Redis server
    db = redis.StrictRedis(host=os.getenv("REDIS_HOST"),
        port=os.getenv("REDIS_PORT"), db=os.getenv("REDIS_DB"))
    return db

def get_image(db):
    """Gets an image from the queue"""
    # monitor queue for jobs and grab one when present
    q = db.blpop(os.getenv("IMAGE_QUEUE_DEXTR"))
    logger.info(q[0])
    q = q[1]
    imageIDs = []
    # deserialize the object and obtain the input image
    q = json.loads(q.decode("utf-8"))
    img_width = q["width"]
    img_height = q["height"]
    image = helpers.base64_decode_image(q["image"],
        os.getenv("IMAGE_DTYPE"),
        (1, img_height, img_width,
            int(os.getenv("IMAGE_CHANS"))))
    points = np.array(q["points"])
    return image, points

def find_contour(image, points, net):
    #  Crop image to the bounding box from the extreme points and resize
    bbox = helpers.get_bbox(image, points=points, pad=pad, zero_pad=True)
    crop_image = helpers.crop_from_bbox(image, bbox, zero_pad=True)
    resize_image = helpers.fixed_resize(crop_image, (512, 512)).astype(np.float32)

    crop_time = time.time() - st
    logger.info(f"crop time: {crop_time}")

    #  Generate extreme point heat map normalized to image values
    extreme_points = extreme_points_ori - [np.min(points[:, 0]), np.min(points[:, 1])] + [pad,
                                                                                                                  pad]
    extreme_points = (512 * extreme_points * [1 / crop_image.shape[1], 1 / crop_image.shape[0]]).astype(np.int)
    extreme_heatmap = helpers.make_gt(resize_image, extreme_points, sigma=10)
    extreme_heatmap = helpers.cstm_normalize(extreme_heatmap, 255)

    heatmap_time = time.time() - crop_time - st
    logger.info(f"heatmap time: {heatmap_time}")
    #  Concatenate inputs and convert to tensor
    input_dextr = np.concatenate((resize_image, extreme_heatmap[:, :, np.newaxis]), axis=2)
    inputs = torch.from_numpy(input_dextr.transpose((2, 0, 1))[np.newaxis, ...])

    # Run a forward pass
    inputs = inputs.to(device)
    outputs = net.forward(inputs)
    outputs = upsample(outputs, size=(512, 512), mode='bilinear', align_corners=True)
    outputs = outputs.to(torch.device('cpu'))

    dextr_time = time.time() - heatmap_time - st
    logger.info(f"dextr time: {dextr_time}")

    pred = np.transpose(outputs.data.numpy()[0, ...], (1, 2, 0))
    pred = 1 / (1 + np.exp(-pred))
    pred = np.squeeze(pred)
    result = helpers.crop2fullmask(pred, bbox, im_size=image.shape[:2], zero_pad=True, relax=pad) > thres

    kernel = np.ones((25,25),np.uint8)
    result = cv2.morphologyEx(result.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    results.append(result)
    logger.info(result.shape)
    #logger.info(result)

    contours,hierarchy = cv2.findContours(255*result.astype(np.uint8), 1, 2)
    [logger.info(len(cnt)) for cnt in contours]
    if len(contours) > 0:
        sizes = [len(cnt) for cnt in contours]
        largest = sizes.index(max(sizes))
    else:
        raise Exception(f"Couldn't find any contours!")

    logger.info(len(contours))
    cnt = contours[largest]
    M = cv2.moments(cnt)
    epsilon = 0.005*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    #cv2.drawContours(255*result.astype(np.uint8), contours, -1, (0,255,0), 3)
    #logger.info(approx)

    contour_time = time.time() - dextr_time - st
    logger.info(f"contour time: {contour_time}")
    return approx

if __name__ == '__main__':

    net = init_model()
    db = init_redis()
    while True:
        image, points = get_image(db)
        poly = find_countour(image, points, net)
