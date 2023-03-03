#!/usr/bin/env python

import os
import sys
import torch
import redis
from collections import OrderedDict
import numpy as np
import cv2
import time
import logging
import json
import base64

from torch.nn.functional import upsample

from dataloaders import helpers as helpers
import networks.deeplab_resnet as resnet
from mypath import Path

logging.basicConfig(
    handlers=[logging.StreamHandler()],
    format="%(asctime)s %(levelname)s:%(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

def base64_decode_image(a, dtype, shape):
    # if this is Python 3, we need the extra step of encoding the
    # serialized NumPy string as a byte object
    if sys.version_info.major == 3:
        a = bytes(a, encoding="utf-8")

    # convert the string to a NumPy array using the supplied data
    # type and target shape
    a = np.frombuffer(base64.decodestring(a), dtype=dtype)
    a = a.reshape(shape)

    # return the decoded image
    return a


def init_model():
    """Initializes the model"""
    modelName = 'dextr_pascal-sbd'
    gpu_id = 0
    device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")

    #  Create the network and load the weights
    net = resnet.resnet101(1, nInputChannels=4, classifier='psp')
    logger.info("Initializing weights from: {}".format(os.path.join(Path.models_dir(), modelName + '.pth')))
    state_dict_checkpoint = torch.load(os.path.join(Path.models_dir(), modelName + '.pth'),
                                       map_location=lambda storage, loc: storage)
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
    return net, device

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
    data = json.loads(q[1].decode("utf-8"))
    metadata = data['metadata']
    # deserialize the object and obtain the input image
    image_id = data["id"]
    img_width = data["width"]
    img_height = data["height"]
    image = base64_decode_image(data["image"],
        os.getenv("IMAGE_DTYPE"),
        (1, img_height, img_width,
            int(os.getenv("IMAGE_CHANS"))))
    points = np.array(metadata["points"], dtype=int)
    return image, points, image_id

def find_contour(image, points, net, device):
    pad = 50
    thres = 0.6
    st = time.time()
    #  Crop image to the bounding box from the extreme points and resize
    bbox = helpers.get_bbox(image, points=points, pad=pad, zero_pad=True)
    crop_image = helpers.crop_from_bbox(image, bbox, zero_pad=True)
    resize_image = helpers.fixed_resize(crop_image, (512, 512)).astype(np.float32)

    crop_time = time.time() - st
    logger.info(f"crop time: {crop_time}")

    #  Generate extreme point heat map normalized to image values
    extreme_points = points - [np.min(points[:, 0]), np.min(points[:, 1])] + [pad,
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
    logger.info(outputs.size())

    pred = np.transpose(outputs.data.numpy()[0, ...], (1, 2, 0))
    pred = 1 / (1 + np.exp(-pred))
    pred = np.squeeze(pred)
    result = helpers.crop2fullmask(pred, bbox, logger, im_size=image.shape[1:3], zero_pad=True, relax=pad) > thres

    kernel = np.ones((25,25),np.uint8)
    result = cv2.morphologyEx(result.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    contours,hierarchy = cv2.findContours(255*result.astype(np.uint8), 1, 2)
    [logger.info(len(cnt)) for cnt in contours]
    if len(contours) > 0:
        sizes = [len(cnt) for cnt in contours]
        largest = sizes.index(max(sizes))
    else:
        logger.error(f"Couldn't find any contours!")
        return []

    cnt = contours[largest]
    M = cv2.moments(cnt)
    epsilon = 0.005*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    approx = np.squeeze(approx).tolist()

    contour_time = time.time() - dextr_time - st
    logger.info(f"contour time: {contour_time}")
    return approx

if __name__ == '__main__':
    with torch.no_grad():
        net, device = init_model()
        db = init_redis()
        logger.info(f"DEXTR worker initialized!")
        while True:
            image, points, image_id = get_image(db)
            poly = find_contour(image, points, net, device)
            db.set(image_id, json.dumps(poly))
