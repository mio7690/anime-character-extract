import argparse
import cv2
import os
import json

from tqdm import tqdm
from pathlib import Path

import numpy as np
from anime_face_detector import create_detector
import ipdb
from PIL import Image

detector = create_detector('yolov3')

def detect_faces(detector,
                 image,
                 score_thres=0.75,
                 ratio_thres=2,
                 debug=False):
    preds = detector(image)  # bgr
    h, w = image.shape[:2]
    facedata = {
        'n_faces': 0,
        'facepos': [],
        'fh_ratio': 0,
        'cropped': False,
    }

    for pred in preds:
        bb = pred['bbox']
        score = bb[-1]
        left, top, right, bottom = [int(pos) for pos in bb[:4]]
        fw, fh = right - left, bottom - top
        # ignore the face if too far from square or too low score
        if (fw / fh > ratio_thres or
                fh / fw > ratio_thres or score < score_thres):
            continue
        facedata['n_faces'] = facedata['n_faces'] + 1
        left_rel = left / w
        top_rel = top / h
        right_rel = right / w
        bottom_rel = bottom / h
        facedata['facepos'].append(
            [left_rel, top_rel, right_rel, bottom_rel])
        if fh / h > facedata['fh_ratio']:
            facedata['fh_ratio'] = fh / h
        if debug:
            cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 255),
                          4)

    return facedata

def has_face(image):

    image = np.array(image)

    if image.shape[2] == 4:
        image = image[:, :, :3].copy()

    h, w = image.shape[:2]

    score_thres = 0.75
    ratio_thres = 2
    debug = True
    facedata = detect_faces(detector,
                            image,
                            score_thres=score_thres,
                            ratio_thres=ratio_thres,
                            debug=debug)
    return facedata['n_faces']


if "__name__" == "__main__":

    test_img_path = './test.png'
    pil_image = Image.open(test_img_path)
    print(has_face(pil_image))
