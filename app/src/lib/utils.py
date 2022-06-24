import cv2
import numpy as np
import random
from src import constants


def bbox_iou_d(box1, box2, x1y1x2y2=True):
    if not x1y1x2y2:

        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)

    inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
                 np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou

def bbox_iou_np(box1, box2, x1y1x2y2=True):
    if not x1y1x2y2:

        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)

    inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
                 np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou

def nms_d(predictions, conf_thres=0.2, nms_thres=0.7, include_conf=False):
    filter_mask = (predictions[:, -1] >= conf_thres)
    predictions = predictions[filter_mask]

    if len(predictions) == 0:
        return np.array([])

    output = []

    while len(predictions) > 0:
        max_index = np.argmax(predictions[:, -1])

        if include_conf:
            output.append(predictions[max_index])
        else:
            output.append(predictions[max_index, :-1])

        ious = bbox_iou_d(np.array([predictions[max_index, :-1]]), predictions[:, :-1], x1y1x2y2=False)

        predictions = predictions[ious < nms_thres]

    return np.stack(output)
def nms_np(predictions, conf_thres=0.2, nms_thres=0.2, include_conf=False):
    filter_mask = (predictions[:, -1] >= conf_thres)
    predictions = predictions[filter_mask]

    if len(predictions) == 0:
        return np.array([])

    output = []

    while len(predictions) > 0:
        max_index = np.argmax(predictions[:, -1])

        if include_conf:
            output.append(predictions[max_index])
        else:
            output.append(predictions[max_index, :-1])

        ious = bbox_iou_np(np.array([predictions[max_index, :-1]]), predictions[:, :-1], x1y1x2y2=False)

        predictions = predictions[ious < nms_thres]

    return np.stack(output)


def preprocess_image_recognizer(img, box):
    ratio = abs((box[2, 0] - box[1, 0]) / (box[3, 1] - box[2, 1]))
    if 1.5 > ratio > 0.8:
        plate_img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(box, constants.PLATE_SQUARE),
                                        (constants.RECOGNIZER_IMAGE_W // 2, constants.RECOGNIZER_IMAGE_H * 2))
        padding = np.ones((constants.RECOGNIZER_IMAGE_H,
                           constants.RECOGNIZER_IMAGE_W // 2, 3), dtype=np.uint8) * constants.PIXEL_MAX_VALUE

        result = np.concatenate((plate_img[:constants.RECOGNIZER_IMAGE_H], padding), axis=1).astype(
            np.uint8)
        return np.ascontiguousarray(
            np.stack(result).astype(np.float32).transpose(
                constants.RECOGNIZER_IMG_CONFIGURATION) / constants.PIXEL_MAX_VALUE)
    else:
        plate_img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(box[2:], constants.PLATE_RECT),
                                        (constants.RECOGNIZER_IMAGE_W, constants.RECOGNIZER_IMAGE_H))
        return np.ascontiguousarray(
            np.stack([plate_img]).astype(np.float32).transpose(
                constants.RECOGNIZER_IMG_CONFIGURATION) / constants.PIXEL_MAX_VALUE)
