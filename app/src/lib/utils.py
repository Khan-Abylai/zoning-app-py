import os.path

import cv2
import numpy as np
import random
import constants
import json


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


def draw_zones(image, parking_slots):
    for slot in parking_slots:
        slot_id = slot['slot_id']
        points = np.array(slot['points']).astype(int)

        tl = points[0]
        bl = points[1]
        br = points[2]
        tr = points[3]
        cv2.circle(image, tl, radius=2, color=(255, 0, 0), thickness=3)
        cv2.circle(image, bl, radius=2, color=(255, 0, 0), thickness=3)
        cv2.circle(image, br, radius=2, color=(255, 0, 0), thickness=3)
        cv2.circle(image, tr, radius=2, color=(255, 0, 0), thickness=3)
        cv2.line(image, tl, bl, (0, 255, 0), thickness=2)
        cv2.line(image, bl, br, (0, 255, 0), thickness=2)
        cv2.line(image, br, tr, (0, 255, 0), thickness=2)
        cv2.line(image, tr, tl, (0, 255, 0), thickness=2)

        color = (255, 255, 255)
        fontScale = 0.7
        thickness = 5
        in_is = f'SLOT-ID: {slot_id}'
        point = (bl[0], bl[1] + 15)
        cv2.putText(image, in_is, point, cv2.FONT_HERSHEY_SIMPLEX, fontScale, color)

    return image


if __name__ == '__main__':
    image_path = '/home/user/parking_zoning/debug/20220627_133111.png'
    config = '/home/user/parking_zoning/dev/config.json'

    with open(config, 'r') as f:
        slots = json.loads(f.read())['object_1'][0]['parking_slots']
    image = cv2.imread(image_path)
    drawn_image = draw_zones(image, slots)
    cv2.imwrite(f"/home/user/parking_zoning/debug/drawn_{os.path.basename(image_path)}", drawn_image)
