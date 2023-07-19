import os
import re
from pathlib import Path

import cv2
import nanoid
import numpy as np
from lib import constants
from lib.point import Point, BBox
# import constants
# from point import Point, BBox

class LicensePlate(object):
    def __init__(self, camera_ip, object_id, img_h, img_w, lp_img, points):
        self.__img_h = img_h
        self.__img_w = img_w
        self.__lp_img = lp_img
        self.__center_point = points.centerPoint
        self.__points = points
        if (points.rightTop.get_x() - points.leftTop.get_x()) / (
                points.leftBottom.get_y() - points.leftTop.get_y()) < constants.SQUARE_LP_RATIO:
            self.__square = True
        else:
            self.__square = False
        self.__camera_ip = camera_ip
        self.__object_id = object_id
        self.__label = None
        self.__snapshot_id = None
        self.__prob = None
        self.__car_box = None
        self.__blue_badge = False

    def get_car_bbox(self):
        return self.__car_box

    def set_car_bbox(self, car_bbox, lower_ratio, upper_ratio):
        mask = [True if self.__center_point.get_x() in range(car_box.p1.get_x(), car_box.p2.get_x()) and
                self.__center_point.get_y() in range(car_box.p1.get_y(), car_box.p2.get_y())
                else False for car_box in car_bbox]

        car_bbox = np.array(car_bbox)[mask]

        if car_bbox.shape[0] == 0:
            return None
        elif car_bbox.shape[0] == 1:
            self.__car_box = car_bbox[0]
        else:
            ls = [pt for pt in car_bbox]
            sorted(ls, key=lambda x: x.get_area()
                   if (x.p2.get_x()-x.p1.get_x())/(x.p2.get_y()-x.p1.get_y()) > upper_ratio and
                   (x.p2.get_x() - x.p1.get_x()) / (x.p2.get_y() - x.p1.get_y()) < lower_ratio else 0, reverse=True)
            self.__car_box = ls[0]

    def get_camera_ip(self):
        return self.__camera_ip

    def get_object_id(self):
        return self.__object_id

    def get_plate_label_prob(self):
        return self.__prob

    def get_center_point(self):
        return self.__center_point

    def get_plate_label(self):
        return self.__label

    def set_plate_label_prob(self, prob):
        self.__prob = prob

    def get_all_points(self):
        return self.__points

    def get_projection_position(self):
        lp_height_cm = constants.SQUARE_LP_H_CM if self.is_squared() else constants.RECT_LP_H_CM
        lp_height_px = ((self.__points.leftBottom.get_y() - self.__points.leftTop.get_y()) + (
                self.__points.rightBottom.get_y() - self.__points.rightTop.get_y())) / 2

        point = self.__center_point.get_y() + lp_height_px * constants.AVERAGE_LP_H_FROM_GROUND_CM / lp_height_cm

        return Point([self.__center_point.get_x(), point], 'gcp')

    def image_save(self):
        facility_folder = Path(os.path.join(constants.STORAGE_FOLDER, self.__object_id))
        if not facility_folder.exists():
            facility_folder.mkdir(parents=True, exist_ok=True)
        camera_folder = Path(os.path.join(facility_folder, self.__camera_ip))
        if not camera_folder.exists():
            camera_folder.mkdir(parents=True, exist_ok=True)

        f_name = re.sub("[^0-9a-zA-Z]+", "", nanoid.generate(size=20)) + '.jpg'
        full_name = os.path.join(camera_folder, f_name)
        cv2.imwrite(full_name, self.__lp_img)

    def get_image(self):
        return self.__lp_img

    def is_squared(self):
        return self.__square

    def set_label(self, label):
        self.__label = label

    def set_snapshot_id(self, snapshot_id):
        self.__snapshot_id = snapshot_id

    def get_inference_image(self):
        plate_imgs = []
        if self.__square:
            lp_img = cv2.resize(self.__lp_img, (constants.RECOGNIZER_IMAGE_W // 2, constants.RECOGNIZER_IMAGE_H * 2))

            padding = np.ones((constants.RECOGNIZER_IMAGE_H, constants.RECOGNIZER_IMAGE_W // 2, 3),
                              dtype=np.uint8) * constants.PIXEL_MAX_VALUE
            first_half = np.concatenate((lp_img[:constants.RECOGNIZER_IMAGE_H], padding), axis=1).astype(np.uint8)
            second_half = np.concatenate((lp_img[constants.RECOGNIZER_IMAGE_H:], padding), axis=1).astype(np.uint8)
            plate_imgs.append(first_half)
            plate_imgs.append(second_half)
        else:
            lp_img = cv2.resize(self.__lp_img, (constants.RECOGNIZER_IMAGE_W, constants.RECOGNIZER_IMAGE_H))
            plate_imgs.append(lp_img)
        return np.ascontiguousarray(np.stack(plate_imgs).astype(np.float32).transpose(
            constants.RECOGNIZER_IMG_CONFIGURATION) / constants.PIXEL_MAX_VALUE), self.is_squared()

    def set_blue_badge(self, blue_badge):
        self.__blue_badge = blue_badge

    def get_blue_badge(self):
        return self.__blue_badge
