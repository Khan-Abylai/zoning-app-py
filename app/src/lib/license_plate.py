import cv2
import numpy as np
import constants


class LicensePlate(object):
    def __init__(self, camera_ip, object_id, img_h, img_w, lp_img, points):
        self.__img_h = img_h
        self.__img_w = img_w
        self.__lp_img = lp_img
        self.__center_point = points.centerPoint
        if (points.rightTop.x - points.leftTop.x) / (
                points.leftBottom.y - points.leftTop.y) < constants.SQUARE_LP_RATIO:
            self.__square = True
        else:
            self.__square = False
        self.__camera_ip = camera_ip
        self.__object_id = object_id

    def get_projection_position(self):
        pass

    def image_save(self):
        pass

    def get_image(self):
        pass

    def get_inference_image(self):
        pass
