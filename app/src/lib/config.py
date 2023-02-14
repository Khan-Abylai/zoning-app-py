import json
import os

import numpy as np
from lib.myLogger import myLogger
from lib.point import Point
from sys import exit

logger = myLogger('root')


class Slot(object):
    def __init__(self, slot):
        self.__slot_id = slot['slot_id']
        points = slot['points']
        self.__tl = Point(points[0], 'tl')
        self.__bl = Point(points[1], 'bl')
        self.__br = Point(points[2], 'br')
        self.__tr = Point(points[3], 'tr')

    def get_slot_id(self):
        return self.__slot_id

    def get_all_points(self):
        return np.array([self.get_tl(), self.get_bl(), self.get_br(), self.get_tr()], dtype=int)

    def get_tl(self):
        return self.__tl.get_point()

    def get_bl(self):
        return self.__bl.get_point()

    def get_br(self):
        return self.__br.get_point()

    def get_tr(self):
        return self.__tr.get_point()

    def get_slot_id(self):
        return self.__slot_id


class Camera(object):
    def __init__(self, ip, username, password, result_send_ip, parking_slots, result_interval,
                 car_dim_lower, car_dim_upper, frame_interval):
        self.__ip = ip
        self.__username = username
        self.__password = password
        self.__result_send_ip = result_send_ip
        self.__result_interval = result_interval
        self.__car_dim_lower = car_dim_lower
        self.__car_dim_upper = car_dim_upper
        self.__frame_interval = frame_interval
        self.__parking_slots = []
        for parking_slot in parking_slots:
            if all(k in parking_slot for k in ('slot_id', 'points')):
                slot = Slot(parking_slot)
                self.__parking_slots.append(slot)
        self.__snapshot_postfix = '/cgi-bin/snapshot.cgi'

    def get_snapshot_url(self):
        return 'http://' + self.__ip + self.__snapshot_postfix

    def get_ip(self):
        return self.__ip

    def get_username(self):
        return self.__username

    def get_password(self):
        return self.__password

    def get_result_send_ip(self):
        return self.__result_send_ip

    def get_result_interval(self):
        return self.__result_interval

    def get_car_dim_lower(self):
        return self.__car_dim_lower

    def get_frame_interval(self):
        return self.__frame_interval-1

    def get_car_dim_upper(self):
        return self.__car_dim_upper

    def get_parking_slots(self):
        return self.__parking_slots



class Facility(object):
    def __init__(self, facility_id, camera_data):
        self.__facility_id = facility_id
        self.__cameras = []
        for camera_item in camera_data:
            if all(k in camera_item for k in ('camera_ip', 'username', 'password', 'result_send_ip', 'parking_slots',
                                              'result_send_interval_secs', 'car_dim_ratio_lower', 'car_dim_ratio_upper',
                                              'interval_between_frames')):
                camera = Camera(ip=camera_item['camera_ip'],
                                username=camera_item['username'],
                                password=camera_item['password'],
                                result_send_ip=camera_item['result_send_ip'],
                                parking_slots=camera_item['parking_slots'],
                                result_interval=camera_item['result_send_interval_secs'],
                                car_dim_lower=camera_item['car_dim_ratio_lower'],
                                car_dim_upper=camera_item['car_dim_ratio_lower'],
                                frame_interval=camera_item['interval_between_frames'])
                self.__result_interval = camera_item['result_send_interval_secs']
                self.__cameras.append(camera)

    def __len__(self):
        return len(self.__cameras)

    def get_facility_id(self):
        return self.__facility_id

    def get_cameras(self):
        return self.__cameras

    def get_result_interval(self):
        return self.__result_interval

class Config(object):
    def __init__(self, config_file):
        if not os.path.exists(config_file):
            exit()
        with open(config_file, 'r') as f:
            content = json.loads(f.read())

        self.__facilities = []

        for facility_id in content:
            data = content[facility_id]
            facility = Facility(facility_id=facility_id, camera_data=data)
            self.__facilities.append(facility)
            self.__interval = facility.get_result_interval()

    def get_facilities(self):
        return self.__facilities

    def get_interval(self):
        return self.__interval