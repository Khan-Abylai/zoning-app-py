import os

import cv2
import numpy as np

try:
    import constants
except:
    from lib import constants
from scipy.spatial import distance


def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


class FrameData(object):
    def __init__(self, snapshot_paths, camera_config, facility_id, distance_threshold=10):
        self.__snapshots_paths = snapshot_paths
        self.__snapshots_ids = [x for x in range(len(snapshot_paths))]
        self.__camera_ip = camera_config.get_ip()
        self.__result_send_ip = camera_config.get_result_send_ip()
        self.__camera_slots = camera_config.get_parking_slots()
        self.__facility_id = facility_id
        self.__license_plate_to_snapshot = {}
        self.__distance_threshold = distance_threshold

    def clean(self):
        for snapshot_path in self.__snapshots_paths:
            os.remove(snapshot_path)

    def set_license_plate2snapshot(self, license_plates, snapshot_id):
        self.__license_plate_to_snapshot[snapshot_id] = license_plates

    def get_camera_ip(self):
        return self.__camera_ip

    def get_facility_id(self):
        return self.__facility_id

    def get_images(self):
        return np.array([cv2.imread(x) for x in self.__snapshots_paths])

    def get_camera_slots(self):
        return self.__camera_slots

    def get_snapshot_ids(self):
        return self.__snapshots_ids

    def get_accumulated_license_plates(self):
        all_license_plates = []
        for snapshot_id in self.__license_plate_to_snapshot:
            for lp in self.__license_plate_to_snapshot[snapshot_id]:
                all_license_plates.append(lp)
        all_license_plates.sort(key=lambda x: x.get_plate_label())
        all_plate_labels = set([x.get_plate_label() for x in all_license_plates])

        result_license_plates = []

        for unique_plate in all_plate_labels:
            found_license_plates = [x for x in all_license_plates if
                                    x.get_plate_label() == unique_plate or levenshteinDistance(x.get_plate_label(),
                                                                                               unique_plate) == 1]
            if len(found_license_plates) == 1:
                result_license_plates.append(
                    found_license_plates[0]
                )
            else:

                center_points = [x.get_center_point() for x in found_license_plates]
                distances = np.stack(np.array([[distance.euclidean(x.get_point(), y.get_point()) for id_y, y in
                                                zip(range(len(center_points)), center_points) if id_y != id_x] for
                                               id_x, x
                                               in
                                               zip(range(len(center_points)), center_points)]))
                exceeds = distances[distances > self.__distance_threshold].shape
                if isinstance(exceeds, tuple):
                    use_this_label = False if exceeds[0] > 1 else True
                else:
                    use_this_label = False if exceeds > 1 else True
                found_license_plates.sort(key=lambda x: x.get_plate_label_prob(), reverse=True)
                result_license_plates.append(
                    found_license_plates[0]
                )
        return result_license_plates
