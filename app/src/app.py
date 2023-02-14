import os
import re
import time
from pathlib import Path
import cv2
import nanoid
import numpy as np

from lib.point import BBox
from lib import constants
from lib.camera_worker import CameraWorker
from lib.config import Config
from lib.detector_engine import DetectionEngine
from lib.frame import FrameData
from lib.myLogger import myLogger
from lib.recognizer_engine import RecognizerEngine
from lib.template_matching import TemplateMatching
from lib.yolov5_engine import YOLOv5_Engine
from shapely.geometry import Polygon as shapely_poly
from lib.SimpleQueue import SimpleQueue
import base64
from PIL import Image

logger = myLogger('root')

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# TODO: union intersection occupation for parking slots
# TODO: package sender in queue
# TODO: send image with occupied and free zones


class App(object):
    def __init__(self, config_file):
        self.config = Config(config_file=config_file)
        logger.info("Loading Detection Engine")
        self.detection_engine = DetectionEngine('detection_weights_cvt.np')
        logger.info("Loading Recognizer Engine")
        self.recognizer_engine = RecognizerEngine('recognizer_weights.np')
        logger.info("Loading YOLOv5 Engine")
        self.yolov5_engine = YOLOv5_Engine('yolov5s-simple.onnx')
        self.workers_dict = {}
        self.template_matcher = TemplateMatching()
        self.queue = SimpleQueue("/home/user/parking_zoning/dev/config.json")

        for facility in self.config.get_facilities():
            for camera in facility.get_cameras():
                camera_worker = CameraWorker(worker_config=camera, facility_id=facility.get_facility_id(),
                                             interval_between_frames=camera.get_frame_interval())
                logger.info(f"Camera Worker with ip: {camera.get_ip()}")
                camera_worker.setDaemon(True)
                camera_worker.start()
                self.workers_dict[camera.get_ip()] = camera_worker
                self.upper_ratio = camera.get_car_dim_upper()
                self.lower_ratio = camera.get_car_dim_lower()
    def get_license_plates(self, prediction, snapshot_id, car_bboxes):
        car_boxes = [BBox(pts) for pts in car_bboxes]
        if len(prediction) != 0:
            for idx, sample in enumerate(prediction):
                plate_imgs, is_squared = sample.get_inference_image()
                plate_labels, probs = self.recognizer_engine.predict(plate_imgs)

                if is_squared:
                    plate_label = self.template_matcher.process_square_lp(plate_labels[0], plate_labels[1])
                    prob = probs[0] * probs[1]
                else:
                    plate_label = plate_labels[0]
                    prob = probs[0]
                # sample.image_save()
                prediction[idx].set_label(plate_label)
                prediction[idx].set_snapshot_id(snapshot_id)
                prediction[idx].set_plate_label_prob(prob)
            prediction = [x for x in prediction if x.get_plate_label_prob() >= constants.RECOGNIZER_THRESHOLD]
            for idx, _ in enumerate(prediction):
                prediction[idx].set_car_bbox(car_boxes, self.lower_ratio, self.upper_ratio)
            return prediction
        else:
            return []


    def lp2slot(self, license_plates, slots, image):
        lp2slot = {}
        lpinslot = []
        lpdouble = {}
        success, encoded_image = cv2.imencode('.png', image)
        encoded_image = encoded_image.tobytes()
        encoded_image = base64.b64encode(encoded_image)
        encoded_image = str(encoded_image)
        encoded_image = encoded_image[2:-1]
        f = open("/home/user/parking_zoning/dev/test.txt", "w")
        f.write(encoded_image)
        f.close()
        for lp in license_plates:
            llp = lp.get_car_bbox()
            for slot in slots:
                if llp is None:
                    continue
                x1 = llp.p1.get_point()[0]
                y1 = llp.p1.get_point()[1]
                x2 = llp.p2.get_point()[0]
                y2 = llp.p2.get_point()[1]
                pol1_xy = [(x1, y2), (x1, y1), (x2, y1), (x2, y2)]
                pol2_xy = slot.get_all_points()
                pol2_xy = list(pol2_xy)
                polygon1_shape = shapely_poly(pol1_xy)
                polygon2_shape = shapely_poly(pol2_xy)

                # int_coords = lambda x: np.array(x).round().astype(np.int32)
                # exterior = [int_coords(polygon1_shape.exterior.coords)]
                # overlay = image.copy()
                # cv2.fillPoly(overlay, exterior, color=(255, 255, 0))
                # cv2.addWeighted(overlay, 0.5, image, 1 - 0.5, 0, image)
                # cv2.imwrite("Polygon.jpg", image)

                polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
                polygon_union = polygon1_shape.union(polygon2_shape).area
                polygon_car = polygon1_shape.area
                iou = polygon_intersection / polygon_union
                if iou > 0.1 and lp.get_plate_label() not in lpinslot:
                    lp2slot[slot.get_slot_id()] = lp.get_plate_label()
                    lpinslot.append(lp.get_plate_label())
                elif iou > 0.1 and lp.get_plate_label() in lpinslot:
                    # if lp.get_plate_label()
                    lpdouble[slot.get_slot_id()] = lp.get_plate_label()
        return lp2slot, lpdouble, encoded_image

    def lp2slot_projection(self, license_plates, slots, img, camera_ip, facility_id):
        lp2slot = {}
        image = img.copy()
        for lp in license_plates:
            ground_projection_point = lp.get_projection_position().get_point()
            # for slot in slots:
            #     points = slot.get_all_points()
            #     black_frame = np.zeros_like(image).astype(np.uint8)
            #     cv2.fillConvexPoly(black_frame, points, (255, 255, 255))
            #     contact_point = black_frame[ground_projection_point[1], ground_projection_point[0]]
            #     if contact_point[0] == 255 and contact_point[1] == 255 and contact_point[2] == 255:
            #         lp2slot[slot.get_slot_id()] = lp.get_plate_label()

        facility_folder = Path(os.path.join(constants.DEBUG_FOLDER, facility_id))
        if not facility_folder.exists():
            facility_folder.mkdir(parents=True, exist_ok=True)
        camera_folder = Path(os.path.join(facility_folder, camera_ip))
        if not camera_folder.exists():
            camera_folder.mkdir(parents=True, exist_ok=True)
        # cv2.circle(black_frame, ground_projection_point, 2, (255, 0, 255), -1)
        f_name = re.sub("[^0-9a-zA-Z]+", "", nanoid.generate(size=20)) + '.jpg'
        full_name = os.path.join(camera_folder, f_name)
        # cv2.imwrite(full_name, black_frame)

        for lp in license_plates:
            ground_projection_point = lp.get_projection_position().get_point()

            cv2.circle(image, ground_projection_point, 2, (255, 0, 255), -1)
            cv2.circle(image, lp.get_center_point().get_point(), 2, (0, 255, 255), -1)

            points = lp.get_all_points()
            cv2.circle(image, points.leftTop.get_point(), 2, (0, 255, 255), -1)
            cv2.circle(image, points.leftBottom.get_point(), 2, (0, 255, 255), -1)
            cv2.circle(image, points.rightTop.get_point(), 2, (0, 255, 255), -1)
            cv2.circle(image, points.rightBottom.get_point(), 2, (0, 255, 255), -1)
            color = (255, 255, 255)
            fontScale = 0.5
            in_is = f'{lp.get_plate_label()}'
            point = (points.rightBottom.get_point()[0], points.rightBottom.get_point()[1] + 15)
            cv2.putText(image, in_is, point, cv2.FONT_HERSHEY_SIMPLEX, fontScale, color)
            car_bbox = lp.get_car_bbox()
            if car_bbox is not None:
                cv2.rectangle(image, car_bbox.p1.get_point(), car_bbox.p2.get_point(), (255, 0, 0), thickness=1, lineType=4)

        for slot in slots:
            points = slot.get_all_points()
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
            in_is = f'SLOT-ID: {slot.get_slot_id()}'
            point = (bl[0], bl[1] + 30)
            cv2.putText(image, in_is, point, cv2.FONT_HERSHEY_SIMPLEX, fontScale, color)

        facility_folder = Path(os.path.join(constants.DEBUG_FOLDER, facility_id))
        if not facility_folder.exists():
            facility_folder.mkdir(parents=True, exist_ok=True)
        camera_folder = Path(os.path.join(facility_folder, camera_ip))
        if not camera_folder.exists():
            camera_folder.mkdir(parents=True, exist_ok=True)

        f_name = re.sub("[^0-9a-zA-Z]+", "", nanoid.generate(size=20)) + '.jpg'
        full_name = os.path.join(camera_folder, f_name)
        cv2.imwrite(full_name, image)

        return lp2slot

    def run(self):
        while True:
            time.sleep(0.001)
            for ip, worker in self.workers_dict.items():
                data = worker.get_snapshots()
                if data is None:
                    continue
                else:
                    frames_paths = [x for x in data if x is not None]
                    if len(frames_paths) == 0:
                        continue
                    else:
                        frame_data = FrameData(snapshot_paths=frames_paths, camera_config=worker.get_worker_config(),
                                               facility_id=worker.get_facility_id())
                        images = frame_data.get_images()
                        snapshot_ids = frame_data.get_snapshot_ids()
                        for image, snapshot_id in zip(images, snapshot_ids):
                            prediction = self.detection_engine.make_prediction(image=image,
                                                                               camera_ip=frame_data.get_camera_ip(),
                                                                               object_id=frame_data.get_facility_id())

                            car_bboxes = self.yolov5_engine.make_prediction(img=image)
                            license_plates = self.get_license_plates(prediction, snapshot_id, car_bboxes)
                            frame_data.set_license_plate2snapshot(license_plates, snapshot_id)
                        unique_license_plates = frame_data.get_accumulated_license_plates()
                        lp2slot = self.lp2slot_projection(unique_license_plates, frame_data.get_camera_slots(), images[0],
                                               frame_data.get_camera_ip(), frame_data.get_facility_id())
                        package = {"facility_id": worker.get_facility_id(), "camera_ip": ip, "lp2slot": lp2slot}

                        lp2slot, double_space, snap = self.lp2slot(unique_license_plates, frame_data.get_camera_slots(), images[0])
                        package = {"camera_ip": ip, "lp2slot": lp2slot, "double_spaced": double_space, "image": snap}
                        # package = {"camera_ip": ip, "lp2slot": lp2slot, "double_spaced": double_space}
                        # print(package)
                        # self.queue.enqueue(package)
                        frame_data.clean()
