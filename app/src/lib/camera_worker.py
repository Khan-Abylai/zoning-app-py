import io
import os
import re
import time
from datetime import datetime
from queue import Queue
from threading import Thread

import nanoid
import requests
from PIL import Image
from requests.auth import HTTPDigestAuth

try:
    import constants
except:
    from lib import constants


class CameraWorker(Thread):
    def __init__(self, worker_config, facility_id, waiting_time=10, timeout=5, n_snapshots=4):
        Thread.__init__(self)
        self.__ip = worker_config.get_ip()
        self.__queue = Queue()
        self.__url = worker_config.get_snapshot_url()
        self.__camera_user = worker_config.get_username()
        self.__camera_pass = worker_config.get_password()
        self.__waiting_time = waiting_time
        self.__worker_config = worker_config
        self.__request_timeout = timeout
        self.__num_snapshots = n_snapshots
        self.__facility_id = facility_id

    def get_size_of_queue(self):
        return self.__queue.qsize()

    def get_worker_config(self):
        return self.__worker_config

    def get_facility_id(self):
        return self.__facility_id

    def run(self):
        if self.__url is not None:
            while True:
                snapshots = []
                for snapshot_idx in range(self.__num_snapshots):
                    try:
                        response = requests.get(url=self.__url,
                                                auth=HTTPDigestAuth(self.__camera_user, self.__camera_pass),
                                                timeout=self.__request_timeout)
                        if response.status_code == 200:
                            image = Image.open(io.BytesIO(response.content))
                            image_path = os.path.join(constants.STORAGE_FOLDER,
                                                      re.sub("[^0-9a-zA-Z]+", "", nanoid.generate(size=20)) + '.jpg')
                            image.save(image_path)
                            snapshots.append(image_path)
                            now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                            print(f"Snapshot ({snapshot_idx}) taken from ip:{self.__ip} at:{now}")
                        else:
                            print("Request finished unsuccessfully")
                    except Exception as E:
                        print(f"Error during take snapshot:{E}")
                        continue
                self.__queue.put(snapshots)
                time.sleep(self.__waiting_time)
        else:
            print(f"Snapshot url was set incorrect:{self.__url}")

    def clean(self):
        if self.__queue.qsize() > 0:
            with self.__queue.mutex:
                self.__queue.queue.clear()

    def get_snapshots(self):
        if self.__queue.qsize() > 0:
            return self.__queue.get()
        return None
