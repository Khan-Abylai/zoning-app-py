import requests
from lib.config import Config
import time
import json

class SimpleQueue(object):
    def __init__(self, config_file):
        self.items = []
        self.config = Config(config_file=config_file)
        self.last_send = 0
        self.entry_old_1 = []
        self.entry_old_2 = []
    def isEmpty(self):
        return self.items == []

    def enqueue(self, result):
        self.items.insert(0, result)
        # print(self.items)
        self.sendrequest()

    def dequeue(self, value):
        return self.items.pop(value)

    def size(self):
        return len(self.items)

    def current_time(self):
        return time.time()

    def sendrequest(self):
        interval = self.config.get_interval()
        while not self.isEmpty():
            if self.current_time()-self.last_send < interval:
                break
            try:
                for facility in self.config.get_facilities():
                    for camera in facility.get_cameras():
                        for entry in self.items:
                            # import random
                            # slots = ['200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210']
                            # cars = ['001SPT01', '001SPT02', '001SPT03', '001SPT04', '001SPT05']
                            # random.shuffle(slots)
                            # random.shuffle(cars)
                            # dict1 = {}
                            # dict_double = {}
                            # for key in slots:
                            #     for value in cars:
                            #         if not dict_double and cars[1] != value:
                            #             dict_double[key] = cars[1]
                            #             break
                            #         dict1[key] = value
                            #         cars.remove(value)
                            #         break
                            #
                            # entry1 = [{
                            #      "facility_id": "object_1",
                            #      "camera_ip": "10.65.5.30",
                            #      "lp2slot": dict1,
                            #      "double_spaced": dict_double,
                            #      "empty_slots": {}
                            #     }
                            #     ]
                            if camera.get_ip() == '172.27.14.98':
                                entry_old = self.entry_old_1
                            elif camera.get_ip() == '172.27.14.99':
                                self.entry_old = self.entry_old_2
                            if self.last_send == 0:
                                entry_old = []
                            elif entry["camera_ip"] != camera.get_ip():
                                continue
                            else:
                                try:
                                    left_cars = {k: entry_old[0]["lp2slot"][k] for k, _ in
                                            set(entry_old[0]["lp2slot"].items()) - set([entry][0]["lp2slot"].items())}
                                except:
                                    left_cars = {}
                                entry["empty_slots"] = left_cars
                            send_url = "http://%s:8888/rest/parking-space/add" % (camera.get_result_send_ip())
                            # print([entry["lp2slot"]])
                            headers = {
                                'Content-type': 'application/json',
                                'Accept': 'application/json'
                            }
                            requests.post(send_url, json=[entry], headers=headers)
                            # print(response.content)
                            if entry["camera_ip"] == '172.27.14.98':
                                self.entry_old_1 = [entry]
                            elif entry["camera_ip"] == '172.27.14.99':
                                self.entry_old_2 = [entry]
                            self.dequeue(self.items.index(entry))
                            self.last_send = self.current_time()
            except Exception as E:
                print("error sending request:", E)

