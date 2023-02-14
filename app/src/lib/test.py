import random

entry_old = {'camera_ip': '172.27.14.98', 'lp2slot': {'1': '35fb673', '3': '649aah02'}, 'double_spaced': {}}
# entry_old = []
slots = ['200', '201']
cars = ['001SPT01', '001SPT02', '001SPT03', '001SPT04', '001SPT05']
random.shuffle(slots)
random.shuffle(cars)
dict1 = {}
dict_double = {}
left_cars = {}
for key in slots:
    for value in cars:
        if not dict_double and cars[1] != value:
            dict_double[key] = cars[1]
            break
        dict1[key] = value
        cars.remove(value)
        break

entry1 = [{
         "facility_id": "object_1",
         "camera_ip": "10.65.5.30",
         "lp2slot": dict1,
         "double_spaced": dict_double
        }
        ]
# print(entry1[0]["lp2slot"])
entry1 = {'camera_ip': '172.27.14.98', 'lp2slot': {'1': '35fb673', '3': '649aah02'}, 'double_spaced': {}}
value = { k : entry_old[0]["lp2slot"][k] for k,_ in set(entry_old[0]["lp2slot"].items()) - set(entry1[0]["lp2slot"].items()) }
print(value)
for i, (sl, car) in enumerate(entry1[0]["lp2slot"].items()):
    try:
        if entry_old[0]["lp2slot"]["{}".format(sl)] != car:
            left_cars[sl] = entry_old[0]["lp2slot"]["{}".format(sl)]
    except:
        continue

entry1 = [{
         "facility_id": "object_1",
         "camera_ip": "10.65.5.30",
         "lp2slot": dict1,
         "double_spaced": dict_double,
         "empty_slots": left_cars
        }
        ]
print(entry_old)
print(entry1)
# import cv2
#
# image = cv2.imread("/home/user/parking_zoning/app/src/snapshot.jpeg")
#
# tl=(693, 559) #165 133
# bl=(491, 718) #117.171
# br=(882, 710) #210.169
# tr=(962, 550) #229.131
#
#
# cv2.circle(image, tl, radius=10, color=(255, 0, 0), thickness=3)
# cv2.circle(image, bl, radius=20, color=(255, 0, 0), thickness=3)
# cv2.circle(image, br, radius=30, color=(255, 0, 0), thickness=3)
# cv2.circle(image, tr, radius=40, color=(255, 0, 0), thickness=3)
#
# cv2.imwrite("/home/user/parking_zoning/app/src/test/im.jpg", image)
