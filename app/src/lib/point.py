import numpy as np


class Point(object):
    def __init__(self, point, name=None):
        self.__x = int(point[0])
        self.__y = int(point[1])
        if name is None:
            self.__name = ''
        else:
            self.__name = name

    def get_x(self):
        return self.__x

    def get_y(self):
        return self.__y

    def get_point(self):
        return np.array([self.__x, self.__y], dtype=int)

    def __getitem__(self, item):
        return np.array([self.__x, self.__y], dtype=int)


class Points(object):
    def __init__(self, points, center_point):
        self.centerPoint = Point(center_point, 'cp')
        self.leftTop = Point(points[0], 'tl')
        self.leftBottom = Point(points[1], 'bl')
        self.rightTop = Point(points[2], 'tr')
        self.rightBottom = Point(points[3], 'br')
