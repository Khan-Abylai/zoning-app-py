import os
import time

import cv2
import numpy as np
import tensorrt as trt
import tqdm
from numba import cuda
from scipy.special import expit as sigmoid
import constants
from utils import nms_np, nms_d


class DetectionEngine(object):
    ENGINE_NAME = 'detection.engine'

    def __init__(self, weights_name, img_w=512, img_h=512, plate_grid_size=16, car_grid_size=64, create_engine=False):
        self.img_w = 512
        self.img_h = 512
        self.plate_attr = 13
        self.car_attr = 5
        self.coordinate_sizes = [self.plate_attr, self.car_attr]
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.plate_grid_size = 16
        self.car_grid_size = 64
        self.img_sizes = (3, img_w, img_h)
        self.plate_grid_w = self.img_w // plate_grid_size
        self.plate_grid_h = self.img_h // plate_grid_size
        self.car_grid_w = self.img_w // car_grid_size
        self.car_grid_h = self.img_h // car_grid_size
        self.engine_path = os.path.join(constants.DEBUG_FOLDER, DetectionEngine.ENGINE_NAME)
        self.plate_x_y_offset = np.stack(np.meshgrid(np.arange(self.plate_grid_w), np.arange(self.plate_grid_h)),
                                         axis=2)
        if create_engine or not os.path.exists(self.engine_path):
            if os.path.exists(self.engine_path):
                os.remove(self.engine_path)
            self.engine = self.__create_engine(
                np.fromfile(os.path.join(constants.MODELS_FOLDER, weights_name), dtype=np.float32))
            with open(self.engine_path, 'wb') as f:
                f.write(self.engine.serialize())
        else:
            with open(self.engine_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
        self.execution_context = self.engine.create_execution_context()

    def predict(self, imgs):
        cuda_stream = cuda.stream()
        batch_size = 1

        plate_output = np.empty((self.plate_attr, self.plate_grid_h, self.plate_grid_w), dtype=np.float32)
        car_output = np.empty((self.car_attr, self.car_grid_h, self.car_grid_w), dtype=np.float32)

        cuda_imgs = cuda.to_device(imgs, cuda_stream)
        cuda_plate_output = cuda.to_device(plate_output, cuda_stream)
        cuda_car_output = cuda.to_device(car_output, cuda_stream)

        bindings = [cuda_imgs.device_ctypes_pointer.value, cuda_plate_output.device_ctypes_pointer.value,
                    cuda_car_output.device_ctypes_pointer.value]
        self.execution_context.execute_async(batch_size, bindings, cuda_stream.handle.value, None)
        cuda_stream.synchronize()

        cuda_plate_output.copy_to_host(plate_output, stream=cuda_stream)
        cuda_car_output.copy_to_host(car_output, stream=cuda_stream)

        plate_output = plate_output.reshape(batch_size, self.plate_attr, self.plate_grid_h, self.plate_grid_w)
        _ = car_output.reshape(batch_size, self.car_attr, self.car_grid_h, self.car_grid_w)

        plate_output = np.transpose(plate_output, (0, 2, 3, 1))

        plate_output[..., :2] = sigmoid(plate_output[..., :2])
        plate_output[..., -1] = sigmoid(plate_output[..., -1])

        plate_output[..., 2:4] = np.exp(plate_output[..., 2:4])
        plate_output[..., :2] = plate_output[..., :2] + self.plate_x_y_offset

        plate_output[..., :-1] = plate_output[..., :-1] * self.plate_grid_size

        result_plate = plate_output.reshape(batch_size, self.plate_grid_h * self.plate_grid_w, self.plate_attr)
        return result_plate

    def __create_engine(self, weights):
        with trt.Builder(
                self.trt_logger) as builder, builder.create_network() as network, builder.create_builder_config() as builder_config:
            builder.max_batch_size = 1
            builder_config.max_workspace_size = 1 << 30
            input_layer = network.add_input('data', trt.DataType.FLOAT, self.img_sizes)
            kernel_size = (3, 3)
            stride = (1, 1)
            padding = (1, 1)
            channels = [8, 16, 32, 64, 128, 256, 512]
            index = 0

            prev_layer = input_layer
            features = []
            for i in tqdm.tqdm(range(len(channels))):
                for j in range(2):
                    conv_weights_count = prev_layer.shape[0] * channels[i] * kernel_size[0] * kernel_size[1]
                    conv_weights = weights[index:index + conv_weights_count]
                    index += conv_weights_count

                    conv_biases_count = channels[i]
                    conv_biases = weights[index:index + conv_biases_count]
                    index += conv_biases_count

                    conv_layer = network.add_convolution(prev_layer, channels[i], kernel_size, conv_weights,
                                                         conv_biases)
                    conv_layer.stride = stride
                    conv_layer.padding = padding

                    scale = weights[index:index + channels[i]]
                    index += channels[i]

                    bias = weights[index:index + channels[i]]
                    index += channels[i]

                    mean = weights[index:index + channels[i]]
                    index += channels[i]

                    var = weights[index:index + channels[i]]
                    index += channels[i]

                    combined_scale = scale / np.sqrt(var + 1e-5)
                    combined_bias = bias - mean * combined_scale

                    bn = network.add_scale(conv_layer.get_output(0), trt.ScaleMode.CHANNEL, combined_bias,
                                           combined_scale, np.ones_like(combined_bias))

                    activation = network.add_activation(bn.get_output(0), trt.ActivationType.RELU)
                    prev_layer = activation.get_output(0)

                if i == 4 or i == 6:
                    features.append(prev_layer)

                if i < len(channels) - 1:
                    pooling = network.add_pooling(prev_layer, trt.PoolingType.MAX, (2, 2))
                    pooling.stride = (2, 2)
                    prev_layer = pooling.get_output(0)
            for i, prev_layer in enumerate(features):
                conv_weights_count = prev_layer.shape[0] * self.coordinate_sizes[i]
                conv_weights = weights[index:index + conv_weights_count]
                index += conv_weights_count

                conv_biases_count = self.coordinate_sizes[i]
                conv_biases = weights[index:index + conv_biases_count]
                index += conv_biases_count

                conv_layer = network.add_convolution(prev_layer, self.coordinate_sizes[i], (1, 1), conv_weights,
                                                     conv_biases)

                network.mark_output(conv_layer.get_output(0))

            engine = builder.build_engine(network, builder_config)
            return engine


if __name__ == '__main__':
    detection_engine = DetectionEngine(weights_name='detection_weights_cvt.np')
    image_path = os.path.join(constants.DEBUG_FOLDER, 'image.png')
    if os.path.exists(image_path):
        img_w, img_h = constants.DETECTION_IMAGE_W, constants.DETECTION_IMAGE_H
        image = cv2.imread(image_path)
        model_image = cv2.resize(image, (img_h, img_w))
        model_image = model_image.transpose((2, 0, 1))
        model_image = 2 * (model_image / 255.0 - 0.5)
        model_image = model_image.astype(np.float32)
        model_image = np.ascontiguousarray(model_image)
        t1 = time.time()
        plate_output = detection_engine.predict(model_image)
        t2 = time.time()
        print(f"file:{image_path} exec time: {t2 - t1}")
        rx = float(image.shape[1]) / img_w
        ry = float(image.shape[0]) / img_h
        plates = nms_np(plate_output[0], conf_thres=0.7, include_conf=True)
        if len(plates) > 0:
            plates[..., [4, 6, 8, 10]] += plates[..., [0]]
            plates[..., [5, 7, 9, 11]] += plates[..., [1]]
            ind = np.argsort(plates[..., -1])

            for plate, ind_ in zip(plates, ind):
                box = np.copy(plate[:12]).reshape(6, 2)
                prob = plate[-1]
                if prob >= 0.7:
                    plate_box = np.array(
                        [(int((plate[4]) * rx), int((plate[5]) * ry)), (int((plate[6]) * rx), int((plate[7]) * ry)),
                         (int((plate[8]) * rx), int((plate[9]) * ry)), (int((plate[10]) * rx), int((plate[11]) * ry))],
                        dtype=np.float32)

                    cv2.circle(image, (int(plate[0] * rx), int(plate[1] * ry)), 2, (0, 255, 255), -1)
                    cv2.circle(image, (int((plate[4]) * rx), int((plate[5]) * ry)), 2, (0, 255, 0), -1)
                    cv2.circle(image, (int((plate[6]) * rx), int((plate[7]) * ry)), 2, (0, 255, 0), -1)
                    cv2.circle(image, (int((plate[8]) * rx), int((plate[9]) * ry)), 2, (0, 255, 0), -1)
                    cv2.circle(image, (int((plate[10]) * rx), int((plate[11]) * ry)), 2, (0, 255, 0), -1)

            cv2.imwrite(os.path.join(os.path.dirname(image_path), os.path.basename(image_path).replace('.', '-detected.')), image)
