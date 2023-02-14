import os
import string
from glob import glob

import cv2
import numpy as np
import tensorrt as trt
from numba import cuda

try:
    import constants
except:
    from lib import constants


class RecognizerEngine(object):
    ENGINE_NAME = 'recognizer.engine'

    def __init__(self, weights_name, create_engine=False):
        self.ALPHABET = '-' + string.digits + string.ascii_lowercase
        self.BLANK_INDEX = 0

        self.SEQUENCE_SIZE = 30
        self.ALPHABET_SIZE = len(self.ALPHABET)
        self.HIDDEN_SIZE = 256

        self.OUTPUT_SHAPE = [self.SEQUENCE_SIZE, self.ALPHABET_SIZE]

        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.max_batch_size = 2
        self.engine_path = os.path.join(constants.DEBUG_FOLDER, RecognizerEngine.ENGINE_NAME)
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

    def __create_engine(self, weights):
        with trt.Builder(self.trt_logger) as builder, builder.create_network() as network:

            input_layer = network.add_input('data', trt.DataType.FLOAT, (constants.IMG_C,
                                                                         constants.RECOGNIZER_IMAGE_H,
                                                                         constants.RECOGNIZER_IMAGE_W))

            kernel_size = (3, 3)
            stride = (1, 1)
            channels = [128, 256, 512]
            index = 0

            prev_layer = input_layer

            for i in range(3):
                for j in range(3):
                    conv_weights_count = prev_layer.shape[0] * channels[i] * kernel_size[0] * kernel_size[1]
                    conv_weights = weights[index:index + conv_weights_count]
                    index += conv_weights_count

                    conv_biases_count = channels[i]
                    conv_biases = weights[index:index + conv_biases_count]
                    index += conv_biases_count

                    conv_layer = network.add_convolution(prev_layer, channels[i], kernel_size, conv_weights,
                                                         conv_biases)
                    conv_layer.stride = stride

                    if i == 2 and j == 2:
                        conv_layer.padding = (0, 0)
                    else:
                        conv_layer.padding = (1, 1)

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

                    bn = network.add_scale(conv_layer.get_output(0), trt.ScaleMode.CHANNEL,
                                           combined_bias,
                                           combined_scale,
                                           np.ones_like(combined_bias))

                    activation = network.add_activation(bn.get_output(0), trt.ActivationType.RELU)
                    prev_layer = activation.get_output(0)

                if i < 2:
                    pooling = network.add_pooling(prev_layer, trt.PoolingType.MAX, (2, 2))
                    pooling.stride = (2, 2)
                    prev_layer = pooling.get_output(0)
            shuffle = network.add_shuffle(prev_layer)
            shuffle.reshape_dims = (prev_layer.shape[0] * prev_layer.shape[1], prev_layer.shape[2])
            shuffle.second_transpose = (1, 0)
            embedding_shape = (self.HIDDEN_SIZE, shuffle.get_output(0).shape[1])
            embedding_weights = weights[index:index + embedding_shape[0] * embedding_shape[1]]
            index += embedding_shape[0] * embedding_shape[1]
            embedding = network.add_constant(embedding_shape, embedding_weights)
            matrix_multiplication = network.add_matrix_multiply(
                shuffle.get_output(0),
                trt.MatrixOperation.NONE,
                embedding.get_output(0),
                trt.MatrixOperation.TRANSPOSE
            )

            embedding_bias_shape = (1, self.HIDDEN_SIZE)
            embedding_bias_weights = weights[index:index + embedding_bias_shape[1]]
            index += embedding_bias_shape[1]
            bias = network.add_constant(embedding_bias_shape, embedding_bias_weights)
            add_bias = network.add_elementwise(matrix_multiplication.get_output(0), bias.get_output(0),
                                               trt.ElementWiseOperation.SUM)
            activation = network.add_activation(add_bias.get_output(0), trt.ActivationType.RELU)
            lstm_input_size = self.HIDDEN_SIZE

            lstm = network.add_rnn_v2(activation.get_output(0),
                                      1, self.HIDDEN_SIZE, self.SEQUENCE_SIZE, trt.RNNOperation.LSTM)

            gates = [trt.RNNGateType.INPUT, trt.RNNGateType.FORGET, trt.RNNGateType.CELL, trt.RNNGateType.OUTPUT]

            input_weights = weights[index:index + lstm_input_size * len(gates) * self.HIDDEN_SIZE]
            index += lstm_input_size * len(gates) * self.HIDDEN_SIZE
            rec_weights = weights[index:index + len(gates) * self.HIDDEN_SIZE * self.HIDDEN_SIZE]
            index += len(gates) * self.HIDDEN_SIZE * self.HIDDEN_SIZE
            input_bias = weights[index:index + self.HIDDEN_SIZE * len(gates)]
            index += self.HIDDEN_SIZE * len(gates)
            rec_bias = weights[index:index + self.HIDDEN_SIZE * len(gates)]
            index += self.HIDDEN_SIZE * len(gates)
            hidden_2 = self.HIDDEN_SIZE ** 2
            hidden_2_2 = lstm_input_size * self.HIDDEN_SIZE

            for i in range(len(gates)):
                lstm.set_weights_for_gate(0, gates[i], True,
                                          input_weights[i * hidden_2_2:i * hidden_2_2 + hidden_2_2])
                lstm.set_weights_for_gate(0, gates[i], False,
                                          rec_weights[i * hidden_2:i * hidden_2 + hidden_2])
                lstm.set_bias_for_gate(0, gates[i], True,
                                       input_bias[i * self.HIDDEN_SIZE:i * self.HIDDEN_SIZE + self.HIDDEN_SIZE])
                lstm.set_bias_for_gate(0, gates[i], False,
                                       rec_bias[i * self.HIDDEN_SIZE:i * self.HIDDEN_SIZE + self.HIDDEN_SIZE])

            ### LAST LINEAR LAYER
            embedding_shape = (self.ALPHABET_SIZE, self.HIDDEN_SIZE)
            embedding_weights = weights[index:index + embedding_shape[0] * embedding_shape[1]]
            index += embedding_shape[0] * embedding_shape[1]

            embedding = network.add_constant(embedding_shape, embedding_weights)
            matrix_multiplication = network.add_matrix_multiply(
                lstm.get_output(0),
                trt.MatrixOperation.NONE,
                embedding.get_output(0),
                trt.MatrixOperation.TRANSPOSE
            )

            embedding_bias_shape = (1, self.ALPHABET_SIZE)
            embedding_bias_weights = weights[index:index + embedding_bias_shape[1]]
            index += embedding_bias_shape[1]

            bias = network.add_constant(embedding_bias_shape, embedding_bias_weights)
            add_bias = network.add_elementwise(matrix_multiplication.get_output(0), bias.get_output(0),
                                               trt.ElementWiseOperation.SUM)

            dimensions = network.add_input('dimensions', trt.DataType.INT32, (self.SEQUENCE_SIZE, 1))
            softmax = network.add_ragged_softmax(add_bias.get_output(0), dimensions)

            builder.max_batch_size = self.max_batch_size
            builder.max_workspace_size = 1 << 30
            network.mark_output(softmax.get_output(0))
            return builder.build_cuda_engine(network)

    def predict(self, imgs):
        cuda_stream = cuda.stream()

        batch_size = imgs.shape[0]

        softmax_dims = np.ones((batch_size, self.SEQUENCE_SIZE, 1), dtype=np.int32) * self.ALPHABET_SIZE

        output = np.empty([batch_size] + self.OUTPUT_SHAPE, dtype=np.float32)

        cuda_imgs = cuda.to_device(imgs, cuda_stream)
        cuda_dims = cuda.to_device(softmax_dims, cuda_stream)
        cuda_output = cuda.to_device(output, cuda_stream)

        bindings = [cuda_imgs.device_ctypes_pointer.value, cuda_dims.device_ctypes_pointer.value,
                    cuda_output.device_ctypes_pointer.value]

        with self.engine.create_execution_context() as execution_context:
            execution_context.execute_async(batch_size, bindings, cuda_stream.handle.value, None)
        cuda_stream.synchronize()

        cuda_output.copy_to_host(output, stream=cuda_stream)

        raw_labels = output.argmax(2)
        raw_probs = output.max(2)
        labels = []
        probs = []

        for i in range(batch_size):
            current_prob = 1.0
            current_label = []
            for j in range(self.SEQUENCE_SIZE):

                if raw_labels[i][j] != self.BLANK_INDEX and not (
                        j > 0 and raw_labels[i][j] == raw_labels[i][j - 1]):
                    current_label.append(self.ALPHABET[raw_labels[i][j]])
                    current_prob *= raw_probs[i][j]

            if not current_label:
                current_label.append(self.ALPHABET[self.BLANK_INDEX])
                current_prob = 0.0
            labels.append(''.join(current_label))
            probs.append(current_prob)

        return labels, probs


if __name__ == '__main__':
    correct = 0
    tot = 1
    recognizer_engine = RecognizerEngine(weights_name='recognizer_weights.np')
    # ls = glob(os.path.join(constants.STORAGE_FOLDER, "*.jpg"))
    ls = glob(os.path.join("/home/user/data/", "*"))
    for l in ls:
        images = glob(os.path.join(l, "*_plate.jpg"))
        for image_path in images:
            tot += 1
            image_name = image_path.rsplit("/", 1)[1]
            plate_name = image_path.replace(image_name, "plate.txt")
            f = open(plate_name, "r")
            lp = f.read().replace(" ", "").replace("-", "").lower()
            image = cv2.imread(image_path)
            resized_image = cv2.resize(image, (128, 32))
            preprocessed_image = np.ascontiguousarray(
                np.stack([resized_image]).astype(np.float32).transpose((0, 3, 1, 2)) / 255)
            result = recognizer_engine.predict(preprocessed_image)
            color = (0, 0, 255)
            fontScale = 0.5
            cv2.putText(image, result[0][0], (20, 20), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color)
            cv2.imwrite(os.path.join(os.path.dirname(image_path),
                        os.path.basename(image_path).replace('.', 'found.')), image)
            if result[0][0].lower() == lp:
                correct += 1
    print(correct)
    print(tot)
