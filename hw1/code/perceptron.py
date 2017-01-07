#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
import math
import time
from cs475_types import Predictor

class Perceptron(Predictor):
    def __init__(self, max_index, online_learing_rate):
        self._max_size = max_index + 1
        self._w = np.zeros(self._max_size)
        self._learning_rate = float(online_learing_rate)
        # self._w_averaged = csr_matrix((1, self._max_size), dtype=np.float32)
        # self._averaged = averaged

    def train(self, instances, iterations):
        # size = instances[0].get_feature_vector()._size
        # self._w = csr_matrix((1, size), dtype=np.float32)

        for i in range(0, iterations):
            for instance in instances:
                ysign = self.predict(instance)

                if instance.get_label() != ysign:
                    y_i = 1 if instance._label._class > 0 else -1
                    x = instance.get_full_features(self._max_size)
                    # x = csr_matrix(instance.get_feature_vector().get_features())

                    update = self._learning_rate * y_i * x
                    self._w += update

        #     if self._averaged:
        #         self._w_averaged += self._w
        # if self._averaged:
        #     self._w = self._w_averaged

    def predict(self, instance):
        # print('starting prediction')
        # w = self._w
        # feature_vector = instance.get_full_features(self._max_size)

        # # make the sizes of the vectors match
        # if feature_vector._size < self._max_size:
        #     feature_vector._size = self._max_size
        #     temp = feature_vector.get_features()
        #     feature_vector._vector = lil_matrix((1, self._max_size), dtype=np.float32)
        #     sv = coo_matrix(temp)
        #     for i, j, v in zip(sv.row, sv.col, sv.data):
        #         # print('v: ')
        #         # print(v)
        #         feature_vector._vector[i, j] = v
        # elif feature_vector._size > self._max_size:
        #     self._max_size = feature_vector._size
        #     temp = self._w
        #     new_matrix = lil_matrix((1, feature_vector._size), dtype=np.float32)
        #     sv = coo_matrix(temp)
        #     for i, j, v in zip(sv.row, sv.col, sv.data):
        #         # print('v: ')
        #         # print(v)
        #         new_matrix[i, j] = v
        #
        #     self._w = csr_matrix(new_matrix)

        # if feature_vector._size != self._max_size:
        #     print('feature vector size: %d' % feature_vector._size)
        #     print('w size: %d' % self._max_size)

        # x = csr_matrix(feature_vector.get_features())
        x = instance.get_full_features(self._max_size)
        # w_dot_x = self._w.dot(x.transpose())[0, 0]
        w_dot_x = np.dot(self._w, x)

        y = 1 if w_dot_x >= 0 else -1
        ysign = 1 if w_dot_x >= 0 else 0

        return ysign




