#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
import math
import time
from cs475_types import Predictor


class MCPerceptron(Predictor):

    def __init__(self, max_index, online_training_iterations):
        self._max_label = 0
        self._max_size = max_index + 1
        self._w = np.zeros(self._max_size)
        self._num_poss_labels = 0
        self._online_training_iterations = online_training_iterations
        self._label_map = {}

    def train(self, instances):

        once = True
        # initializing everything
        # possible_labels = self.get_possible_labels(instances)
        # self._num_poss_labels = len(possible_labels)
        self._max_label = self.get_max_label(instances)
        # self._w = np.zeros((self._max_size, self._num_poss_labels))
        self._w = np.zeros((self._max_size, self._max_label + 1))

        # flipping the dictionary
        # self._label_map = {v: k for k, v in possible_labels.iteritems()}
        # print(self._label_map)

        for i in range(0, self._online_training_iterations):
            # print('iteration: %d' % i)
            for instance in instances:
                x_vector = instance.get_full_features(self._max_size)
                predicted_label = self.predict(instance)
                correct_label = instance.get_label()
                # predict_label_index = possible_labels[self.predict(instance)]
                # correct_label_index = possible_labels[instance.get_label()]
                # print('correct label: %d' % correct_label)
                if predicted_label != correct_label:
                    # print('updating')
                    # update
                    self._w[:, correct_label] = self._w[:, correct_label] + x_vector
                    self._w[:, predicted_label] = self._w[:, predicted_label] - x_vector
                    # print(self._w)
                if once:
                    # self._w[:, predict_label_index] = self._w[:, predict_label_index] - x_vector
                    # print self._w
                    once = False

    def predict(self, instance):

        max_val = float("-inf")
        # not actually the maximum k value, just the k associated with the max dot product
        max_k_index = 0

        x_vect = instance.get_full_features(self._max_size)

        # print('predicting')
        #
        # print(self._w)

        for k in range(1, self._max_label + 1):
            # print('k: %d' % k)
            w_vect = self._w[:, k]
            # print('w and x')
            # print(w_vect)
            # print(x_vect)
            val = np.dot(w_vect, x_vect)
            # val = self.dot(w_vect, x_vect)
            # print('val')
            # print(val)
            if val > max_val:
                max_val = val
                max_k_index = k
            # elif val == max_val:
            #     # see which k value has the lower index TODO: NOT SURE IF THIS IS NECCESSARY
            #     max_k_index = k if self._label_map[k] < self._label_map[max_k_index] else max_k_index

        # print('predicted index: %d' % max_k_index)

        # getting the actual label associated with the max
        # max_k_index = self._label_map[max_k_index]

        # print('predicted label: %d' % max_k_index)

        return max_k_index

    # @staticmethod
    # def get_possible_labels(instances):
    #     possible_labels = {0: 0}
    #
    #     index = 1
    #     for instance in instances:
    #         label = instance.get_label()
    #         if label not in possible_labels:
    #             possible_labels[label] = index
    #             index += 1
    #
    #     return possible_labels

    def dot(self, vect1, vect2):
        size = len(vect1)
        sum = 0.0
        for i in range(0, size):
            sum += vect1[i] * vect2[i]

        return sum

    @staticmethod
    def get_max_label(instances):

        max_label = 0
        for instance in instances:
            label = instance.get_label()
            if label > max_label:
                max_label = label

        return max_label





