#!/usr/bin/env python
#-*- coding:utf-8 -*-

from cs475_types import Predictor
from abc import ABCMeta, abstractmethod
import numpy as np
# from scipy.sparse import lil_matrix, csr_matrix, coo_matrix
import operator
import math


class Knn(Predictor):
    def __init__(self, num_neighbors, max_max_index, algorithm='knn'):
        # self._max_size = max_size
        self._k = num_neighbors
        self._instances = None
        self._algorithm = algorithm
        self._max_max_index = max_max_index
        self._max_size = max_max_index + 1

    def train(self, instances):
        self._instances = instances
        return self

    def predict(self, instance):
        # input is an instance
        # print('starting prediction')
        feature_vector = instance.get_full_features(self._max_size)
        # array of tuples s.t. (distance, label)
        tuples = None
        tuples_is_empty = True

        for inst in self._instances:
            # this for loop is used to create an array of tuples of all the instances
            other_vector = inst.get_full_features(self._max_size)
            distance = self.euc_dist(feature_vector, other_vector)
            # print('distance')
            # print(distance)
            label = inst.get_label()
            tuple = [distance, label]
            if not tuples_is_empty:
                tuples = np.vstack((tuples, tuple))
            else:
                # if the array of tuples hasn't been initialized, initialize it
                tuples = np.array(tuple)
                tuples_is_empty = False

        # print('tuples')
        # print(tuples)

        # sorting by the first column of the matrix, i.e. the distance
        tuples = tuples[tuples[:,0].argsort()]
        r, c = tuples.shape

        knn_tuples = tuples[:self._k,:]

        rows, cols = knn_tuples.shape

        d = {knn_tuples[0][1] : 1}

        # iterate through the tuples, and keep track of the number of occurences
        # of each label using a dictionary
        if self._algorithm == 'knn':
            for i in range(1, rows):
                label = knn_tuples[i][1]
                if label not in d:
                    d[label] = 1
                else:
                    d[label] = d[label] + 1
        elif self._algorithm == 'distance_knn':
            for i in range(1, rows):
                label = knn_tuples[i][1]
                distance = knn_tuples[i][0]
                weight = 1.0 / (1.0 + math.pow(distance, 2))
                if label not in d:
                    d[label] = weight
                else:
                    d[label] = d[label] + weight

        # if there's a tie, this max function returns the one with the lower key value
        # i.e. the lower label
        prediction = max(d.iteritems(), key=operator.itemgetter(1))[0]

        return int(prediction)

    def euclidian_distance(self, vect1, vect2):
        max_index = 0
        if vect1._max_index > vect2._max_index:
            max_index = vect1._max_index
        else:
            max_index = vect2._max_index
        sum = 0.0
        for i in range(0, max_index):
            sum += math.pow(vect1.get(i) - vect2.get(i), 2)
        # diff = vect1 - vect2
        distance = math.sqrt(sum)
        return distance
        # return np.sqrt(diff.multiply(diff).sum())


    def euc_dist(self, v1, v2):
        """ Euclidian distance between 2 numpy arrays of the same shape. """
        return np.linalg.norm(v1 - v2)





