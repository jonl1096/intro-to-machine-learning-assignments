#!/usr/bin/env python
#-*- coding:utf-8 -*-

from abc import ABCMeta, abstractmethod
import numpy as np
# from scipy.sparse import lil_matrix, csr_matrix, coo_matrix
import operator
import math

# abstract base class for defining labels
class Label:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __str__(self): pass

       
class ClassificationLabel(Label):
    def __init__(self, label):
        # this is either an int or a float. idk
        self._class = label
        
    def __str__(self):
        return str(int(self._class))

class FeatureVector:
    def __init__(self):
        self._max_index = 0
        self._vector = {}

    def add(self, index, value):
        # vector starts at index one, because it reads from the file and the file
        # always has the index of the features start at 1
        self._vector[index] = value
        if index > self._max_index:
            self._max_index = index

    def get(self, index):
        # if the index doesn't exist in the dict, return 0 because it's sparse anyways
        if index in self._vector:
            return self._vector[index]
        return 0

    def get_sparse_vector(self):
        return self._vector
        # return self._vector.keys()

    def get_full_vector(self, size = None):
        """ Returns a full vector of features as a numpy array. """
        size = (self._max_index + 1) if size == None else size
        full_vector = np.zeros(size) # 0 indexed
        for key, value in self._vector.iteritems():
            full_vector[key] = value

        return full_vector

    def __str__(self):
        return str(self._vector)


class Instance:
    def __init__(self, feature_vector, label):
        # convert to csr so later we won't have to becasue lil -> csr is relatively time intensive
        # self._feature_vector = feature_vector.convert_to_csr()
        self._feature_vector = feature_vector
        self._label = label

    def get_features(self):
        return self._feature_vector

    def get_full_features(self, size = None):
        return self._feature_vector.get_full_vector(size)

    def get_max_index(self):
        return self._feature_vector._max_index

    def __str__(self):
        return str(self._feature_vector)

# abstract base class for defining predictors
class Predictor:
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, instances): pass

    @abstractmethod
    def predict(self, instance): pass




