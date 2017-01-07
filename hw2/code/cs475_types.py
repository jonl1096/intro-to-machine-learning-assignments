from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix

# abstract base class for defining labels
class Label:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __str__(self): pass

       
class ClassificationLabel(Label):
    def __init__(self, label):
        self._class = label
        
    def __str__(self):
        return str(self._class)

class FeatureVector:
    def __init__(self):
        self._size = 100
        self._vector = lil_matrix((1, self._size), dtype=np.float32)
        
    def add(self, index, value):
        if index >= self._size:
            while index >= self._size:
                self._size = self._size * 2
            temp = self._vector
            self._vector = lil_matrix((1, self._size), dtype=np.float32)
            sv = coo_matrix(temp)
            for i, j, v in zip(sv.row, sv.col, sv.data):
                self._vector[i, j] = v

            self._vector[0, index] = value;
        else:
            self._vector[0, index] = value;
        
    def get(self, index):
        return self._vector[0, index]

    def get_features(self):
        return self._vector
        

class Instance:
    def __init__(self, feature_vector, label):
        self._feature_vector = feature_vector
        self._label = label

    def get_feature_vector(self):
        return self._feature_vector

# abstract base class for defining predictors
class Predictor:
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, instances): pass

    @abstractmethod
    def predict(self, instance): pass

class Perceptron(Predictor):
    def __init__(self, max_size, mode, online_learing_rate):
        self._max_size = max_size
        self._w = csr_matrix((1, max_size), dtype=np.float32)
        self._learning_rate = online_learing_rate
        self._w_averaged = csr_matrix((1, max_size), dtype=np.float32) if mode == "averaged_perceptron" else None
        self._w_margin = csr_matrix((1, max_size), dtype=np.float32) if mode == "margin_perceptron" else None
        self._mode = mode

    def train(self, instances, iterations):
        for i in range(0, iterations):
            for instance in instances:
                ysign = self.predict(instance)
                y_i = 1 if instance._label._class > 0 else -1

                if self._mode != 'margin_perceptron':
                    if instance._label._class != ysign:
                        x = csr_matrix(instance.get_feature_vector().get_features())
                        update = self._learning_rate * y_i * x
                        self._w += update
                elif self._mode == 'margin_perceptron':
                    feature_vector = instance.get_feature_vector()
                    x = csr_matrix(feature_vector.get_features())
                    w_dot_x = self._w.dot(x.transpose())[0, 0]
                    if (y_i * w_dot_x < 1):
                        update = self._learning_rate * y_i * x
                        self._w += update
 
            if self._mode == 'averaged_perceptron':
                self._w_averaged += self._w

        if self._mode == 'averaged_perceptron':
            """If the mode is averaged perceptron, return the averaged w"""
            self._w = self._w_averaged

    def predict(self, instance):
        # print('starting prediction')
        # w = self._w
        feature_vector = instance.get_feature_vector()

        # make the sizes of the vectors match
        if feature_vector._size < self._max_size:
            feature_vector._size = self._max_size
            temp = feature_vector.get_features()
            feature_vector._vector = lil_matrix((1, self._max_size), dtype=np.float32)
            sv = coo_matrix(temp)
            for i, j, v in zip(sv.row, sv.col, sv.data):
                # print('v: ')
                # print(v)
                feature_vector._vector[i, j] = v
        elif feature_vector._size > self._max_size:
            self._max_size = feature_vector._size
            temp = self._w
            new_matrix = lil_matrix((1, feature_vector._size), dtype=np.float32)
            sv = coo_matrix(temp)
            for i, j, v in zip(sv.row, sv.col, sv.data):
                # print('v: ')
                # print(v)
                new_matrix[i, j] = v

            self._w = csr_matrix(new_matrix)

        if feature_vector._size != self._max_size:
            print('feature vector size: %d' % feature_vector._size)
            print('w size: %d' % self._max_size)

        x = csr_matrix(feature_vector.get_features())
        w_dot_x = self._w.dot(x.transpose())[0, 0]

        y = 1 if w_dot_x >= 0 else -1
        ysign = 1 if w_dot_x >= 0 else 0

        return ysign




