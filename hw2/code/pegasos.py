from cs475_types import Perceptron
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix

class Pegasos(Perceptron):

    def __init__(self, max_size, pegasos_lambda):
        super(Pegasos, self).__init__(max_size, '', 0)
        self._pegasos_lambda = pegasos_lambda

    def train(self, instances, iterations):
        t = 1.0
        for i in range(0, iterations):
            for instance in instances:
                ysign = self.predict(instance)
                y_i = 1 if instance._label._class > 0 else -1

                feature_vector = instance.get_feature_vector()
                x = csr_matrix(feature_vector.get_features())
                w_dot_x = self._w.dot(x.transpose())[0, 0]

                indicator_function = y_i * w_dot_x
                update_second_half = 0

                # If the indicator function is less than 1, it takes a value of 1.
                if indicator_function < 1:
                    update_second_half = (1.0 / (self._pegasos_lambda * t)) * y_i * x

                update_whole = (1.0 - (1.0 / t)) * self._w + update_second_half
                self._w = update_whole
                
                t += 1.0