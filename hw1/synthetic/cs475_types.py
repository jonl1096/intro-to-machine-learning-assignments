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
        self._size = 50
        self._vector = lil_matrix((1, self._size), dtype=np.float32)
        
    def add(self, index, value):
        # print("index = %f" % index)
        # print("value = %f" % value)
        # print("value at index %f is %f" % (index, value))
        if index >= self._size:
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
    def __init__(self):
        self._w = None
        self._learning_rate = 1

    def train(self, instances):
        size = instances[0].get_feature_vector()._size
        # max_index = indices[indices.size - 1]

        self._w = csr_matrix((1, size), dtype=np.float32)
        
        for instance in instances:
            ysign = self.predict(instance)

            if instance._label._class != ysign:
                y_i = 1 if instance._label._class > 0 else -1
                x = csr_matrix(instance.get_feature_vector().get_features())

                update = self._learning_rate * y_i * x
                self._w += update
                # print('x = %f' % x)
                # print('real label = %d' % (instance._label._class))
                # print('predicted label = %d' % ysign)
                # print(self._w)
                # print("w CHANGED to %f after a change of %f" % (self._w, update))

            # TODO: when to update
            # y_i = 1 if instance._label._class > 0 else -1
            # x = csr_matrix(instance.get_feature_vector().get_lil_matrix())
            # update = self._learning_rate * y_i * x
            # self._w += update

        pass

    def predict(self, instance):
        w = csr_matrix(self._w)
        x = csr_matrix(instance.get_feature_vector().get_features())
        w_dot_x = w.dot(x.transpose())[0, 0]

        y = 1 if w_dot_x >= 0 else -1
        ysign = 1 if w_dot_x >= 0 else 0

        # count = 0

        # wx = 0
        # # Converting to a a coo_matrix to iterate fastest
        # feature_vector = coo_matrix(instance.get_feature_vector().get_lil_matrix())
        # for i, j, x in zip(feature_vector.row, feature_vector.col, feature_vector.data):
        #     wx += self._w * x        
            
        # y = 1 if wx >= 0 else -1
        # ysign = 1 if wx >= 0 else 0
            

        return ysign




