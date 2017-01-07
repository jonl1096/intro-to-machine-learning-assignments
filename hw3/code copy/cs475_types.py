from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix
import operator

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
        if (index > self._max_index):
            self._max_index = index

    def get(self, index):
        # if the index doesn't exist in the dict, return 0 because it's sparse anyways
        if index in self._vector:
            return self._vector[index]
        return 0

    def get_feature_vector(self):
#         return self._vector.keys()
        return self._vector

    def __str__(self):
        return str(self._vector)


class FeatureVector:
    def __init__(self):
        self._size = 100
        self._vector = lil_matrix((1, self._size), dtype=np.float32)
        self._max_index = 0
        
    def add(self, index, value):
        # vector starts at index one, because it reads from the file and the file
        # always has the index of the features start at 1
        if index >= self._size:
            while index >= self._size:
                self._size = self._size * 2
            temp = self._vector
            self._vector = lil_matrix((1, self._size), dtype=np.float32)
            sv = coo_matrix(temp)
            for i, j, v in zip(sv.row, sv.col, sv.data):
                self._vector[i, j] = v

            self._vector[0, index] = value;
            # check if the new index is bigger than the recorded max index
            self._max_index = index if index > self._max_index else self._max_index

        else:
            self._vector[0, index] = value;
            # check if the new index is bigger than the recorded max index
            self._max_index = index if index > self._max_index else self._max_index

        
    def get(self, index):
        return self._vector[0, index]

    def get_features(self):
        return self._vector

    def convert_to_csr(self):
        self._vector = csr_matrix(self._vector)
        return self
        

class Instance:
    def __init__(self, feature_vector, label):
        # convert to csr so later we won't have to becasue lil -> csr is relatively time intensive
        self._feature_vector = feature_vector.convert_to_csr()
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

class Knn(Predictor):
    def __init__(self, max_size, num_neighbors):
        self._max_size = max_size
        self._k = num_neighbors
        self._instances = None

    def train(self, instances):
        self._instances = instances
        return self

    def predict(self, input):
        # input is an instance
        # print('starting prediction')
        feature_vector = input.get_feature_vector()
        # array of tuples s.t. (distance, label)
        tuples = None

        for instance in self._instances:
            # this for loop is used to create an array of tuples of all the instances
            distance = self.euclidian_distance(feature_vector._vector, instance.get_feature_vector()._vector)
            label = instance._label._class
            tuple = [distance, label]
            if tuples != None:
                tuples = np.vstack((tuples, tuple))
            else:
                # if the array of tuples hasn't been initialized, initialize it
                tuples = np.array(tuple)

        # sorting by the first column of the matrix, i.e. the distance
        tuples = tuples[tuples[:,0].argsort()]
        r, c = tuples.shape

        knn_tuples = tuples[:self._k,:]

        rows, cols = knn_tuples.shape

        d = {knn_tuples[0][1] : 1}

        # iterate through the tuples, and keep track of the number of occurences
        # of each label using a dictionary
        for i in range(1, rows):
            label = knn_tuples[i][1]
            if label not in d:
                d[label] = 1
            else:
                d[label] = d[label] + 1

        # if there's a tie, this max function returns the one with the lower key value
        # i.e. the lower label
        prediction = max(d.iteritems(), key=operator.itemgetter(1))[0]

        # print("knn_tuples")
        # print(knn_tuples)

        # print("votes")
        # print(d)

        # print("prediction")
        # print(prediction)

        return int(prediction)

    def euclidian_distance(self, vect1, vect2):
        # Assuming both vectors are csr or csv matricies
        diff = vect1 - vect2
        return np.sqrt(diff.multiply(diff).sum())






