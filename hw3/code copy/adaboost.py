from cs475_types import Predictor
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix
import operator

class Adaboost(Predictor):

    def __init__(self, max_size, max_max_index, iterations):
        # max size is the maximum amount of features from any of the examples
        self._max_size = max_size
        # max max index is the maximum index of all the features
        self._max_max_index = max_max_index
        # iterations is the number of boosting iterations
        self._iterations = iterations
        # the instances
        self._instances = None
        # n is the number of training examples
        self._n = None
        # d is the distribution where d[i] is the weight of this distribution 
        # on training example i on iteration t.
        self._d = None
        # c cache is the cache of all the c values for each feature j
        self._c_cache = {}

    def calculate_c_cache(self, instances):
        """ 
        Funciton that caluclates the set of possible c for the jth feature, 
        given the set of possible values for that feature, and saves it to a l
        logic:  Take the average of every adjacent set of values. 
                These averages are the set of values that c can take for the jth feature.
                Source: piazza post 104
        jth_feature_values is an ordered set of the possible values for the jth feature
        """
        print('creating c cache')
        for instance in instances:
            vector = instance.get_feature_vector()._vector
            for j in range(1, instance.get_feature_vector()._max_index + 1):
                # iterating through the features
                value = vector[0, j]
                if j in self._c_cache:
                    # if j is already in the cache, just append the value
                    self._c_cache[j].append(value)
                else:
                    self._c_cache[j] = [value]
        # print(self._c_cache)


        for k, v in self._c_cache.iteritems():
            # sort and only get the unique possible values for each feature
            v = np.unique(v)
            c = []
            if len(v) > 1:
                for i in range(0, len(v) - 1):
                    num = (v[i] + v[i + 1]) / 2
                    c.append(num)
                self._c_cache[k] = c

        print(self._c_cache)

        print('finished creating c cache')
        return self


    def train(self, instances):
        # pretty sure instances is 0 indexed, and so as is self._d
        # instance, h, and d are all 0 indexed and their indices align
        print('staring training')
        self._instances = instances
        self.calculate_c_cache(instances)

        self._n = len(instances)
        print("num instances: %d" % self._n)
        # setting D[i] = 1 / n
        self._d = np.full(self._n, (1 / self._n), dtype=np.float)

        

        # for each iteration
        for t in range(1, self._iterations + 1):
            min_error = None
            # h is 0 indexed
            h_t = []
            for j in range(1, self._max_max_index + 1):
                # loop through c
                c = self._c_cache[j]
                print("size of c: %d" % len(c))
                for value in c:
                    print('value in c: %f' % value)
                    # for each feature, there is a different h
                    # for k in range(1, self._max_size + 1):
                    h_error = 0;
                    h = []
                    for i in range(0, len(instances)):
                        # h_error and h will be constructed after this for loop
                        instance = instances[i]
                        h_x = self.predict(instance, j, value)
                        h.append(h_x)
                        if h != instance._label:
                            h_error = h_error + self._d[i]
                    if min_error == None:
                        min_error = h_error
                        h_t = h
                    elif h_error < min_error:
                        min_error = h_error
                        h_t = h

            if min_error < 0.000001:
                break

            alpha = (0.5) * math.log((1 - min_error) / min_error)
            z = self.calculate_z(alpha, h_t)

            for i in range(0, len(self._d)):
                instance = instances[i]
                y_i = 1 if instance._label._class == 1 else -1
                h = 1 if h_t[i] == 1 else -1
                self._d[i] = (1 / z) * self._d[i] * math.exp(-1 * alpha * y_i * h)

        print('finshed training')

        return self


    def predict(self, instance, j, c):
        vector = instance.get_feature_vector()._vector
        prediction = None
        # counts is a dicitonary that maps from a key to how many time it ocurred
        counts = {}
        if vector[0, j] > c:
            for inst in self._instances:
                vect = inst.get_feature_vector()._vector
                if vect[0, j] > c:
                    self.add_to_counts(inst._label._class, counts)
            prediction = max(counts.iteritems(), key=operator.itemgetter(1))[0]
        else:
            for inst in self._instances:
                vect = inst.get_feature_vector()._vector
                if vect[0, j] <= c:
                    self.add_to_counts(inst._label._class, counts)
            prediction = max(counts.iteritems(), key=operator.itemgetter(1))[0]

        return prediction

    def add_to_counts(self, key, dict):
        # dictionaries are mutable
        if key in dict:
            dict[key] = dict[key] + 1
        else:
            dict[key] = 1

    def calculate_z(self, alpha, h_t):
        z = 0
        for i in range(0, len(self._instances)):
            instance = self._instances[i]
            y_i = 1 if instance._label._class == 1 else -1
            h = 1 if h_t[i] == 1 else -1
            z = z + self._d[i] * math.exp(-1 * alpha * y_i * h)

        return z






        