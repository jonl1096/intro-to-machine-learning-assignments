#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
from cs475_types import Predictor

class LambdaMeans(Predictor):
    def __init__(self, input_lambda, max_max_index, iterations):
        self._lambda = input_lambda
        # The maximum feature index of all the instances
        self._max_max_index = max_max_index
        # _prototypes is a list of numpy arrays, initialized with a zero vector so it can be 1 indexed
        self._prototypes = [np.array([0])]
        # self._K = 0
        # self._iterations = iterations
        self._iterations = iterations
        # Number of instances is len(instances) because instances is 0 indexed
        # self._instances = None
        self._indicators = []

    def train(self, instances):
        max = self._max_max_index + 1
        # self._instances = instances
        self.init_prototypes(instances)
        if (self._lambda == 0.0):
            self.init_lambda(instances)

        asdf = 1

        for i in range(1, self._iterations):
            self._indicators = []
            # The E step
            for instance in instances:
                self.predict(instance)

            # print(self._indicators)
            # print(self._prototypes)

            # The M step
            for i in range(1, len(self._prototypes)):
                numerator = np.zeros(max)
                denominator = 0
                # if asdf == 1:
                    # print('prototype index')
                    # print(i)

                for j in range(0, len(instances)):
                    vector = instances[j].get_full_features(max)
                    # if asdf == 1:
                    #     print(self._indicators[j])
                    #     asdf = 2
                    
                    if self._indicators[j] == i:
                        numerator += vector
                        denominator += 1

                # asdf = 1
                # if asdf == 1:
                #     print(numerator)
                #     print(denominator)
                #     asdf = 2
                if denominator == 0:
                    # if there are no instances in this cluster
                    self._prototypes[i] = 0
                    print('ITS 0')
                    print(i)
                else:
                    self._prototypes[i] = numerator / denominator

        return self

    def predict(self, instance):
        max = self._max_max_index + 1
        vector = None
        if self._max_max_index >= instance.get_max_index():
            vector = instance.get_full_features(max)
        else:
            vector = instance.get_full_features()
            max += 1
            self._max_max_index = instance.get_max_index()

        prototype = self._prototypes[1]
        # make the two vectors the same size if they aren't
        self.make_same_size(vector, prototype)
        # initialize the mins to the first prototype
        min_sqdist = self.euc_dist(vector, prototype)**2
        min_prototype = 1
        for i in range(1, len(self._prototypes)):
            # iterating through the means/prototypes to see which one gives the smallest
            # sqdist from the instance
            prototype = self._prototypes[i] # u^k
            self.make_same_size(vector, prototype)
            sqdist = self.euc_dist(vector, prototype)**2
            if sqdist < min_sqdist:
                min_sqdist = sqdist
                min_prototype = i
        if min_sqdist <= self._lambda:
            self._indicators.append(min_prototype)
            # min_list.append((min_prototype, min_sqdist))
        elif min_sqdist > self._lambda:
            # if after iterating through all prototypes the min_sqdist is still greater than
            # lambda, create a new cluster and set the prototype to the instance feature vector
            self._prototypes.append(vector)
            min_prototype = len(self._prototypes) - 1 # need -1 because going from len to index (min_prototype is an index)
            self._indicators.append(min_prototype)

        return min_prototype


    def init_prototypes(self, instances):
        max = self._max_max_index + 1
        # Number of instances is len(instances) because instances is 0 indexed
        N = len(instances)
        mean_of_instances = np.zeros(max, dtype = np.float64)
        for instance in instances:
            mean_of_instances += instance.get_full_features(max)

        mean_of_instances = (1.0 / N) * mean_of_instances 

        # This should set _prototypes[1] equal to mean_of_instances
        self._prototypes.append(mean_of_instances)


    def init_lambda(self, instances):
        max = self._max_max_index + 1
        # Number of instances is len(instances) because instances is 0 indexed
        N = len(instances)
        # The mean of all instances is the first index in the prototype vector
        mean_of_instances = self._prototypes[1]
        sum = 0
        for instance in instances:
            sum += self.euc_dist(instance.get_full_features(max), mean_of_instances)**2

        self._lambda = (1.0 / N) * sum


    def euc_dist(self, v1, v2):
        """ Euclidian distance between 2 numpy arrays of the same shape. """
        return np.linalg.norm(v1 - v2)

    def make_same_size(self, v1, v2):
        """ Makes two numpy arrays of different sizes, into the same size. """
        if len(v1) > len(v2):
            v2.resize(v1.shape, refcheck=False)
        elif len(v1) < len(v2):
            v1.resize(v2.shape, refcheck=False)

        # return (v1, v2)






