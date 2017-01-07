#!/usr/bin/env python
#-*- coding:utf-8 -*-

from cs475_types import Predictor
from abc import ABCMeta, abstractmethod
import numpy as np
# from scipy.sparse import lil_matrix, csr_matrix, coo_matrix
import operator
import math
import time

class Adaboost(Predictor):

    def __init__(self,  max_max_index, iterations):
        # # max size is the maximum amount of features from any of the examples
        # self._max_size = max_size
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
        self._d = []
        # c cache is the cache of all the c values for each feature j
        self._c_cache = {}
        self._min_j = {}
        self._min_c = {}
        self._h_t_list = {}
        self._alpha_list = {}
        self._0s_less_than_c = {}
        self._1s_less_than_c = {}
        self._0s_more_than_c = {}
        self._1s_more_than_c = {}

        self._max_iteration = self._iterations


    def calculate_c_cache(self, instances):
        """ 
        Funciton that caluclates the set of possible c for the jth feature, 
        given the set of possible values for that feature, and saves it to a list
        logic:  Take the average of every adjacent set of values. 
                These averages are the set of values that c can take for the jth feature.
                Source: piazza post 104
        jth_feature_values is an ordered set of the possible values for the jth feature
        """
        print('creating c cache')
        for instance in instances:
            vector = instance.get_feature_vector()
            for j in range(1, vector._max_index + 1):
                # iterating through the features
                value = vector.get(j)
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
            else:
                self._c_cache[k] = [v[0]]

        # print(self._c_cache)

        print('finished creating c cache')
        return self


    def train(self, instances):
        # pretty sure instances is 0 indexed, and so as is self._d
        # instance, h, and d are all 0 indexed and their indices align
        print('staring training')
        self._instances = instances
        self.calculate_c_cache(instances)

        self._n = len(instances)
        # print("num instances: %d" % self._n)
        # setting D[i] = 1 / n
        # print('n: %f' % self._n)

        self._d = np.full(self._n, (1.0 / self._n), dtype=np.float)

        # print('dist')
        # print(self._d)

        # for each iteration
        for t in range(1, self._iterations + 1):
            min_error = None
            # h is 0 indexed
            h_t = None
            count_0s_less_than_c = 0
            count_1s_less_than_c = 0
            count_0s_more_than_c = 0
            count_1s_more_than_c = 0
            h = None
            min_c = None
            min_j = None

            # print('max max index: %d' % self._max_max_index)
            for j in range(1, self._max_max_index + 1):
                # loop through c
                c = self._c_cache[j]
                # print("size of c: %d" % len(c))
                for c_value in c:
                    # print('value in c: %f' % value)
                    # for each feature, there is a different h
                    # for k in range(1, self._max_size + 1):
                    h_error = 0
                    # h = []
                    
                    count_1_less_c = 0
                    count_1_more_c = 0

                    count_0_less_c = 0
                    count_0_more_c = 0

                    for i in range(0, len(instances)):
                        instance = instances[i]
                        feature_value = instance.get_feature_vector().get(j)
                        label = instance.get_label()
                        # print('feature value: %f' % feature_value)
                        # print('c: %f' % value)
                        if feature_value > c_value:
                            if label == 1:
                                count_1_more_c += 1
                            else:
                                count_0_more_c += 1
                        else:
                            if label == 1:
                                count_1_less_c += 1
                            else:
                                count_0_less_c += 1

                    # print('1l 1m 0l 0m')
                    # print(count_1_less_c)
                    # print(count_1_more_c)
                    # print(count_0_less_c)
                    # print(count_0_more_c)


                    for i in range(0, len(instances)):
                        # h_error and h will be constructed after this for loop
                        instance = instances[i]
                        feature_value = instance.get_feature_vector().get(j)
                        y_popular = None
                        if feature_value > c_value:
                            if count_1_more_c > count_0_more_c:
                                y_popular = 1
                            else:
                                y_popular = 0
                        else:
                            if count_1_less_c > count_0_less_c:
                                y_popular = 1
                            else:
                                y_popular = 0
                        # print('prediction: %d' % y_popular)
                        # print('actual label %d' % instance._label._class)

                        # h_x = self.get_h_x(instance, j, value)
                        # h.append(y_popular)
                        if y_popular != instance.get_label():
                            # print(self._d[i])
                            h_error = h_error + self._d[i]
                        # print('error: %f' % h_error)

                    if min_error == None:
                        min_error = h_error
                        # h_t = h
                        h_t = y_popular
                        min_c = c_value
                        min_j = j
                        count_0s_less_than_c = count_0_less_c
                        count_1s_less_than_c = count_1_less_c
                        count_0s_more_than_c = count_0_more_c
                        count_1s_more_than_c = count_1_more_c
                    elif h_error < min_error:
                        min_error = h_error
                        # h_t = h
                        h_t = y_popular
                        min_c = c_value
                        min_j = j
                        count_0s_less_than_c = count_0_less_c
                        count_1s_less_than_c = count_1_less_c
                        count_0s_more_than_c = count_0_more_c
                        count_1s_more_than_c = count_1_more_c

            # print('iteration: %d' % t)
            # print('min_error: %f' % min_error)
            if min_error < 0.000001:
                # don't use the current iteration
                self._max_iteration = t - 1
                break

            alpha = (0.5) * math.log((1.0 - min_error) / min_error)
            z = self.calculate_z(alpha, h_t)

            # print('min_j')
            # print(min_j)

            self._min_c[t] = min_c
            self._min_j[t] = min_j
            self._alpha_list[t] = alpha
            self._h_t_list[t] = h_t

            self._0s_less_than_c[t] = count_0s_less_than_c
            self._1s_less_than_c[t] = count_1s_less_than_c
            self._0s_more_than_c[t] = count_0s_more_than_c
            self._1s_more_than_c[t] = count_1s_more_than_c

            for i in range(0, len(self._d)):
                instance = instances[i]
                y_i = 1 if instance.get_label() == 1 else -1
                h = 1 if h_t == 1 else -1
                self._d[i] = (1.0 / z) * self._d[i] * math.exp(-1 * alpha * y_i * h)
        # print('min_js')
        # print(self._min_j)

        print('finshed training')

        return self


    def predict(self, instance):
        # print("strrt predict")
        vector = instance.get_feature_vector()
        prediction = None
        # sums is a dicitonary that maps from a key to how many time it ocurred
        sums = {}
        zero_bucket = 0.0
        one_bucket = 0.0
        for t in range(1, self._max_iteration + 1):
            # print('predicting ideration: %d' % t)
            j = self._min_j[t]
            c = self._min_c[t]
            h = self._h_t_list[t]
            l0 = self._0s_less_than_c[t]
            l1 = self._1s_less_than_c[t]
            g0 = self._0s_more_than_c[t]
            g1 = self._1s_more_than_c[t]
            alpha = self._alpha_list[t]
            feature_value = vector.get(j)

            # print(c)
            # print('c: %d' % (c))

            y_popular = None
            if feature_value > c:
                if g1 > g0:
                    one_bucket += alpha
                else:
                    zero_bucket += alpha
            else:
                if l1 > l0:
                    one_bucket += alpha
                else:
                    zero_bucket += alpha
        # print('end predict')
        return 1 if one_bucket > zero_bucket else 0

    def add_to_counts(self, key, dict, alpha = 1):
        # dictionaries are mutable
        if key in dict:
            dict[key] = dict[key] + alpha
        else:
            dict[key] = alpha

    def get_h_x(self, instance, j, c):
        vector = instance.get_feature_vector()
        prediction = None
        # counts is a dicitonary that maps from a key to how many time it ocurred
        counts = {}
        if vector.get(j) > c:
            for inst in self._instances:
                vect = inst.get_feature_vector()
                if vect.get(j) > c:
                    self.add_to_counts(inst.get_label(), counts)
            prediction = max(counts.iteritems(), key=operator.itemgetter(1))[0]
        else:
            for inst in self._instances:
                vect = inst.get_feature_vector()
                if vect.get(j) <= c:
                    self.add_to_counts(inst.get_label(), counts)
            prediction = max(counts.iteritems(), key=operator.itemgetter(1))[0]

        return prediction

    def calculate_z(self, alpha, h_t):
        z = 0
        for i in range(0, len(self._instances)):
            instance = self._instances[i]
            y_i = 1 if instance.get_label() == 1 else -1
            h = 1 if h_t == 1 else -1
            z = z + self._d[i] * math.exp(-1.0 * alpha * y_i * h)

        return z






        