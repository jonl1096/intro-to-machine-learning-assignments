#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
import scipy.stats
import math
import time
from cs475_types import Predictor

class NaiveBayes(Predictor):
    def __init__(self, num_clusters, max_max_index, iterations):
        # The number of clusters, K
        self._num_clusters = num_clusters
        # The maximum feature index of all the instances
        self._max_max_index = max_max_index
        self._max_size = self._max_max_index + 1
        # _means is a dict of numpy arrays, 0 indexed, so index 0 refers to the mean for the 0th cluster
        self._means = {}
        # _variances is a dict of numpy arrays, keys are clusters
        self._variances = {}
        # _phis is a dict of numpy arrays, keys are clusters
        self._phis = {}

        # A list of lists of instances, 0 indexed
        self._folds = {} # YOU CAN GET RID OF THIS
        self._iterations = iterations
        # Number of instances is len(instances) because instances is 0 indexed
        self._indicators = []

        self._s = None

        # array keeping track of which instances are in each cluster, keys are cluster indices
        # values are arrays of instances
        self._clusters = {}


    def train(self, instances):
        max = self._max_max_index + 1
        num_instances = len(instances)
        # start = time.time()
        self.init_clusters(instances)
        # run_time = time.time() - start
        # print('Time taken for initialization: %f' % run_time)

        # print('means')
        # print(self._means)
        # print('variances')
        # print(self._variances)

        for i in range(1, self._iterations + 1):

            # print('means')
            # print(self._means)
            # print('variances')
            # print(self._variances)
            # print('phis')
            # print(self._phis)

            # print('ITERATION %d' %i)
            self._clusters = {}
            # The E step
            # start = time.time()
            for instance in instances:
                self.predict(instance)
            # run_time = time.time() - start
            # print('Time taken for E step: %f' % run_time)
            # print('clusters')
            # # print(self._clusters)
            # for key, value in self._clusters.iteritems():
            #     print(key)
            #     for i in range(0, len(value)):
            #         print(value[i].get_features())

            # print(self._indicators)
            # print(self._prototypes)

            # The M step
            # start = time.time()
            for k in range(0, self._num_clusters):
                if k in self._clusters:
                    Nk = len(self._clusters[k])
                    # updating phi
                    self._phis[k] = (Nk + 1.0) / (num_instances + self._num_clusters) 

                    # updating means
                    mean_of_cluster = self.calculate_mean(self._clusters[k])
                    # mean_of_cluster = np.zeros(max, dtype = np.float64)
                    # if len(self._clusters[k]) > 0:
                    #     # if cluster is not empty
                    #     for instance in self._clusters[k]:
                    #         mean_of_cluster += instance.get_full_features(max)
                    #     mean_of_cluster = (1.0 / Nk) * mean_of_cluster

                    # updating variances
                    if len(self._clusters[k]) >= 2:
                        # print('BOOOASDOFAOSDF')
                        # var_of_cluster = np.zeros(max, dtype = np.float64)
                        # for instance in self._clusters[k]:
                        #     var_of_cluster += self.euc_dist(instance.get_full_features(max), 
                        #         mean_of_cluster)**2

                        var_of_cluster = self.calculate_variance(self._clusters[k], mean_of_cluster)
                        # print(var_of_cluster)
                        self.check_variance_bound(var_of_cluster)
                        # # Making sure none of the variances go below that of S
                        # for j in range(1, len(var_of_cluster)):
                        #     feature_var = var_of_cluster[j]
                        #     if feature_var < self._s[j]:
                        #         var_of_cluster[j] = self._s[j]
                    else:
                        # if there are 0 or 1 instances in the cluster
                        # print('S')
                        # print(self._s)
                        var_of_cluster = self._s
                else:
                    mean_of_cluster = np.zeros(max, dtype = np.float64)
                    var_of_cluster = self._s


                self._means[k] = mean_of_cluster
                self._variances[k] = var_of_cluster
            # run_time = time.time() - start
            # print('Time taken for M step: %f' % run_time)

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

        # print('vector')
        # print(vector)
        max_lprob = float("-inf")
        max_lprob_cluster_index = None
        for i in range(0, self._num_clusters):
            # iterating through the clusters to see which one gives the highest probability
            mean = self._means[i]
            # print('mean')
            # print(mean)
            # std_dev = np.sqrt(self._variances[i]) # element-wise square root
            variance = self._variances[i] # element-wise square root
            # print('variance')
            # print(variance)


            # calculating the gaussian probabilities
            # gprobs = scipy.stats.norm(mean, std_dev).pdf(vector) # element-wise calculations

            # if len(mean) < len(vector):
            #     # only a problem when vector len is greater than mean/std_dev len because 
            #     # we iterate through all of vector
            #     mean.resize(vector.shape)
            #     std_dev.resize(vector.shape)

            sum = 0 # max of sum of log probabilities = max of product of probabilities
            # start = time.time()
            for j in range(1, len(vector)): # features are 1 indexed
                # gprob = scipy.stats.norm(mean[j], std_dev[j]).pdf(vector[j])
                # if variance[j] == 0:
                #     print('VAR WAS 0:')
                #     print(i)
                #     print(j)
                if variance[j] != 0:
                    log_g_prob = self.log_gaussian_probability(mean[j], variance[j], vector[j])
                    # print('WORK %d' % j)
                    # print(math.log(gprob))
                    # print(sum)
                    # print('log_g_prob')
                    # print(log_g_prob)
                    sum += log_g_prob
            # run_time = time.time() - start
            # print('Time taken for calculating gaussian prob: %f' % run_time)

            # print(sum)

            # print(math.log(self._phis[i]))
            sum += math.log(self._phis[i])

            # print('joint lprob of cluster %d' % i)
            # print(sum)
            # print('mean')
            # print(mean)
            # print('std_dev')
            # print(std_dev)

            if sum > max_lprob:
                max_lprob = sum
                max_lprob_cluster_index = i

        if max_lprob_cluster_index in self._clusters:
            self._clusters[max_lprob_cluster_index].append(instance)
        else:
            self._clusters[max_lprob_cluster_index] = [instance]
            # print('CLUSTER CREATED: %d' % max_lprob_cluster_index)

        return max_lprob_cluster_index


    def init_clusters(self, instances):
        """ Function for initializing clusters, means, variances, and phis. """
        max = self._max_max_index + 1
        # initializing the folds
        for i in range(0, len(instances)):
            k = i % self._num_clusters
            if k not in self._folds:
                # if the kth cluster has not been initialized, initialize it to an array of instances
                self._folds[k] = [instances[i]]
            else:
                self._folds[k].append(instances[i])

        # initializing s
        mean_of_instances = self.calculate_mean(instances)
        # mean_of_instances = np.zeros(max, dtype = np.float64)
        # for instance in instances:
        #     mean_of_instances += instance.get_full_features(max)
        # mean_of_instances = (1.0 / N) * mean_of_instances

        var_of_instances = self.calculate_variance(instances, mean_of_instances)
        self._s = 0.01 * var_of_instances
        # print('S')
        # print(self._s)

        # initializing the means and variances and phis
        for i in range(0, self._num_clusters):
            # initializing means
            N = len(self._folds[i])
            mean_of_fold = np.zeros(max, dtype = np.float64)
            for instance in self._folds[i]:
                mean_of_fold += instance.get_full_features(max)
            mean_of_fold = (1.0 / N) * mean_of_fold 
            self._means[i] = mean_of_fold

            # initializing variances
            variance_of_fold = self.calculate_variance(self._folds[i], mean_of_fold)
            self.check_variance_bound(variance_of_fold)
            self._variances[i] = variance_of_fold

            # initializing phis
            phi = (N + 1.0) / (len(instances) + self._num_clusters)
            self._phis[i] = phi

        

        # var_of_instances = np.zeros(max, dtype = np.float64)
        # for instance in instances:
        #     var_of_instances += self.euc_dist(instance.get_full_features(max), mean_of_instances)**2
        # var_of_instances = (1.0 / (N - 1)) * var_of_instances
        # self._s = 0.01 * var_of_instances

    def calculate_mean(self, cluster):
        Nk = len(cluster)
        mean = np.zeros(self._max_size, dtype = np.float64)
        for instance in cluster:
            mean += instance.get_full_features(self._max_size)
        mean = (1.0 / Nk) * mean
        return mean


    def calculate_variance(self, cluster, mean):
        Nk = len(cluster)
        variance = np.zeros(self._max_size, dtype = np.float64)
        # need another for loop because we need to have finished calculating the mean
        # in order to calculate the variance
        for instance in cluster:
            difference = instance.get_full_features(self._max_size) - mean
            variance += difference**2
        variance = (1.0 / (Nk - 1)) * variance
        return variance


    def check_variance_bound(self, variance):
        # Making sure none of the variances go below that of S
        for j in range(1, len(variance)):
            feature_var = variance[j]
            if feature_var < self._s[j]:
                # print('VARIANCE TOO LOW')
                # print(j)
                variance[j] = self._s[j]


    def log_gaussian_probability(self, mean, variance, value):
        """ Returns a scalar. """
        term1 = math.log(1 / math.sqrt(2.0 * variance * math.pi))
        term2 = (value - mean)**2 / (2.0 * variance)
        return term1 - term2


    def make_same_size(self, v1, v2):
        """ Makes two numpy arrays of different sizes, into the same size. """
        if len(v1) > len(v2):
            v2.resize(v1.shape, refcheck=False)
        elif len(v1) < len(v2):
            v1.resize(v2.shape, refcheck=False)

        # return (v1, v2)


