#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os
import argparse
import sys
import pickle
import time

# from pudb import set_trace; set_trace()
from cs475_types import ClassificationLabel, FeatureVector, Instance, Predictor
from lambdameans import LambdaMeans
from naivebayes import NaiveBayes
# from scipy.sparse import lil_matrix, csr_matrix, coo_matrix

def load_data(filename):
    """Function for loading the features from a file into instances"""
    instances = []
    with open(filename) as reader:
        # print('adding to the instances')
        global max_max_index
        max_max_index = 0
        for line in reader:
            if len(line.strip()) == 0:
                continue
            
            # Divide the line into features and label.
            split_line = line.split(" ")
            label_string = split_line[0]

            int_label = -1
            try:
                int_label = int(label_string)
            except ValueError:
                raise ValueError("Unable to convert " + label_string + " to integer.")

            label = ClassificationLabel(int_label)
            feature_vector = FeatureVector()
            
            for item in split_line[1:]:
                try:
                    index = int(item.split(":")[0])

                except ValueError:
                    raise ValueError("Unable to convert index " + item.split(":")[0] + " to integer.")
                try:
                    value = float(item.split(":")[1])
                except ValueError:
                    raise ValueError("Unable to convert value " + item.split(":")[1] + " to float.")
                
                if value != 0.0:
                    feature_vector.add(index, value)

            instance = Instance(feature_vector, label)
            instances.append(instance)
            # if feature_vector._size > max_size:
            #     max_size = feature_vector._size
            if feature_vector._max_index > max_max_index:
                max_max_index = feature_vector._max_index
        # print('finished adding')

    return instances


def get_args():
    parser = argparse.ArgumentParser(description="This is the main test harness for your algorithms.")

    parser.add_argument("--data", type=str, required=True, help="The data to use for training or testing.")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"],
                        help="Operating mode: train or test.")
    parser.add_argument("--model-file", type=str, required=True,
                        help="The name of the model file to create/load.")
    parser.add_argument("--predictions-file", type=str, help="The predictions file to create.")

    parser.add_argument("--algorithm", type=str, help="The name of the algorithm for training.")

    parser.add_argument("--cluster-lambda", type=float, help="The value of lambda in lambda-means.",
        default=0.0)

    parser.add_argument("--num-clusters", type=int, 
        help="The number of clusters in Naive Bayes clustering.", default=3)

    parser.add_argument("--clustering-training-iterations", type=int, 
        help="The number of clustering iterations.", default=10)

    args = parser.parse_args()
    check_args(args)

    return args


def check_args(args):
    if args.mode.lower() == "train":
        if args.algorithm is None:
            raise Exception("--algorithm should be specified in mode \"train\"")
    else:
        if args.predictions_file is None:
            raise Exception("--prediction-file should be specified in mode \"test\"")
        if not os.path.exists(args.model_file):
            raise Exception("model file specified by --model-file does not exist.")


def train(instances):
    print('starting training')
    p = None
    if args.algorithm == 'lambda_means':
        p = LambdaMeans(args.cluster_lambda, max_max_index, args.clustering_training_iterations)
        p.train(instances)
    elif args.algorithm == 'nb_clustering':
        p = NaiveBayes(args.num_clusters, max_max_index, args.clustering_training_iterations)
        p.train(instances)
        
    print('ending training')
    return p


def write_predictions(predictor, instances, predictions_file):
    try:
        print("Writing predictions")
        with open(predictions_file, 'w') as writer:
            for instance in instances:
                label = predictor.predict(instance)
        
                writer.write(str(label))
                writer.write('\n')
        print("finished writing predictiosn")
    except IOError:
        raise Exception("Exception while opening/writing file for writing predicted labels: " + predictions_file)

def main():
    global args 
    args = get_args()

    if args.mode.lower() == "train":
        start = time.time()
        # Load the training data.
        instances = load_data(args.data)

        # Train the model.
        predictor = train(instances)
        # print("w  = %f, learning_rate = %f" % (predictor._w, predictor._learning_rate))
        run_time = time.time() - start
        print('Time taken for training: %f' % run_time)
        try:
            with open(args.model_file, 'wb') as writer:
                pickle.dump(predictor, writer)
        except IOError:
            raise Exception("Exception while writing to the model file.")        
        except pickle.PickleError:
            raise Exception("Exception while dumping pickle.")
            
    elif args.mode.lower() == "test":
        start = time.time()
        # Load the test data.
        instances = load_data(args.data)
        run_time = time.time() - start
        print('Time taken for testing: %f' % run_time)

        predictor = None
        # Load the model.
        try:
            with open(args.model_file, 'rb') as reader:
                predictor = pickle.load(reader)
                # print("AASDFASDFASDFASD" if predictor == None else "HELLOOOO")
                # print(predictor._w)
        except IOError:
            raise Exception("Exception while reading the model file.")
        except pickle.PickleError:
            raise Exception("Exception while loading pickle.")

        # print("LEARNGIN RATE: %d" % predictor._learning_rate)
            
        write_predictions(predictor, instances, args.predictions_file)
        # print('finished testing and writing predictions')
    else:
        raise Exception("Unrecognized mode.")

if __name__ == "__main__":
    main()

