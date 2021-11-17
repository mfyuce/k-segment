"""
A K-means clustering program using MLlib.
spark-submit kmeans_mllib.py <the path to dataset> <the number of desired clusters>

Usage of train function:
    - k is the number of desired clusters.
    - maxIterations is the maximum number of iterations to run.
    - initializationMode specifies either random initialization or initialization via k-means||.
    - runs is the number of times to run the k-means algorithm (k-means is not guaranteed to find a globally optimal solution,
      and when run multiple times on a given dataset, the algorithm returns the best clustering result).
    - initializationSteps determines the number of steps in the k-means|| algorithm.
    - epsilon determines the distance threshold within which we consider k-means to have converged.
    - initialModel is an optional set of cluster centers used for initialization. If this parameter is supplied, only one
      run is performed.
"""
from __future__ import print_function

import sys

import numpy as np
from pyspark import SparkContext
from pyspark.mllib.clustering import KMeans


def parseVector(line):
    return np.array([float(x) for x in line.split(' ')])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Invalid number of parameters! "
              "Usage: kmeans <file> <k>", file=sys.stderr)
        exit(-1)
    sc = SparkContext(appName="KMeans")
    lines = sc.textFile(sys.argv[1])
    data = lines.map(parseVector)
    k = int(sys.argv[2])
    model = KMeans.train(data, k, maxIterations=10, runs=10, initializationMode="random")
    print("Total Cost: " + str(model.computeCost(data)))
    print("Final centers: " + str(model.clusterCenters))
    sc.stop()