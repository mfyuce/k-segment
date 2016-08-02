__author__ = 'Anton'
from weighted_kmeans import KMeans
import numpy as np
from pyspark import SparkContext
import utils

def parseVector(line):
    return np.array([float(x) for x in line.split(' ')])

def distanceToClosest(p, centers):
    closest = float("+inf")
    for i in range(len(centers)):
        tempDist = np.sum((p - centers[i]) ** 2)
        if tempDist < closest:
            closest = tempDist
    return closest

if __name__ == "__main__":
    points = np.loadtxt("coreset_points.txt",dtype=np.float64)
    weights = np.loadtxt("coreset_weights.txt",dtype=np.float64)
    org = np.loadtxt("small_dataset.txt",dtype=np.float64)
    k = 2

    means = KMeans(points, np.expand_dims(weights, axis=0), k, rounds=20)
    means = means.compute()
    real_cost = (np.sum(utils.get_dist_to_centers(org, KMeans(org, np.expand_dims(np.ones(org.shape[0]), axis=0), k, rounds=20).compute())))
    print real_cost



    sc = SparkContext(appName="test_results ")    # start from here.
    sc._jsc.hadoopConfiguration().set("fs.s3n.awsAccessKeyId", "123")
    sc._jsc.hadoopConfiguration().set("fs.s3n.awsSecretAccessKey", "456")
    points = sc.textFile("small_dataset.txt").map(parseVector)
    closest = points.map(lambda p: (distanceToClosest(p, means)))
    cs_result = closest.reduce(lambda a, b: a+b)
    print cs_result
    print "mistake: ", (1-real_cost/cs_result)
    sc.stop()