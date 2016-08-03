#!/usr/bin/env python
__author__ = 'Anton'
import numpy as np
import pandas as pd
import ConfigParser
from cStringIO import StringIO
from pyspark import SparkContext

from coreset import Coreset     # Should i put it inside some conf?

k = 2

def init_spark(open_sc=1):
    config = ConfigParser.RawConfigParser()
    config.read('config.ini')
    infile = config.get("conf", "input_s3_url")
    sc = None
    def addFiles(sc, files):
        for x in files:
            sc.addFile(x)
    if open_sc:
        sc = SparkContext(appName="coreset")
        sc._jsc.hadoopConfiguration().set("fs.s3n.awsAccessKeyId", config.get("conf", "awsAccessKeyId"))
        sc._jsc.hadoopConfiguration().set("fs.s3n.awsSecretAccessKey", config.get("conf", "awsSecretAccessKey"))
        addFiles(sc, config.get("conf", "files").split(','))
    return sc, config, infile

if __name__ == "__main__":
    def readPointBatch(Int,iterator):
        return [(Int/2, [pd.read_csv(StringIO("\n".join(iterator)), header=None, delim_whitespace=True,dtype=np.float64).as_matrix(), None])]

    def merge(a, b):
        points = np.vstack((a[0], b[0]))
        weights = None
        if a[1] is None and b[1] is not None:
            a[1] = np.ones(a[0].shape[0])

        if a[1] is not None and b[1] is None:
            b[1] = np.ones(b[0].shape[0])

        if a[1] is not None and b[1] is not None:
            weights = np.hstack((a[1], b[1]))

        size = (a[0].shape[0]+b[0].shape[0])/2
        return points, weights, size

    def reduce(data, size):
        c = Coreset(data[0], k, data[1])
        p, w = c.compute(size)
        return [p, w] # needs to be a python iterator for spark.

    def coreset(a, b):
        p, w , size = merge(a, b)
        c = reduce([p, w], size)
        return c

    # start from here.
    sc, config, infile = init_spark()
    initial_num_of_partitions = config.getint("conf", "numOfPartitions")
    
    points = sc.textFile(infile, initial_num_of_partitions).mapPartitionsWithIndex(readPointBatch) # from text file to (key,numpy_array)
    def computeTree(rdd, f):
        while rdd.getNumPartitions() != 1:
            rdd = (rdd
                   .reduceByKey(f)  # merge couple and reduce by half
                   .map(lambda x: (x[0] / 2, x[1]))  # set new keys
                   .partitionBy(rdd.getNumPartitions() / 2))    # reduce num of partitions

        return rdd.reduceByKey(f).first()[1] #for case its not a complete binary tree. first is actually everything now..
        #return the corest as a numpy array

    result = (computeTree(points, coreset))
    np.savetxt("coreset_points.txt", result[0])
    np.savetxt("coreset_weights.txt", result[1])

