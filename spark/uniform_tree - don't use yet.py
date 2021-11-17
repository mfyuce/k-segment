#!/usr/bin/env python
__author__ = 'Anton'
import numpy as np
from pyspark import SparkContext
 
 
 
if __name__ == "__main__":
    def readPointBatch(Int,iterator):
        return [(Int/2, np.loadtxt(iterator))] # np.loadtext gets a text file and makes it a numpy array
 
    def randomSampling(a, b):
        c = np.vstack((a, b)) # concatenate arrays
        size = (a.shape[0]+b.shape[0])/2 # size of a should be size the of b, so it's half of c actually
        idx = np.random.choice(range(0, c.shape[0]), size, replace=True) # sample indexes
        return c[idx] # return the sampled array. if a+b is of size 2x,  we will return only x points back.
 
    sc = SparkContext(appName="uniform_coreset")    # start from here.
    initial_num_of_partitions = 10 # If we are using HDFS with 5mb per block its not relevant..
    sc._jsc.hadoopConfiguration().set("fs.s3n.awsAccessKeyId", "123")
    sc._jsc.hadoopConfiguration().set("fs.s3n.awsSecretAccessKey", "456")
    points = sc.textFile("small_dataset.txt", initial_num_of_partitions).mapPartitionsWithIndex(readPointBatch) # from text file to (key,numpy_array)
 
    def reduce(rdd, f):
        while rdd.getNumPartitions() != 1:
            rdd = (rdd
                   .reduceByKey(f)  # merge couple and reduce by half
                   .map(lambda x: (x[0] / 2, x[1]))  # set new keys
                   .partitionBy(rdd.getNumPartitions() / 2))    # reduce num of partitions
 
        return rdd.reduceByKey(f).first()[1] #for case its not a complete binary tree. first is actually everything now..
 

	res = reduce(points, randomSampling)
    np.savetxt("myoutput_uniform.txt", reduce(points, randomSampling))