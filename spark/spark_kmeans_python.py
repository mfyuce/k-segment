"""
LOOK AT:
http://spark.apache.org/docs/latest/mllib-clustering.html#k-means

Not sure whether this is the fastest way for doing this,
But it's good enough at the moment.
"""
import ConfigParser
from pyspark.mllib.clustering import KMeans, KMeansModel
from numpy import array
#from math import sqrt
from pyspark import SparkContext
import tree

def skp_rdd(k, rdd, num_partitions, rounds=10, initMode="k-means||", initSteps=1):
    return KMeans.train(rdd, k, maxIterations=rounds, runs=1, initializationMode=initMode, initializationSteps=initSteps)    

def skp(k, sc, infile, num_partitions, rounds=10, initMode="k-means||", initSteps=1):    
    data = sc.textFile(infile, num_partitions)
    parsedData = data.map(lambda line: array([float(x) for x in line.split(' ')]))
    # Build the model (cluster the data)
    clusters = KMeans.train(parsedData, k, maxIterations=rounds, runs=1, initializationMode=initMode, initializationSteps=initSteps)    
    return clusters, parsedData

def skp_cost(k, sc, infile, num_partitions):
    clusters, parsedData = skp(k, sc, infile, num_partitions)

    # Evaluate clustering by computing Within Set Sum of Squared Errors
    def error(point):
        center = clusters.centers[clusters.predict(point)]        
        return (sum([x**2 for x in (point - center)]))

    WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
    return WSSSE     

if __name__ == "__main__":
    sc, config, infile = tree.init_spark()
    num_partitions = config.getint("conf", "numOfPartitions")
    k = 2

    WSSSE = skp_cost(k, sc, infile, 2)#num_partitions)
    print("Within Set Sum of Squared Error = " + str(WSSSE))

