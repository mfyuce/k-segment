#!/usr/bin/env python
__author__ = 'Ahmad'

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--num_means', default=2, help="k for k-means")
    parser.add_argument('-i', '--input', default="2c300k_dataset.txt", help="input txt file")
    parser.add_argument('-v', '--verbose', default=3, help="verbose printing. 1: header, 2: cost, 3: centers, 6: points")
    args = parser.parse_args()
    return args
 
def load_test(f):
    p = np.loadtxt(f)
    w = np.ones(p.shape[0])
    return p, w  

def do_kmeans(k, datafile, n_init=10, plot=True, v=1):
    p, w = load_test(datafile)
    p = np.array(p, dtype='float64')
    
    alg = KMeans(n_clusters=k, n_init=n_init).fit(p)
    means = alg.cluster_centers_
    cost = alg.inertia_
    
    if v>0:
        print "K-means using sklearn for ", datafile, ", with k=", k, "."
        if v>1: print "reference cost: ", cost
        if v>2: print "centers: ", means
        if v>3: print "centers str: ", str(means).replace("\n", ";")
        if v>5: print "points: ", p
    
    if plot:
        plt.plot(p[:,0], p[:,1],'go')
        plt.plot(means[:,0], means[:,1],'ro')
        plt.show()
        
    return cost, means, p.shape[0]

def main(argv):
    args = get_args()
    k = int(args.num_means)
    datafile = args.input
    prnt = int(args.verbose)
    do_kmeans(k, datafile, v=prnt)

if __name__ == "__main__":
    main(sys.argv)

