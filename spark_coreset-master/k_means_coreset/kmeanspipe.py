#!/usr/bin/env python
__author__ = 'Amir'
from random import random
from random import randint
from bisect import bisect
import utils
import numpy as np
from kmeanspp import KMeanspp
import math

class KMeanspipe():
    def __init__(self, points, k, initializationSteps):
        self.points = points
        self.k = k
        self.init_steps = initializationSteps

    def points_cost(self, points, centers):
        """
        this function will return the minimal distance of each point from its closest center

        :param points: a list of points with dimension d
        :param centers: a list of centers from which we'll take the miniaml distance
        :return: the minimal distance of each point from its closest center
        """
        minDist = float("inf")
        for center in centers:
            center = [np.array(center)]
            tmpDistances = utils.get_sq_distances(x=points, y=center)
            minDist = np.minimum(minDist, np.amin(tmpDistances))
        return minDist

    def weighted_choice(self, choices):
        """
        will randonly choose a cell, using the probability in choices.
        :param choices: a list of tuples (<string>, <probability>). prob shouldn't be normalized. example: [("WHITE",90), ("RED",8), ("GREEN",2)]
        :return: picked VALUE
        """
        values, weights = zip(*choices)
        total = 0
        cum_weights = []
        for w in weights:
            total += w
            cum_weights.append(total)
        x = random() * total
        i = bisect(cum_weights, x)
        return values[i]

    def sample_independently_bahman(self, points, centers, overSamplingFactor):
        """
        will return a set of center candidates using formulat described in calling function
        :param points:
        :param centers:
        :param overSamplingFactor: how many new centers will we sample
        :return: center candidates
        """
        C_prime = []
        sq_min_dist_array = utils.get_sq_distances(x=points, y=centers).ravel()
        phy_x_c = sum(sq_min_dist_array)
        if phy_x_c is 0:
            phy_x_c = 0.0000001 #used to handle a singular case where all points are the same point
        for i in range(0, len(points) - 1):
            tmp = sq_min_dist_array[i]
            p_x = overSamplingFactor*sq_min_dist_array[i]/phy_x_c
            rand_x=random()
            if rand_x <= p_x:
                C_prime.append(points[i])
        return list(set(C_prime)) #set removes duplicates


    def pick_first_k_clusters(self):
        '''
        Main function of this Class, executes KMEANS||:
        Algorithm 2 k-means||(k, l) initialization.
        1: C <- sample a point uniformly at random from X
        2: PSI <- distance of all points from C)
        3: for O(log PSI) times do:
        4:      C' <- sample each point x in X independently with probability px =l*d^2(x,C)/PSI_X(C)
        5:      C += C'
        6: end for
        7: For x in C, set wx to be the number of points in X closer to x than any other point in C
        8: Recluster the weighted points in C into k clusters (using KMeans++)
        :return: k initial centers
        '''
        l = 2*self.k
        # 1
        arg_rand = randint(0, len(self.points)-1)
        C = []
        C.append(self.points[arg_rand])
        print "initial center is %s" %(C, )

        # 2 (currently doing nothing with phi, since we used fixed initializationSteps=5)
        #phi = self.points_cost(np.asarray(self.points) , C)

        #3-6
        initializationSteps = self.init_steps #np.log10(phi) - modified original formula in order to align to spark's implementation - 5
        print "number of iterations will be %d" %initializationSteps
        for i in range(0, initializationSteps):
            C_prime = self.sample_independently_bahman(points=self.points, centers=C, overSamplingFactor=l)
            C += C_prime

        #7-8
        print "KMeans|| output before running kmeans++ is %s" %(C, )

        initializedCenters = KMeanspp(np.asarray(C), self.k).seed()
        print "KMeans|| result is %s" %(initializedCenters.tolist(), )
        return initializedCenters

#usage example
points = [(1,3), (100,100), (50,60), (50, 61) , (0,3)]
myInstance = KMeanspipe(points, 2, 5)
myInstance.pick_first_k_clusters()
