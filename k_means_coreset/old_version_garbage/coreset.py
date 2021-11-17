#!/usr/bin/env python
__author__ = 'Anton'
import utils
import numpy as np
from bicriteria import Bicriteria
import weighted_kmeans as w_kmeans

class Coreset(object):
    def __init__(self, points, weights, k):
        self.points = points
        self.weights = weights
        self.k = k


    def get_bicriteria(self):
        m = self.k
        bi = Bicriteria(self.points, self.weights, m)
        return bi.compute()

    #magic
    def find_cluester_size_weighted(self, c,W):
        counts = np.bincount(c.ravel())
        output = counts[c.ravel()]
        return output

    def compute(self, size):
        """
        self.points is a vector with n rows and d cols
        bi its a vector of with klogn rows and d dols
        dist(i) represents the sens(p_i) as in the formula discussed.
        """
        e = w_kmeans.Kmeans(self.points, np.expand_dims(self.weights, axis=0), self.k, 10)
        bi = e.compute()

        dist = utils.get_dist_to_centers(self.points, bi) #find distance of each point to its nearset cluster
        if self.weights is not None: # its always not none!!!
            dist /= np.sum(dist) #norm
        dist *= 2
        c = utils.get_centers(self.points, bi)#get centers
        c = self.find_cluester_size_weighted(c, W=self.weights)#get weighted size of center's cluster
        dist += ((4.0)/(c)) #add to each point the size of its cluster as at the formula
        t = np.sum(dist*self.weights)
        weights = 1/(dist*size)
        weights *= t
        # print (t)
        dist *= self.weights
        dist /= np.sum(dist)
        prob = dist # its actually the sampling probability
        points, weights = utils.sample(self.points, prob, size, weights=weights)
        return points, weights
