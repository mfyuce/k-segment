#!/usr/bin/env python
__author__ = 'Anton'
import numpy as np
import utils
import weighted_kmeans as w_KMeans

class Coreset():
    def __init__(self, points, k, weights=None):
        self.p = points
        self.k = k
        if weights is not None:
            self.w = weights
        else:
            self.w = np.ones(points.shape[0], dtype=np.float64)

    def _find_cluster_size(self, c):
        counts = np.bincount(c, self.w)
        output = counts[c]
        return output

    def compute(self, size, grnds=10, ginit=1):
        q = w_KMeans.KMeans(self.p, np.expand_dims(self.w , axis=0), self.k, grnds, ginit).compute() # this is my kmeans for the coreset.
        sq_d = utils.get_sq_distances(self.p, q) # Squared distances from each point to each center
        dist = utils.get_dist_to_centers(d=sq_d) # I get the sq dist from each point its center.
        dist /= np.sum(dist) # Norm 
        dist *= 2 # according to the paper
        c = utils.get_centers(d=sq_d) # I get the index of the center
        c = self._find_cluster_size(c) # Find the size of the cluster for each point.
        s = dist + 4.0/c # I add it, the 4 is according to the paper.
        t = np.sum(s*self.w) # This is the t from the paper.
        u = t/(s*size) # the new weights for coreset.
        prob = s*self.w/t # the probability for sampling
        p, w = utils.sample(self.p, size, prob=prob, weights=u) # sample coreset: points + weights.
        return p, w

