#!/usr/bin/env python
__author__ = 'Anton'
import numpy as np
import utils

class KMeanspp():
    def __init__(self,points, k, weights=None, n_init=1):
        self.p = points
        if weights is None:
            weights = np.ones(points.shape[0],dtype=np.float64)
        self.w = weights
        self.k = k
        self.n_init = n_init

    def seed(self):
        k = self.k-1
        centers = []
        prob = self.w/np.sum(self.w)
        center = utils.sample(self.p, 1, prob)
        centers.append(center[0])
        min_dist = None
        while k > 0:
            np_centers = np.array(centers)
            if min_dist is None:
                d = utils.get_sq_distances(x=self.p, y=np_centers).ravel()
                min_dist = d
            else:
                d = utils.get_sq_distances(x=self.p, y=np.array([np_centers[-1]])).ravel()
                min_dist = np.minimum(min_dist, d)
            dist = np.array(min_dist)
            dist *= self.w
            prob = dist / np.sum(dist)
            center = utils.sample(self.p, 1, prob)
            centers.append(center[0])
            k -= 1
        return np.array(centers, dtype=np.float64)
        
    def compute(self):
        best_cent = self.seed()
        if self.n_init==1: return best_cent         
        best_cost = np.sum(utils.get_dist_to_centers(self.p, best_cent)*self.w)
        for i in range(self.n_init - 1):
            cent = self.seed()
            cost = np.sum(utils.get_dist_to_centers(self.p, cent)*self.w)
            if cost < best_cost:
                best_cost = cost
                best_cent = cent
        return best_cent

