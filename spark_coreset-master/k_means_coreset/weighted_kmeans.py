#!/usr/bin/env python
__author__ = 'Anton'
import numpy as np
import utils
from kmeanspp import KMeanspp

class KMeans(object):
    def __init__(self, points, weights, k, rounds=10, n_init=1, epsilon=0.0001):
        self.p = points
        self.w = weights
        self.k = k
        self.e = rounds
        self.centers = None
        self.ni = n_init
        self.epsilon = epsilon

    def _rand_seeds(self):
        prob = self.w.ravel()/np.sum(self.w)
        p1, tmp = utils.sample(self.p, self.k, prob, prob) #hack weights param
        return np.array(p1, dtype=np.float64)
    
    def compute(self, kmpp=True):
        if kmpp: self.centers = KMeanspp(self.p, self.k, self.w.ravel(), n_init=self.ni).compute()
        else:    self.centers = self._rand_seeds()
        np.reshape(self.centers, (self.k, self.p.shape[1]))   #just fix the shape
            
        dist, d = utils.get_centers_d(self.p, self.centers)
        xcost = np.sum(utils.get_dist_to_centers(self.p, self.centers, d)*self.w)
        points = self.p
        weights = self.w.T
        for j in range(0, self.e):
            for i in range(0, self.k):
                x = [dist == i]
                a = points[x]
                w = weights[x]
                c = a*w
                new_center = np.sum(c, axis=0, keepdims=1)
                if np.sum(w) == 0:
                    print "not nice"
                    continue
                new_center /= np.sum(w)
                self.centers[i] = new_center
            dist, d = utils.get_centers_d(self.p, self.centers)
            cost = np.sum(utils.get_dist_to_centers(self.p, self.centers, d)*self.w)
            if abs(xcost - cost) < self.epsilon:
                break
            xcost = cost

        return self.centers
