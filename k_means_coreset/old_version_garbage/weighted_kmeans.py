#!/usr/bin/env python
__author__ = 'Anton'
import numpy as np
import utils

class Kmeans(object):
    def __init__(self, points, weights, k, rounds=10):
        self.p = points
        self.w = weights
        self.k = k
        self.e = rounds
        self.centers = None

    def compute(self):
        self.centers, temp = utils.sample(self.p, None, n=self.k)     # random k centers
        np.reshape(self.centers, (self.k, 2))   #just fix the shape
        dist = utils.get_centers(self.p ,self.centers)
        points = self.p
        weights = self.w.T
        for j in range(0, self.e):
            for i in range(0, self.k):
                x = [dist == i]
                a = points[x]
                w = weights[x]
                c = a*w
                new_center = np.sum(c ,axis=0, keepdims=1)
                if np.sum(w) == 0:
                    continue
                new_center /= np.sum(w)
                self.centers[i] = new_center
            dist = utils.get_centers(self.p, self.centers)

        return self.centers



