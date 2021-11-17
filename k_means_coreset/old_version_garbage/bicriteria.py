#!/usr/bin/env python
__author__ = 'Anton'
import utils
import numpy as np

class Bicriteria(object):
    def __init__(self, p, w, m):
        self.p = p
        self.w = w
        self.m = m

    #TODO: fix this to work with weights
    def drop_half_points(self, points, weights, M):
        d = utils.get_dist_to_centers(points, M)
        median = np.median(d)
        points = points[d>median]
        if weights is not None:
            weights = weights[d>median]
        return points, weights


    def drop_half_weighted_points(self, points, weights, M, W):
        left = W
        points_to_drop=[]
        d = utils.get_dist_to_centers(points, M)
        idx = np.argsort(d)
        i = 0
        while left > 0:
            index = idx[i]
            if weights[index] > left:
                weights[index] -= left
                left = 0
            else:
                left -= weights[index]
                points_to_drop.append(index)
            i += 1

        points = np.delete(points,points_to_drop,axis=0)
        weights = np.delete(weights,points_to_drop)
        return points, weights

    def compute(self):
        bi = None
        wi = None
        points = self.p
        weights = np.array(self.w)
        W = np.sum(weights) / 2 # I should drop half of weight
        while W > self.m:
            prob = weights*1.0 / np.sum(weights) #Sums to 1
            M, w = utils.sample(points, prob, self.m, self.w) #sample points
            #if-else to concatane points to current dataset
            if bi is None:
                bi = M
                wi = w
            else:
                bi = np.vstack((bi,M))
                wi = np.hstack((wi,w))
            points, weights = self.drop_half_weighted_points(points, weights, M, W)
            if points.shape[0] < self.m:
                break
            W = np.sum(weights) / 2
            W = int(W) #TODO: is that good?
        return bi,wi
