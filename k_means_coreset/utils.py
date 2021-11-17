#!/usr/bin/env python
__author__ = 'Anton'
import numpy as np
from scipy.spatial import distance

def get_sq_distances(x, y):
    return distance.cdist(x, y, 'sqeuclidean')

def get_centers_d(x=None, y=None, d=None):
    if d is None:
        d = get_sq_distances(x, y)
    return np.argmin(d, axis=1), d

def get_centers(x=None, y=None, d=None):
    c, tmp = get_centers_d(x, y, d)
    return c

def get_dist_to_centers(x=None, y=None, d=None):
    if d is None:
        d = get_sq_distances(x, y)
    return np.min(d, axis=1)

def sample(arr, size, prob=None, weights=None):
    if prob is None:
        s = np.random.choice(arr.shape[0], size)
    else:
        s = np.random.choice(arr.shape[0], size, p=prob)
    if weights is None:
        return arr[s]
    else:
        return arr[s], weights[s]
