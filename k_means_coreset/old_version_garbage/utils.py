#!/usr/bin/env python
__author__ = 'Anton'
import numpy as np

def get_sq_distances(X, Y):
    a = np.sum(np.square(X),axis=1,keepdims=1)
    b = np.ones((1,Y.shape[0]))
    c = a.dot(b)
    a = np.ones((X.shape[0],1))
    b = np.sum(np.square(Y),axis=1,keepdims=1).T
    c += a.dot(b)
    c -= 2*X.dot(Y.T)
    return c

def get_centers(X, Y):
    d = get_sq_distances(X, Y)
    return np.argmin(d, axis=1)

def get_dist_to_centers(X, Y):
    d = get_sq_distances(X, Y)
    return np.min(d, axis=1)


def sample(p, w, n, weights=None):
    if w is None:
        s = np.random.choice(range(0, p.shape[0]), n, replace=True)
        return p[s], None
    else:
        s = np.random.choice(range(0, p.shape[0]), n, replace=True, p=w)
        points = p[s]
        if weights is not None:
            weights = weights[s]
        return points, weights
