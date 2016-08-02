#!/usr/bin/env python
__author__ = 'Anton & Ahmad'
from sklearn.datasets import make_blobs
import numpy as np
import sys
"""
In this dataset you will have 3 types of clusters:
  Big - 10K points
  Medium - 100 points
  Small - 1 points
scaled by a factor

You can set the params(b,m,s) to determinate the number of these clusters, otherwise its random.

"""

def gen_dataset(b=None,m=None,s=None):
    points_max_range = 1000
    min0 = 0 #-points_max_range
    fact = 100
    if len(sys.argv) > 1:
        x = int(sys.argv[1])
        fact = max(x, fact)
    print "using factor=", fact

    if b is None:
        b = np.random.randint(0,10)
    if m is None:
        m = np.random.randint(0,10)
    if s is None:
        s = np.random.randint(0,10)

    p = None

    for i in range(0,b):
        x = np.random.randint(min0, points_max_range)
        y = np.random.randint(min0, points_max_range)
        points,yy = make_blobs(1000*fact, centers=[(x,y)],cluster_std=4)
        if p is None:
            p = points
        else:
            p = np.vstack((p,points))

    for i in range(0,m):
        x = np.random.randint(min0, points_max_range)
        y = np.random.randint(min0, points_max_range)
        points,yy = make_blobs(100*fact, centers=[(x,y)],cluster_std=2)
        if p is None:
            p = points
        else:
            p = np.vstack((p,points))

    for i in range(0,s):
        x = np.random.randint(min0, points_max_range)
        y = np.random.randint(min0, points_max_range)
        points,yy = make_blobs(1*fact, centers=[(x,y)],cluster_std=0.5)
        if p is None:
            p = points
        else:
            p = np.vstack((p,points))

    return p, b+m+s

p, k  = gen_dataset(1,3,10)
f="dataset.txt"
print 'saving to file:', f
np.savetxt(f, p)

