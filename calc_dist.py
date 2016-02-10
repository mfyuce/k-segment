import numpy as np
import math
from numpy import linalg


def sqr_sum_axis_func(x):
    return sum(x*x)

def calc_sq_dist(points, subspace):
	subspace = np.matrix(subspace, float)
	points = np.matrix(points, float)
	transition_vector = subspace[0]
	subspace = subspace - transition_vector
	points = points - transition_vector
	print "points:\n", points
	subspace = np.delete(subspace, 0,0)
	subspace = subspace.T
	AtA = subspace.T*subspace
	inv = linalg.inv(AtA)
	p=subspace*inv*subspace.T*points.T
	print "p = \n" ,p
	prjct = points.T - p
	print "prjct= \n", prjct 
	sum_sqr_prjct = np.apply_along_axis(sqr_sum_axis_func, axis=0, arr=prjct)
	print sum_sqr_prjct
	distances = np.sqrt(sum_sqr_prjct);
	print "distances = \n" ,distances
	distances_sum = np.sum(distances)
	return distances_sum;

# x is a list of points in the subspace

x = [(2,3,2,3),(4,8,1,3),(6,3,9,3)]
print "x="
print x

# A is a list of points

A = [(0,1,0,3),(1,1,1,3),(3,8,4,3)]

print "A="
print A

print ("%.7f" % calc_sq_dist(A,x))
#print y.T                                    # Transpose of A.
#print A*x                                    # Matrix multiplication of A and x.
#print A.I                                    # Inverse of A.
