import numpy as np
import utils
import ksegment
import Coreset

def random_data(N, dimension):
    return np.random.random_integers(0, 100, (N,dimension))

def example1(n):
    #3 straight lines with noise
    x1 = np.mgrid[1:9:2*n/6j]
    y1 = np.mgrid[-5:3:2*n/6j]
    x2 = np.mgrid[23:90:n/2j]
    y2 = np.mgrid[43:0:n/2j]
    x3 = np.mgrid[80:60:n/6j]
    y3 = np.mgrid[90:100:n/6j]

    x = np.r_[x1,x2,x3]
    y = np.r_[y1,y2,y3]
    x += np.random.normal(size=x.shape) * 4
    #y += np.random.normal(size=y.shape) * 4
    return np.c_[x,y]

def example2():
    x1 = np.mgrid[1:9:100j]
    y1 = np.mgrid[-5:3:100j]
    x1 += np.random.normal(size=x1.shape) * 4
    return np.c_[x1,y1]

def main():
    # generate points
    N = 600
    dimension = 2
    k = 3

    #data = random_data(N, dimension)
    # for example1 choose N that divides by 6
    data = example1(N)

    P = np.c_[np.mgrid[1:N+1], data]

    coreset = Coreset.build_coreset(P, k, 8)
    dividers = ksegment.coreset_k_segment(coreset, k)
    #dividers = ksegment.k_segment(P, k)
    #W = Coreset.PiecewiseCoreset(N, utils.s_func, 0.1)
    #bicriteria_est = Coreset.bicriteria(P,k)
    #print "BiCritetria estimated distance sum: ", bicriteria_est
    utils.visualize_3d(P, dividers)

main()