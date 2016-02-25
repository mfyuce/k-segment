import numpy as np
import utils
import ksegment
import BiCritetria
import BalancedPartition

def main():
    # generate points
    N = 140
    dimension = 2
    k = 3

    #random
    #data = np.random.random_integers(0, 100, (N,dimension))

    #3 straight lines with noise
    x1 = np.mgrid[1:9:40j]
    y1 = np.mgrid[-5:3:40j]
    x2 = np.mgrid[23:90:80j]
    y2 = np.mgrid[43:0:80j]
    x3 = np.mgrid[80:60:20j]
    y3 = np.mgrid[90:100:20j]

    x = np.r_[x1,x2,x3]
    y = np.r_[y1,y2,y3]
    x += np.random.normal(size=x.shape) * 4
    #y += np.random.normal(size=y.shape) * 4
    data = np.c_[x,y]

    P = np.c_[np.mgrid[1:N+1], data]

    #dividers = ksegment.k_segment(P, k)
    #bicriteria_est = BiCritetria.bicriteria(P,k)
    #print "BiCritetria estimated distance sum: ", bicriteria_est
    #utils.visualize_3d(P, dividers)
    
    res = BalancedPartition.BalancedPartition(P, 1, 40)
    return res

main()