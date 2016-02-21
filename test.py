import numpy as np
import utils
import ksegment
import BiCritetria

def main():
    # generate points
    N = 500
    dimension = 2
    k = 300

    data = np.random.random_integers(0, 100, (N,dimension))
    P = np.c_[np.mgrid[1:N+1], data]

    dividers = ksegment.k_segment(P, k)
    bicriteria_est = BiCritetria.bicriteria(P,k)
    print "BiCritetria estimated distance sum: ", bicriteria_est
    utils.visualize_3d(P, dividers)
    

main()