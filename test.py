import numpy as np
import utils
import ksegment
import BiCritetria

def main():
    # generate points
    N = 100
    dimension = 2
    data = np.random.random_integers(0, 100, (N,dimension))
    P = np.c_[np.mgrid[1:N+1], data]

    dividers = ksegment.k_segment(P, 4)
    utils.visualize_3d(P, dividers)

main()