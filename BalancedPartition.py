import numpy as np
import BiCritetria
import utils

def BalancedPartition(P, a, b):
    Q = []
    D = []
    arbitrary_p = np.zeros_like(P[0])
    arbitrary_p[0] = len(P) + 1
    points = np.vstack((P, arbitrary_p))
    n = P.shape[0]
    for i in xrange(n):
        Q.append(P[i])
        cost = utils.best_fit_line_cost(np.asarray(Q))
        if cost > b or i == (n - 1) :
            T = Q[:-1]
            C = [] # TODO 
            g = utils.calc_best_fit_line(np.asarray(T))
            b = i - len(T) + 1
            e = i
            D.append([C, g, b , e])
            Q = [Q[-1]]
    return D

