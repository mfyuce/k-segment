import numpy as np
import math
import utils

def Coreset(P, k, eps):
    h = bicriteria(P, k)
    b = (eps**2 * h) / (100*k*np.log2(P.shape[0]))
    return BalancedPartition(P, eps, b)

def one_seg (P):
    return utils.best_fit_line_cost(P)

def bicriteria (P,k):
    # sort array by first index (t) so segments will be continous
    P = np.array(sorted(P, key=lambda point: point[0]))
    #print "P = \n", P

    # Lines  1 - 3
    if (len(P) <= (2 * k + 1)):
        return one_seg(P)

    # line 5 - 9
    m = int(math.floor(len(P)/(2*k)))
    i = 0
    j = m

    # one_seg_res will hold segment starting index and result (squred distance sum)
    one_seg_res = []
    
    # partition to 2k segments and call 1-segment for each
    while (i < len(P)):
        one_seg_res.append((one_seg(P[i:j,1:]), int(i)))
        i = i + m
        j = j + m

    #sort result
    one_seg_res = sorted(one_seg_res, key=lambda res: res[0])

    # res = the distances of the min k+1 segments
    res = 0

    # sum distances of k+1 min segments and make a list of point to delete from P to get P \ Q from the algo'
    rows_to_delete = []
    for a in xrange(k+1):
        res = res + one_seg_res[a][0]
        for b in xrange(m):  
            rows_to_delete.append(one_seg_res[a][1]+b)
    
    # lines 10-11
    P = np.delete(P, rows_to_delete, axis=0)
    return res + bicriteria(P, k)

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

