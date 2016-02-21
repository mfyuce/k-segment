import numpy as np
import math
import random as rnd
import utils

def one_seg (P):
    best_fit_line = utils.calc_best_fit_line(P)
    return utils.sqrd_dist_sum(P, best_fit_line)

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
    for a in range(k+1):
        res = res + one_seg_res[a][0]
        for b in range(m):  
            rows_to_delete.append(one_seg_res[a][1]+b)
    
    # lines 10-11
    P = np.delete(P, rows_to_delete, axis=0)
    return res + bicriteria(P, k)

#p = [(0,2,2),(1,0,0),(5,0,1),(3,2,6),(2,0,0),(7,0,1),(10,4,7),(15,10,1),(4,9,2),(11,3,8),(12,6,5),(6,2,1),(8,3,4),(9,2,6),(13,7,8),(14,3,2),(16,9,9),(17,6,1),(18,3,3),
#     (19,15,3),(20,7,8),(21,16,7),(22,30,18),(23,8,9),(24,1,1),(25,2,2),(26,3,3),(27,4,5),(28,8,10),(29,8,10),(30,8,10),(31,9,11),(32,15,7),(33,10,6),(34,5,5),(35,3,3),
#     (36,12,12),(37,8,5),(38,12,1),(39,10,2)]
#print bicriteria(p,3)
