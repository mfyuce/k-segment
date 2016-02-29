import numpy as np
import utils

'''
nodes_info brief:
the array info holds for each point an array of k size meaning the lowest cost
it takes to reach the current point in i (0<i<=k) steps, for each cost there's
a place to hold the point from which it came.
how to access array:
    - first index : point number
    - second index: [0] - the value of the current minimum path in i steps.
                    [1] - the number of the point from which this value originated.
    - third index : i
'''
class nodes_info:
    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.info = np.zeros((n, 2, k))
        for i in xrange(n):
            self.info[i][0] = float("inf")
    
def update_node_info(nodes, prep, cur_n, next_n):
    # update the path from the first node to next_n (path len 1)
    nodes.info[next_n, 0, 0] = prep[0 ,next_n]
    nodes.info[next_n ,1, 0] = 0
    # update the path from cur_n to next_n
    for i, cur_cost in np.ndenumerate(nodes.info[next_n, 0, 1:]):
        index = i[0]+1
        # if such path doesn't exist
        if (float("inf") == nodes.info[cur_n, 0, index-1] or 
            float("inf") == prep[cur_n, next_n]):
            continue
        updt_cost = nodes.info[cur_n, 0, index-1] + prep[cur_n, next_n]
        if (updt_cost < cur_cost):
            nodes.info[next_n, 0, index] = updt_cost
            nodes.info[next_n, 1, index] = cur_n

def calc_partitions(prep_dist, n, k):
    nodes = nodes_info(n, k)
    for i in xrange(n):
        for j in xrange(i+1, n):
            update_node_info(nodes, prep_dist, i, j)
	print nodes
    return nodes

def get_x_val_dividers(p, k, nodes):
    next = len(nodes.info) - 1
    result = np.array(p[next][0])
    for i in reversed(xrange(0,k)):
        next = int(nodes.info[next, 1, i])
        x_value = p[next][0]
        result = np.insert(result, 0, x_value)
    return result

def get_x_val_dividers_coreset(D, k, nodes):
    next = len(nodes.info) - 1
    result = np.array(D[next].e)
    for i in reversed(xrange(0,k)):
        next = int(nodes.info[next, 1, i])
        x_value = D[next].b
        result = np.insert(result, 0, x_value)
    return result

def calc_prep_dist(P):
    prep_dist = np.full((len(P), len(P)),float("inf"))
    for index,value in np.ndenumerate(prep_dist):
        if (index[0]<index[1]):
            segment = P[index[0]:index[1]+1,:]
            best_fit_line = utils.calc_best_fit_line(segment)
            prep_dist[index] = utils.sqrd_dist_sum(segment, best_fit_line)
    return prep_dist

def k_segment(P, k):
    prep_dist = calc_prep_dist(P)
    #print "distances for each block:\n%s\n" % prep_dist
    result = calc_partitions(prep_dist, len(P), k)
    #print "dynamic programming (belman) result:\n%s\n" % result.info
    dividers = get_x_val_dividers(P, k,result)
    #print "the x values that divivde the pointset to k segments are:\n%s" % dividers
    return dividers

def calc_coreset_prep_dist(D):
    prep_dist = np.full((len(D), len(D)), float("inf"))
    for (first_coreset, second_coreset),value in np.ndenumerate(prep_dist):
        # we only want to calculate for segments that start in
        # starting coreset endpoints and end in ending coreset endpoints
        if first_coreset <= second_coreset:
            segment = np.array([])
            C = []
            W = []
            for coreset in D[first_coreset:second_coreset+1]:
                segment = np.vstack([segment, coreset.C]) if segment.size else coreset.C
                C.append(coreset.C)
                W.append(coreset.W)
            best_fit_line = utils.calc_best_fit_line(segment)
            fitting_cost = 0
            for i in xrange(len(C)):
                fitting_cost += utils.sqrd_dist_sum(C[i], best_fit_line)*W[i]
            prep_dist[first_coreset, second_coreset] = fitting_cost
    return prep_dist

def coreset_k_segment(D, k):
    prep_dist = calc_coreset_prep_dist(D)
    #print "distances for each block:\n%s\n" % prep_dist
    result = calc_partitions(prep_dist, len(D), k)
    #print "dynamic programming (belman) result:\n%s\n" % result.info
    dividers = get_x_val_dividers_coreset(D, k, result)
    #print "the x values that divivde the pointset to k segments are:\n%s" % dividers
    return dividers
