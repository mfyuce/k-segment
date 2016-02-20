import numpy as np

'''
nodes_info brief:
the array info holds for each point an array of k size meaning the lowest cost
it takes to reach the current point in i (0<i<=k) steps, for each cost theres
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
    return nodes

def get_x_val_dividers(p, nodes):
    k = len(nodes.info[0][0])
    next = len(nodes.info) - 1
    result = np.array(p[next][0])
    for i in reversed(xrange(0,k)):
        next = int(nodes.info[next, 1, i])
        x_value = p[next][0]
        result = np.insert(result, 0, x_value)
    return result

def calc_prep_dist(sorted_p):
    #prep_dist = np.full((len(points), len(points)),float("inf"))
    #prep_dist = np.random.rand(len(points),len(points)) *50
    #prep_dist = prep_dist.astype(int)
    #for index,value in np.ndenumerate(prep_dist):
    #    if (index[0]>=index[1]):
    #        prep_dist[index] = 9999999
    #prep_dist = prep_dist.astype(int)
    #### ^ RANDOM ^ ############# v ARBITRARY EXAMPLE v ################
    prep_dist = np.array(
        [[float('inf'),1,4,16,29,31],
        [float('inf'),float('inf'),3,17,23,28],            
        [float('inf'),float('inf'),float('inf'),4,10,17],
        [float('inf'),float('inf'),float('inf'),float('inf'),7,20],
        [float('inf'),float('inf'),float('inf'),float('inf'),float('inf'),4],
        [float('inf'),float('inf'),float('inf'),float('inf'),float('inf'),float('inf')]])
    return prep_dist

def k_segment(p, k):
    points = np.array(p, dtype=np.float)
    print "input points:\n%s\n" % points

    sorted_pts = points[points[:,0].argsort()]
    print "sorted points:\n%s\n" % sorted_pts

    prep_dist = calc_prep_dist(sorted_pts)
    print "distances for each block(currently arbitrary or random):\n%s\n" % prep_dist

    result = calc_partitions(prep_dist, len(sorted_pts), k)
    print "dynamic programming (belman) result:\n%s\n" % result.info

    dividers = get_x_val_dividers(sorted_pts, result)
    print "the x values that divivde the pointset to k segments are:\n%s" % dividers

def main():
    p = [[4,2,6],[1,4,-2],[2,7,0],[15,-2,16],[17,4,13],[-2,-5,-109]]
    k_segment(p,3)

main()