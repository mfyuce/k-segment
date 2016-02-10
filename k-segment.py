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





import numpy as np
import scipy.optimize as optimize
#from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import scikits.statsmodels.api as sm

x = np.matrix([
    [2.4,30.6    ],
    [2.7,75      ],
    [1,6.4       ],
    [1.7,32.5    ],
    [-5.3,104.4  ],
    [5.5,40.8    ],
    [1.8,91      ],
    [8.5,56.8    ],
    [2.7,52.8    ],
    [1.8,15.4    ],
    [2.5,52.5    ],
    [4.9,35.6    ],
    [6.6,52.7    ],
    [0.5,112.6   ],
    [1.9,38.3    ],
    [1.1,50.6    ],
    [2,47.9      ],
    [8.3,7.1     ],
    [2.8,47.7    ],
    [2,93.7      ],
    [7,32.5      ],
    [3,55.6      ],
    [-7.1,106.5  ],
    [4.1,24.5    ],
    [1.6,81.6    ],
    [2.9,118.9   ],
    [6.3,48.5    ],
    [6.5,26.2    ],
    [1.4,104.3   ],
    [0.4,110.9   ],
    [1.3,120.4   ],
    [-0.6,174.8  ],
    [2.6,61.9    ],
    [7.5,9.9     ],
    [5.5,42.1    ],
    [5.9,43.5    ],
    [1.7,16.8    ],
    [5.1,51.8    ],
    [7,66.6      ],
    [1.8,84      ],
    [3.8,36.4    ],
    [6.4,23.7    ],
    [17.5,46.9   ],
    [5,56.8      ],
    [3.9,33.9    ],
    [1,66        ],
    [1.1,63.9    ],
    [7.4,3       ],
    [1.2,20.2    ],
    [5.5,5.1     ],
    [6.9,19.5    ],
    [-1.6,92.5   ],
    [4.3,9.3     ],
    [5,75.3      ],
    [5.2,110.2   ],
    [3.2,45.5    ],
    [0.4,55.2    ],
    [3.7,38.3    ],
    [1.9,27.7    ],
    [0.1,30.2    ],
    [-2.6,21.9   ],
    [-2,44       ],
    [8.8,45.9    ],
    [6.6,42.7    ],
    [5.2,27.4    ],
    [1,101.2     ],
    [1.8,81.8    ],
    [6.5,46.8    ]
    ])

#2.4,30.6        ,5.3
#2.7,75          ,6.7
#1,6.4           ,1.1
#1.7,32.5        ,14.6
#-5.3,104.4      ,20.3
#5.5,40.8        ,3
#1.8,91          ,7.6
#8.5,56.8        ,8.4
#2.7,52.8        ,24.1
#1.8,15.4        ,2.2
#2.5,52.5        ,9.5
#4.9,35.6        ,0.2
#6.6,52.7        ,15.2
#0.5,112.6       ,5.7
#1.9,38.3        ,3.8
#1.1,50.6        ,4.2
#2,47.9          ,10.4
#8.3,7.1         ,0.2
#2.8,47.7        ,3.4
#2,93.7          ,5.8
#7,32.5          ,4.4
#3,55.6          ,5.3
#-7.1,106.5      ,17.3
#4.1,24.5        ,12.5
#1.6,81.6        ,8.3
#2.9,118.9       ,15.4
#6.3,48.5        ,24.7
#6.5,26.2        ,7.7
#1.4,104.3       ,10.1
#0.4,110.9       ,12.2
#1.3,120.4       ,31.5
#-0.6,174.8      ,15.7
#2.6,61.9        ,7.9
#7.5,9.9         ,1.8
#5.5,42.1        ,4.5
#5.9,43.5        ,6.4
#1.7,16.8        ,1.1
#5.1,51.8        ,9.6
#7,66.6          ,10.4
#1.8,84          ,7.9
#3.8,36.4        ,8.5
#6.4,23.7        ,2.5
#17.5,46.9       ,1
#5,56.8          ,3.1
#3.9,33.9        ,5
#1,66            ,4.1
#1.1,63.9        ,3.8
#7.4,3           ,6.6
#1.2,20.2        ,1.3
#5.5,5.1         ,0.4
#6.9,19.5        ,5.7
#-1.6,92.5       ,9.7
#4.3,9.3         ,1.5
#5,75.3          ,7.7
#5.2,110.2       ,0
#3.2,45.5        ,5.2
#0.4,55.2        ,7.5
#3.7,38.3        ,2.9
#1.9,27.7        ,3.5
#0.1,30.2        ,5.9
#-2.6,21.9       ,8.4
#-2,44           ,5.8
#8.8,45.9        ,15.6
#6.6,42.7        ,5.8
#5.2,27.4        ,5.2
#1,101.2         ,8.7
#1.8,81.8        ,12.8
#6.5,46.8        ,8.1

y = np.matrix ([
    5.3   ,
    6.7   ,
    1.1   ,
    14.6  ,
    20.3  ,
    3     ,
    7.6   ,
    8.4   ,
    24.1  ,
    2.2   ,
    9.5   ,
    0.2   ,
    15.2  ,
    5.7   ,
    3.8   ,
    4.2   ,
    10.4  ,
    0.2   ,
    3.4   ,
    5.8   ,
    4.4   ,
    5.3   ,
    17.3  ,
    12.5  ,
    8.3   ,
    15.4  ,
    24.7  ,
    7.7   ,
    10.1  ,
    12.2  ,
    31.5  ,
    15.7  ,
    7.9   ,
    1.8   ,
    4.5   ,
    6.4   ,
    1.1   ,
    9.6   ,
    10.4  ,
    7.9   ,
    8.5   ,
    2.5   ,
    1     ,
    3.1   ,
    5     ,
    4.1   ,
    3.8   ,
    6.6   ,
    1.3   ,
    0.4   ,
    5.7   ,
    9.7   ,
    1.5   ,
    7.7   ,
    0     ,
    5.2   ,
    7.5   ,
    2.9   ,
    3.5   ,
    5.9   ,
    8.4   ,
    5.8   ,
    15.6  ,
    5.8   ,
    5.2   ,
    8.7   ,
    12.8  ,
    8.1   
    ])
print x
print y.transpose()
print np.linalg.pinv(x.transpose()*x)*x.transpose()*y.transpose()
