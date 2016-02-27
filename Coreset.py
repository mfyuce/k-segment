import numpy as np
import math
import utils

def coreset(P, k, eps):
    h = bicriteria(P, k) *10 # TODO temporary multiply bicriteria gives wrong est
    b = (eps**2 * h) / (100*k*np.log2(P.shape[0]))
    return BalancedPartition(P, eps, b)

def one_seg (P):
    return utils.best_fit_line_cost(P)

def bicriteria(P,k):
    if (len(P) <= (2 * k + 1)):
        return 0
    m = int(math.floor(len(P)/(2*k)))
    i = 0
    j = m
    # one_seg_res will hold segment starting index and result (squred distance sum)
    one_seg_res = []
    # partition to 2k segments and call 1-segment for each
    while (i < len(P)):
        one_seg_res.append((one_seg(P[i:j,:]), int(i)))
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
            if len(Q[:-1]) > 8:  # TODO temporary condition, beta isn't good enough currently
                T = Q[:-1]
                C = OneSegmentCorset(T)
                g = utils.calc_best_fit_line(np.asarray(T))
                b = i - len(T) + 1
                e = i
                D.append([C, g, b , e])
                Q = [Q[-1]]
    return D

def OneSegmentCorset(P):
    #add 1's to the first column
    X = np.insert(P, 0, values=1, axis=1)
    U, s, V = np.linalg.svd(X, full_matrices=False)
    #reshape S
    S = np.diag(s) 
    #calculate SV
    SVt = np.dot(S, V)
    u = SVt[:,0] # u is leftmost column of SVt
    w = (np.linalg.norm(u)**2)/X.shape[1]
    q = np.identity(X.shape[1]) #q - temporary matrix to build an identity matrix with leftmost column - u
    q[:,0] = u
    Q = np.linalg.qr(q)[0] #QR decomposition returns in Q what is requested
    #calculate Y
    y = np.identity(X.shape[1]) # y - temporary matrix to build an identity matrix with leftmost column
    y[:,0] = math.sqrt(w)/np.linalg.norm(u) #set y's first column to be sqrt of w divided by u's normal
    # set all the vectors to be orthogonal to the first vector
    for i in xrange(1, X.shape[1]):
        y[:,i] = y[:,i] - np.dot(y[:,0],np.linalg.norm(y[:,i],axis=0))*np.linalg.norm(y[:,i],axis=0)
    #compute Y with QR decompression - first column will not change - it is already normalized
    Y = np.linalg.qr(y)[0] 
    YQtSVt= np.dot(np.dot(Y,Q.T),SVt)
    YQtSVt= YQtSVt/math.sqrt(w)
    #set B to the d+1 rightmost columns
    B = YQtSVt[:,1:]
    return [B, w]

def PiecewiseCoreset(n, s, eps):
    s_arr = [s(i, n) for i in xrange(1, n+1)]
    t = sum(s_arr)
    B = []
    b_list = []
    W = np.zeros(n)
    for i in xrange(1, n+1):
        b = math.ceil(sum(s_arr[0:i])/(t*eps))
        if b not in b_list:
            B.append(i)
        b_list.append(b)
    for j in B:
        I = [i+1 for i, b in enumerate(b_list) if b == b_list[j-1]]
        print 1
        W[j-1]=(1./s_arr[j-1])*sum([s_arr[i-1] for i in I])
    return W
    