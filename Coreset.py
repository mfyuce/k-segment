import numpy as np
import math
import utils


class coreset:
    def __init__(self, C, W, g, b, e):
        self.C = C  # 1-segment coreset
        self.W = W  # 1-segment coreset weight
        self.g = g  # best line
        self.b = b  # coreset beginning index
        self.e = e  # coreset ending index


def build_coreset(P, k, eps):
    h = bicriteria(P, k)
    b = (eps ** 2 * h) / (100 * k * np.log2(P.shape[0]))
    return BalancedPartition(P, eps, b)


def one_seg(P, is_coreset=False):
    return utils.best_fit_line_cost(P, is_coreset)


def bicriteria(P, k, is_coreset=False):
    if len(P) <= (2 * k + 1):
        return 0
    m = int(math.floor(len(P) / (2 * k)))
    i = 0
    j = m
    # one_seg_res will hold segment starting index and result (squred distance sum)
    one_seg_res = []
    # partition to 2k segments and call 1-segment for each
    while i < len(P):
        if is_coreset:
            partition_set = one_seg(P[i:j])
        else:
            partition_set = one_seg(P[i:j, :])
        one_seg_res.append((partition_set, int(i)))
        i += m
        j += m
    # sort result
    one_seg_res = sorted(one_seg_res, key=lambda res: res[0])
    # res = the distances of the min k+1 segments
    res = 0
    # sum distances of k+1 min segments and make a list of point to delete from P to get P \ Q from the algo'
    rows_to_delete = []
    for a in xrange(k + 1):
        res = res + one_seg_res[a][0]
        for b in xrange(m):
            rows_to_delete.append(one_seg_res[a][1] + b)
    P = np.delete(P, rows_to_delete, axis=0)
    return res + bicriteria(P, k)


def BalancedPartition(P, a, b):
    Q = []
    D = []
    arbitrary_p = np.zeros_like(P[0])
    arbitrary_p[0] = len(P) + 1
    points = np.vstack((P, arbitrary_p))
    n = points.shape[0]
    for i in xrange(1, n + 1):
        Q.append(points[i - 1])
        cost = utils.best_fit_line_cost(np.asarray(Q))
        if (cost > b or i == n) and len(Q) > 2:
            # if current number of points can be turned into a coreset
            T = Q[:-1]
            if len(Q[:-1]) > len(Q[0]):
                C = OneSegmentCorset(T)
            # if small number of points
            else:
                C = [np.array(Q[:-1]), 1]
            g = utils.calc_best_fit_line(np.asarray(T))
            b = i - len(Q[:-1])
            e = i - 1
            D.append(coreset(C[0], C[1], g, b, e))
            Q = [Q[-1]]
    return D


def OneSegmentCorset(P):
    # add 1's to the first column
    X = np.insert(P, 0, values=1, axis=1)
    U, s, V = np.linalg.svd(X, full_matrices=False)
    # reshape S
    S = np.diag(s)
    # calculate SV
    SVt = np.dot(S, V)
    u = SVt[:, 0]  # u is leftmost column of SVt
    w = (np.linalg.norm(u) ** 2) / X.shape[1]
    q = np.identity(X.shape[1])  # q - temporary matrix to build an identity matrix with leftmost column - u
    q[:, 0] = u
    Q = np.linalg.qr(q)[0]  # QR decomposition returns in Q what is requested
    # calculate Y
    y = np.identity(X.shape[1])  # y - temporary matrix to build an identity matrix with leftmost column
    y[:, 0] = math.sqrt(w) / np.linalg.norm(u)  # set y's first column to be sqrt of w divided by u's normal
    # set all the vectors to be orthogonal to the first vector
    for i in xrange(1, X.shape[1]):
        y[:, i] = y[:, i] - np.dot(y[:, 0], np.linalg.norm(y[:, i], axis=0)) * np.linalg.norm(y[:, i], axis=0)
    # compute Y with QR decompression - first column will not change - it is already normalized
    Y = np.linalg.qr(y)[0]
    YQtSVt = np.dot(np.dot(Y, Q.T), SVt)
    YQtSVt = YQtSVt / math.sqrt(w)
    # set B to the d+1 rightmost columns
    B = YQtSVt[:, 1:]
    return [B, w]


def PiecewiseCoreset(n, s, eps):
    s_arr = [s(i, n) for i in xrange(1, n + 1)]
    t = sum(s_arr)
    B = []
    b_list = []
    W = np.zeros(n)
    for i in xrange(1, n + 1):
        b = math.ceil(sum(s_arr[0:i]) / (t * eps))
        if b not in b_list:
            B.append(i)
        b_list.append(b)
    for j in B:
        I = [i + 1 for i, b in enumerate(b_list) if b == b_list[j - 1]]
        print 1
        W[j - 1] = (1. / s_arr[j - 1]) * sum([s_arr[i - 1] for i in I])
    return W
