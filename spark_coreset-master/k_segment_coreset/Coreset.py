import numpy as np
import math
import utils


class OneSegCoreset():
    def __init__(self, repPoints, weight, SVt):
        self.repPoints = repPoints
        self.weight = weight
        self.SVt = SVt


class coreset:
    def __init__(self, C, g, b, e):
        self.C = C  # 1-segment coreset
        self.g = g  # best line
        self.b = b  # coreset beginning index
        self.e = e  # coreset ending index


def build_coreset(P, k, eps, is_coreset=False):
    h = bicriteria(P, k, is_coreset)
    b = (eps ** 2 * h) / (100 * k * np.log2(len(P)))
    return BalancedPartition(P, eps, b, is_coreset)


def one_seg_cost(P, is_coreset=False):
    if is_coreset:
        oneSegmentCoreset = OneSegmentCorset(P, is_coreset)
        return utils.best_fit_line_cost(oneSegmentCoreset.repPoints, is_coreset) * oneSegmentCoreset.weight
    else:
        return utils.best_fit_line_cost(P, is_coreset)


def bicriteria(P, k, is_coreset=False):
    if len(P) <= (4 * k + 1):
        return 0
    m = int(math.floor(len(P) / (4 * k)))
    i = 0
    j = m
    # one_seg_res will hold segment starting index and result (squred distance sum)
    one_seg_res = []
    # partition to 4k segments and call 1-segment for each
    while i < len(P):
        partition_set = one_seg_cost(P[i:j], is_coreset)
        one_seg_res.append((partition_set, int(i)))
        i += m
        j += m
    # sort result
    one_seg_res = sorted(one_seg_res, key=lambda res: res[0])
    # res = the distances of the min k+1 segments
    res = 0
    # sum distances of k+1 min segments and make a list of point to delete from P to get P \ Q from the algo'
    rows_to_delete = []
    for i in xrange(k + 1):
        res += one_seg_res[i][0]
        for j in xrange(m):
            rows_to_delete.append(one_seg_res[i][1] + j)
    P = np.delete(P, rows_to_delete, axis=0)
    return res + bicriteria(P, k, is_coreset)


def BalancedPartition(P, a, bicritiriaEst, is_coreset=False):
    Q = []
    D = []
    points = P
    # add arbitrary item to list
    dimensions = points[0].C.repPoints.shape[1] if is_coreset else points.shape[1]
    if is_coreset:
        points.append(P[0])  # arbitrary coreset n+1
    else:
        points = np.vstack((points, np.zeros(dimensions)))  # arbitrary point n+1
    n = len(points)
    for i in xrange(0, n):
        Q.append(points[i])
        cost = one_seg_cost(np.asarray(Q), is_coreset)
        # if current number of points can be turned into a coreset - 3 conditions :
        # 1) cost passed threshold
        # 2) number of points to be packaged greater than dimensions + 1
        # 3) number of points left greater then dimensions + 1 (so they could be packaged later)
        if cost > bicritiriaEst and (is_coreset or (len(Q) > dimensions + 1 and dimensions + 1 <= n - 1 - i)) or i == n - 1:
            if is_coreset and len(Q) == 1:
                if i != n - 1:
                    D.append(Q[0])
                    Q = []
                continue
            T = Q[:-1]
            C = OneSegmentCorset(T, is_coreset)
            g = utils.calc_best_fit_line(OneSegmentCorset(np.asarray(T), is_coreset).repPoints)
            if is_coreset:
                b = T[0].b
                e = T[-1].e
            else:
                b = T[0][0]  # signal index of first item in T
                e = T[-1][0]  # signal index of last item in T
            D.append(coreset(C, g, b, e))
            Q = [Q[-1]]
    return D


def OneSegmentCorset(P, is_coreset=False):
    if len(P) < 2:
        return P[0].C
    if is_coreset:
        svt_to_stack = []
        for oneSegCoreset in P:
            svt_to_stack.append(oneSegCoreset.C.SVt)
        X = np.vstack(svt_to_stack)
    else:
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
    q[:, 0] = u / np.linalg.norm(u)
    Q = np.linalg.qr(q)[0]  # QR decomposition returns in Q what is requested
    if np.allclose(Q[:, 0], -q[:, 0]):
        Q = -Q
    # assert matrix is as expected
    assert ((np.allclose(Q[:, 0], q[:, 0])))
    # calculate Y
    y = np.identity(X.shape[1])  # y - temporary matrix to build an identity matrix with leftmost column
    yLeftCol = math.sqrt(w) / np.linalg.norm(u)
    y[:, 0] = yLeftCol  # set y's first column to be sqrt of w divided by u's normal
    # compute Y with QR decompression - first column will not change - it is already normalized
    Y = np.linalg.qr(y)[0]
    if np.allclose(Y[:, 0], -y[:, 0]):
        Y = -Y
    # assert matrix is as expected
    assert ((np.allclose(Y[:, 0], y[:, 0])))
    YQtSVt = np.dot(np.dot(Y, Q.T), SVt)
    YQtSVt /= math.sqrt(w)
    # set B to the d+1 rightmost columns
    B = YQtSVt[:, 1:]
    # return [B, w, SVt]
    return OneSegCoreset(repPoints=B, weight=w, SVt=SVt)


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
