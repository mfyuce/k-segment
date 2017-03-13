import numpy as np
import Coreset
import test
import ksegment

def best_fit_line_cost(P, is_coreset=False):
    best_fit_line = calc_best_fit_line_polyfit(P, is_coreset)
    return sqrd_dist_sum(P, best_fit_line, is_coreset)


def best_fit_line_cost_weighted(P, W, is_coreset=False):
    best_fit_line = calc_best_fit_line_polyfit(P, W, is_coreset)
    return sqrd_dist_sum_weighted(P,best_fit_line, W, is_coreset)


def calc_best_fit_line(P):
    """
    calc_best_fit_line -
        input - set of points
        return- matrix [dX2] such as that for each column C[i] -
            C[i,0] is the slope
            C[i,1] is the intercept with the i-th dimensional axis
    """
    try:
        n = len(P)
        time_array = P[:, 0]
        A = np.vstack([time_array, np.ones(n)]).T
        data = P[:, 1:]
        return np.linalg.lstsq(A, data)[0]
    except:
        print "error in calc_best_fit_line"


def calc_best_fit_line_polyfit(P, W=False, is_coreset=False):
    if type(W) == bool:
        W = [1] * len(P)
        if W:
            is_coreset = True
    try:
        n = len(P)
        time_array = P[:, 0]
        # A = np.vstack([time_array, np.ones(n)]).T
        data = P[:, 1:]
        return np.polyfit(time_array, data, 1, w=W)
    except:
        print "error in calc_best_fit_line"


def sqrd_dist_sum(P, line,is_coreset=False):
    try:
        time_array = P[:, 0]
        A = np.vstack([time_array, np.ones(len(time_array))]).T
        data = P[:, 1:]
        projected_points = np.dot(A, line)
        norm_vector = np.apply_along_axis(np.linalg.norm, axis=1, arr=data - projected_points)
        squared_norm_distances = np.square(norm_vector)
        return sum(squared_norm_distances)
    except:
        print "error in sqrd_dist_sum"


def sqrd_dist_sum_weighted(P, line, w, is_coreset=False):
    try:
        time_array = P[:, 0]
        A = np.vstack([time_array, np.ones(len(time_array))]).T
        data = P[:, 1:]
        projected_points = np.dot(A, line)
        norm_vector = np.apply_along_axis(np.linalg.norm, axis=1, arr=data - projected_points)
        squared_norm_distances = np.square(norm_vector)
        return sum(squared_norm_distances * (w ** 2))
    except:
        print "error in sqrd_dist_sum"


def pt_on_line(x, line):
    coordinates = [x]
    for i in xrange(len(line[0])):
        coordinates.append(line[0, i] * x + line[1, i])
    return coordinates


def calc_cost_dividers(P, dividers):
    cost = 0.0
    for i in xrange(len(dividers) - 1):
        segment = P[dividers[i] - 1: dividers[i + 1], :]
        cost += sqrd_dist_sum(segment, calc_best_fit_line_polyfit(segment))
    return cost


def lines_from_dividers(P, dividers):
    lines = []
    for i in xrange(len(dividers) - 1):
        segment = P[dividers[i] - 1:dividers[i + 1], :]
        lines.append(calc_best_fit_line_polyfit(segment))
    return lines


def visualize_3d(P, dividers):
    first_index = P[0, 0]
    last_index = P[P.shape[0] - 1, 0]
    line_pts_list = []
    all_sgmnt_sqrd_dist_sum = 0
    for i in xrange(len(dividers) - 1):
        line_start_arr_index = dividers[i] - 1
        line_end_arr_index = dividers[i + 1] - 1 if i != len(dividers) - 2 else dividers[i + 1]
        segment = P[line_start_arr_index:line_end_arr_index, :]
        best_fit_line = calc_best_fit_line_polyfit(segment)
        line_pts_list.append([pt_on_line(dividers[i], best_fit_line),
                              pt_on_line(dividers[i + 1] - (1 if i != len(dividers) - 2 else 0), best_fit_line)])
        all_sgmnt_sqrd_dist_sum += sqrd_dist_sum(segment, best_fit_line)
    # print "real squared distance sum: ", all_sgmnt_sqrd_dist_sum


def is_unitary(m):
    return np.allclose(np.eye(len(m)), m.dot(m.T.conj()))
