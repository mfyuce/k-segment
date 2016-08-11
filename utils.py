import numpy as np
import scipy as sc
import mpl_toolkits.mplot3d as m3d
import matplotlib.pyplot as plt

'''
calc_best_fit_line -
    input - set of points
    return- matrix [dX2] such as that for each column C[i] -
        C[i,0] is the slope
        C[i,1] is the intercept with the i-th dimensional axis
'''
def calc_best_fit_line(P):
    try:
        n = len(P)
        time_array = P[:, 0]
        A = np.vstack([time_array, np.ones(n)]).T
        data = P[:, 1:]
        return np.linalg.lstsq(A, data)[0]
    except:
        print "error in calc_best_fit_line"


def calc_best_fit_line_polyfit(P):
    try:
        n = len(P)
        time_array = P[:, 0]
        # A = np.vstack([time_array, np.ones(n)]).T
        data = P[:, 1:]
        return np.polyfit(time_array, data, 1)
    except:
        print "error in calc_best_fit_line"


def calc_best_fit_line_coreset(C1, C2):
    try:
        n1 = len(C1[0])
        n2 = len(C2[0])
        time_array1 = C1[0][:, 0]
        time_array2 = C2[0][:, 0]
        wieghts_vector = np.concatenate((np.full(n1, C1[1]), np.full(n2, C2[1])))
        A = np.concatenate((time_array1, time_array2))
        data = np.vstack([C1[0][:, 1:], C2[0][:, 1:]])
        return np.polyfit(A, data, 1, w=wieghts_vector)
    except:
        print "error in calc_best_fit_line"


def calc_best_fit_line_coreset_optimize(C1, C2):
    try:
        n1 = len(C1[0])
        n2 = len(C2[0])
        time_weight_array1 = np.vstack([C1[0][:, 0], np.ones(n1)])
        time_weight_array2 = np.vstack([C2[0][:, 0], np.ones(n2)])
        A = np.hstack([time_weight_array1, time_weight_array2]).T
        data = np.vstack([C1[0][:, 1:], C2[0][:, 1:]])
        return np.linalg.lstsq(A, data)[0]
    except:
        print "error in calc_best_fit_line"


def sqrd_dist_sum(P, line):
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


def pt_on_line(x, line):
    coordinates = [x]
    for i in xrange(len(line[0])):
        coordinates.append(line[0, i] * x + line[1, i])
    return coordinates


def lines_from_dividers(P, dividers):
    lines = []
    for i in xrange(len(dividers) - 1):
        segment = P[dividers[i] - 1:dividers[i + 1], :]
        lines.append(calc_best_fit_line(segment))
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
        best_fit_line = calc_best_fit_line(segment)
        line_pts_list.append([pt_on_line(dividers[i], best_fit_line),
                              pt_on_line(dividers[i + 1] - (1 if i != len(dividers) - 2 else 0), best_fit_line)])
        all_sgmnt_sqrd_dist_sum += sqrd_dist_sum(segment, best_fit_line)
    # print "real squared distance sum: ", all_sgmnt_sqrd_dist_sum

    ax = m3d.Axes3D(plt.figure())
    ax.scatter3D(*P.T)
    for line in line_pts_list:
        lint_pts_arr = np.asarray(line)
        ax.plot3D(*lint_pts_arr.T)

    ax.set_xlabel('time axis')
    ax.set_ylabel('x1 axis')
    ax.set_zlabel('x2 axis')

    plt.show()


def best_fit_line_cost(P, is_coreset=False):
    best_fit_line = calc_best_fit_line(P)
    return sqrd_dist_sum(P, best_fit_line)


def is_unitary(M):
    return np.allclose(np.eye(len(M)), M.dot(M.T.conj()))