import numpy as np
import mpl_toolkits.mplot3d as m3d
import matplotlib.pyplot as plt
import BiCritetria

'''
calc_best_fit_line -
    input - set of points
    return- matrix [dX2] such as that for each column C[i] - 
        C[i,0] is the slope
        C[i,1] is the intercept with the i-th dimensional axis
'''
def calc_best_fit_line (P):
    time_array = P[:,0]
    A = np.vstack([time_array, np.ones(len(time_array))]).T
    data = P[:,1:]
    return np.linalg.lstsq(A, data)[0]

def sqrd_dist_sum(P, line):
    time_array = P[:,0]
    A = np.vstack([time_array, np.ones(len(time_array))]).T
    data = P[:,1:]
    projected_points = np.dot(A, line)
    return sum(np.linalg.norm(data - projected_points, axis=1)**2)

def pt_on_line(x, line):
        coordinates = [x]
        for i in xrange(len(line[0])):
            coordinates.append(line[0,i]*x + line[1,i])
        return coordinates

def lines_from_dividers(P, dividers):
    lines = []
    for i in xrange(len(dividers)-1):
        # dividers[i]-1 because signal index starts from 1 not 0
        segment = P[dividers[i]-1:dividers[i+1],:]
        lines.append(calc_best_fit_line(segment))
    return lines

def visualize_3d (P, dividers):
    first_index = P[0,0]
    last_index = P[P.shape[0]-1, 0]
    line_pts_list = []
    all_sgmnt_sqrd_dist_sum = 0
    for i in xrange(len(dividers)-1):
        # dividers[i]-1 because signal index starts from 1 not 0
        segment = P[dividers[i]-1:dividers[i+1],:]
        best_fit_line = calc_best_fit_line(segment)
        line_pts_list.append(pt_on_line(dividers[i], best_fit_line))
        all_sgmnt_sqrd_dist_sum += sqrd_dist_sum(segment, best_fit_line)
    line_pts_list.append(pt_on_line(dividers[i+1], best_fit_line))
    lint_pts_arr = np.asarray(line_pts_list)
    
    print "real squared distance sum: ", all_sgmnt_sqrd_dist_sum

    ax = m3d.Axes3D(plt.figure())
    ax.scatter3D(*P.T)
    ax.plot3D(*lint_pts_arr.T)

    ax.set_xlabel('time axis')

    plt.show()

#P = np.array([[0,0,0],[1,1,2],[2,2,4],[3,3,6],[4,4,8],[5,5,10],[6,6,12],[7,7,20]])
#best_fit_line = calc_best_fit_line(P)
#visualize_3d (P, best_fit_line)