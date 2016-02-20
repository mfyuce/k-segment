import numpy as np
import mpl_toolkits.mplot3d as m3d
import matplotlib.pyplot as plt

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

def visualize_3d (P, best_fit_line):
    first_index = P[0,0]
    last_index = P[P.shape[0]-1, 0]
    line_pts = np.array([[first_index, best_fit_line[0,0]*first_index + best_fit_line[1,0], best_fit_line[0,1]*first_index + best_fit_line[1,1]]
                        ,[last_index, best_fit_line[0,0]*last_index + best_fit_line[1,0], best_fit_line[0,1]*last_index + best_fit_line[1,1]]])
    print best_fit_line[0,0],best_fit_line[0,1]
    print best_fit_line[1,0],best_fit_line[1,1]
    ax = m3d.Axes3D(plt.figure())
    ax.scatter3D(*P.T)
    ax.plot3D(*line_pts.T)
    plt.show()

P = np.array([[0,0,0],[1,1,2],[2,2,4],[3,3,6],[4,4,8],[5,5,10],[6,6,12],[7,7,20]])
best_fit_line = calc_best_fit_line(P)
visualize_3d (P, best_fit_line)
