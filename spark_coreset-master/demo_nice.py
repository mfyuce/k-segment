__author__ = 'Anton & Ahmad'

# Std imports
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import KMeans
#from scipy.spatial import distance

# our own imports
from coreset import Coreset
import weighted_kmeans as w_KMeans
import utils

def create_test():
    x = np.random.randint(0,1000,30)
    y = np.random.randint(0,1000,30)
    p = np.hstack((y,x)).reshape(30,2)
    x = np.random.randint(50000,60000,10000)
    y = np.random.randint(50000,60000,10000)
    p1 = np.hstack((y,x)).reshape(10000,2)
    p = np.vstack((p,p1))
    return p, np.ones(10030)

p, w = create_test()
p = np.array(p, dtype='float64')
plt.plot(p[:,0], p[:,1],'go')
model = KMeans(n_clusters=2)
alg = model.fit(p)
aaa = w_KMeans.KMeans(p, np.expand_dims(w, axis=0), 2)
print "weighted_kmeans centers ravel(): ", aaa.compute().ravel()
means = alg.cluster_centers_
print "sklearn K-means centers ravel(): ", means.ravel()
cost = alg.inertia_
print "real cost (sklearn): ", cost
plt.plot(means[:,0], means[:,1],'ro')
plt.show()

t = 50
delta = 100
print "regressing sample size in [50, 2000] w/ jumps of", delta, "each w/", t, "trials..."
x = []
y = []
y_uni = []
for size in range(50, 2000, delta):
    c_mistake = 0
    u_mistake = 0
    x.append(size)
    print "size:", size, "trials",
    for i in range(0,t):
        s = np.random.choice(range(0,10030),size)
        s = p[s]
        centers = model.fit(s).cluster_centers_
        uni_cost = (np.sum(utils.get_dist_to_centers(p, centers)))
        u_mistake += (1 - cost/uni_cost)
        p_cset, w_cset = Coreset(p, 2, w).compute(size)
        e = w_KMeans.KMeans(p_cset, np.expand_dims(w_cset, axis=0), 2, 10)
        e = e.compute()
        res = (np.sum(utils.get_dist_to_centers(p, e)))
        c_mistake += (1-cost/res)
        sys.stdout.write(".")
        sys.stdout.flush()
    c_mistake /= t
    y.append(c_mistake)
    u_mistake /= t
    y_uni.append(u_mistake)
    print "mistakes for uniform:", round(u_mistake, 3), "coreset:", round(c_mistake, 3)
    u_mistake = c_mistake = 0
plt.plot(x, y, 'r')
plt.plot(x, y_uni, 'b')
plt.show()

