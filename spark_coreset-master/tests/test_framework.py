__author__ = 'Anton'

import sys
import numpy as np
import weighted_kmeans as w_KMeans
import utils
from stream import Stream
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from coreset import Coreset
class testCoreset():
    def __init__(self,points):
        self.p = points
        self.w = np.ones(points.shape[0])

    def _real(self, k):
        model = KMeans(n_clusters=k)
        alg = model.fit(self.p)
        means = alg.cluster_centers_
        cost = alg.inertia_
        return cost, means

    def _compute_cost(self, p, means):
        return np.sum(utils.get_dist_to_centers(p, means))

    def _test(self,real_cost, k, sizes, trials, z):
        x = []
        y_cor = []
        y_uni = []
        for size in sizes:
            uni_cost = []
            cor_cost = []
            print "size:", size, "trials: ",
            x.append(size)
            for t in range(0, trials):
                #uni sampling:
                s = np.random.choice(range(0, self.p.shape[0]),size)
                s = self.p[s]
                centers = KMeans(n_clusters=k).fit(s).cluster_centers_
                res = self._compute_cost(self.p, centers)
                uni_cost.append(1-real_cost/res)

                #non uni sampling:
                p_cset, w_cset = Coreset(self.p, k, self.w).compute(size)
                best_cost = float("inf")
                for zz in range(0, z):
                    e = w_KMeans.KMeans(p_cset, np.expand_dims(w_cset, axis=0), k, 300).compute()
                    res = self._compute_cost(self.p, e)
                    if res < best_cost:
                        best_cost = res
                res = best_cost
                cor_cost.append(1-real_cost/res)

                sys.stdout.write(".")
                sys.stdout.flush()

            c_mistake = np.average(cor_cost)
            u_mistake = np.average(uni_cost)
            y_uni.append(u_mistake)
            y_cor.append(c_mistake)
            print "  mistake for uniform:", round(u_mistake, 10), "coreset:", round(c_mistake, 10)
        return x, y_uni, y_cor

    def _test_tree(self,real_cost, k, sizes, trials, z, chunks):
        x = []
        y_cor = []
        y_uni = []
        weights_avg = []
        for size in sizes:
            uni_cost = []
            cor_cost = []
            print "size:", size, "trials: ",
            x.append(size)
            for t in range(0, trials):
                #uni sampling:
                s = np.random.choice(range(0, self.p.shape[0]),size)
                s = self.p[s]
                centers = KMeans(n_clusters=k).fit(s).cluster_centers_
                res = self._compute_cost(self.p, centers)
                uni_cost.append(1-real_cost/res)

                #non uni sampling with tree:
                stream = Stream(Coreset,chunks,size,k)
                stream.add_points(self.p)
                p_cset, w_cset = stream.get_unified_coreset()
                weights_avg.append(np.sum(w_cset))
                best_cost = float("inf")
                for zz in range(0, z):
                    e = w_KMeans.KMeans(p_cset, np.expand_dims(w_cset, axis=0), k, 300).compute()
                    res = self._compute_cost(self.p, e)
                    if res < best_cost:
                        best_cost = res
                res = best_cost
                cor_cost.append(1-real_cost/res)

                sys.stdout.write(".")
                sys.stdout.flush()

            c_mistake = np.average(cor_cost)
            u_mistake = np.average(uni_cost)
            y_uni.append(u_mistake)
            y_cor.append(c_mistake)

            print "  mistake for uniform:", round(u_mistake, 10), "coreset:", round(c_mistake, 10)
        print "weight average mistake for all(!) of the trails(should be clost to 0): ", 1-np.sum(self.w)/np.average(weights_avg)
        return x, y_uni, y_cor

    def run_test(self,k,test_range, num_trials, cset_kmeans_trials=5, tree=False, num_chunks = 4):
        cost, means = self._real(k)
        if not tree:
            x, y_uni, y_cor = self._test(cost, k, test_range, num_trials, cset_kmeans_trials)
        else:
            x, y_uni, y_cor = self._test_tree(cost, k, test_range, num_trials, cset_kmeans_trials, num_chunks)

        plt.plot(self.p[:,0], self.p[:,1],'go')
        plt.plot(means[:,0], means[:,1],'ro')
        plt.show()

        print "red is coreset, blue is uniform"
        plt.plot(x, y_cor, 'r')
        plt.plot(x, y_uni, 'b')
        plt.show()




