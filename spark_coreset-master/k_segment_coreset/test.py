import numpy as np
import utils
import ksegment
import Coreset
import unittest
import cProfile
import re


class KSegmentTest(unittest.TestCase):
    cProfile.run('re.compile("test_coreset_merging")')
    def test_basic_demo(self):
        # generate points
        N = 600
        dimension = 2
        k = 3
        epsilon = 10

        # data = random_data(N, dimension)
        # for example1 choose N that divides by 6
        # data = example1(N)
        data = np.genfromtxt("input.csv", delimiter=" ")
        P = np.c_[np.mgrid[1:N + 1], data]

        coreset = Coreset.build_coreset(P, k, epsilon)
        print len(coreset)
        print coreset
        dividers = ksegment.coreset_k_segment(coreset, k)
        print dividers
        utils.visualize_3d(P, dividers)

    def test_coreset_merging(self):
        # generate points
        N = 600
        dimension = 2
        k = 3
        epsilon = 0.1

        # data = random_data(N, dimension)
        # for example1 choose N that divides by 6
        data = example1(N)

        P = np.c_[np.mgrid[1:N + 1], data]

        coreset = Coreset.build_coreset(P, k, epsilon)
        coreset_of_coreset = Coreset.build_coreset(coreset, k, epsilon, is_coreset=True)
        dividers = ksegment.coreset_k_segment(coreset_of_coreset, k)
        utils.visualize_3d(P, dividers)

    def test_bicritiria(self):
        # generate points
        N = 180
        dimension = 2
        k = 3
        epsilon = 0.1

        # for example1 choose N that divides by 6
        data = example1(N)

        P = np.c_[np.mgrid[1:N + 1], data]

        #real_cost = utils.best_fit_line_cost(P)
        bicritiria_cost = Coreset.bicriteria(P, 3)
        real_cost = utils.calc_cost_dividers(P, ksegment.k_segment(P, 3))
        self.assertGreaterEqual(real_cost, bicritiria_cost)

    def test_best_fit_line_multiple_coresets(self):
        # generate points
        N = 1200
        # for example1 choose N that divides by 6
        data = example1(N)

        P = np.c_[np.mgrid[1:N + 1], data]
        P1 = P[:1000]
        P2 = P[1000:]

        C = Coreset.OneSegmentCorset(P)
        C1 = Coreset.OneSegmentCorset(P1)
        C2 = Coreset.OneSegmentCorset(P2)

        best_fit_line_P = utils.calc_best_fit_line(P)
        best_fit_line_C = utils.calc_best_fit_line(C.repPoints)
        best_fit_line_P1 = utils.calc_best_fit_line(P1)
        best_fit_line_C1 = utils.calc_best_fit_line(C1.repPoints)

        original_cost_not_best_fit_line = utils.sqrd_dist_sum(P, best_fit_line_P)
        single_coreset_cost = utils.sqrd_dist_sum(C.repPoints, best_fit_line_P) * C.weight
        C1_cost = int(utils.sqrd_dist_sum(C1.repPoints, best_fit_line_P) * C1.weight)
        P1_cost = int(utils.sqrd_dist_sum(P1, utils.calc_best_fit_line(P1)))
        C2_cost = int(utils.sqrd_dist_sum(C2.repPoints, best_fit_line_P) * C2.weight)
        dual_coreset_cost = C1_cost + C2_cost

        self.assertEqual(int(original_cost_not_best_fit_line), int(single_coreset_cost))
        self.assertEqual(C1_cost, P1_cost)
        self.assertEqual(int(original_cost_not_best_fit_line), int(dual_coreset_cost))

        res2 = utils.calc_best_fit_line_coreset(C1, C2)

        self.assertEqual(best_fit_line_P, res2)

    def test_OneSegmentCoreset_Cost(self):
        # generate points
        N = 1200
        # for example1 choose N that divides by 6
        data = example1(N)

        P = np.c_[np.mgrid[1:N + 1], data]
        P1 = P[:1000]
        C1 = Coreset.OneSegmentCorset(P1)

        best_fit_line_P = utils.calc_best_fit_line(P)
        best_fit_line_P1 = utils.calc_best_fit_line(P1)
        best_fit_line_C1 = utils.calc_best_fit_line(C1.repPoints)

        self.assertEqual(best_fit_line_P1.all(), best_fit_line_C1.all())

        original_cost_not_best_fit_line = utils.sqrd_dist_sum(P1, best_fit_line_P)
        original_cost_best_fit_line = utils.sqrd_dist_sum(P1, best_fit_line_P1)
        single_coreset_cost_not_best_fit_line = utils.sqrd_dist_sum(C1.repPoints, best_fit_line_P) * C1.weight
        single_coreset_cost_best_fit_line = utils.sqrd_dist_sum(C1.repPoints, best_fit_line_C1) * C1.weight

        self.assertEqual(int(original_cost_best_fit_line), int(single_coreset_cost_best_fit_line))
        self.assertEqual(int(original_cost_not_best_fit_line), int(single_coreset_cost_not_best_fit_line))

    def test_OneSegmentCoreset_bestFitLineIdentical_diferrentWeights(self):
        # generate points
        N = 1200

        # for example1 choose N that divides by 6
        data = example1(N)

        P = np.c_[np.mgrid[1:N + 1], data]
        P1 = P[:5]
        P2 = P[5:20]
        P3 = P[20:30]
        P4 = P[30:]

        C = Coreset.OneSegmentCorset(P)
        C1 = Coreset.OneSegmentCorset(P1)
        C2 = Coreset.OneSegmentCorset(P2)
        C3 = Coreset.OneSegmentCorset(P3)
        C4 = Coreset.OneSegmentCorset(P4)
        C1_C2 = [C1,C2]
        C3_C4 = [C3,C4]
        coreset_of_coresets1 = Coreset.OneSegmentCorset(C1_C2, True)
        coreset_of_coresets2 = Coreset.OneSegmentCorset(C3_C4, True)
        coreset_of_coresetrs = [coreset_of_coresets1, coreset_of_coresets2]
        coreset_of_coresets3 = Coreset.OneSegmentCorset(coreset_of_coresetrs, True)

        original_points_best_fit_line = utils.calc_best_fit_line(P)
        single_coreset_best_fit_line = utils.calc_best_fit_line(C.repPoints)
        coreset_of_coresetes_best_fit_line = utils.calc_best_fit_line(coreset_of_coresets3.repPoints)
        np.testing.assert_allclose(original_points_best_fit_line, coreset_of_coresetes_best_fit_line)
        np.testing.assert_allclose(coreset_of_coresetes_best_fit_line, single_coreset_best_fit_line)

    def test_generate_input_file(self):
        def make_input_file(N):
            data = example1(N)
            # P = np.c_[np.mgrid[1:N + 1], data]    # add time to points
            np.savetxt('input.csv', data, '%.5f', delimiter=' ')
        make_input_file(600)

    def test_calc_best_fit_line_weighted(self):
        data = np.array([[1,1],
                [2,3],
                [3,4],
                [4,4]])
        w = [1,0,0,1]
        best_fit_Line = utils.calc_best_fit_line_polyfit(data,w)
        print best_fit_Line

    def test_calc_sqr_dist_weighted(self):
        data = np.array([[1, 1],
                         [2, 3],
                         [3, 4],
                         [4, 4]])
        w = [1, 0, 0, 1]
        best_fit_line_cost_weighted = utils.best_fit_line_cost_weighted(data,w)
        print best_fit_line_cost_weighted

def random_data(N, dimension):
    return np.random.random_integers(0, 100, (N, dimension))


# 3 straight lines with noise
# choose N that divides by 6
def example1(n):
    x1 = np.mgrid[1:9:2 * n / 6j]
    y1 = np.mgrid[-5:3:2 * n / 6j]
    x2 = np.mgrid[23:90:n / 2j]
    y2 = np.mgrid[43:0:n / 2j]
    x3 = np.mgrid[80:60:n / 6j]
    y3 = np.mgrid[90:100:n / 6j]

    x = np.r_[x1, x2, x3]
    y = np.r_[y1, y2, y3]
    x += np.random.normal(size=x.shape) * 10
    y += np.random.normal(size=y.shape) * 10
    return np.c_[x, y]

# random
def example2():
    x1 = np.mgrid[1:9:100j]
    y1 = np.mgrid[-5:3:100j]
    x1 += np.random.normal(size=x1.shape) * 4
    return np.c_[x1, y1]

