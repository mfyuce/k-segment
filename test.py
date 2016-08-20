import numpy as np
import utils
import ksegment
import Coreset
import unittest


class ksegment_test(unittest.TestCase):
    def test_basic_demo(self):
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
        dividers = ksegment.coreset_k_segment(coreset, k)
        # dividers = ksegment.k_segment(P, k)
        # W = Coreset.PiecewiseCoreset(N, utils.s_func, 0.1)
        # bicriteria_est = Coreset.bicriteria(P,k)
        # print "BiCritetria estimated distance sum: ", bicriteria_est
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
        best_fit_line_C = utils.calc_best_fit_line(C[0])
        best_fit_line_P1 = utils.calc_best_fit_line(P1)
        best_fit_line_C1 = utils.calc_best_fit_line(C1[0])

        original_cost_not_best_fit_line = utils.sqrd_dist_sum(P, best_fit_line_P)
        single_coreset_cost = utils.sqrd_dist_sum(C[0], best_fit_line_P) * C[1]
        C1_cost = int(utils.sqrd_dist_sum(C1[0], best_fit_line_P) * C1[1])
        P1_cost = int(utils.sqrd_dist_sum(P1, utils.calc_best_fit_line(P1)))
        C2_cost = int(utils.sqrd_dist_sum(C2[0], best_fit_line_P) * C2[1])
        dual_coreset_cost = C1_cost + C2_cost

        self.assertEqual(int(original_cost_not_best_fit_line), int(single_coreset_cost))
        self.assertEqual(C1_cost, P1_cost)
        self.assertEqual(int(original_cost_not_best_fit_line), int(dual_coreset_cost))

        res2 = utils.calc_best_fit_line_coreset(C1, C2)

        self.assertEqual(best_fit_line_P, res2)

    def test_best_polyfit_compared_to_linalg(self):
        # generate points
        N = 1200
        k = 3

        # for example1 choose N that divides by 6
        data = example1(N)

        P = np.c_[np.mgrid[1:N + 1], data]

        res1 = utils.calc_best_fit_line(P)
        res2 = utils.calc_best_fit_line_polyfit(P)

        self.assertEqual(res1, res2)

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
        best_fit_line_C1 = utils.calc_best_fit_line(C1[0])

        self.assertEqual(best_fit_line_P1.all(), best_fit_line_C1.all())    # TODO doesn't work

        original_cost_not_best_fit_line = utils.sqrd_dist_sum(P1, best_fit_line_P)
        original_cost_best_fit_line = utils.sqrd_dist_sum(P1, best_fit_line_P1)
        single_coreset_cost_not_best_fit_line = utils.sqrd_dist_sum(C1[0], best_fit_line_P) * C1[1]
        single_coreset_cost_best_fit_line = utils.sqrd_dist_sum(C1[0], best_fit_line_C1) * C1[1]

        self.assertEqual(int(original_cost_best_fit_line), int(single_coreset_cost_best_fit_line))
        self.assertEqual(int(original_cost_not_best_fit_line), int(single_coreset_cost_not_best_fit_line))

    def test_OneSegmentCoreset_on_multiple_coresets(self):
        # generate points
        N = 1200
        dimension = 2
        k = 3
        epsilon = 0.1

        # for example1 choose N that divides by 6
        data = example1(N)

        P = np.c_[np.mgrid[1:N + 1], data]
        P1 = np.c_[np.mgrid[1:1000], data[0:999]]
        P2 = np.c_[np.mgrid[1001: N + 1], data[1000:]]

        res1 = utils.calc_best_fit_line(P)

        C1 = Coreset.OneSegmentCorset(P1)
        C2 = Coreset.OneSegmentCorset(P2)
        C3 = Coreset.OneSegmentCorset(P)

        coreset_of_coresets = Coreset.OneSegmentCorset_weights(C1, C2)

        res2 = utils.calc_best_fit_line(coreset_of_coresets[0])

        res3 = utils.calc_best_fit_line(C3[0])
        self.assertEqual(res2, res3)


def random_data(N, dimension):
    return np.random.random_integers(0, 100, (N, dimension))


def example1(n):
    # 3 straight lines with noise
    x1 = np.mgrid[1:9:2 * n / 6j]
    y1 = np.mgrid[-5:3:2 * n / 6j]
    x2 = np.mgrid[23:90:n / 2j]
    y2 = np.mgrid[43:0:n / 2j]
    x3 = np.mgrid[80:60:n / 6j]
    y3 = np.mgrid[90:100:n / 6j]

    x = np.r_[x1, x2, x3]
    y = np.r_[y1, y2, y3]
    x += np.random.normal(size=x.shape) * 4
    # y += np.random.normal(size=y.shape) * 4
    return np.c_[x, y]


def example2():
    x1 = np.mgrid[1:9:100j]
    y1 = np.mgrid[-5:3:100j]
    x1 += np.random.normal(size=x1.shape) * 4
    return np.c_[x1, y1]
