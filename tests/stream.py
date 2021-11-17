"""
Online coresets.
"""

from collections import namedtuple
import numpy as np
from stack import Stack

StackItem = namedtuple("StackItem", "coreset level")
WeightedPointSet = namedtuple("WeightedPointSet", "points weights")


class Stream(object):

    def __init__(self, coreset_alg, leaf_size, coreset_size, k):
        self.coreset_alg = coreset_alg
        self.leaf_size = leaf_size
        self.last_leaf = []
        self.coreset_size = coreset_size
        self.stack = Stack()
        self.coreset_size = coreset_size
        self.k = k


    def _merge(self, pset1, pset2):
        points = np.vstack([pset1.points, pset2.points])
        weights = np.hstack([pset1.weights, pset2.weights])
        cset = self.coreset_alg(points, self.k, weights)
        coreset, weights = cset.compute(self.coreset_size)
        return WeightedPointSet(coreset,weights)

    def _add_leaf(self, points, weights):
        if weights is None:
            weights = np.ones((points.shape[0])).ravel()
        self._insert_into_tree(WeightedPointSet(points, weights))

    def _is_correct_level(self, level):
        if self.stack.is_empty():
            return True
        elif self.stack.top().level > level:
            return True
        elif self.stack.top().level == level:
            return False
        else:
            raise Exception("New level should be smaller")

    def _insert_into_tree(self, coreset):
        level = 1
        while not self._is_correct_level(level):
            last = self.stack.pop()
            coreset = self._merge(last.coreset, coreset)
            level += 1
        self.stack.push(StackItem(coreset, level))
    def add_points(self, points):
        """Add a set of points to the stream.

        If the set is larger than leaf_size, it is split
        into several sets and a coreset is constructed on each set.
        """

        for split in np.array_split(points, self.leaf_size):
                self._add_leaf(split,None)

    def get_unified_coreset(self):
        solution = None
        while not self.stack.is_empty():
            coreset = self.stack.pop().coreset
            if solution is None:
                solution = coreset
            else:
                solution = self._merge(solution, coreset)
        return solution.points, solution.weights
