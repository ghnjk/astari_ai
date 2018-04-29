#!/usr/bin/python
# -*- coding: UTF-8 -*-

import random
from collections import deque
import numpy as np


class Memory(object):
    def __init__(self):
        pass

    def append(self, item):
        pass

    def choose_sample(self, sample_count):
        pass

    def data_count(self):
        pass


class RandomMemory(Memory):
    def __init__(self, memory_size):
        super(RandomMemory, self).__init__()
        self.memory_size = memory_size
        self.memory = deque(maxlen=self.memory_size)

    def append(self, item):
        self.memory.append(item)

    def choose_sample(self, sample_count):
        sample_count = sample_count if sample_count < len(self.memory) else len(self.memory)
        idx = random.sample(range(len(self.memory)), sample_count)
        sample = []
        for i in idx:
            sample.append(self.memory[i])
        return sample

    def data_count(self):
        return len(self.memory)


class SumTree(object):
    def __init__(self, memory_size):
        self.data_pointer = 0
        self.memory_size = memory_size
        self.tree = np.zeros(2 * self.memory_size - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: memory_size - 1                       size: memory_size
        self.data = np.zeros(self.memory_size, dtype=object)
        # [--------------data frame-------------]
        #             size: memory_size

    def add(self, p, data):
        tree_idx = self.data_pointer + self.memory_size - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, p)
        self.data_pointer += 1
        if self.data_pointer >= self.memory_size:
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) / 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]

        """
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx
        data_idx = leaf_idx - self.memory_size + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total_p(self):
        return self.tree[0]


class SumTreeMemory(Memory):
    def __init__(self, memory_size):
        super(Memory, self).__init__()
        self.tree = SumTree(memory_size)
        self.max_abs_error = 1
        self.beta = 0.4  # importance-sampling, from initial value increasing to 1
        self.beta_increment_per_sampling = 0.001
        self.epsilon = 0.01  # small amount to avoid zero priority
        self.alpha = 0.6  # [0~1] convert the importance of TD error to priority
        self._data_count = 0

    def append(self, item):
        max_p = np.max(self.tree.tree[-self.tree.memory_size:])
        if max_p == 0:
            max_p = self.max_abs_error
        self.tree.add(max_p, item)  # set the max p for new p
        self._data_count = np.minimum(self._data_count + 1, self.tree.memory_size)

    def choose_sample(self, sample_count):
        b_idx = np.empty((sample_count,), dtype=np.int32)
        b_weights = np.empty((sample_count, 1))
        b_memory = []
        p_seg = self.tree.total_p() / sample_count
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1
        min_prob = np.min(self.tree.tree[-self.tree.memory_size:]) / self.tree.total_p()
        for i in range(sample_count):
            #  优先级区间[a,b]
            a, b = p_seg * i, p_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p()
            b_weights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i] = idx
            b_memory.append(data)
        return b_idx, b_memory, b_weights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.max_abs_error)
        ps = np.power(clipped_errors, self.alpha)
        for idx, p in zip(tree_idx, ps):
            self.tree.update(idx, p)

    def data_count(self):
        return self._data_count
