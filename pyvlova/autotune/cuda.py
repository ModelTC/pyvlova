# Copyright 2020 Jiang Shenghu
# SPDX-License-Identifier: Apache-2.0
from functools import reduce
from typing import Tuple
import os

import tvm
import numpy
from tvm import autotvm

from .settings import default_tune_eval_settings
from ..codegen import lower_tvm_stmt
from ..poly import cuda_tile, ScheduleTree
from ..utils import load_best, slugify, get_unnamed_tuples, cuda_settings


class CUDATileConfigEntity(list):
    def __init__(self, index, total, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index = index
        self.total = total

    def valid(self):
        return True

    def to_json_dict(self):
        return {'index': str(self.index), 'total': str(self.total), 'tile': list(map(str, self))}

    @staticmethod
    def from_json_dict(d):
        return CUDATileConfigEntity(int(d['index']), int(d['total']), list(map(int, d['tile'])))

    def get_flatten_feature(self):
        return numpy.array(list(self), dtype=numpy.float32)

    def get_other_option(self):
        return {}


class CUDATileConfigSpace(object):
    def __init__(self, n, b):
        self.n = n
        self.b = list(b) + [n]
        self.dim = len(self.b) - 1

        self.a = []
        self.al = []
        cur = 1
        while cur <= n:
            self.a.append(cur)
            self.al.append(n // cur - n // (cur + 1))
            if cur >= n:
                break
            cur = n // (n // (cur + 1))
        self.m = len(self.a)
        self.ra = {self.a[i]: i for i in range(self.m)}

        self.ft = [[0 for _ in range(self.m)] for _ in range(self.dim)]
        self.gt = [[[0 for _ in range(self.m)] for _ in range(self.m)] for _ in range(self.dim)]
        for i in range(self.dim):
            for j in range(self.m):
                self.ft[i][j] = self._f(i, self.a[j])
        for i in range(self.dim):
            for j in range(self.m):
                for k in range(self.m):
                    self.gt[i][j][k] = self.g(i, self.a[j], self.a[k])

    @property
    def space_map(self):
        return {'tile': self}

    def get(self, index):
        return CUDATileConfigEntity(index, len(self), self[index])

    def __len__(self):
        return self.size()

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, i):
        return self.rev(i)

    def __contains__(self, i):
        return 0 <= i < len(self)

    def size(self):
        return self.f(self.dim - 1, self.n)

    def rank(self, x):
        x = list(x)
        assert len(x) == self.dim
        r = 0
        n = self.n
        for i in range(self.dim - 1, -1, -1):
            r += self.g(i, n, x[i] - 1)
            n //= x[i]
        return r

    def rev(self, x):
        res = [0] * self.dim
        c = x
        n = self.n
        for i in range(self.dim - 1, -1, -1):
            left, right = 1, min(n, self.b[i]) + 1
            while right - left > 1:
                mid = (left + right) // 2
                if self.g(i, n, mid - 1) <= c:
                    left = mid
                else:
                    right = mid
            res[i] = left
            c -= self.g(i, n, left - 1)
            n //= left
        return res

    def g(self, k, n, x):
        assert n in self.ra
        if k < 0 or n <= 0 or x <= 0:
            return 0
        if k <= 0:
            return min(n, self.b[0], x)
        if x <= 1:
            return self.f(k - 1, n)
        q = n // (n // x + 1)
        r = x - q
        return self.gt[k][self.ra[n]][self.ra[q]] + r * self.f(k - 1, n // x)

    def _f(self, k, n):
        if k < 0 or n <= 0:
            return 0
        if k <= 0:
            return min(n, self.b[0])
        bd = self.b[k]
        res, cur = 0, 1
        while cur <= n:
            right = min(n // cur, bd)
            left = n // (cur + 1)
            if right - left >= 1:
                res += (right - left) * self.f(k - 1, cur)
            if cur >= n:
                break
            cur = n // (n // (cur + 1))
        return res

    def bf(self, k, n):
        if k < 0 or n <= 0:
            return 0
        if k <= 0:
            return min(n, self.b[0])
        res = 0
        for i in range(1, min(self.b[k], n) + 1):
            res += self.bf(k - 1, n // i)
        return res

    def f(self, k, n):
        if 0 <= k < self.dim and n in self.ra:
            return self.ft[k][self.ra[n]]
        return self._f(k, n)


class CUDATileTask(autotvm.task.Task):
    def __init__(self, name, tree, kernel_args, parser):
        autotvm.task.template(name, lambda: None)
        super().__init__(name, [])
        self.func = None
        self.tree = tree
        self.kernel_args = kernel_args
        self.parser = parser
        box_size, lower, stride = tree.outermost_band_box()
        band_size = [-(-i // j) for i, j in zip(box_size, stride)]
        self.config_space = CUDATileConfigSpace(cuda_settings['max_threads'], band_size)
        self.target = tvm.target.create('cuda')
        self.flop = 0
        self._init_flop()

    def _init_flop(self):
        self.flop = 0
        # TODO: better flop prediction
        domain = self.tree.domain()
        stmt_instances = list()
        domain.foreach_set(stmt_instances.append)
        box = list(map(lambda x: x.simple_fixed_box_hull().size(), stmt_instances))
        for i in box:
            b, *_ = get_unnamed_tuples(i)
            self.flop += reduce(float.__mul__, map(float, b))

    def instantiate(self, config):
        tree = self.tree.copy()
        cuda_tile(tree, config)
        stmt = self.parser.parse(tree)
        module = lower_tvm_stmt(stmt, self.kernel_args, name=self.name)
        return module, self.kernel_args


def tune_cuda_tile(name, tree, kernel_args, parser, n_trial=40,
                   tuner=None, measure_option=None, callbacks=None, preserve_log=False):
    tmp_file_name = slugify(name) + '.cuda_tile.log'
    task = CUDATileTask(name, tree.copy(), kernel_args, parser)
    from random import randint
    stmt, args = task.instantiate(task.config_space.get(randint(0, len(task.config_space) - 1)))
    kernel = tvm.build(stmt, name=name, target='cuda')

    if n_trial > 0:
        if tuner is None:
            tuner = autotvm.tuner.XGBTuner(task, feature_type='knob')
        else:
            tuner = tuner(task)

        tuner.tune(
            n_trial=n_trial,
            measure_option={
                'builder': autotvm.LocalBuilder(),
                'runner': autotvm.LocalRunner(timeout=20, **default_tune_eval_settings),
                **(measure_option or {}),
            },
            callbacks=[
                autotvm.callback.progress_bar(n_trial, prefix=f'CUDATile {name}'),
                autotvm.callback.log_to_file(tmp_file_name),
                *(callbacks or [])
            ]
        )

    best, best_cost = load_best(tmp_file_name, task)
    
    if not best:
        raise Exception('failed to build kernel')

    best = CUDATileConfigEntity.from_json_dict(best)

    print('CUDATile %s: best %s, best cost %.12f' % (name, repr(best), best_cost))

    if not preserve_log:
        os.remove(tmp_file_name)

    return best, best_cost
