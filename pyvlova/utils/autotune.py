# Copyright 2020 Jiang Shenghu
# SPDX-License-Identifier: Apache-2.0
import json
import numpy


def load_best(file_name, task):
    best, best_cost = None, None
    with open(file_name) as f:
        for line in f:
            if line.startswith('#'):
                continue
            row = json.loads(line)
            if row['input'][1] == task.name:
                costs, error_no, *_ = row['result']
                if error_no:
                    continue
                mean_cost = float(numpy.mean(costs))
                if best_cost is None or best_cost > mean_cost:
                    best, best_cost = row['config'], mean_cost
    return best, best_cost
